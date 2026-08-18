"""
03_offline_indexing.py — Offline Compression and FAISS Index Building
======================================================================
Compresses 10,000 ESM-2 protein embeddings using six FAISS strategies and
builds a FAISS index for each. Records build time and index size.

Strategy 7 (AVQ / ScaNN) runs under WSL2/Linux — see 03b_offline_indexing_avq.py.

Compression strategies:
  1. Mean Pooling (Baseline)          — no compression, 1280-dim FlatL2
  2. PCA + FlatL2                     — sklearn PCA, 64-dim FlatL2
     Math: tilde_x = x @ V_d  (top-64 eigenvectors of sample covariance)
     Jolliffe (2002), Principal Component Analysis, Springer.
  3. Product Quantization (PQ)        — FAISS IndexPQ, 1280-dim quantized
  4. DCT Fingerprinting               — SciPy DCT, 64-dim FlatL2
     Math: X_k = sqrt(2/N) * sum_n x_n * cos[pi/N * (n+0.5) * k]
     Adapted from: Iovino, Tang & Ye (2024), Genome Research
  5. TEA Alphabet                     — K-Means Vector Quantization, 20-dim histogram
     Math: t_i = argmin_k ||x_i - c_k||_2  (Pantolini et al., 2026)
     Trains KMeans on all database residues; saves codebook to disk.
  6. Knowledge Distillation (BiLSTM)  — Student model, 256-dim FlatL2
     Math: L_MSE = (1/B) * sum_i ||f_T(S_i) - teacher_head(f_S(S_i))||_2^2
     Adapted from: Keluskar et al. (2026), TM-Vec 2 / TM-Vec 2s
     Trains BiLSTM student via MSE against ESM-2 mean-pooled targets;
     saves trained weights to disk.

  NOTE: Strategy 7 — Anisotropic Vector Quantisation (AVQ) — is implemented
  in a separate script (03b_offline_indexing_avq.py) running under WSL2/Linux
  because ScaNN does not support Windows natively. The mean-pooled vectors
  produced by this script (data/mean_pooled_10k.npy) are shared as AVQ input.

Outputs
-------
  data/mean_pooled_10k.npy     — Mean-pooled float32 vectors (shared with AVQ script)
  data/pca_model.pkl           — Fitted sklearn PCA model (64 components)
  data/indices/*.index         — FAISS index files for online evaluation
  data/tea_kmeans_codebook.pkl — Trained TEA K-Means codebook (20 centroids)
  data/kd_student.pt           — Trained KD BiLSTM student model weights
  results/indexing_results.csv — Build time and index size per strategy
  results/total_build_time.txt — Total wall-clock time for all builds
"""

import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import faiss
import joblib
from scipy.fft import dct
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR  = os.path.dirname(SRC_DIR)
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
INDICES_DIR = os.path.join(DATA_DIR, "indices")

INPUT_FILE        = os.path.join(DATA_DIR, "raw_embeddings_10k.pt")
OUTPUT_CSV        = os.path.join(RESULTS_DIR, "indexing_results.csv")
TOTAL_TIME_FILE   = os.path.join(RESULTS_DIR, "total_build_time.txt")
KD_MODEL_PATH     = os.path.join(DATA_DIR, "kd_student.pt")
TEA_CODEBOOK_PATH = os.path.join(DATA_DIR, "tea_kmeans_codebook.pkl")
# PCA artefacts — also shared with AVQ script via mean_pooled_10k.npy
PCA_MODEL_PATH    = os.path.join(DATA_DIR, "pca_model.pkl")
MEAN_POOLED_NPY   = os.path.join(DATA_DIR, "mean_pooled_10k.npy")  # shared with 03b

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(INDICES_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# KD training hyper-parameters
KD_HIDDEN_DIM  = 128
KD_OUT_DIM     = 256   # compact embedding dimension
KD_N_EPOCHS    = 15
KD_LR          = 1e-3
KD_MAX_SEQ_LEN = 512   # truncate to avoid very long sequences

# TEA hyper-parameters
TEA_N_CLUSTERS        = 20
TEA_MAX_RESIDUES      = 500_000  # cap to keep KMeans tractable
TEA_BATCH_SIZE_KMEANS = 10_000

# PCA hyper-parameters
PCA_N_COMPONENTS = 64   # matches DCT output dim for direct comparability

# DCT hyper-parameters
DCT_N_COEFFS = 64


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def save_index_and_get_size(index_obj, filename):
    """Write FAISS index to disk; return its size in GB."""
    filepath = os.path.join(INDICES_DIR, filename)
    faiss.write_index(index_obj, filepath)
    return os.path.getsize(filepath) / (1024 ** 3)


# ---------------------------------------------------------------------------
# Strategy 5 — Knowledge Distillation student model
# (architecture mirrors TM-Vec 2s: Keluskar et al., 2026)
# ---------------------------------------------------------------------------
class StudentBiLSTM(nn.Module):
    """
    Bidirectional LSTM student model for knowledge distillation.

    The model maps variable-length per-residue ESM-2 embeddings to a
    compact fixed-size vector (out_dim=256) by concatenating the final
    hidden states of both LSTM directions.

    A `teacher_head` linear layer projects the compact vector back to
    1280-dim during training so that MSE loss against ESM-2 mean-pooled
    targets can be computed.  The teacher_head is saved alongside the
    student weights but is NOT used during index building or online
    querying — only the `compact` output is used.

    Training objective (Keluskar et al., 2026):
        L_MSE(θ) = (1/B) Σ_i ||f_T(S_i) - teacher_head(f_S(S_i; θ))||²
    """
    def __init__(self,
                 input_dim=1280,
                 hidden_dim=KD_HIDDEN_DIM,
                 out_dim=KD_OUT_DIM,
                 teacher_dim=1280):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            batch_first=True, bidirectional=True)
        self.fc           = nn.Linear(hidden_dim * 2, out_dim)
        self.teacher_head = nn.Linear(out_dim, teacher_dim)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        _, (hn, _) = self.lstm(x)
        # Concatenate final hidden states from both directions
        pooled      = torch.cat((hn[-2], hn[-1]), dim=1)   # [batch, hidden*2]
        compact     = self.fc(pooled)                        # [batch, out_dim]
        pred_teacher = self.teacher_head(compact)            # [batch, teacher_dim]
        return compact, pred_teacher


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")
    print(f"Loading embeddings from {INPUT_FILE}...")
    embeddings = torch.load(INPUT_FILE, weights_only=False)
    n = len(embeddings)
    print(f"Loaded {n} per-residue embedding tensors.")

    results = []

    # -----------------------------------------------------------------------
    # Pre-computation: Mean Pooling (used by strategies 1, 3, and KD targets)
    # -----------------------------------------------------------------------
    print("\n[Pre-compute] Mean Pooling all sequences...")
    mean_pooled = np.vstack([
        emb.mean(dim=0).numpy() for emb in embeddings
    ]).astype(np.float32)
    dim = mean_pooled.shape[1]  # 1280
    print(f"  → mean_pooled matrix: {mean_pooled.shape}")

    # Save mean-pooled matrix for AVQ script (03b_offline_indexing_avq.py)
    np.save(MEAN_POOLED_NPY, mean_pooled)
    print(f"  ✓ mean_pooled_10k.npy saved → {MEAN_POOLED_NPY}  (shared with AVQ script)")

    # -----------------------------------------------------------------------
    # Pre-computation: Fit PCA on full database embeddings
    # (Jolliffe, 2002 — eigendecomposition of sample covariance)
    # tilde_x = x @ V_d,  V_d = top-64 eigenvectors of (1/N) X^T X
    # -----------------------------------------------------------------------
    print(f"\n[Pre-compute] Fitting PCA (n_components={PCA_N_COMPONENTS})...")
    pca = PCA(n_components=PCA_N_COMPONENTS, random_state=42)
    pca.fit(mean_pooled)
    joblib.dump(pca, PCA_MODEL_PATH)
    pca_vectors = pca.transform(mean_pooled).astype(np.float32)
    print(f"  Explained variance (first 5 components): "
          f"{pca.explained_variance_ratio_[:5].round(4)}")
    print(f"  Cumulative explained variance (64 components): "
          f"{pca.explained_variance_ratio_.sum():.4f}")
    print(f"  ✓ PCA model saved → {PCA_MODEL_PATH}")

    # -----------------------------------------------------------------------
    # Pre-computation: Train TEA K-Means codebook on database residues
    # (Pantolini et al., 2026)
    # Vector Quantization: t_i = argmin_k ||x_i - c_k||_2
    # -----------------------------------------------------------------------
    print("\n[Pre-compute] Training TEA K-Means codebook on database residues...")
    print(f"  Collecting residues from all {n} sequences "
          f"(max {TEA_MAX_RESIDUES:,} total)...")

    all_residues = np.vstack([emb.numpy() for emb in embeddings])
    if len(all_residues) > TEA_MAX_RESIDUES:
        rng = np.random.default_rng(seed=22)
        idx = rng.choice(len(all_residues), TEA_MAX_RESIDUES, replace=False)
        sample = all_residues[idx]
    else:
        sample = all_residues

    print(f"  Training MiniBatchKMeans (k={TEA_N_CLUSTERS}) "
          f"on {len(sample):,} residues...")
    kmeans = MiniBatchKMeans(
        n_clusters=TEA_N_CLUSTERS,
        random_state=42,
        n_init=5,
        batch_size=TEA_BATCH_SIZE_KMEANS,
        verbose=0
    )
    kmeans.fit(sample)
    joblib.dump(kmeans, TEA_CODEBOOK_PATH)
    print(f"  ✓ TEA codebook saved → {TEA_CODEBOOK_PATH}")

    # -----------------------------------------------------------------------
    # Pre-computation: Train KD BiLSTM student model
    # (Keluskar et al., 2026 — TM-Vec 2s distillation)
    # -----------------------------------------------------------------------
    print(f"\n[Pre-compute] Training KD BiLSTM student (Max {KD_N_EPOCHS} epochs with Early Stopping)...")
    student = StudentBiLSTM().to(DEVICE)
    optimizer = torch.optim.Adam(student.parameters(), lr=KD_LR)
    criterion = nn.MSELoss()
    targets   = torch.tensor(mean_pooled, dtype=torch.float32).to(DEVICE)

    # 90-10 Train/Validation Split for Early Stopping
    np.random.seed(42)
    indices = np.random.permutation(n)
    split_idx = int(0.9 * n)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in range(KD_N_EPOCHS):
        student.train()
        train_loss = 0.0
        for i in train_idx:
            # Truncate to KD_MAX_SEQ_LEN residues to bound compute
            seq = embeddings[i][:KD_MAX_SEQ_LEN].unsqueeze(0).to(DEVICE)  # [1, L, 1280]
            target = targets[i].unsqueeze(0)                      # [1, 1280]
            optimizer.zero_grad()
            _, pred_teacher = student(seq)
            loss = criterion(pred_teacher, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train = train_loss / len(train_idx)

        # Validation loop
        student.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in val_idx:
                seq = embeddings[i][:KD_MAX_SEQ_LEN].unsqueeze(0).to(DEVICE)
                target = targets[i].unsqueeze(0)
                _, pred_teacher = student(seq)
                loss = criterion(pred_teacher, target)
                val_loss += loss.item()
        avg_val = val_loss / len(val_idx)

        print(f"  Epoch {epoch+1:>2}/{KD_N_EPOCHS}  Train MSE: {avg_train:.6f}  Val MSE: {avg_val:.6f}")

        # Early Stopping logic
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(student.state_dict(), KD_MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [Early Stopping] Validation loss did not improve for {patience} epochs. Stopping early!")
                break

    # Load the best weights from disk for the remainder of the script
    student.load_state_dict(torch.load(KD_MODEL_PATH, weights_only=True))
    print(f"  ✓ Best KD student weights saved → {KD_MODEL_PATH}")

    student.eval()

    # -----------------------------------------------------------------------
    # Generate compressed vectors using the trained student
    # -----------------------------------------------------------------------
    print("\n[KD] Generating distilled 256-dim vectors for all sequences...")
    distilled_vectors = []
    with torch.no_grad():
        for emb in embeddings:
            seq = emb[:KD_MAX_SEQ_LEN].unsqueeze(0).to(DEVICE)
            compact, _ = student(seq)
            distilled_vectors.append(compact.squeeze(0).cpu().numpy())
    distilled_vectors = np.vstack(distilled_vectors).astype(np.float32)

    # -----------------------------------------------------------------------
    # Strategy 1 — Mean Pooling (FlatL2 Baseline)
    # -----------------------------------------------------------------------
    print("\n[1/6] Building Mean Pooling (FlatL2) index...")
    t0 = time.time()
    index_flat = faiss.IndexFlatL2(dim)
    index_flat.add(mean_pooled)
    build_time = time.time() - t0
    size_gb = save_index_and_get_size(index_flat, "mean_pool.index")
    results.append({
        "Method": "Mean Pooling (FlatL2)",
        "Build Time (s)": build_time,
        "Index Size (GB)": size_gb,
    })
    print(f"  ✓ {build_time:.2f}s  |  {size_gb:.6f} GB")

    # -----------------------------------------------------------------------
    # Strategy 2 — PCA + FlatL2
    # Closed-form linear projection; zero codebook; variance-ordered truncation.
    # Control method: any accuracy ceiling reveals the limit of dimensionality
    # reduction alone, with no confounds from training or quantisation.
    # (Jolliffe, 2002 — Principal Component Analysis)
    # -----------------------------------------------------------------------
    print(f"\n[2/6] Building PCA + FlatL2 index (d={PCA_N_COMPONENTS})...")
    t0 = time.time()
    index_pca = faiss.IndexFlatL2(PCA_N_COMPONENTS)
    index_pca.add(pca_vectors)
    build_time = time.time() - t0
    size_gb = save_index_and_get_size(index_pca, "pca.index")
    results.append({
        "Method": "PCA + FlatL2",
        "Build Time (s)": build_time,
        "Index Size (GB)": size_gb,
    })
    print(f"  ✓ {build_time:.2f}s  |  {size_gb:.6f} GB")

    # -----------------------------------------------------------------------
    # Strategy 3 — Product Quantization (PQ)
    # (FAISS IndexPQ; Fahmy et al., 2026)
    # -----------------------------------------------------------------------
    print("\n[3/6] Building Product Quantization (PQ) index...")
    m     = 32   # number of sub-quantizers
    nbits = 8    # bits per sub-quantizer
    t0 = time.time()
    index_pq = faiss.IndexPQ(dim, m, nbits)
    index_pq.train(mean_pooled)
    index_pq.add(mean_pooled)
    build_time = time.time() - t0
    size_gb = save_index_and_get_size(index_pq, "pq.index")
    results.append({
        "Method": "Product Quantization (PQ)",
        "Build Time (s)": build_time,
        "Index Size (GB)": size_gb,
    })
    print(f"  ✓ {build_time:.2f}s  |  {size_gb:.6f} GB")

    # -----------------------------------------------------------------------
    # Strategy 3 — DCT Fingerprinting
    # (Iovino, Tang & Ye, 2024; Type-II orthogonal DCT via SciPy)
    # X_k = sqrt(2/N) * Σ_n x_n * cos[π/N * (n+0.5) * k],  k=0..K-1
    # -----------------------------------------------------------------------
    print(f"\n[4/6] Building DCT Fingerprinting index (K={DCT_N_COEFFS})...")
    t0 = time.time()
    dct_vectors = []
    for emb in embeddings:
        seq_1d = emb.numpy().mean(axis=1)          # collapse feature dim → [L]
        coeffs = dct(seq_1d, norm='ortho')          # orthogonal Type-II DCT
        if len(coeffs) >= DCT_N_COEFFS:
            feat = coeffs[:DCT_N_COEFFS]
        else:
            feat = np.pad(coeffs, (0, DCT_N_COEFFS - len(coeffs)))
        dct_vectors.append(feat)
    dct_vectors = np.vstack(dct_vectors).astype(np.float32)
    index_dct = faiss.IndexFlatL2(DCT_N_COEFFS)
    index_dct.add(dct_vectors)
    build_time = time.time() - t0
    size_gb = save_index_and_get_size(index_dct, "dct.index")
    results.append({
        "Method": "DCT Fingerprinting",
        "Build Time (s)": build_time,
        "Index Size (GB)": size_gb,
    })
    print(f"  ✓ {build_time:.2f}s  |  {size_gb:.6f} GB")

    # -----------------------------------------------------------------------
    # Strategy 4 — TEA Alphabet (Vector Quantization)
    # (Pantolini et al., 2026)
    # t_i = argmin_k ||x_i - c_k||_2  → 20-bin histogram per protein
    # -----------------------------------------------------------------------
    print(f"\n[5/6] Building TEA Alphabet index (k={TEA_N_CLUSTERS} tokens)...")
    t0 = time.time()
    tea_vectors = []
    for emb in embeddings:
        labels = kmeans.predict(emb.numpy())                      # one label per residue
        hist, _ = np.histogram(labels, bins=TEA_N_CLUSTERS,
                               range=(0, TEA_N_CLUSTERS))
        tea_vectors.append(hist.astype(np.float32))
    tea_vectors = np.vstack(tea_vectors).astype(np.float32)
    faiss.normalize_L2(tea_vectors)                               # L2 normalise histograms
    index_tea = faiss.IndexFlatL2(TEA_N_CLUSTERS)
    index_tea.add(tea_vectors)
    build_time = time.time() - t0
    size_gb = save_index_and_get_size(index_tea, "tea.index")
    results.append({
        "Method": "TEA Alphabet",
        "Build Time (s)": build_time,
        "Index Size (GB)": size_gb,
    })
    print(f"  ✓ {build_time:.2f}s  |  {size_gb:.6f} GB")

    # -----------------------------------------------------------------------
    # Strategy 5 — Knowledge Distillation (BiLSTM student)
    # (Keluskar et al., 2026 — TM-Vec 2s)
    # Uses trained student weights (already computed above)
    # -----------------------------------------------------------------------
    print(f"\n[6/6] Building Knowledge Distillation index ({KD_OUT_DIM}-dim)...")
    t0 = time.time()
    index_kd = faiss.IndexFlatL2(KD_OUT_DIM)
    index_kd.add(distilled_vectors)
    build_time = time.time() - t0
    size_gb = save_index_and_get_size(index_kd, "kd.index")
    results.append({
        "Method": "Knowledge Distillation (BiLSTM)",
        "Build Time (s)": build_time,
        "Index Size (GB)": size_gb,
    })
    print(f"  ✓ {build_time:.2f}s  |  {size_gb:.6f} GB")

    # -----------------------------------------------------------------------
    # Save results CSV
    # -----------------------------------------------------------------------
    print(f"\nSaving indexing results → {OUTPUT_CSV}")
    fieldnames = ["Method", "Build Time (s)", "Index Size (GB)"]
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    total_time = sum(r["Build Time (s)"] for r in results)
    with open(TOTAL_TIME_FILE, 'w') as f:
        f.write(f"Total Index Build Time: {total_time:.2f} seconds\n")

    print("\n=== Indexing Results ===")
    print(f"{'Method':<35} {'Build Time (s)':>15} {'Index Size (GB)':>16}")
    print("-" * 68)
    for r in results:
        print(f"{r['Method']:<35} {r['Build Time (s)']:>15.2f} {r['Index Size (GB)']:>16.6f}")
    print(f"\nTotal Build Time: {total_time:.2f}s")
    print(f"\nArtifacts saved:")
    print(f"  mean_pooled_10k.npy → {MEAN_POOLED_NPY}  ← input for 03b_offline_indexing_avq.py")
    print(f"  PCA model     → {PCA_MODEL_PATH}")
    print(f"  TEA codebook  → {TEA_CODEBOOK_PATH}")
    print(f"  KD weights    → {KD_MODEL_PATH}")
    print(f"  FAISS indices → {INDICES_DIR}/")
    print(f"\n  Next step (WSL2): python 03b_offline_indexing_avq.py")


if __name__ == "__main__":
    main()
