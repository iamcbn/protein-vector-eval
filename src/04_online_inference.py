"""
04_online_inference.py — Online Query Evaluation and Benchmarking
=================================================================
Runs 1,000 held-out query proteins against all six FAISS indices
and computes the two ground-truth metrics agreed in the methodology:

  Metric 1 — ANN Recall@10 (Mathematical ground truth)
      For each query, the exact Top-10 from the uncompressed
      FlatL2 index defines the ground truth.  ANN Recall@10 is
      the fraction of queries where the compressed index returns
      at least one of those same Top-10 proteins.
      Protocol: ann-benchmarks standard (Aumuller et al., 2020).

  Metric 2 — Biological Accuracy@1 (Biological ground truth)
      Ground truth: SCOPe 2.08 Superfamily co-membership, parsed
      directly from the FASTA headers (zero extra compute).
      A query is successful if its Top-1 result belongs to the
      same SCOPe Superfamily as the query.
      Precedent: TM-Vec (Hamamsy et al., 2022), which uses SCOP
      Superfamily labels for retrieval evaluation at scale.

Latency measured as: query compression time + FAISS search time
(ESM-2 inference excluded per methodology §3.7).

AVQ (ScaNN) is evaluated in a separate script — 04b_online_inference_avq.py —
running under WSL2/Linux.  This script serialises the shared data files
that the AVQ script depends on:
  data/mean_pooled_queries.npy   — query mean-pooled vectors
  data/flatl2_ground_truth.pkl   — FlatL2 top-10 sets per query
  data/scope_labels.pkl          — SCOPe label dicts for db and queries

Prerequisite files (produced by the offline pipeline):
  data/scope_10k_subset.fasta       — database FASTA with SCOPe headers
  data/queries_1000.fasta           — query FASTA with SCOPe headers
  data/raw_embeddings_10k.pt        — database per-residue embeddings
  data/pca_model.pkl                — fitted PCA model (64 components)
  data/tea_kmeans_codebook.pkl      — trained TEA K-Means codebook
  data/kd_student.pt                — trained KD BiLSTM student weights
  data/indices/mean_pool.index      — FlatL2 exact baseline index
  data/indices/pca.index            — PCA + FlatL2 index
  data/indices/pq.index             — Product Quantization index
  data/indices/dct.index            — DCT Fingerprinting index
  data/indices/tea.index            — TEA Alphabet index
  data/indices/kd.index             — Knowledge Distillation index

Outputs:
  data/raw_embeddings_queries.pt    — query embeddings (generated if absent)
  data/mean_pooled_queries.npy      — query mean-pooled vectors (shared with AVQ script)
  data/flatl2_ground_truth.pkl      — FlatL2 top-10 ground truth (shared with AVQ script)
  data/scope_labels.pkl             — SCOPe label dicts (shared with AVQ script)
  results/benchmark_results.csv     — full benchmark results table (averages)
  results/raw_latencies.csv         — per-query latency (ms) for all 6 methods
  results/raw_ann_hits.csv          — per-query ANN hit/miss (0/1) for all 6 methods
  results/raw_bio_hits.csv          — per-query Bio hit/miss (0/1) for all 6 methods
"""

import os
import csv
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
import faiss
import joblib
from transformers import EsmModel, EsmTokenizer
from scipy.fft import dct as scipy_dct

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR  = os.path.dirname(SRC_DIR)
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
INDICES_DIR = os.path.join(DATA_DIR, "indices")

FASTA_DB          = os.path.join(DATA_DIR, "scope_10k_subset.fasta")
FASTA_QUERIES     = os.path.join(DATA_DIR, "queries_1000.fasta")
EMBEDDINGS_DB     = os.path.join(DATA_DIR, "raw_embeddings_10k.pt")
EMBEDDINGS_QUERIES = os.path.join(DATA_DIR, "raw_embeddings_queries.pt")
TEA_CODEBOOK_PATH = os.path.join(DATA_DIR, "tea_kmeans_codebook.pkl")
KD_MODEL_PATH     = os.path.join(DATA_DIR, "kd_student.pt")
PCA_MODEL_PATH    = os.path.join(DATA_DIR, "pca_model.pkl")
# Files shared with the AVQ WSL2 script (04b_online_inference_avq.py)
MEAN_POOLED_QUERIES_NPY = os.path.join(DATA_DIR, "mean_pooled_queries.npy")
FLATL2_GT_PKL           = os.path.join(DATA_DIR, "flatl2_ground_truth.pkl")
SCOPE_LABELS_PKL        = os.path.join(DATA_DIR, "scope_labels.pkl")
OUTPUT_CSV           = os.path.join(RESULTS_DIR, "benchmark_results.csv")
RAW_LATENCIES_CSV    = os.path.join(RESULTS_DIR, "raw_latencies.csv")
RAW_ANN_CSV          = os.path.join(RESULTS_DIR, "raw_ann_hits.csv")
RAW_BIO_CSV          = os.path.join(RESULTS_DIR, "raw_bio_hits.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_NAME   = "facebook/esm2_t33_650M_UR50D"
BATCH_SIZE   = 4
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K        = 10
DCT_N_COEFFS = 64
PCA_N_COMPONENTS = 64
KD_OUT_DIM   = 256
KD_HIDDEN    = 128
KD_MAX_LEN   = 512
TEA_CLUSTERS = 20


# ---------------------------------------------------------------------------
# Knowledge Distillation student model
# Must match the architecture defined in 03_offline_indexing.py exactly
# ---------------------------------------------------------------------------
class StudentBiLSTM(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=KD_HIDDEN,
                 out_dim=KD_OUT_DIM, teacher_dim=1280):
        super().__init__()
        self.lstm         = nn.LSTM(input_dim, hidden_dim,
                                    batch_first=True, bidirectional=True)
        self.fc           = nn.Linear(hidden_dim * 2, out_dim)
        self.teacher_head = nn.Linear(out_dim, teacher_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        pooled      = torch.cat((hn[-2], hn[-1]), dim=1)
        compact     = self.fc(pooled)
        pred_teacher = self.teacher_head(compact)
        return compact, pred_teacher


# ---------------------------------------------------------------------------
# FASTA parsing
# ---------------------------------------------------------------------------
def parse_fasta_with_headers(filename):
    """
    Return list of (header_line, sequence_string) tuples in file order.

    BUG FIX: SCOPe ASTRAL sequences are all-lowercase. ESM-2 tokenizer only
    recognises uppercase amino acid codes. .upper() is applied to all sequences.
    """
    entries, header, seq_parts = [], None, []
    with open(filename, 'r') as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('>'):
                if header is not None:
                    entries.append((header, ''.join(seq_parts).upper()))
                header, seq_parts = line, []
            else:
                seq_parts.append(line)
    if header is not None:
        entries.append((header, ''.join(seq_parts).upper()))
    return entries


def parse_scope_superfamily(header):
    """
    Extract the SCOPe Superfamily code (first 3 levels) from a FASTA header.

    SCOPe 2.08 ASTRAL header format:
        >d1dlwa_ a.1.1.1 (A:) Myoglobin {Sperm whale...}
                 ^^^^^^^
                 class.fold.superfamily.family

    Returns 'a.1.1' (superfamily = class.fold.superfamily).
    Returns None if the header cannot be parsed.
    """
    parts = header.lstrip('>').split()
    if len(parts) < 2:
        return None
    code   = parts[1]           # e.g. 'a.1.1.1'
    levels = code.split('.')
    if len(levels) < 3:
        return None
    return '.'.join(levels[:3]) # e.g. 'a.1.1'


# ---------------------------------------------------------------------------
# ESM-2 Inference
# ---------------------------------------------------------------------------
def generate_embeddings(sequences, model, tokenizer, device, batch_size=4):
    """
    Run ESM-2 inference and return a list of per-residue embedding tensors.
    Each tensor has shape [seq_len, 1280].
    CLS and EOS tokens are stripped.

    NOTE: sequences must already be uppercase (enforced by parse_fasta_with_headers).
    """
    all_embs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            # Sequences MUST be uppercase for ESM-2 tokenizer
            # (SCOPe ASTRAL FASTA is all-lowercase by default)
            batch = [s.upper() for s in batch]
            inputs = tokenizer(batch, return_tensors="pt",
                               padding=True, truncation=True,
                               max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            lhs  = outputs.last_hidden_state          # [B, L, 1280]
            mask = inputs['attention_mask']
            for j in range(len(batch)):
                valid_len = mask[j].sum().item()
                # strip <cls> (idx 0) and <eos> (last valid idx)
                emb = lhs[j, 1:valid_len - 1, :].cpu()
                all_embs.append(emb)
            if (i // batch_size) % 25 == 0:
                print(f"  Embedded {i + len(batch):>5} / {len(sequences)} sequences")
    return all_embs


# ---------------------------------------------------------------------------
# Compression helpers (must mirror 03_offline_indexing.py exactly)
# ---------------------------------------------------------------------------
def compress_mean_pool(emb):
    """Return the 1280-dim mean-pooled vector as float32 array."""
    return emb.mean(dim=0).numpy().astype(np.float32)


def compress_pca(emb, pca_model):
    """
    Project a mean-pooled ESM-2 embedding to PCA_N_COMPONENTS dimensions.
    Mirrors the pre-computation in 03_offline_indexing.py:
        tilde_x = x @ V_d  (top-64 eigenvectors of sample covariance)
    (Jolliffe, 2002)
    """
    mp = emb.mean(dim=0).numpy().astype(np.float32)
    return pca_model.transform(mp.reshape(1, -1)).squeeze(0).astype(np.float32)


def compress_dct(emb):
    """
    Apply orthogonal Type-II DCT along the sequence axis and keep the
    first DCT_N_COEFFS coefficients.  Mirrors 03_offline_indexing.py.
    """
    seq_1d = emb.numpy().mean(axis=1)           # [L]
    coeffs = scipy_dct(seq_1d, norm='ortho')
    if len(coeffs) >= DCT_N_COEFFS:
        feat = coeffs[:DCT_N_COEFFS]
    else:
        feat = np.pad(coeffs, (0, DCT_N_COEFFS - len(coeffs)))
    return feat.astype(np.float32)


def compress_tea(emb, kmeans):
    """
    Vector-quantise each residue against the K-Means codebook and build
    a normalised 20-bin histogram.  Mirrors 03_offline_indexing.py.
    """
    labels = kmeans.predict(emb.numpy())
    hist, _ = np.histogram(labels, bins=TEA_CLUSTERS, range=(0, TEA_CLUSTERS))
    v = hist.astype(np.float32)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v


def compress_kd(emb, student, device):
    """
    Pass per-residue embeddings through the trained BiLSTM student and
    return the 256-dim compact vector.  teacher_head output is discarded.
    """
    seq = emb[:KD_MAX_LEN].unsqueeze(0).to(device)  # [1, L, 1280]
    with torch.no_grad():
        compact, _ = student(seq)
    return compact.squeeze(0).cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Online Evaluation — ANN Recall@10 + Biological Accuracy@1")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Parse FASTA files and extract SCOPe Superfamily labels
    # ------------------------------------------------------------------
    print("\n[1/6] Parsing FASTA files and SCOPe labels...")
    db_entries    = parse_fasta_with_headers(FASTA_DB)
    query_entries = parse_fasta_with_headers(FASTA_QUERIES)

    db_labels    = {i: parse_scope_superfamily(h) for i, (h, _) in enumerate(db_entries)}
    query_labels = {i: parse_scope_superfamily(h) for i, (h, _) in enumerate(query_entries)}

    n_db      = len(db_entries)
    n_queries = len(query_entries)
    print(f"  Database : {n_db:,} proteins")
    print(f"  Queries  : {n_queries:,} proteins")

    # Sanity check: how many queries have a parseable SCOPe label?
    parseable = sum(1 for v in query_labels.values() if v is not None)
    print(f"  SCOPe labels parsed for {parseable}/{n_queries} query proteins")

    # ------------------------------------------------------------------
    # 2. Generate or load query embeddings
    # ------------------------------------------------------------------
    print("\n[2/6] Query embeddings...")
    if os.path.exists(EMBEDDINGS_QUERIES):
        print(f"  Loading cached embeddings from {EMBEDDINGS_QUERIES}")
        query_embeddings = torch.load(EMBEDDINGS_QUERIES, weights_only=False)
    else:
        print(f"  Embeddings not found. Running ESM-2 inference on {n_queries} queries.")
        print(f"  (This is a one-time operation; results will be cached.)")
        try:
            tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
            model     = EsmModel.from_pretrained(MODEL_NAME).to(DEVICE)
        except Exception as e:
            print(f"  [Network Error] Could not connect to HuggingFace ({e}).")
            print("  Attempting to load from local cache...")
            tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
            model     = EsmModel.from_pretrained(MODEL_NAME, local_files_only=True).to(DEVICE)

        query_seqs = [seq for _, seq in query_entries]
        query_embeddings = generate_embeddings(
            query_seqs, model, tokenizer, DEVICE, BATCH_SIZE
        )
        torch.save(query_embeddings, EMBEDDINGS_QUERIES)
        print(f"  ✓ Saved → {EMBEDDINGS_QUERIES}")
        del model  # free GPU memory

    # ------------------------------------------------------------------
    # 3. Compute FlatL2 exact baseline (ANN ground truth)
    # ------------------------------------------------------------------
    print("\n[3/6] Computing FlatL2 exact ground truth for all queries...")
    flatl2_index = faiss.read_index(os.path.join(INDICES_DIR, "mean_pool.index"))

    query_mean_pooled = np.vstack([
        emb.mean(dim=0).numpy().astype(np.float32)
        for emb in query_embeddings
    ])

    # flatl2_gt[i] = set of exact top-10 database indices for query i
    flatl2_gt = {}
    for i in range(n_queries):
        qv = query_mean_pooled[i].reshape(1, -1)
        _, I = flatl2_index.search(qv, TOP_K)
        flatl2_gt[i] = set(int(x) for x in I[0] if x >= 0)

    print(f"  ✓ FlatL2 ground truth computed for {n_queries} queries")

    # ------------------------------------------------------------------
    # 4. Load artefacts: PCA model, TEA codebook and KD student model
    # ------------------------------------------------------------------
    print("\n[4/6] Loading offline artefacts (PCA model + TEA codebook + KD student)...")
    pca_model = joblib.load(PCA_MODEL_PATH)
    kmeans    = joblib.load(TEA_CODEBOOK_PATH)
    student   = StudentBiLSTM().to(DEVICE)
    student.load_state_dict(torch.load(KD_MODEL_PATH,
                                       map_location=DEVICE,
                                       weights_only=True))
    student.eval()
    print("  ✓ PCA model, TEA codebook and KD student loaded")

    # ------------------------------------------------------------------
    # 5. Benchmarking loop — all 6 FAISS strategies
    # (AVQ / ScaNN evaluated separately in 04b_online_inference_avq.py)
    # ------------------------------------------------------------------
    strategies = [
        {
            "name":     "Mean Pooling (FlatL2)",
            "index":    "mean_pool.index",
            "compress": lambda emb: compress_mean_pool(emb),
        },
        {
            "name":     "PCA + FlatL2",
            "index":    "pca.index",
            "compress": lambda emb: compress_pca(emb, pca_model),
        },
        {
            "name":     "Product Quantization (PQ)",
            "index":    "pq.index",
            "compress": lambda emb: compress_mean_pool(emb),   # PQ encodes internally
        },
        {
            "name":     "DCT Fingerprinting",
            "index":    "dct.index",
            "compress": lambda emb: compress_dct(emb),
        },
        {
            "name":     "TEA Alphabet",
            "index":    "tea.index",
            "compress": lambda emb: compress_tea(emb, kmeans),
        },
        {
            "name":     "Knowledge Distillation (BiLSTM)",
            "index":    "kd.index",
            "compress": lambda emb: compress_kd(emb, student, DEVICE),
        },
    ]

    print("\n[5/6] Running benchmark across 6 FAISS strategies "
          "(AVQ evaluated separately via 04b_online_inference_avq.py)...")
    all_results = []

    # Containers for raw per-query data (needed for Wilcoxon tests)
    raw_latencies = {}   # method_name -> list of 1000 latency values (ms)
    raw_ann_hits  = {}   # method_name -> list of 1000 binary hits (0 or 1)
    raw_bio_hits  = {}   # method_name -> list of 1000 binary hits (0 or 1)

    for strat in strategies:
        print(f"\n  → {strat['name']}")
        index_path = os.path.join(INDICES_DIR, strat["index"])
        index = faiss.read_index(index_path)

        ann_hits = 0
        bio_hits = 0
        latencies_ms   = []
        per_query_ann  = []   # binary 0/1 per query
        per_query_bio  = []   # binary 0/1 per query

        for i, emb in enumerate(query_embeddings):
            # --- Compress query vector ---
            t_start = time.perf_counter()
            qv = strat["compress"](emb).reshape(1, -1)
            _, I = index.search(qv, TOP_K)
            t_end = time.perf_counter()

            latencies_ms.append((t_end - t_start) * 1000.0)
            retrieved = [int(x) for x in I[0] if x >= 0]

            # Metric 1: ANN Recall@10
            ann_hit = 1 if set(retrieved) & flatl2_gt[i] else 0
            ann_hits += ann_hit
            per_query_ann.append(ann_hit)

            # Metric 2: Biological Accuracy@1
            bio_hit = 0
            if retrieved:
                top1_id = retrieved[0]
                q_label  = query_labels.get(i)
                db_label = db_labels.get(top1_id)
                if q_label is not None and db_label is not None and q_label == db_label:
                    bio_hit = 1
            bio_hits += bio_hit
            per_query_bio.append(bio_hit)

        ann_recall  = ann_hits / n_queries
        bio_acc     = bio_hits / n_queries
        avg_lat_ms  = float(np.mean(latencies_ms))

        all_results.append({
            "Method":                  strat["name"],
            "ANN Recall@10":           round(ann_recall, 4),
            "Biological Accuracy@1":   round(bio_acc,    4),
            "Avg Query Latency (ms)":  round(avg_lat_ms, 4),
        })

        # Store raw arrays keyed by method name
        raw_latencies[strat["name"]] = latencies_ms
        raw_ann_hits[strat["name"]]  = per_query_ann
        raw_bio_hits[strat["name"]]  = per_query_bio

        print(f"     ANN Recall@10          : {ann_recall:.4f}")
        print(f"     Biological Accuracy@1  : {bio_acc:.4f}")
        print(f"     Avg Query Latency (ms) : {avg_lat_ms:.4f}")

    # ------------------------------------------------------------------
    # 6. Save results CSV + serialise shared data for AVQ script
    # ------------------------------------------------------------------
    print(f"\n[6/6] Saving results and serialising shared data for AVQ script...")

    # Serialise mean-pooled query vectors (needed by 04b_online_inference_avq.py)
    np.save(MEAN_POOLED_QUERIES_NPY, query_mean_pooled)
    print(f"  ✓ mean_pooled_queries.npy saved → {MEAN_POOLED_QUERIES_NPY}")

    # Serialise FlatL2 ground truth dict
    with open(FLATL2_GT_PKL, "wb") as _f:
        pickle.dump(flatl2_gt, _f)
    print(f"  ✓ flatl2_ground_truth.pkl saved → {FLATL2_GT_PKL}")

    # Serialise SCOPe label dicts
    with open(SCOPE_LABELS_PKL, "wb") as _f:
        pickle.dump({"db": db_labels, "query": query_labels}, _f)
    print(f"  ✓ scope_labels.pkl saved → {SCOPE_LABELS_PKL}")

    print(f"  Next step (WSL2): python 04b_online_inference_avq.py")
    print(f"  Saving results → {OUTPUT_CSV}")
    # --- Summary averages (unchanged) ---
    fieldnames = [
        "Method",
        "ANN Recall@10",
        "Biological Accuracy@1",
        "Avg Query Latency (ms)",
    ]
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # --- Raw per-query latencies (1000 rows × 5 method columns) ---
    method_names = [s["name"] for s in strategies]
    n_q = len(query_embeddings)

    with open(RAW_LATENCIES_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(method_names)          # header = method names
        for i in range(n_q):
            writer.writerow([raw_latencies[m][i] for m in method_names])
    print(f"  ✓ raw_latencies.csv  written ({n_q} rows × {len(method_names)} methods)")

    with open(RAW_ANN_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(method_names)
        for i in range(n_q):
            writer.writerow([raw_ann_hits[m][i] for m in method_names])
    print(f"  ✓ raw_ann_hits.csv   written ({n_q} rows × {len(method_names)} methods)")

    with open(RAW_BIO_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(method_names)
        for i in range(n_q):
            writer.writerow([raw_bio_hits[m][i] for m in method_names])
    print(f"  ✓ raw_bio_hits.csv   written ({n_q} rows × {len(method_names)} methods)")

    print("\n" + "=" * 70)
    print(f"{'Method':<35} {'ANN R@10':>9} {'Bio Acc@1':>10} {'Latency ms':>11}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['Method']:<35} {r['ANN Recall@10']:>9.4f} "
              f"{r['Biological Accuracy@1']:>10.4f} "
              f"{r['Avg Query Latency (ms)']:>11.4f}")
    print("=" * 70)
    print("\n✓ Benchmark complete. Results saved to:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
