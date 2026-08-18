"""
03b_offline_indexing_avq.py — AVQ / ScaNN Index Building (WSL2 / Linux only)
=============================================================================
Builds a ScaNN index using the Anisotropic Vector Quantisation (AVQ) loss
function (Guo et al., 2020) from the mean-pooled ESM-2 protein embeddings
produced by 03_offline_indexing.py.

WHY A SEPARATE SCRIPT?
    ScaNN (google-research/scann) officially supports Linux and macOS only.
    This script is therefore designed to run inside a WSL2 (Windows Subsystem
    for Linux) environment on the same workstation.  All other six strategies
    run in the native Windows Python environment.  This is a disclosed,
    bounded deviation — one of seven methods — documented in Chapter 3 §3.4.

EXECUTION ORDER:
    1. (Windows)  python src/03_offline_indexing.py
       → produces data/mean_pooled_10k.npy
    2. (WSL2)     python src/03b_offline_indexing_avq.py      ← THIS SCRIPT
       → produces data/indices/avq_scann/ and results/indexing_avq.csv

AVQ LOSS (Guo et al., 2020):
    L_AVQ = E_q [ Σ_x ( <q,x> - <q, x_hat> )² ]
           = r^T E[qq^T] r   where r = x - x_hat
    The `anisotropic_quantization_threshold` parameter activates this loss.
    A value of 0.2 follows the configuration in Guo et al. (2020) Table 1.

PERFORMANCE CLAIMS:
    AVQ latency results are RECORDED but excluded from cross-method Wilcoxon
    comparisons and Pareto plots. Throughput differences between ScaNN and
    FAISS reflect engine-level engineering (SIMD, batching, memory layout),
    not the quantisation algorithm.  Accuracy metrics (ANN Recall@10,
    Biological Accuracy@1) ARE cross-comparable because they depend only on
    the quantisation objective, not on which library executes the search.

Prerequisites (produced by 03_offline_indexing.py — Windows):
    data/mean_pooled_10k.npy   — database mean-pooled float32 vectors

Outputs:
    data/indices/avq_scann/    — serialised ScaNN index directory
    results/indexing_avq.csv   — build time and approximate index size
"""

import os
import csv
import time

import numpy as np

try:
    import scann
except ImportError as exc:
    raise SystemExit(
        "\n[ERROR] ScaNN is not installed.  Install it inside WSL2/Linux with:\n"
        "  pip install scann\n"
        "See: https://github.com/google-research/google-research/tree/master/scann\n"
    ) from exc

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Resolve paths relative to this script's location so the script works
# regardless of the working directory.
SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(SRC_DIR)
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
INDICES_DIR = os.path.join(DATA_DIR, "indices")
AVQ_INDEX_DIR = os.path.join(INDICES_DIR, "avq_scann")

MEAN_POOLED_NPY = os.path.join(DATA_DIR, "mean_pooled_10k.npy")
OUTPUT_CSV      = os.path.join(RESULTS_DIR, "indexing_avq.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(AVQ_INDEX_DIR, exist_ok=True)

# ScaNN / AVQ hyper-parameters
#   num_leaves        : partition count for the tree-AH structure
#   num_leaves_to_search : partitions probed at query time (trade accuracy/speed)
#   score_ah dims     : asymmetric hashing codebook dimensionality (2 = 2-byte codes)
#   anisotropic_quantization_threshold : activates the direction-weighted inner-product
#                        loss (AVQ).  0.2 follows Guo et al. (2020), Table 1.
#   reorder           : number of exact-distance re-scoring candidates after AH
AVQ_NUM_LEAVES          = 100
AVQ_LEAVES_TO_SEARCH    = 10
AVQ_AH_DIMS             = 2
AVQ_ANISO_THRESHOLD     = 0.2   # key parameter — activates AVQ loss
AVQ_REORDER_CANDIDATES  = 50
TOP_K                   = 10    # must match 04_online_inference.py


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("AVQ Offline Index Build — ScaNN (WSL2 / Linux)")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Load shared mean-pooled vectors
    # ------------------------------------------------------------------
    if not os.path.exists(MEAN_POOLED_NPY):
        raise FileNotFoundError(
            f"\n[ERROR] {MEAN_POOLED_NPY} not found.\n"
            "Run 03_offline_indexing.py (Windows) first to produce this file."
        )

    print(f"\n[1/3] Loading mean-pooled vectors from {MEAN_POOLED_NPY} ...")
    mean_pooled = np.load(MEAN_POOLED_NPY)
    n, dim = mean_pooled.shape
    print(f"  Shape : {mean_pooled.shape}  (dtype={mean_pooled.dtype})")

    # Ensure float32 — ScaNN requires it
    if mean_pooled.dtype != np.float32:
        mean_pooled = mean_pooled.astype(np.float32)

    # ------------------------------------------------------------------
    # 2. Build ScaNN index with AVQ loss
    # ------------------------------------------------------------------
    print(f"\n[2/3] Building ScaNN AVQ index ...")
    print(f"  num_leaves          = {AVQ_NUM_LEAVES}")
    print(f"  num_leaves_to_search = {AVQ_LEAVES_TO_SEARCH}")
    print(f"  aniso_threshold     = {AVQ_ANISO_THRESHOLD}  ← activates AVQ loss")
    print(f"  reorder_candidates  = {AVQ_REORDER_CANDIDATES}")

    t0 = time.time()
    searcher = (
        scann.scann_ops_pybind.builder(mean_pooled, TOP_K, "dot_product")
        .tree(
            num_leaves=AVQ_NUM_LEAVES,
            num_leaves_to_search=AVQ_LEAVES_TO_SEARCH,
            training_sample_size=n,
        )
        .score_ah(
            AVQ_AH_DIMS,
            anisotropic_quantization_threshold=AVQ_ANISO_THRESHOLD,
        )
        .reorder(AVQ_REORDER_CANDIDATES)
        .build()
    )
    build_time = time.time() - t0
    print(f"  ✓ ScaNN index built in {build_time:.2f}s")

    # Serialise index
    searcher.serialize(AVQ_INDEX_DIR)
    print(f"  ✓ ScaNN index serialised → {AVQ_INDEX_DIR}")

    # Approximate index size (sum of files in the serialised directory)
    index_size_gb = sum(
        os.path.getsize(os.path.join(AVQ_INDEX_DIR, f))
        for f in os.listdir(AVQ_INDEX_DIR)
        if os.path.isfile(os.path.join(AVQ_INDEX_DIR, f))
    ) / (1024 ** 3)
    print(f"  Approximate index size : {index_size_gb:.6f} GB")

    # ------------------------------------------------------------------
    # 3. Save results CSV
    # ------------------------------------------------------------------
    print(f"\n[3/3] Saving indexing results → {OUTPUT_CSV}")
    row = {
        "Method":          "AVQ (ScaNN)",
        "Build Time (s)":  round(build_time, 4),
        "Index Size (GB)": round(index_size_gb, 6),
        "Note":            (
            "ScaNN AVQ index; latency excluded from cross-method comparisons "
            "(see Chapter 3 §3.7 and methodology note)"
        ),
    }
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Method", "Build Time (s)", "Index Size (GB)", "Note"]
        )
        writer.writeheader()
        writer.writerow(row)

    print(f"\n=== AVQ Indexing Result ===")
    print(f"  Method          : {row['Method']}")
    print(f"  Build Time      : {row['Build Time (s)']:.2f}s")
    print(f"  Index Size (GB) : {row['Index Size (GB)']:.6f}")
    print(f"\n  Next step (WSL2): python src/04b_online_inference_avq.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
