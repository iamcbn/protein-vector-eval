"""
04b_online_inference_avq.py — AVQ / ScaNN Online Query Evaluation (WSL2 / Linux only)
======================================================================================
Runs 1,000 held-out query proteins through the ScaNN AVQ index and records
ANN Recall@10 and Biological Accuracy@1.

WHY A SEPARATE SCRIPT?
    ScaNN officially supports Linux and macOS only.  This script runs inside
    WSL2 on the same workstation.  Six of seven strategies are evaluated in
    the native Windows script (04_online_inference.py).

LATENCY POLICY:
    Per-query latency IS recorded (raw_latencies_avq.csv) but is EXCLUDED
    from the Wilcoxon Signed-Rank latency comparisons and the Pareto
    speed-vs-accuracy plots in 05_statistical_analysis.py.
    Reason: throughput differences between ScaNN and FAISS reflect
    engine-level implementation choices (SIMD, batching, memory layout),
    not the quantisation algorithm.  Mixing them would conflate two distinct
    variables.  See Chapter 3 §3.7 for the full methodological justification.

ACCURACY METRICS (ANN Recall@10, Biological Accuracy@1):
    These ARE cross-comparable across engines because they depend only on
    the quantisation objective — given identical database and query vectors,
    accuracy at a fixed compression ratio is a property of the loss function,
    not of which library executes the search.

EXECUTION ORDER:
    1. (Windows) python src/03_offline_indexing.py
    2. (WSL2)    python src/03b_offline_indexing_avq.py
    3. (Windows) python src/04_online_inference.py
       → produces shared data files for this script
    4. (WSL2)    python src/04b_online_inference_avq.py   ← THIS SCRIPT
    5. (Windows) python src/05_statistical_analysis.py

Prerequisites (produced by the Windows scripts):
    data/indices/avq_scann/      — serialised ScaNN index (from 03b)
    data/mean_pooled_queries.npy — query mean-pooled vectors (from 04)
    data/flatl2_ground_truth.pkl — FlatL2 top-10 ground truth (from 04)
    data/scope_labels.pkl        — SCOPe label dicts for db + queries (from 04)

Outputs:
    results/benchmark_avq.csv        — summary (ANN Recall@10, Bio Acc@1)
    results/raw_latencies_avq.csv    — per-query latency (ms); LATENCY_EXCLUDED flag
    results/raw_ann_hits_avq.csv     — per-query ANN hit/miss (0/1)
    results/raw_bio_hits_avq.csv     — per-query Bio hit/miss (0/1)
"""

import os
import csv
import pickle
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
SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(SRC_DIR)
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
INDICES_DIR = os.path.join(DATA_DIR, "indices")

AVQ_INDEX_DIR           = os.path.join(INDICES_DIR, "avq_scann")
MEAN_POOLED_QUERIES_NPY = os.path.join(DATA_DIR, "mean_pooled_queries.npy")
FLATL2_GT_PKL           = os.path.join(DATA_DIR, "flatl2_ground_truth.pkl")
SCOPE_LABELS_PKL        = os.path.join(DATA_DIR, "scope_labels.pkl")

BENCHMARK_CSV       = os.path.join(RESULTS_DIR, "benchmark_avq.csv")
RAW_LATENCIES_CSV   = os.path.join(RESULTS_DIR, "raw_latencies_avq.csv")
RAW_ANN_CSV         = os.path.join(RESULTS_DIR, "raw_ann_hits_avq.csv")
RAW_BIO_CSV         = os.path.join(RESULTS_DIR, "raw_bio_hits_avq.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)

TOP_K       = 10   # must match 04_online_inference.py
METHOD_NAME = "AVQ (ScaNN)"

# Flag written to the latency CSV header so 05_statistical_analysis.py
# can identify and skip this method in latency comparisons.
LATENCY_EXCLUDED_NOTE = (
    "LATENCY_EXCLUDED: ScaNN throughput reflects engine engineering, "
    "not the AVQ quantisation algorithm. See Chapter 3 §3.7."
)


# ---------------------------------------------------------------------------
# Prerequisite check
# ---------------------------------------------------------------------------
def check_prerequisites():
    required = {
        "AVQ ScaNN index dir":      AVQ_INDEX_DIR,
        "Mean-pooled queries .npy": MEAN_POOLED_QUERIES_NPY,
        "FlatL2 ground truth .pkl": FLATL2_GT_PKL,
        "SCOPe labels .pkl":        SCOPE_LABELS_PKL,
    }
    missing = [label for label, path in required.items() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "\n[ERROR] The following prerequisite files are missing:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\n\nRun the scripts in the order documented in the module docstring."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("AVQ Online Evaluation — ANN Recall@10 + Biological Accuracy@1")
    print("(ScaNN / WSL2 — latency recorded but excluded from comparisons)")
    print("=" * 65)

    check_prerequisites()

    # ------------------------------------------------------------------
    # 1. Load shared data produced by 04_online_inference.py (Windows)
    # ------------------------------------------------------------------
    print(f"\n[1/4] Loading shared prerequisite data ...")

    query_mean_pooled = np.load(MEAN_POOLED_QUERIES_NPY).astype(np.float32)
    n_queries = len(query_mean_pooled)
    print(f"  Query vectors : {query_mean_pooled.shape}")

    with open(FLATL2_GT_PKL, "rb") as f:
        flatl2_gt = pickle.load(f)      # dict: int -> set of int
    print(f"  FlatL2 GT     : {len(flatl2_gt)} queries")

    with open(SCOPE_LABELS_PKL, "rb") as f:
        scope = pickle.load(f)          # {"db": dict, "query": dict}
    db_labels    = scope["db"]
    query_labels = scope["query"]
    print(f"  SCOPe labels  : {len(db_labels)} db / {len(query_labels)} query")

    # ------------------------------------------------------------------
    # 2. Load serialised ScaNN index
    # ------------------------------------------------------------------
    print(f"\n[2/4] Loading ScaNN AVQ index from {AVQ_INDEX_DIR} ...")
    searcher = scann.scann_ops_pybind.load_searcher(AVQ_INDEX_DIR)
    print("  ✓ ScaNN index loaded")

    # ------------------------------------------------------------------
    # 3. Run 1,000 queries
    # ------------------------------------------------------------------
    print(f"\n[3/4] Running {n_queries} queries through ScaNN AVQ index ...")

    latencies_ms  = []
    per_query_ann = []
    per_query_bio = []
    ann_hits = 0
    bio_hits = 0

    for i in range(n_queries):
        qv = query_mean_pooled[i]   # [dim]

        t_start = time.perf_counter()
        # ScaNN search — returns (indices, distances) for a single vector
        neighbours, _ = searcher.search(qv, final_num_neighbors=TOP_K)
        t_end = time.perf_counter()

        latencies_ms.append((t_end - t_start) * 1000.0)
        retrieved = [int(x) for x in neighbours]

        # Metric 1: ANN Recall@10
        ann_hit = 1 if set(retrieved) & flatl2_gt[i] else 0
        ann_hits += ann_hit
        per_query_ann.append(ann_hit)

        # Metric 2: Biological Accuracy@1
        bio_hit = 0
        if retrieved:
            top1_id  = retrieved[0]
            q_label  = query_labels.get(i)
            db_label = db_labels.get(top1_id)
            if q_label is not None and db_label is not None and q_label == db_label:
                bio_hit = 1
        bio_hits += bio_hit
        per_query_bio.append(bio_hit)

        if (i + 1) % 100 == 0:
            print(f"  Queried {i + 1:>4} / {n_queries}")

    ann_recall = ann_hits / n_queries
    bio_acc    = bio_hits / n_queries
    avg_lat_ms = float(np.mean(latencies_ms))

    print(f"\n  ANN Recall@10         : {ann_recall:.4f}")
    print(f"  Biological Accuracy@1 : {bio_acc:.4f}")
    print(f"  Avg Query Latency (ms): {avg_lat_ms:.4f}  [EXCLUDED from comparisons]")

    # ------------------------------------------------------------------
    # 4. Save outputs
    # ------------------------------------------------------------------
    print(f"\n[4/4] Saving results ...")

    # Summary CSV
    summary_row = {
        "Method":                 METHOD_NAME,
        "ANN Recall@10":          round(ann_recall, 4),
        "Biological Accuracy@1":  round(bio_acc,    4),
        "Avg Query Latency (ms)": round(avg_lat_ms, 4),
        "Latency Note":           LATENCY_EXCLUDED_NOTE,
    }
    with open(BENCHMARK_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Method", "ANN Recall@10", "Biological Accuracy@1",
                        "Avg Query Latency (ms)", "Latency Note"]
        )
        writer.writeheader()
        writer.writerow(summary_row)
    print(f"  ✓ {BENCHMARK_CSV}")

    # Raw latencies — first row is the LATENCY_EXCLUDED flag, then data
    with open(RAW_LATENCIES_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([METHOD_NAME])          # method header
        writer.writerow([LATENCY_EXCLUDED_NOTE])  # flag row for 05_statistical_analysis
        for lat in latencies_ms:
            writer.writerow([lat])
    print(f"  ✓ {RAW_LATENCIES_CSV}  (flagged as latency-excluded)")

    # Raw ANN hits
    with open(RAW_ANN_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([METHOD_NAME])
        for hit in per_query_ann:
            writer.writerow([hit])
    print(f"  ✓ {RAW_ANN_CSV}")

    # Raw bio hits
    with open(RAW_BIO_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([METHOD_NAME])
        for hit in per_query_bio:
            writer.writerow([hit])
    print(f"  ✓ {RAW_BIO_CSV}")

    print(f"\n  Next step (Windows): python src/05_statistical_analysis.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
