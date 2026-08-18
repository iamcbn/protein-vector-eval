"""
online_eval.py — Online Evaluation Wrapper
==========================================
Entry point for running the full online benchmarking suite.

Checks that all offline build artefacts exist, then runs
04_online_inference.py and prints a final confirmation.

Usage:
    python online_eval.py

Pipeline order (full run from scratch):
    1. python offline_build.py          ← data prep + embeddings + indexing
    2. python online_eval.py            ← online query evaluation
"""

import subprocess
import sys
import os

# Paths
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
INDICES_DIR = os.path.join(DATA_DIR, "indices")

REQUIRED_FILES = {
    "Database FASTA":         os.path.join(DATA_DIR,    "scope_10k_subset.fasta"),
    "Query FASTA":            os.path.join(DATA_DIR,    "queries_1000.fasta"),
    "Database embeddings":    os.path.join(DATA_DIR,    "raw_embeddings_10k.pt"),
    "TEA codebook":           os.path.join(DATA_DIR,    "tea_kmeans_codebook.pkl"),
    "KD student weights":     os.path.join(DATA_DIR,    "kd_student.pt"),
    "FlatL2 index":           os.path.join(INDICES_DIR, "mean_pool.index"),
    "PQ index":               os.path.join(INDICES_DIR, "pq.index"),
    "DCT index":              os.path.join(INDICES_DIR, "dct.index"),
    "TEA index":              os.path.join(INDICES_DIR, "tea.index"),
    "KD index":               os.path.join(INDICES_DIR, "kd.index"),
    "AVQ index":              os.path.join(INDICES_DIR, "avq_scann"),
}


def check_prerequisites():
    print("Checking offline build artefacts...")
    missing = []
    for label, path in REQUIRED_FILES.items():
        status = "✓" if os.path.exists(path) else "✗ MISSING"
        print(f"  {status}  {label:<25}  {os.path.basename(path)}")
        if not os.path.exists(path):
            missing.append(label)
    if missing:
        print(f"\n[ERROR] {len(missing)} required file(s) not found.")
        print("  Run `python offline_build.py` first to build all artefacts.")
        sys.exit(1)
    print("\nAll prerequisites satisfied.\n")


def run_online_inference():
    script = os.path.join(BASE_DIR, "src", "04_online_inference.py")
    print(f"{'=' * 60}")
    print("Running Online Inference...")
    print(f"{'=' * 60}\n")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\n[ERROR] 04_online_inference.py failed (exit code {result.returncode}).")
        sys.exit(result.returncode)


def run_wsl_script(script_name):
    print(f"\n{'=' * 60}")
    print(f"ACTION REQUIRED: WSL2 Execution")
    print(f"{'=' * 60}")
    print(f"The next step requires Linux (for ScaNN AVQ).")
    print(f"Please open your WSL2 (Ubuntu) terminal and copy/paste these commands:")
    print(f"\n    cd \"/mnt/c/Users/NEW USER/Desktop/Topic 26 - Vector Retrieval Optimization for Protein Embeddings/codebase\"")
    print(f"    source .venv_wsl2/bin/activate")
    print(f"    python3 src/{script_name}\n")
    input("Press Enter once the script has successfully finished in WSL2 to continue...")


def confirm_output():
    results_csv = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    avq_results_csv = os.path.join(RESULTS_DIR, "benchmark_avq.csv")
    if os.path.exists(results_csv) and os.path.exists(avq_results_csv):
        size = os.path.getsize(results_csv)
        avq_size = os.path.getsize(avq_results_csv)
        print(f"\n✓ benchmark_results.csv written ({size} bytes)")
        print(f"✓ benchmark_avq.csv written ({avq_size} bytes)")
    else:
        print("\n[WARNING] Some benchmark result files were not found — check logs above.")


def main():
    print("\n=== Protein Vector Retrieval — Online Evaluation Suite ===\n")
    check_prerequisites()
    run_online_inference()
    run_wsl_script("04b_online_inference_avq.py")
    confirm_output()
    print("\n=== Online Evaluation Completed Successfully! ===\n")


if __name__ == "__main__":
    main()
