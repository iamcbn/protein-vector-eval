"""
run_analysis.py — Statistical Analysis Entry Point
====================================================
Entry point for Phase 2 of the benchmarking pipeline.

Checks all required result files exist, then runs
05_statistical_analysis.py.

Usage:
    python run_analysis.py

Pipeline order (full run from scratch):
    1. python offline_build.py   ← data prep + embeddings + indexing
    2. python online_eval.py     ← online query evaluation (saves raw data)
    3. python run_analysis.py    ← statistical tests + figures  ← YOU ARE HERE
"""

import subprocess
import sys
import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

REQUIRED_FILES = {
    "benchmark_results.csv":  os.path.join(RESULTS_DIR, "benchmark_results.csv"),
    "indexing_results.csv":   os.path.join(RESULTS_DIR, "indexing_results.csv"),
    "raw_latencies.csv":      os.path.join(RESULTS_DIR, "raw_latencies.csv"),
    "raw_ann_hits.csv":       os.path.join(RESULTS_DIR, "raw_ann_hits.csv"),
    "raw_bio_hits.csv":       os.path.join(RESULTS_DIR, "raw_bio_hits.csv"),
    "benchmark_avq.csv":      os.path.join(RESULTS_DIR, "benchmark_avq.csv"),
    "raw_latencies_avq.csv":  os.path.join(RESULTS_DIR, "raw_latencies_avq.csv"),
    "raw_ann_hits_avq.csv":   os.path.join(RESULTS_DIR, "raw_ann_hits_avq.csv"),
    "raw_bio_hits_avq.csv":   os.path.join(RESULTS_DIR, "raw_bio_hits_avq.csv"),
}


def check_prerequisites():
    print("Checking required result files...")
    missing = []
    for label, path in REQUIRED_FILES.items():
        status = "✓" if os.path.exists(path) else "✗ MISSING"
        print(f"  {status}  {label:<30}  {os.path.basename(path)}")
        if not os.path.exists(path):
            missing.append(label)

    if missing:
        print(f"\n[ERROR] {len(missing)} required file(s) not found.")
        print("  Run `python online_eval.py` first to generate all result files.")
        sys.exit(1)
    print("\nAll prerequisites satisfied.\n")


def run_statistical_analysis():
    script = os.path.join(BASE_DIR, "src", "05_statistical_analysis.py")
    print("=" * 60)
    print("Running Statistical Analysis + Visualization...")
    print("=" * 60 + "\n")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\n[ERROR] 05_statistical_analysis.py failed (exit code {result.returncode}).")
        sys.exit(result.returncode)


def confirm_outputs():
    print("\nChecking generated outputs...")
    figures_dir = os.path.join(RESULTS_DIR, "figures")
    expected = [
        os.path.join(RESULTS_DIR,  "wilcoxon_results.csv"),
        os.path.join(RESULTS_DIR,  "full_results.csv"),
        os.path.join(figures_dir,  "pareto_speed_vs_accuracy.png"),
        os.path.join(figures_dir,  "tradeoff_ann_vs_compression.png"),
        os.path.join(figures_dir,  "tradeoff_bio_vs_compression.png"),
        os.path.join(figures_dir,  "latency_boxplot.png"),
    ]
    all_ok = True
    for path in expected:
        status = "✓" if os.path.exists(path) else "✗ MISSING"
        if not os.path.exists(path):
            all_ok = False
        print(f"  {status}  {os.path.relpath(path, BASE_DIR)}")

    if not all_ok:
        print("\n[WARNING] Some output files are missing — check logs above.")
    else:
        print("\nAll outputs generated successfully!")


def main():
    print("\n=== Protein Vector Retrieval — Statistical Analysis Suite ===\n")
    check_prerequisites()
    run_statistical_analysis()
    confirm_outputs()
    print("\n=== Analysis Pipeline Completed Successfully! ===\n")


if __name__ == "__main__":
    main()
