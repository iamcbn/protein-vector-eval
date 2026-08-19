"""
00_sanity_checks.py — Sanity Check Orchestrator
================================================
Top-level entry point that runs all data-integrity and labelling sanity
checks defined in src/06_sanity_checks.py.

Run this:
  • After 01_offline_build.py (with just FASTA files) to confirm the
    stratified split produced zero unwinnable queries.
  • Again after embeddings are generated to confirm no NaNs / zero vectors.
  • After 02_online_eval.py to spot-check labelling against raw results.

Usage:
    python 00_sanity_checks.py

Pipeline context:
    python 01_offline_build.py    ← data prep + embeddings + indices
    python 00_sanity_checks.py    ← run this at any point to verify
    python 02_online_eval.py      ← online evaluation
    python 03_run_analysis.py     ← statistical analysis
"""

import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_script(script_path, label):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"\n[ERROR] {os.path.basename(script_path)} exited with code {result.returncode}.")
        sys.exit(result.returncode)


def main():
    print("\n=== Protein Vector Retrieval — Sanity Checks ===\n")
    run_script(
        os.path.join(BASE_DIR, "src", "06_sanity_checks.py"),
        "Coverage, Integrity, and Label Spot-Check"
    )
    print("\n=== All Sanity Checks Passed ===\n")


if __name__ == "__main__":
    main()
