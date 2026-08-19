"""
07_bootstrap_analysis.py -- Bootstrap CIs, McNemar Tests, and Density-Bucket Breakdown
======================================================================================
Computes all statistical reporting for Biological Accuracy@1.

WHY NOT WALD/WILSON?
--------------------
The Wald/Wilson closed-form CI assumes each query trial is an independent
Bernoulli draw with a fixed success probability.  That assumption is violated
here because queries from the same SCOPe superfamily share structural context:
if one query from superfamily c.37.1 fails, its siblings are more likely to
fail too. This intra-cluster correlation inflates the effective variance
relative to what Wilson assumes, meaning Wilson CIs are *artificially narrow*
and overstate confidence.

The two bootstrap variants here correct for this:
  - Case Bootstrap     : resamples individual queries (corrects for non-normality
                         but not the clustering structure).
  - Cluster Bootstrap  : resamples by *superfamily*, not by query -- the correct
                         unit of replication when success probability varies by SF.
                         This is the number to report.

Statistical tests:
  - McNemar's test (paired) is used for all cross-method comparisons because
    every method is evaluated on the exact same 1,000 queries.  Comparing
    independent CIs would be wrong here.

Density-Bucket Breakdown:
  - Splits queries into four bins by the size of their superfamily in the
    primary DB, then reports per-bin accuracy.  This directly proves that the
    overdispersion (success probability varying by SF) is real and quantifies
    how large the effect is.

Outputs:
  results/bootstrap_ci_report.txt    -- full CI table for all methods
  results/density_bucket_report.txt  -- per-bucket accuracy for all methods
  results/mcnemar_report.txt         -- pairwise McNemar p-values vs baseline

Usage:
    python 07_bootstrap_analysis.py                   # standalone
    python 03_run_analysis.py                          # via top-level orchestrator
"""

import sys

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(SRC_DIR)
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

FASTA_DB        = os.path.join(DATA_DIR, "scope_10k_subset.fasta")
FASTA_QUERIES   = os.path.join(DATA_DIR, "queries_1000.fasta")
SCOPE_LABELS    = os.path.join(DATA_DIR, "scope_labels.pkl")
RAW_BIO_CSV     = os.path.join(RESULTS_DIR, "raw_bio_hits.csv")
N_BOOT          = 5_000   # bootstrap replicates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_fasta_headers(filename):
    headers = []
    with open(filename, "r") as f:
        for line in f:
            if line.startswith(">"):
                headers.append(line.strip())
    return headers


def extract_sf(header):
    parts = header.lstrip(">").split()
    if len(parts) >= 2:
        levels = parts[1].split(".")
        if len(levels) >= 3:
            return ".".join(levels[:3])
    return "unknown"


def case_bootstrap(hits, n_boot=N_BOOT, seed=0):
    rng  = np.random.default_rng(seed)
    hits = np.asarray(hits)
    means = [rng.choice(hits, size=len(hits), replace=True).mean()
             for _ in range(n_boot)]
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def cluster_bootstrap(hits, clusters, n_boot=N_BOOT, seed=1):
    """Resample by superfamily cluster -- the correct unit of replication."""
    rng          = np.random.default_rng(seed)
    hits         = np.asarray(hits)
    cluster_dict = defaultdict(list)
    for h, c in zip(hits, clusters):
        cluster_dict[c].append(h)
    keys   = list(cluster_dict.keys())
    n_keys = len(keys)
    means  = []
    for _ in range(n_boot):
        sampled = rng.choice(keys, size=n_keys, replace=True)
        pool    = []
        for k in sampled:
            pool.extend(cluster_dict[k])
        means.append(np.mean(pool))
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def section(title):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("\n=== Bootstrap Analysis ===")

    # ------------------------------------------------------------------
    # Guard: require result files
    # ------------------------------------------------------------------
    for path, label in [(RAW_BIO_CSV,   "raw_bio_hits.csv"),
                        (SCOPE_LABELS,  "scope_labels.pkl"),
                        (FASTA_DB,      "scope_10k_subset.fasta"),
                        (FASTA_QUERIES, "queries_1000.fasta")]:
        if not os.path.exists(path):
            print(f"\n  [ERROR] Missing required file: {label}")
            print("  Run the full pipeline (01_offline_build -> 02_online_eval) first.")
            return

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    with open(SCOPE_LABELS, "rb") as f:
        labels      = pickle.load(f)
    query_labels    = labels["query"]           # {query_idx: sf_string | None}

    df          = pd.read_csv(RAW_BIO_CSV)
    methods     = list(df.columns)
    n_queries   = len(df)
    baseline    = methods[0]

    # SF per query (aligned to df row index)
    q_sfs = [query_labels.get(i) for i in range(n_queries)]

    # Superfamily density in DB
    db_headers   = parse_fasta_headers(FASTA_DB)
    db_sf_counts = {}
    for h in db_headers:
        sf = extract_sf(h)
        db_sf_counts[sf] = db_sf_counts.get(sf, 0) + 1

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Section 1: Bootstrap CIs
    # ------------------------------------------------------------------
    section("Bootstrap Confidence Intervals (Biological Accuracy@1)")
    print(f"  n_queries = {n_queries} | n_boot = {N_BOOT} | baseline = {baseline}\n")
    print(f"  {'Method':<40} {'Acc':>6}  {'Case Boot 95% CI':>20}  {'Cluster Boot 95% CI':>22}")
    print(f"  {'-'*40} {'-'*6}  {'-'*20}  {'-'*22}")

    ci_lines = []
    for method in methods:
        hits    = df[method].values.astype(int)
        acc     = np.mean(hits)
        cb_lo, cb_hi   = case_bootstrap(hits)
        clb_lo, clb_hi = cluster_bootstrap(hits, q_sfs)
        line = (f"  {method:<40} {acc:>6.4f}  "
                f"[{cb_lo:.4f}, {cb_hi:.4f}]        "
                f"[{clb_lo:.4f}, {clb_hi:.4f}]")
        print(line)
        ci_lines.append(line)

    with open(os.path.join(RESULTS_DIR, "bootstrap_ci_report.txt"), "w") as f:
        f.write("Bootstrap CI Report -- Biological Accuracy@1\n")
        f.write("(Use Cluster Bootstrap CI as the primary reported interval)\n\n")
        f.write(f"  {'Method':<40} {'Acc':>6}  {'Case Boot 95% CI':>20}  {'Cluster Boot 95% CI':>22}\n")
        f.write("\n".join(ci_lines) + "\n")
    print(f"\n  [OK] Saved -> bootstrap_ci_report.txt")

    # ------------------------------------------------------------------
    # Section 2: McNemar Paired Tests (all methods vs baseline)
    # ------------------------------------------------------------------
    section(f"McNemar Paired Tests vs '{baseline}'")
    print(f"\n  {'Method':<40} {'p-value':>12}  Sig?")
    print(f"  {'-'*40} {'-'*12}  ----")

    mcnemar_lines = []
    b_hits = df[baseline].values.astype(int)
    for method in methods:
        if method == baseline:
            continue
        hits  = df[method].values.astype(int)
        table = [
            [int(((hits==1) & (b_hits==1)).sum()), int(((hits==1) & (b_hits==0)).sum())],
            [int(((hits==0) & (b_hits==1)).sum()), int(((hits==0) & (b_hits==0)).sum())],
        ]
        res  = mcnemar(table, exact=False, correction=True)
        sig  = "***" if res.pvalue < 0.001 else ("**" if res.pvalue < 0.01
               else ("*" if res.pvalue < 0.05 else "n.s."))
        line = f"  {method:<40} {res.pvalue:>12.4e}  {sig}"
        print(line)
        mcnemar_lines.append(line)

    with open(os.path.join(RESULTS_DIR, "mcnemar_report.txt"), "w") as f:
        f.write(f"McNemar Paired Tests vs baseline '{baseline}'\n\n")
        f.write(f"  {'Method':<40} {'p-value':>12}  Sig?\n")
        f.write("\n".join(mcnemar_lines) + "\n")
    print(f"\n  [OK] Saved -> mcnemar_report.txt")

    # ------------------------------------------------------------------
    # Section 3: Failure Clustering
    # ------------------------------------------------------------------
    section("Failure Clustering (queries failed by ALL methods)")
    failure_counts = (df == 0).sum(axis=1)
    all_fail_idx   = np.where(failure_counts == len(methods))[0]
    sf_fail_counts: dict[str, int] = defaultdict(int)
    for i in all_fail_idx:
        sf_fail_counts[q_sfs[i]] += 1

    print(f"\n  Queries failed by ALL {len(methods)} methods: {len(all_fail_idx)} / {n_queries}")
    print("  Top 10 superfamilies responsible:")
    for sf, cnt in sorted(sf_fail_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    {sf}: {cnt}")

    # ------------------------------------------------------------------
    # Section 4: Density-Bucket Breakdown (all methods)
    # ------------------------------------------------------------------
    section("Density-Bucket Breakdown by Superfamily DB Representation")
    BUCKETS = [("1-5", 1, 5), ("6-20", 6, 20), ("21-50", 21, 50), ("51+", 51, 10**9)]

    # Build bucket assignment per query (based on DB SF density)
    bucket_label = []
    for sf in q_sfs:
        density = db_sf_counts.get(sf, 0)
        for bname, lo, hi in BUCKETS:
            if lo <= density <= hi:
                bucket_label.append(bname)
                break
        else:
            bucket_label.append("unknown")

    print(f"\n  {'Bucket':<10} {'n_queries':>10}  " +
          "  ".join(f"{m[:20]:>22}" for m in methods))
    print(f"  {'-'*10} {'-'*10}  " + "  ".join(["-"*22]*len(methods)))

    bucket_lines = []
    for bname, _, _ in BUCKETS:
        idx = [i for i, b in enumerate(bucket_label) if b == bname]
        n   = len(idx)
        if n == 0:
            continue
        accs = [np.mean(df[m].values[idx]) for m in methods]
        acc_str = "  ".join(f"{a:>21.2%}" for a in accs)
        line = f"  {bname:<10} {n:>10}  {acc_str}"
        print(line)
        bucket_lines.append(line)

    with open(os.path.join(RESULTS_DIR, "density_bucket_report.txt"), "w") as f:
        f.write("Density-Bucket Breakdown -- Accuracy by Superfamily DB Size\n\n")
        f.write(f"  {'Bucket':<10} {'n_queries':>10}  " +
                "  ".join(f"{m[:22]:>22}" for m in methods) + "\n")
        f.write("\n".join(bucket_lines) + "\n")
    print(f"\n  [OK] Saved -> density_bucket_report.txt")

    print("\n=== Bootstrap Analysis Complete ===\n")


if __name__ == "__main__":
    main()
