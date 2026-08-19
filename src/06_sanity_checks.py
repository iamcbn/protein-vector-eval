"""
06_sanity_checks.py - Coverage and Representation + Data Integrity Checks
==========================================================================
Runs three sanity-check modules against the data files and results
produced by the pipeline:

  MODULE A - Coverage and Representation (runs on FASTA files alone)
    - Confirms every query superfamily has >=1 representative in the
      primary 10k set (should be 0 unwinnable after stratified split).
    - Flags any remaining unwinnable queries to a separate report.
    - Reports superfamily density statistics and the top-10 most
      represented superfamilies (used to interpret overdispersion).

  MODULE B - Data Integrity / Leakage + Embedding QC (requires .pt files)
    - Confirms no exact sequence overlap between the query set and the
      primary set.
    - Confirms no duplicate sequences within the primary set.
    - Scans raw embedding tensors for NaNs, zero vectors, and truncated
      embeddings (<=1 residue) that could silently bias results.

  MODULE C - Label Spot-Check (30 random queries)
    - Randomly samples 30 query headers and prints the SCOPe superfamily
      code extracted by the automated parser so you can visually confirm
      the labelling logic is correct.

Outputs (written to results/):
  coverage_report.txt      -- full SF density table + summary stats
  unwinnable_queries.txt   -- list of any queries with no DB representative

Usage:
    python 06_sanity_checks.py                   # standalone
    python 00_sanity_checks.py                   # via top-level orchestrator
"""

import sys
import os

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pickle
from collections import Counter
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(SRC_DIR)
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

FASTA_DB        = os.path.join(DATA_DIR, "scope_10k_subset.fasta")
FASTA_QUERIES   = os.path.join(DATA_DIR, "queries_1000.fasta")
EMBEDDINGS_DB   = os.path.join(DATA_DIR, "raw_embeddings_10k.pt")
EMBEDDINGS_Q    = os.path.join(DATA_DIR, "raw_embeddings_queries.pt")
SCOPE_LABELS    = os.path.join(DATA_DIR, "scope_labels.pkl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_fasta(filename):
    headers, sequences = [], []
    header, seq_parts = None, []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    headers.append(header)
                    sequences.append("".join(seq_parts).upper())
                header, seq_parts = line, []
            else:
                seq_parts.append(line)
    if header is not None:
        headers.append(header)
        sequences.append("".join(seq_parts).upper())
    return headers, sequences


def extract_sf(header):
    parts = header.lstrip(">").split()
    if len(parts) >= 2:
        levels = parts[1].split(".")
        if len(levels) >= 3:
            return ".".join(levels[:3])
    return "unknown"


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Module A - Coverage and Representation
# ---------------------------------------------------------------------------
def module_a_coverage():
    section("MODULE A - Coverage and Representation")

    if not os.path.exists(FASTA_DB) or not os.path.exists(FASTA_QUERIES):
        print("  [SKIP] FASTA files not found. Run 01_prepare_data.py first.")
        return

    db_headers, _ = parse_fasta(FASTA_DB)
    q_headers,  _ = parse_fasta(FASTA_QUERIES)

    db_sf_counts = Counter(extract_sf(h) for h in db_headers)
    q_sf_counts  = Counter(extract_sf(h) for h in q_headers)

    # Winnability check
    unwinnable = [(h, extract_sf(h)) for h in q_headers
                  if extract_sf(h) not in db_sf_counts]
    winnable   = len(q_headers) - len(unwinnable)

    print(f"\n  Total queries            : {len(q_headers):,}")
    print(f"  Winnable (SF in DB)      : {winnable:,}")
    print(f"  Unwinnable (SF absent)   : {len(unwinnable):,}  <- target: 0")
    if unwinnable:
        print("\n  [WARNING] Unwinnable queries detected:")
        for h, sf in unwinnable[:10]:
            print(f"    {h.split()[0][1:]}  ->  SF {sf}")
        if len(unwinnable) > 10:
            print(f"    ... and {len(unwinnable) - 10} more (see unwinnable_queries.txt)")
    else:
        print("  [PASS] All query superfamilies are represented in the primary set.")

    # Density statistics
    counts = list(db_sf_counts.values())
    print(f"\n  Unique SFs in primary set : {len(db_sf_counts):,}")
    print(f"  Unique SFs in query set   : {len(q_sf_counts):,}")
    print(f"  Max domains per SF        : {max(counts)}")
    print(f"  Min domains per SF        : {min(counts)}")
    print(f"  Mean domains per SF       : {sum(counts)/len(counts):.2f}")
    print("\n  Top 10 most represented superfamilies:")
    for sf, n in db_sf_counts.most_common(10):
        print(f"    {sf}: {n}")

    # Write reports
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "coverage_report.txt"), "w") as f:
        f.write("Coverage and Representation Report\n")
        f.write("==================================\n\n")
        f.write(f"Total queries           : {len(q_headers)}\n")
        f.write(f"Winnable               : {winnable}\n")
        f.write(f"Unwinnable             : {len(unwinnable)}\n\n")
        f.write(f"Unique SFs in primary  : {len(db_sf_counts)}\n")
        f.write(f"Max / Min / Mean       : {max(counts)} / {min(counts)} / {sum(counts)/len(counts):.2f}\n\n")
        f.write("Superfamily Densities (all):\n")
        for sf, n in db_sf_counts.most_common():
            f.write(f"  {sf}: {n}\n")

    with open(os.path.join(RESULTS_DIR, "unwinnable_queries.txt"), "w") as f:
        f.write("Unwinnable Queries\n==================\n")
        for h, sf in unwinnable:
            f.write(f"SF: {sf} | {h}\n")

    print(f"\n  Reports written to {RESULTS_DIR}/")


# ---------------------------------------------------------------------------
# Module B - Data Integrity / Leakage + Embedding QC
# ---------------------------------------------------------------------------
def module_b_integrity():
    section("MODULE B - Data Integrity / Leakage + Embedding QC")

    if not os.path.exists(FASTA_DB) or not os.path.exists(FASTA_QUERIES):
        print("  [SKIP] FASTA files not found.")
        return

    db_headers, db_seqs = parse_fasta(FASTA_DB)
    q_headers,  q_seqs  = parse_fasta(FASTA_QUERIES)

    # Leakage
    overlap = set(db_seqs) & set(q_seqs)
    status  = "[PASS]" if not overlap else "[FAIL]"
    print(f"\n  {status} Exact sequence overlaps (query <-> DB) : {len(overlap)}  <- target: 0")

    # Duplicates within DB
    db_dups = len(db_seqs) - len(set(db_seqs))
    status  = "[PASS]" if not db_dups else "[FAIL]"
    print(f"  {status} Duplicates within primary set          : {db_dups}  <- target: 0")

    # Embedding QC
    for path, label in [(EMBEDDINGS_DB, "DB embeddings"),
                        (EMBEDDINGS_Q,  "Query embeddings")]:
        if not os.path.exists(path):
            print(f"\n  [SKIP] {label} file not found (run 01_offline_build.py first).")
            continue
        print(f"\n  Checking {label}...")
        embs        = torch.load(path, map_location="cpu", weights_only=False)
        nan_count   = sum(1 for e in embs if torch.isnan(e).any())
        zero_count  = sum(1 for e in embs if torch.all(e == 0))
        short_count = sum(1 for e in embs if e.shape[0] <= 1)
        print(f"    Total     : {len(embs):,}")
        print(f"    NaNs      : {nan_count}   <- target: 0")
        print(f"    Zero vecs : {zero_count}   <- target: 0")
        print(f"    Truncated : {short_count}   <- target: 0")
        if nan_count or zero_count or short_count:
            print("    [FAIL] Malformed embeddings detected.")
        else:
            print("    [PASS] All embeddings are healthy.")


# ---------------------------------------------------------------------------
# Module C - Label Spot-Check
# ---------------------------------------------------------------------------
def module_c_labels():
    section("MODULE C - Label Spot-Check (30 random queries)")

    if not os.path.exists(FASTA_QUERIES):
        print("  [SKIP] Query FASTA not found.")
        return

    q_headers, _ = parse_fasta(FASTA_QUERIES)

    np.random.seed(99)
    idx = np.random.choice(len(q_headers), size=min(30, len(q_headers)), replace=False)
    print("\n  Header -> extracted superfamily:")
    for i in sorted(idx):
        sf = extract_sf(q_headers[i])
        print(f"    {q_headers[i][:55]:<55}  ->  {sf}")
    print("\n  Visually confirm the 3-level SCOPe code matches the header.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print("\n=== Sanity Checks ===")
    module_a_coverage()
    module_b_integrity()
    module_c_labels()
    print("\n=== Sanity Checks Complete ===\n")


if __name__ == "__main__":
    main()
