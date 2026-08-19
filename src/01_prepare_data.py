"""
01_prepare_data.py — Stratified Data Preparation
=================================================
Downloads the SCOPe ASTRAL 40% FASTA file from Berkeley and produces a
10,000-sequence primary set and a 1,000-sequence query set using a
*coverage-guaranteed stratified split*.

Why stratified?
---------------
A purely random split (the naive approach) leaves ~9.5% of queries whose
true SCOPe superfamily is entirely absent from the primary set.  Those
queries are structurally unwinnable — no retrieval method can ever return
a correct match — so they depress every method's Biological Accuracy
equally and add noise, not signal.

The stratified split guarantees that for every superfamily that appears in
the query set, at least one member is present in the primary set.
The *natural, uneven* representation of SCOPe superfamilies (some have 200
domains, others only 2) is deliberately preserved so that the correlation
between superfamily density and retrieval accuracy can be studied.

Split algorithm (seed=22):
  1. Group all 15k sequences by SCOPe superfamily.
  2. Singletons (only 1 member) → primary set only; they can never be queries.
  3. For every superfamily with ≥2 members, reserve 1 member as a guaranteed
     "correct answer" in the primary set; all remaining members are query
     candidates.
  4. Randomly draw 1,000 sequences from the candidate pool → query set.
  5. Fill the primary set to exactly 10,000 by random sampling from the
     remaining candidate pool, ensuring the reserved representatives are
     never displaced.

Outputs:
  data/scope_10k_subset.fasta   — 10,000-sequence primary (database) set
  data/queries_1000.fasta       — 1,000-sequence query set (zero unwinnable)

NOTE: The original non-stratified split files, if present, will be
overwritten.  Delete data/*.pt, data/*.pkl, data/*.npy and re-run the
full pipeline after running this script.
"""

import os
import random
import requests
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
URL = "https://scop.berkeley.edu/downloads/scopeseq-2.08/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DOWNLOADED_FILE = os.path.join(DATA_DIR, "astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa")
OUTPUT_FILE  = os.path.join(DATA_DIR, "scope_10k_subset.fasta")
QUERIES_FILE = os.path.join(DATA_DIR, "queries_1000.fasta")
SUBSET_SIZE  = 10_000
QUERIES_SIZE = 1_000
RANDOM_SEED  = 22


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def download_file(url, local_filename):
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(local_filename):
        print(f"  Already downloaded: {os.path.basename(local_filename)}")
        return
    print(f"  Downloading {url} ...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("  Download complete.")


def parse_fasta(filename):
    sequences, current_header, current_seq = [], "", []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header:
                    sequences.append((current_header, "".join(current_seq)))
                current_header, current_seq = line, []
            else:
                current_seq.append(line)
    if current_header:
        sequences.append((current_header, "".join(current_seq)))
    return sequences


def extract_superfamily(header):
    """Return the 3-level SCOPe code (e.g. 'a.1.1') or 'unknown'."""
    parts = header.split()
    if len(parts) > 1:
        levels = parts[1].split(".")
        if len(levels) >= 3:
            return ".".join(levels[:3])
    return "unknown"


def write_fasta(path, entries):
    with open(path, "w") as f:
        for header, seq in entries:
            f.write(f"{header}\n{seq}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("\n[1/4] Downloading SCOPe ASTRAL 40% FASTA...")
    download_file(URL, DOWNLOADED_FILE)

    print("\n[2/4] Parsing FASTA file...")
    sequences = parse_fasta(DOWNLOADED_FILE)
    print(f"  {len(sequences):,} total sequences found.")

    print("\n[3/4] Stratified split (seed={RANDOM_SEED})...")
    random.seed(RANDOM_SEED)

    # Group by superfamily, shuffle within each group for reproducibility
    sf_groups: dict[str, list] = defaultdict(list)
    for header, seq in sequences:
        sf_groups[extract_superfamily(header)].append((header, seq))
    for seqs in sf_groups.values():
        random.shuffle(seqs)

    all_sfs = list(sf_groups.keys())
    random.shuffle(all_sfs)

    # Singletons go straight to primary; only SF with ≥2 members can yield queries
    eligible_sfs   = [sf for sf in all_sfs if len(sf_groups[sf]) >= 2]
    singleton_sfs  = [sf for sf in all_sfs if len(sf_groups[sf]) == 1]

    primary_reserved = []   # guaranteed representatives, one per eligible SF
    query_candidates = []   # all non-representative members

    for sf in eligible_sfs:
        seqs = sf_groups[sf]
        primary_reserved.append(seqs[0])    # 1 guaranteed DB member
        query_candidates.extend(seqs[1:])   # rest are query candidates

    for sf in singleton_sfs:
        primary_reserved.extend(sf_groups[sf])

    # Draw query set from candidates
    random.shuffle(query_candidates)
    query_set = query_candidates[:QUERIES_SIZE]
    remainder = query_candidates[QUERIES_SIZE:]

    # Fill primary to SUBSET_SIZE: lock guaranteed reps, sample from remainder
    query_sfs_used = {extract_superfamily(h) for h, _ in query_set}
    locked, fillable = [], []
    seen_sfs: set[str] = set()
    for h, s in primary_reserved:
        sf = extract_superfamily(h)
        if sf in query_sfs_used and sf not in seen_sfs:
            locked.append((h, s))
            seen_sfs.add(sf)
        else:
            fillable.append((h, s))

    fillable.extend(remainder)
    random.shuffle(fillable)
    needed = SUBSET_SIZE - len(locked)
    final_primary = locked + fillable[:needed]

    # Final shuffle so order conveys no information
    random.shuffle(final_primary)
    random.shuffle(query_set)

    print(f"  Primary set : {len(final_primary):,} sequences")
    print(f"  Query set   : {len(query_set):,} sequences")

    # Verify zero unwinnable queries
    primary_sfs = {extract_superfamily(h) for h, _ in final_primary}
    unwinnable = [(h, extract_superfamily(h)) for h, _ in query_set
                  if extract_superfamily(h) not in primary_sfs]
    if unwinnable:
        print(f"  [WARNING] {len(unwinnable)} unwinnable queries — check logic!")
    else:
        print("  ✓ All query superfamilies represented in primary set.")

    print("\n[4/4] Saving FASTA files...")
    os.makedirs(DATA_DIR, exist_ok=True)
    write_fasta(OUTPUT_FILE, final_primary)
    print(f"  ✓ {OUTPUT_FILE}")
    write_fasta(QUERIES_FILE, query_set)
    print(f"  ✓ {QUERIES_FILE}")

    print("\n  Data preparation complete.")
    print("  Next step: run 01_offline_build.py to regenerate embeddings and indices.")


if __name__ == "__main__":
    main()
