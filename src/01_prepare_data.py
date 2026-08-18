import os
import requests
import random

# Configuration
URL = "https://scop.berkeley.edu/downloads/scopeseq-2.08/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DOWNLOADED_FILE = os.path.join(DATA_DIR, "astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa")
OUTPUT_FILE = os.path.join(DATA_DIR, "scope_10k_subset.fasta")
QUERIES_FILE = os.path.join(DATA_DIR, "queries_1000.fasta")
SUBSET_SIZE = 10000
QUERIES_SIZE = 1000
RANDOM_SEED = 22


# Downloading protein file from Berkeley's SCOP database 
def download_file(url, local_filename):
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading {url} to {local_filename}...")
    if os.path.exists(local_filename):
        print("File already exists. Skipping download.")
        return
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Download complete.")

# Parsing FASTA file
def parse_fasta(filename):
    sequences = []
    current_header = ""
    current_seq = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header:
                    sequences.append((current_header, "".join(current_seq)))
                current_header = line
                current_seq = []
            else:
                current_seq.append(line)
        if current_header:
            sequences.append((current_header, "".join(current_seq)))
            
    return sequences


def main():
    download_file(URL, DOWNLOADED_FILE)
    
    print("Parsing FASTA file...")
    sequences = parse_fasta(DOWNLOADED_FILE)
    print(f"Found {len(sequences)} sequences in total.")
    
    print(f"Randomly selecting {SUBSET_SIZE} sequences (seed={RANDOM_SEED})...")
    random.seed(RANDOM_SEED)
    random.shuffle(sequences)
    
    subset = sequences[:SUBSET_SIZE]
    queries = sequences[SUBSET_SIZE:SUBSET_SIZE + QUERIES_SIZE]
    
    print(f"Saving subset to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for header, seq in subset:
            f.write(f"{header}\n{seq}\n")
            
    print(f"Saving queries to {QUERIES_FILE}...")
    with open(QUERIES_FILE, 'w') as f:
        for header, seq in queries:
            f.write(f"{header}\n{seq}\n")
            
    print("Data preparation complete.")

if __name__ == "__main__":
    main()
