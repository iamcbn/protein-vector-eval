import torch
from transformers import EsmModel, EsmTokenizer
import time
import os


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FASTA_FILE = os.path.join(DATA_DIR, "scope_10k_subset.fasta")
OUTPUT_FILE = os.path.join(DATA_DIR, "raw_embeddings_10k.pt")
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
BATCH_SIZE = 32  # Safe for >16 GB VRAM (RTX 3090/4090/A100/L40). Drop to 16 if OOM on a 16 GB card.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_fasta(filename):
    """
    Parse a FASTA file and return a list of uppercase amino acid sequences.

    SCOPe ASTRAL FASTA files store sequences in all-lowercase. The ESM-2
    tokenizer only recognises uppercase single-letter amino acid codes;
    lowercase characters are mapped to <unk>, which collapses all embeddings
    to the same unknown-token vector. Converting to uppercase here fixes this.
    """
    sequences = []
    current_seq = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    # BUG FIX: .upper() — SCOPe ASTRAL stores sequences in
                    # lowercase; ESM-2 tokenizer needs uppercase amino acid codes.
                    sequences.append(''.join(current_seq).upper())
                current_seq = []
            else:
                current_seq.append(line)
        if current_seq:
            sequences.append(''.join(current_seq).upper())
    return sequences

def main():
    if os.path.exists(OUTPUT_FILE):
        print(f"Embeddings already exist at {OUTPUT_FILE}. Skipping generation.")
        return

    print(f"Using device: {DEVICE}")
    
    print(f"Loading {MODEL_NAME}...")
    try:
        tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
        model = EsmModel.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"[Network Error] {e}. Falling back to local cache.")
        tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        model = EsmModel.from_pretrained(MODEL_NAME, local_files_only=True)

    model.to(DEVICE)
    model.eval()
    
    print("Loading sequences...")
    sequences = load_fasta(FASTA_FILE)
    print(f"Loaded {len(sequences)} sequences.")

    # Pre-flight validation: confirm sequences are uppercase and non-empty
    n_lower    = sum(1 for s in sequences if any(c.islower() for c in s))
    n_empty    = sum(1 for s in sequences if len(s) == 0)
    avg_len    = sum(len(s) for s in sequences) / max(len(sequences), 1)
    print(f"  Sequence validation:")
    print(f"    Total sequences   : {len(sequences)}")
    print(f"    Lowercase present : {n_lower} (should be 0 after .upper() fix)")
    print(f"    Empty sequences   : {n_empty} (should be 0)")
    print(f"    Avg sequence len  : {avg_len:.1f} residues")
    if n_lower > 0:
        raise RuntimeError(
            f"[BUG] {n_lower} sequences still contain lowercase characters. "
            "ESM-2 tokenizer requires uppercase amino acid codes."
        )
    if n_empty > 0:
        print(f"  WARNING: {n_empty} empty sequences — these will produce zero-dim embeddings.")
    
    all_embeddings = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(0, len(sequences), BATCH_SIZE):
            batch_seqs = sequences[i:i+BATCH_SIZE]
            
            # Tokenize batch
            inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            # Forward pass
            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state # [batch_size, seq_len, hidden_dim]
            
            # Extract actual sequence embeddings (remove CLS/EOS padding tokens)
            attention_mask = inputs['attention_mask']
            
            for j in range(len(batch_seqs)):
                # 1 for valid tokens, including CLS and EOS. We typically drop CLS (index 0) and EOS (last valid index)
                # But for simplicity or to be safe, we can just extract based on the mask and remove the first and last
                seq_len = attention_mask[j].sum().item()
                # Extract sequence excluding <cls> and <eos>
                # The actual sequence length is seq_len - 2
                res_embedding = last_hidden_state[j, 1:seq_len-1, :].cpu()
                # Sanity check: embedding should have >0 residues
                if res_embedding.shape[0] == 0:
                    print(f"  WARNING: sequence {i+j} produced a 0-residue embedding "
                          f"(seq_len={seq_len}). Skipping.")
                all_embeddings.append(res_embedding)
            
            if (i + BATCH_SIZE) % 100 == 0 or i == 0:
                print(f"Processed {i + len(batch_seqs)} / {len(sequences)} sequences. Elapsed: {time.time() - start_time:.2f}s")
                
    # Post-generation shape sanity check
    shapes = [e.shape[0] for e in all_embeddings]
    n_single = sum(1 for s in shapes if s <= 1)
    print(f"\nEmbedding shape summary:")
    print(f"  Total embeddings   : {len(all_embeddings)}")
    print(f"  Min seq-len        : {min(shapes)}")
    print(f"  Max seq-len        : {max(shapes)}")
    print(f"  Mean seq-len       : {sum(shapes)/len(shapes):.1f}")
    print(f"  Embeddings with <= 1 residue : {n_single} (should be ~0)")
    if n_single > len(all_embeddings) * 0.05:
        print("  [WARNING] >5% of embeddings have <=1 residue. "
              "Check for tokenizer issues or very short sequences.")

    print(f"\nSaving embeddings to {OUTPUT_FILE}...")
    torch.save(all_embeddings, OUTPUT_FILE)
    print("Done!")

if __name__ == "__main__":
    main()
