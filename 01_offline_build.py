import subprocess
import os
import sys

def run_script(script_name):
    print(f"\n{'='*50}\nRunning {script_name}...\n{'='*50}")
    # The scripts are inside the 'src' directory relative to this file
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", script_name)
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"Error: {script_name} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    print("=== Starting Protein Vector Retrieval: Offline Build ===")
    run_script("01_prepare_data.py")
    run_script("02_generate_embeddings.py")
    run_script("03_offline_indexing.py")
    run_script("03b_offline_indexing_avq.py")
    print("\n=== Offline Build Completed Successfully! ===")

if __name__ == "__main__":
    main()
