import os
import argparse

import subprocess
from concurrent.futures import ProcessPoolExecutor

def generate_missing_hhm(args):
    fasta_path, db_path, output_hhm = args

    db_path = os.path.join(db_path, "UniRef30_2023_02")
    
    cmd = [
        "hhblits",
        "-i", fasta_path,
        "-d", db_path,
        "-ohhm", output_hhm,
        "-n", "2",
        "-cpu", "4",
        "-v", "1"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return f"SUCCESS: {os.path.basename(fasta_path)}"
    except subprocess.CalledProcessError as e:
        return f"ERROR: {os.path.basename(fasta_path)} - {e.stderr}"


def main(args):

    os.makedirs(args.hhm_output_dir, exist_ok=True)

    tasks = []
    for fasta_file in os.listdir(args.fasta_dir):
        if fasta_file.endswith(".fasta"):
            protein_id = fasta_file.split(".")[0]
            fasta_path = os.path.join(args.fasta_dir, fasta_file)
            output_hhm = os.path.join(args.hhm_output_dir, f"{protein_id}.hhm")
            
            if not os.path.exists(output_hhm):
                tasks.append((fasta_path, args.uniref_db_path, output_hhm))

    print(f"Generating HHM for {len(tasks)} missing proteins...")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(generate_missing_hhm, tasks))

    for res in results:
        if "ERROR" in res:
            print(res)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta_dir", type=str, required=True, help="path to fasta directory for saving")
    parser.add_argument("--uniref_db_path", type=str, help="path to uniref database")
    parser.add_argument("--hmm_dir", type=str, help="path to hmm directory for saving")
    parser.add_argument("--workers", type=int, required=True, help="number of parallel workers")
    parser.add_argument('--jobs', nargs='+', choices=['pssm', 'hmm'], help="jobs to run [pssm, hmm]")
    parser.add_argument('--hhm_output_dir', type=str, required=True, help="path to hhm output directory for saving")
    args = parser.parse_args()

    main(args)