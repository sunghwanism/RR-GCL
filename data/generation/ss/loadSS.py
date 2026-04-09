import os
import argparse 
import pandas as pd
import time

import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from helper import get_pdb_redo_cif, process_pdb

import warnings
warnings.filterwarnings('ignore')

MAX_SASA = {
    'ALA': 129.0, 'ARG': 274.0, 'ASN': 195.0, 'ASP': 193.0,
    'CYS': 167.0, 'GLN': 225.0, 'GLU': 223.0, 'GLY': 104.0,
    'HIS': 224.0, 'ILE': 197.0, 'LEU': 201.0, 'LYS': 236.0,
    'MET': 224.0, 'PHE': 240.0, 'PRO': 159.0, 'SER': 155.0,
    'THR': 172.0, 'TRP': 285.0, 'TYR': 263.0, 'VAL': 174.0
}

SEC_STRUCT_MAP = {
    'H': 'Helix', 'G': 'Helix', 'I': 'Helix', 'P': 'Helix',
    'B': 'Sheet', 'E': 'Sheet',
    'T': 'Loop', 'S': 'Loop', '-': 'Loop', ' ': 'Loop', '.': 'Loop'
}

def getParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument('--tmp_dir', type=str, required=True)
    parser.add_argument('--resume_pdb', type=str, default=None, help="PDB to resume (txt file)")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of parallel processes")
    return parser.parse_args()


def worker(pdb_code, group, args):
    """
    Worker function executed by each process for a single PDB.
    Returns (pdb_code, features_list, success_bool)
    """
    targets = list(zip(
        group['chain_id'], 
        group['auth_residue_number'], 
        group['pdb_raw_resi']
    ))
    
    time.sleep(np.random.uniform(0, 2)) 

    cif_file = get_pdb_redo_cif(pdb_code, args)
    features = []
    success = False

    if cif_file:
        try:
            features = process_pdb(pdb_code, targets, cif_file, MAX_SASA, SEC_STRUCT_MAP)
            success = True
            if os.path.exists(cif_file):
                os.remove(cif_file)
        except Exception as e:
            success = False
    
    return pdb_code, features, success


def main(args):
    # Setup directories
    output_dir = os.path.dirname(args.output_file) or "."
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    if not os.path.exists(args.tmp_dir): os.makedirs(args.tmp_dir)
    failed_log_path = os.path.join(output_dir, "failed_pdb.txt")

    # Load data
    print(f"Loading data...")
    df = pd.read_csv(args.input_file)
    nodes1 = df[['pdb_code', 'auth_chain1', 'pdb_auth_resi1', 'pdb_raw_resi1']].rename(
        columns={'auth_chain1': 'chain_id', 'pdb_auth_resi1': 'auth_residue_number', 'pdb_raw_resi1': 'pdb_raw_resi'})
    nodes2 = df[['pdb_code', 'auth_chain2', 'pdb_auth_resi2', 'pdb_raw_resi2']].rename(
        columns={'auth_chain2': 'chain_id', 'pdb_auth_resi2': 'auth_residue_number', 'pdb_raw_resi2': 'pdb_raw_resi'})
    unique_nodes = pd.concat([nodes1, nodes2]).drop_duplicates()
    
    pdb_groups = list(unique_nodes.groupby('pdb_code'))

    # Resume Logic
    if args.resume_pdb and os.path.exists(args.resume_pdb):
        with open(args.resume_pdb, "r") as f:
            processed = {line.strip() for line in f}
        pdb_groups = [g for g in pdb_groups if g[0] not in processed]

    print(f"Starting parallel processing with {args.num_workers} workers...")

    total_allocated_cpus = len(os.sched_getaffinity(0)) 
    print(f"Total allocated CPUs: {total_allocated_cpus} || Real Use {min(args.num_workers, total_allocated_cpus)}")

    # Parallel Execution
    with ProcessPoolExecutor(max_workers=min(args.num_workers, total_allocated_cpus)) as executor:
        # Submit all tasks
        futures = {executor.submit(worker, code, group, args): code for code, group in pdb_groups}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Total Progress"):
            pdb_code, feats, success = future.result()
            
            if success:
                if feats:
                    out_df = pd.DataFrame(feats)
                    file_exists = os.path.isfile(args.output_file)
                    out_df.to_csv(args.output_file, mode='a', index=False, header=not file_exists)
            else:
                with open(failed_log_path, "a") as f:
                    f.write(f"{pdb_code}\n")

if __name__ == "__main__":
    args = getParser()
    main(args)