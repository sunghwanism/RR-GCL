import os
import gzip
import json
import argparse
import re
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial

from utils.graphfunction import load_graph
from data.reference import residue1to3

def get_parser():
    parser = argparse.ArgumentParser(description="Parallel AlphaMissense Feature Extractor")
    parser.add_argument("--amPATH", type=str, required=True, help="Path to AlphaMissense_hg38.tsv.gz")
    parser.add_argument("--graphPATH", type=str, required=True, help="Path to graph pkl file")
    parser.add_argument("--savePATH", type=str, required=True, help="Path to save the feature JSON")
    parser.add_argument("--num_workers", type=int, default=cpu_count(), help="Number of CPU cores to use")
    parser.add_argument("--add_am_score_folder", type=str default='data/reference', help='Additional AM score Table (tsv)')
    return parser.parse_args()

def process_am_chunk(target_subset, tsv_gz_path):
    """
    Worker function to scan the large TSV file for a subset of UniProt IDs.
    """
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
    variant_pattern = re.compile(r'([A-Z])(\d+)([A-Z])')
    local_dict = {}
    
    # Using pandas chunking to maintain memory efficiency while scanning
    chunks = pd.read_csv(
        tsv_gz_path, 
        sep='\t', 
        compression='gzip', 
        comment='#',
        header=None,
        names=['CHROM', 'POS', 'REF', 'ALT', 'genome', 'uniprot_id', 
               'transcript_id', 'protein_variant', 'am_pathogenicity', 'am_class'],
        usecols=['uniprot_id', 'protein_variant', 'am_pathogenicity'],
        chunksize=2000000
    )

    target_set = set(target_subset)

    for chunk in chunks:
        filtered = chunk[chunk['uniprot_id'].isin(target_set)]
        
        if not filtered.empty:
            for _, row in filtered.iterrows():
                u_id = row['uniprot_id'].lower()
                variant = row['protein_variant']
                score = float(row['am_pathogenicity'])
                
                match = variant_pattern.match(variant)
                if match:
                    ref_res = match.group(1)
                    position = match.group(2)
                    alt_res = match.group(3)
                    
                    res_type_3 = residue1to3.get(ref_res, 'unk').lower()
                    node_id = f"{u_id}_{position}_{res_type_3}"
                    
                    if node_id not in local_dict:
                        local_dict[node_id] = [0.0] * 20
                    
                    if alt_res in aa_to_idx:
                        local_dict[node_id][aa_to_idx[alt_res]] = score
    return local_dict

def main(args):
    # Load graph and extract unique UniProt IDs from nodes
    G = load_graph(args.graphPATH)
    nodes = list(G.nodes())
    
    # Extract protein IDs (handling potential isoforms or suffixes)
    selected_proteins = [node.split('_')[0].split('-')[0] for node in nodes]
    selected_proteins = list(set(selected_proteins))
    original_protein_set = {p.lower() for p in selected_proteins}

    print(f"Total unique proteins to process: {len(selected_proteins)}")

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    print(f"SLURM_CPUS_PER_TASK: {slurm_cpus}")
    if slurm_cpus:
        effective_workers = int(slurm_cpus)
    else:
        effective_workers = args.num_workers
        
    print(f"Parallel workers: {effective_workers}")

    if not selected_proteins:
        print("Error: No proteins found in the graph.")
        return

    # Split protein list for parallel workers
    target_ids_upper = [str(_id).upper() for _id in selected_proteins]
    n = len(target_ids_upper)
    chunk_size = (n + effective_workers - 1) // effective_workers
    id_chunks = [target_ids_upper[i:i + chunk_size] for i in range(0, n, chunk_size)]

    # Execute parallel processing
    with Pool(effective_workers) as pool:
        func = partial(process_am_chunk, tsv_gz_path=args.amPATH)
        results = pool.map(func, id_chunks)

    # Merge results and track found protein IDs
    print("Merging worker results...")
    total_am_dict = {}
    found_protein_ids = set()
    
    for res in results:
        total_am_dict.update(res)
        for node_id in res.keys():
            # Extract raw protein ID from the node_id string
            found_protein_ids.add(node_id.split('_')[0].lower())

    # Identify IDs that were not found in AlphaMissense file
    failed_ids = list(original_protein_set - found_protein_ids)

    additional_uniprot = [f.split("-")[1][:-4].lower() for f in os.listdir(args.add_am_score_folder) if 'AlphaMissense' in f]

    for fail_uniprot in failed_ids:
        if fail_uniprot in additional_uniprot:
            print(fail_uniprot)
            am_path = os.path.join(args.add_am_score_folder, f"AlphaMissense-{fail_uniprot.upper()}.tsv")
            am_df = pd.read_csv(am_path, sep='\t',)
            aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}

            grouped = am_df.groupby(['position', 'a.a.1'])
            for (pos, ref_aa), group in grouped:
                res_type_3 = residue1to3.get(ref_aa, ref_aa.lower())
                node_id = f"{fail_uniprot}_{pos}_{res_type_3.lower()}"
                score_list = [0.0] * 20

                for _, row in group.iterrows():
                    alt_aa = row['a.a.2']
                    score = float(row['pathogenicity score'])
                    if alt_aa in aa_to_idx:
                        score_list[aa_to_idx[alt_aa]] = score

                # total_am_dict[node_id] = score_list
                total_am_dict[node_id] = score_list

                found_protein_ids.add(fail_uniprot.lower())

    # Save Feature JSON
    am_path = args.savePATH + "/am_features.json"
    with open(am_path, 'w', encoding='utf-8') as jf:
        json.dump(total_am_dict, jf)

    # Identify IDs that were not found in AlphaMissense file
    failed_ids = list(original_protein_set - found_protein_ids)
    
    # Save Failed IDs JSON (filename_failed.json)
    failed_json_path = args.savePATH + "/am_failed.json"
    with open(failed_json_path, 'w', encoding='utf-8') as fjf:
        json.dump(failed_ids, fjf, indent=4)
    
    # Final Summary Report
    print(f"\n" + "="*40)
    print(f"SUCCESS: {len(found_protein_ids)} proteins | {len(total_am_dict)} nodes")
    print(f"FAILED : {len(failed_ids)} proteins")
    print(f"Feature JSON: {am_path}")
    print(f"Failed IDs JSON: {failed_json_path}")
    print("="*40)

if __name__ == "__main__":
    args = get_parser()
    main(args)