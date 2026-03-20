import os
import sys

import re
import json

import pandas as pd
import numpy as np

from Bio import SearchIO
from data.reference import residue1to3

def hmm_to_df(file_path):
    uniprot_id = os.path.basename(file_path).split('.')[0]
    
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    trans_cols = ['M->M', 'M->I', 'M->D', 'I->M', 'I->I', 'D->M', 'D->D']
    
    data = []
    index_labels = []

    with open(file_path, 'r') as f:
        content = f.read()

    parts = re.split(r'HMM\s+A\s+C\s+D', content)
    if len(parts) < 2:
        print(f"Cannot Find any Data in {file_path}")
        return None
    
    hmm_body = parts[1].strip()
    lines = hmm_body.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line == "//":
            i += 1
            continue
            
        row_parts = line.split()
        
        if len(row_parts) >= 22 and row_parts[0].isalpha() and len(row_parts[0]) == 1 and row_parts[1].isdigit():
            res_name = row_parts[0]
            res_pos = row_parts[1]
            index_labels.append(f"{res_name}{res_pos}")
            
            em_scores = [2 ** (-int(s) / 1000) if s != '*' else 0.0 for s in row_parts[2:22]]
            
            i += 1
            if i < len(lines):
                trans_parts = lines[i].split()
                
                transitions = [2 ** (-int(s) / 1000) if s != '*' else 0.0 for s in trans_parts[:7]]
                
                neff = 0.0
                if len(trans_parts) >= 8:
                    s_neff = trans_parts[7]
                    if s_neff != '*':
                        neff = float(s_neff) / 100.0
                
                data.append(em_scores + transitions + [neff])
        
        i += 1

    feature_names = amino_acids + trans_cols + ['Neff']
    df = pd.DataFrame(data, columns=feature_names, index=index_labels)
    df.reset_index(inplace=True)
    
    df['resType'] = df['index'].apply(lambda x: residue1to3.get(x[0], 'unk').lower())
    df['position'] = df['index'].apply(lambda x: x[1:])
    df['ID'] = uniprot_id + '_' + df['position'].astype(str) + '_' + df['resType']

    df = pd.concat([df[['resType', 'position', 'ID']], df[feature_names]], axis=1)

    return df

def merge_hmm_to_json(input_dir, output_json):
    total_hhm_dict = {}
    
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    trans_cols = ['M->M', 'M->I', 'M->D', 'I->M', 'I->I', 'D->M', 'D->D']
    all_feature_cols = amino_acids + trans_cols + ['Neff']
    
    hhm_files = [f for f in os.listdir(input_dir) if f.endswith('.hhm')]
    print(f"Found {len(hhm_files)} HHM files. Starting process...")

    for filename in hhm_files:
        file_path = os.path.join(input_dir, filename)
        df = hmm_to_df(file_path)
        
        if df is not None:
            for _, row in df.iterrows():
                total_hhm_dict[row['ID']] = row[all_feature_cols].tolist()
            print(f"Processed: {filename}")
        else:
            print(f"Failed to process: {filename}")

    with open(output_json, 'w', encoding='utf-8') as jf:
        json.dump(total_hhm_dict, jf)
    
    print(f"\nAll data saved to {output_json}")
    print(f"Feature Dimension: {len(all_feature_cols)} (Emissions 20 + Transitions 7 + Neff 1)")


def pssm_to_df(file_path):
    amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                   'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    
    uniprot_id = os.path.basename(file_path).split('.')[0].lower()
    
    data = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.split()
        
        if len(parts) > 40 and parts[0].isdigit():
            pos = parts[0]      
            ref_aa = parts[1]   
            
            log_odds = [int(x) for x in parts[2:22]]
            
            entropy = float(parts[-2])
            
            res_type_3 = residue1to3.get(ref_aa, 'unk').lower()
            node_id = f"{uniprot_id}_{pos}_{res_type_3}"
            
            data.append([node_id, res_type_3, pos, entropy] + log_odds)
            
    columns = ['ID', 'resType', 'position', 'Entropy'] + amino_acids
    df = pd.DataFrame(data, columns=columns)
    
    return df

def merge_pssm_to_json(input_dir, output_json):
    total_pssm_dict = {}
    
    amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                   'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    
    feature_cols = ['Entropy'] + amino_acids
    
    pssm_files = [f for f in os.listdir(input_dir) if f.endswith('.pssm')]
    print(f"Found {len(pssm_files)} PSSM files. Starting process...")

    for filename in pssm_files:
        file_path = os.path.join(input_dir, filename)
        
        df = pssm_to_df(file_path)
        
        if df is not None:
            for _, row in df.iterrows():
                total_pssm_dict[row['ID']] = row[feature_cols].tolist()
            print(f"Processed: {filename}")
        else:
            print(f"Failed to process: {filename}")

    with open(output_json, 'w', encoding='utf-8') as jf:
        json.dump(total_pssm_dict, jf)
    
    print(f"\nAll PSSM data saved to {output_json}")
    print(f"Feature Dimension: {len(feature_cols)} (Entropy 1 + Log-odds 20)")