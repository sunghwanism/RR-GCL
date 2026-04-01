import os

import json
import pickle
import networkx as nx
import pandas as pd
import numpy as np

import networkx as nx

import matplotlib.pyplot as plt
from utils.functions import load_yaml
from utils.graphfunction import load_graph, get_sample, get_uniprot_from_nodes, get_pos_from_nodes, get_res_from_nodes, get_node_id_rm_copy

config = load_yaml("../../../config/RRGCL.yaml")
DATABASE = config.DATABASE


def main():
    y_df = pd.read_csv(f"{DATABASE}/node_features_with_location_nodeid_v031026.csv")

    # Load Graph for weighted Augmentation
    finalG = load_graph((f"{DATABASE}/cleaned_weighted_graph.pkl"))
    edge_df = nx.to_pandas_edgelist(finalG)

    ################################################
    # 1. Float SS Augmentation
    ################################################
    T = 5.0 # For Temperature in NaN Augmentation (2)

    float_cols = ['rel_sasa', 'ss_helix', 'ss_sheet','ss_loop', 'depth', 'hse_up', 'hse_down',
                  'dssp_accessibility', 'dssp_TCO', 'dssp_kappa','dssp_alpha', 'dssp_phi', 'dssp_psi',]

    float_ss_df = y_df[['node_id'] + float_cols].copy()
    float_ss_df['cleaned_rm_node_id']= float_ss_df['node_id'].map(get_node_id_rm_copy)

    float_ss_df['uniprot'] = float_ss_df['cleaned_rm_node_id'].apply(lambda x: x.split('_')[0])
    float_ss_df['res'] = float_ss_df['cleaned_rm_node_id'].apply(lambda x: x.split('_')[2])
    float_ss_df['pos'] = float_ss_df['cleaned_rm_node_id'].apply(lambda x: int(x.split('_')[1]))
    
    # NaN Augmentation (1) - Origin Residue based Augmentaiton for copy
    float_ss_df.set_index('node_id', inplace=True)

    for col in float_cols:
        group_means = float_ss_df.groupby('cleaned_rm_node_id')[col].mean()
        is_na = float_ss_df[col].isna()
        float_ss_df.loc[is_na, col] = float_ss_df.loc[is_na, 'cleaned_rm_node_id'].map(group_means)

    float_ss_df.reset_index(inplace=True)

    # NaN Augmentation (2) - Neighbor & Energy based Augmentaiton

    edges_fwd = edge_df[['source', 'target', 'cleaned_total_energy']].rename(
        columns={'source': 'node', 'target': 'neighbor'}
    )
    edges_rev = edge_df[['target', 'source', 'cleaned_total_energy']].rename(
        columns={'target': 'node', 'source': 'neighbor'}
    )
    all_edges = pd.concat([edges_fwd, edges_rev], ignore_index=True)
    all_edges['scaled_neg_energy'] = -all_edges['cleaned_total_energy'] / T

    max_energy_per_node = all_edges.groupby('node')['scaled_neg_energy'].transform('max')

    all_edges['safe_exponent'] = all_edges['scaled_neg_energy'] - max_energy_per_node
    all_edges['weight'] = np.exp(all_edges['safe_exponent'])

    edge_weights = all_edges.groupby(['node', 'neighbor'])['weight'].sum().reset_index()
    float_ss_df = float_ss_df.set_index('node_id')

    for col in float_cols:
        print(f"[{col}] processing...")
        
        missing_nodes = float_ss_df[float_ss_df[col].isna()].index
        
        if len(missing_nodes) == 0:
            continue
            
        rel_edges = edge_weights[edge_weights['node'].isin(missing_nodes)].copy()
        

        unique_mapping = float_ss_df.groupby(level=0)[col].mean()
        rel_edges['neighbor_val'] = rel_edges['neighbor'].map(unique_mapping)
        
        valid_edges = rel_edges.dropna(subset=['neighbor_val'])
        
        if valid_edges.empty:
            continue

        valid_edges['weighted_val'] = valid_edges['neighbor_val'] * valid_edges['weight']
        
        agg_df = valid_edges.groupby('node').agg(
            sum_val=('weighted_val', 'sum'),
            sum_wt=('weight', 'sum')
        )
        
        agg_df['weighted_mean'] = agg_df['sum_val'] / agg_df['sum_wt']
        float_ss_df.loc[agg_df.index, col] = agg_df['weighted_mean']

    float_ss_df = float_ss_df.reset_index()
    float_ss_df.drop(columns=['uniprot', 'res', 'pos', 'cleaned_rm_node_id'], inplace=True)
    merge_float_df = float_ss_df.groupby('node_id').mean().reset_index()

    ################################################
    # 2. Str SS Augmentation
    ################################################

    str_list = ['node_id',
                'dssp_sec_struct', # Class
                'dssp_helix_3_10', 'dssp_helix_alpha', 'dssp_helix_pi', 'dssp_helix_pp', 'dssp_bend',
                'dssp_chirality', 'dssp_sheet', 'dssp_strand', 'dssp_ladder_1','dssp_ladder_2',]

    str_ss_df = y_df[str_list].copy()  # str
    str_ss_df['cleaned_rm_node_id']= str_ss_df['node_id'].map(get_node_id_rm_copy)





if __name__ == "__main__":
    main()