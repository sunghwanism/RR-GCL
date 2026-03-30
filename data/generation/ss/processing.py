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

    ################################################
    # 1. Float SS Augmentation
    ################################################
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

    # NaN Augmentation (2) - Neighbor based Augmentaiton
    for col in float_cols:
        na_rows = float_ss_df[float_ss_df[col].isna()]
        
        for idx, row in na_rows.iterrows():
            u_id = row['uniprot']
            res_type = row['res']
            current_pos = row['pos']
            mask = (
                (float_ss_df['uniprot'] == u_id) & 
                (float_ss_df['res'] == res_type) & 
                (float_ss_df['pos'] >= current_pos - 20) & 
                (float_ss_df['pos'] <= current_pos + 20)
            )
            
            nearby_mean = float_ss_df.loc[mask, col].mean()
            
            if pd.notna(nearby_mean):
                float_ss_df.at[idx, col] = nearby_mean

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