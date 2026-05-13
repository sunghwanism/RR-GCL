import os
import json
import numpy as np
import pandas as pd

import torch

import networkx as nx
from sklearn.manifold import TSNE

import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

from utils.functions import load_yaml
from utils.graphfunction import load_graph, get_uniprot_from_nodes, get_res_from_nodes, get_pos_from_nodes, save_graph

from data.reference import residue1to3

def main():
    # Load Configuration
    config = load_yaml("config/RRGCL.yaml")
    DATABASE = config.DATABASE

    config = load_yaml('config/DGI.yaml')

    # Load Graph Data
    finalG = load_graph((f"{DATABASE}/cleaned_weighted_graph_wAtt_v050126.pkl"))

    # Feature DataFrame
    org_res_feat_df = pd.read_csv(f'{DATABASE}/merged_feature_data_v041226.csv')
    org_res_feat_df['avg_am'] = org_res_feat_df.filter(like='am').mean(axis=1)

    # Cancer Driver DataFrame
    mut_df = pd.read_csv('models/matched_cancer_driver_df.csv')


    models = {
            "All": '9oxu0esk',
            "exCategory": '4kckha9d',
            'exEvol': '26bsk9qs',
            'exSS': 'pyav8l3c',
            'All+Egy': '39rgottw'
            }


    meta_df = mut_df[['node_id', 'label']].merge(org_res_feat_df[['node_id', 'avg_am']], on='node_id', how='left')
    meta_df = meta_df.set_index('node_id')
    def process_model(args):
        model_name, wandb_id = args
        print(f"Processing model: {model_name} ({wandb_id})...")
        npz_path = f"{config.SAVEPATH}/DGI/{wandb_id}/test/test_embeddings.npz"
        
        if not os.path.exists(npz_path):
            print(f"No file found for {model_name} at {npz_path}")
            return model_name, None
            
        data = np.load(npz_path)
        node_ids = data['node_ids']
        embeddings = data['embeddings']
        
        if embeddings.ndim > 2:
            embeddings = embeddings.reshape(embeddings.shape[0], -1)
            
        tsne = TSNE(
            n_components=2, 
            random_state=42, 
            perplexity=100, 
            early_exaggeration=20,
            max_iter=1000,
            n_jobs=-1,
            init='pca',
            learning_rate='auto'
        )
        emb_2d = tsne.fit_transform(embeddings)
        
        vis_df = pd.DataFrame({
            'node_id': node_ids,
            'tsne_1': emb_2d[:, 0],
            'tsne_2': emb_2d[:, 1]
        })
        
        vis_df = vis_df.join(meta_df, on='node_id', how='left')
        vis_df.to_csv(f"{config.SAVEPATH}/DGI/{wandb_id}/tSNE_result_driver+am.csv", index=False)
        return model_name, vis_df

    vis_dfs = {}

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(process_model, models.items()))


    # 1. Please directly specify the columns to be used for visualization as a list!
    feature_cols = [
        # Structural-based features
        'rel_sasa', 'ss_helix', 'ss_sheet', 'ss_loop', 'depth', 'hse_up', 'hse_down', 
        # DSSP features (float)
        'dssp_accessibility', 'dssp_TCO', 'dssp_kappa', 'dssp_alpha', 'dssp_phi', 'dssp_psi', 
        # AAindex1
        'aa1_KYTJ820101', 'aa1_KLEP840101', 'aa1_BHAR880101', 
        'aa1_JANJ780101', 'aa1_CHOP780201', 'aa1_GRAR740102', 'aa1_GRAR740103', 
        # PSSM
        'pssm_A', 'pssm_C', 'pssm_D', 'pssm_E', 'pssm_F', 'pssm_G', 'pssm_H', 
        'pssm_I', 'pssm_K', 'pssm_L', 'pssm_M', 'pssm_N', 'pssm_P', 'pssm_Q', 
        'pssm_R', 'pssm_S', 'pssm_T', 'pssm_V', 'pssm_W', 'pssm_Y', 'pssm_entropy', 
        # HMM
        'hmm_A', 'hmm_C', 'hmm_D', 'hmm_E', 'hmm_F', 'hmm_G', 'hmm_H', 'hmm_I', 
        'hmm_K', 'hmm_L', 'hmm_M', 'hmm_N', 'hmm_P', 'hmm_Q', 'hmm_R', 'hmm_S', 
        'hmm_T', 'hmm_V', 'hmm_W', 'hmm_Y', 'hmm_MM', 'hmm_MI', 'hmm_MD', 'hmm_IM', 
        'hmm_II', 'hmm_DM', 'hmm_DD', 'hmm_neff'
    ]

    # 2. Get the list of node_ids from the npz file in the test folder (arbitrarily using test_embeddings.npz from the "All" model)
    npz_path = f"{config.SAVEPATH}/DGI/{models['All']}/test/test_embeddings.npz"
    data = np.load(npz_path)
    test_node_ids = data['node_ids'].tolist()
            
    # 3. Filter only the rows corresponding to test_node_ids from org_res_feat_df
    filtered_org_df = org_res_feat_df[org_res_feat_df['node_id'].isin(test_node_ids)].copy()
    # If there are missing values (NaN), fill them with 0 as t-SNE will raise an error (this can be modified to your preferred missing value handling method)
    features = filtered_org_df[feature_cols].fillna(0).values

    # 4. Calculate t-SNE based on the original features specified by the user
    print(f"Running TSNE on {len(features)} nodes with {len(feature_cols)} features...")
    tsne = TSNE(
        n_components=2, 
        random_state=42, 
        perplexity=100, 
        early_exaggeration=20,
        max_iter=1500,
        n_jobs=-1,
        init='pca',
        learning_rate='auto'
    )

    org_emb_2d = tsne.fit_transform(features)

    # 5. Create a DataFrame for visualization (including avg_am)
    vis_org_df = pd.DataFrame({
        'node_id': filtered_org_df['node_id'].values,
        'tsne_1': org_emb_2d[:, 0],
        'tsne_2': org_emb_2d[:, 1],
        'avg_am': filtered_org_df['avg_am'].values
    })

    # Merge with mut_df to get label information
    vis_org_df = vis_org_df.merge(mut_df[['node_id', 'label']], on='node_id', how='left')
    vis_org_df.to_csv(f'{config.SAVEPATH}/DGI/tSNE_result_all-feat_org.csv')

if __name__ == '__main__':
    main()