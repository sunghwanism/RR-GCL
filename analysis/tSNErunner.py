import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from utils.functions import load_yaml

# ==========================================
# Constants and Configurations
# ==========================================

# Define columns to be used for visualization
FEATURE_COLS = [
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

# Model mapping (Model Name -> Wandb ID)
MODELS = {
    "All": '9oxu0esk',
    "exCategory": '4kckha9d',
    'exEvol': '26bsk9qs',
    'exSS': 'pyav8l3c',
    'All+Egy': '39rgottw'
}

# ==========================================
# Utility Functions
# ==========================================

def compute_tsne(features, n_components=2, perplexity=100, max_iter=1000):
    """Utility function to compute t-SNE embeddings."""
    tsne = TSNE(
        n_components=n_components, 
        random_state=42, 
        perplexity=perplexity, 
        early_exaggeration=20,
        max_iter=max_iter,
        n_jobs=-1,
        init='pca',
        learning_rate='auto'
    )
    return tsne.fit_transform(features)

def process_model_embeddings(model_info, savepath, meta_df):
    """Processes a single model's embeddings and saves the t-SNE results."""
    model_name, wandb_id = model_info
    print(f"Processing model: {model_name} ({wandb_id})...")
    
    npz_path = f"{savepath}/DGI/{wandb_id}/test/test_embeddings.npz"
    if not os.path.exists(npz_path):
        print(f"No file found for {model_name} at {npz_path}")
        return model_name, None
        
    data = np.load(npz_path)
    node_ids = data['node_ids']
    embeddings = data['embeddings']
    
    if embeddings.ndim > 2:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
        
    emb_2d = compute_tsne(embeddings, max_iter=1000)
    
    vis_df = pd.DataFrame({
        'node_id': node_ids,
        'tsne_1': emb_2d[:, 0],
        'tsne_2': emb_2d[:, 1]
    })
    
    vis_df = vis_df.join(meta_df, on='node_id', how='left')
    
    output_path = f"{savepath}/DGI/{wandb_id}/tSNE_result_driver+am.csv"
    vis_df.to_csv(output_path, index=False)
    print(f"Saved t-SNE results for {model_name} to {output_path}")
    
    return model_name, vis_df

def process_original_features(savepath, org_res_feat_df, mut_df):
    """Calculates t-SNE based on the original features for test nodes."""
    print("Starting t-SNE processing for original features...")
    
    # 1. Get the list of node_ids from the npz file in the test folder 
    # (arbitrarily using test_embeddings.npz from the "All" model)
    all_model_id = MODELS['All']
    npz_path = f"{savepath}/DGI/{all_model_id}/test/test_embeddings.npz"
    
    if not os.path.exists(npz_path):
        print(f"Cannot process original features: {npz_path} not found.")
        return
        
    data = np.load(npz_path)
    test_node_ids = data['node_ids'].tolist()
            
    # 2. Filter only the rows corresponding to test_node_ids from org_res_feat_df
    filtered_org_df = org_res_feat_df[org_res_feat_df['node_id'].isin(test_node_ids)].copy()
    
    # If there are missing values (NaN), fill them with 0 as t-SNE will raise an error
    features = filtered_org_df[FEATURE_COLS].fillna(0).values

    # 3. Calculate t-SNE based on the original features
    print(f"Running TSNE on {len(features)} original nodes with {len(FEATURE_COLS)} features...")
    org_emb_2d = compute_tsne(features, max_iter=1500)

    # 4. Create a DataFrame for visualization (including max_am)
    vis_org_df = pd.DataFrame({
        'node_id': filtered_org_df['node_id'].values,
        'tsne_1': org_emb_2d[:, 0],
        'tsne_2': org_emb_2d[:, 1],
        'max_am': filtered_org_df['max_am'].values
    })

    # Merge with mut_df to get label information
    vis_org_df = vis_org_df.merge(mut_df[['node_id', 'label']], on='node_id', how='left')
    
    output_path = f'{savepath}/DGI/tSNE_result_all-feat_org.csv'
    vis_org_df.to_csv(output_path, index=False)
    print(f"Saved original features t-SNE to {output_path}")

# ==========================================
# Main Execution Flow
# ==========================================

def main():
    # Load Configurations
    global_config = load_yaml("config/RRGCL.yaml")
    DATABASE = global_config.DATABASE
    SAVEPATH = getattr(global_config, 'SAVEPATH', DATABASE) # fallback to DATABASE if not found

    # Load Feature DataFrame
    print(f"Loading feature data from {DATABASE}...")
    org_res_feat_df = pd.read_csv(f'{DATABASE}/merged_feature_data_v041226.csv')
    org_res_feat_df['max_am'] = org_res_feat_df.filter(like='am').max(axis=1)

    # Load Cancer Driver DataFrame
    print("Loading cancer driver data...")
    mut_df = pd.read_csv('models/matched_cancer_driver_df.csv')

    # Create metadata DataFrame for merging
    meta_df = mut_df[['node_id', 'label']].merge(org_res_feat_df[['node_id', 'max_am']], on='node_id', how='left')
    meta_df = meta_df.set_index('node_id')

    # Run parallel processing for model embeddings
    print("Starting parallel t-SNE processing for models...")
    process_func = partial(process_model_embeddings, savepath=SAVEPATH, meta_df=meta_df)
    
    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(process_func, MODELS.items()))

    # Run processing for original features
    process_original_features(SAVEPATH, org_res_feat_df, mut_df)
    
    print("All tasks completed successfully.")

if __name__ == '__main__':
    main()