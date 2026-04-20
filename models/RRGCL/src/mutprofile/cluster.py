import hdbscan
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import argparse
import os
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from joblib import Parallel, delayed
import itertools

from utils.functions import load_yaml
from data.augmentation import ss_aug
from data.scaler import sin_cos_transform, pca_transform

import warnings
warnings.filterwarnings('ignore')


ss_cols = ['node_id','rel_sasa', 'depth', 'hse_up', 'hse_down', 'dssp_accessibility', 'dssp_TCO', 'dssp_alpha', 'dssp_phi', 'dssp_psi']

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, choices=['residue', 'neighbor'], required=True, help='Target of clustering (residue or neighbor)')
    parser.add_argument('--ngb_feat_path', type=str, default='data/proc_data/ngb_avg_feat.csv', help='Path to neighbor features')
    parser.add_argument("--config", type=str, default='config/RRGCL.yaml', help='Path to config file')
    parser.add_argument("--n_jobs", type=int, default=1, help='Number of jobs for parallel processing')
    parser.add_argument("--param", type=str, default='models/RRGCL/src/mutprofile/param.yaml', help='Path to param file')
    return parser.parse_args()

def load_data(DATABASE, target, used_cols):

    print(f"Loading data from {DATABASE}...")
    feat_df = pd.read_csv(DATABASE)
    
    # AAindex1
    print("Processing AAindex1 features...")
    aa1_cols = [col for col in feat_df.columns if col.startswith('aa1')]
    AA1 = feat_df[['node_id']+aa1_cols].copy()
    if 'aa1' in AA1.columns:
        AA1.drop('aa1', axis=1, inplace=True)

    pca_cols = [col for col in aa1_cols if col != 'aa1']

    pca_aa1_df, pca = pca_transform(AA1, pca_cols, 2)
    print("AAindex Explained variance:", pca.explained_variance_ratio_)

    # Secondary Structure
    print("Processing Secondary Structure features...")
    ss_feat_df = feat_df[ss_cols]
    only_ss_cols = [col for col in ss_cols if col != 'node_id']

    ## Augmentation
    print("Augmenting Secondary Structure features...")
    aug_ss_df = ss_aug(ss_feat_df, only_ss_cols, num_ref=5)

    if 'dssp_phi' in ss_cols:
        aug_ss_df = sin_cos_transform(aug_ss_df, 'dssp_phi', 'sin')
        aug_ss_df = sin_cos_transform(aug_ss_df, 'dssp_phi', 'cos')
        aug_ss_df.drop(['dssp_phi'], axis=1, inplace=True)
    
    if 'dssp_psi' in ss_cols:
        aug_ss_df = sin_cos_transform(aug_ss_df, 'dssp_psi', 'sin')
        aug_ss_df = sin_cos_transform(aug_ss_df, 'dssp_psi', 'cos')
        aug_ss_df.drop(['dssp_psi'], axis=1, inplace=True)  


    if 'dssp_alpha' in ss_cols:
        aug_ss_df = sin_cos_transform(aug_ss_df, 'dssp_alpha', 'sin')
        aug_ss_df = sin_cos_transform(aug_ss_df, 'dssp_alpha', 'cos')
        aug_ss_df.drop(['dssp_alpha'], axis=1, inplace=True)

    feat_for_sim = pd.merge(aug_ss_df, pca_aa1_df, on='node_id', how='left')

    # Correlation Visualization
    print("Generating correlation visualization...")
    corr = feat_for_sim.select_dtypes(include=['number']).corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Clustering Features Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('clustering_feat_corr.png', dpi=600, bbox_inches='tight')
    plt.close()

    # Feature Selection
    aa1_pca_col = list(pca_aa1_df.columns)

    sim_cols = aa1_pca_col + used_cols

    final_sim_df = feat_for_sim[sim_cols]

    id_cols = ['node_id']
    feature_cols = [col for col in final_sim_df.columns if col not in id_cols]

    node_X = final_sim_df[feature_cols].values

    print("Scaling features...")
    scaler = StandardScaler()
    node_X_scaled = scaler.fit_transform(node_X)

    final_sim_df.to_csv(f'{target}_feat_for_cluster_df.csv', index=False)

    return node_X_scaled, final_sim_df['node_id']

def evaluate_hdbscan(eps, mcs, node_feat_scaled):
    clusterer = hdbscan.HDBSCAN(
        cluster_selection_epsilon=eps,
        min_cluster_size=mcs,
        gen_min_span_tree=True
    )
    labels = clusterer.fit_predict(node_feat_scaled)
    valid_labels = [l for l in labels if l >= 0]
    num_clusters = len(set(valid_labels))    
    
    noise_count = np.sum(labels < 0)
    noise_ratio = noise_count / len(labels)
    
    dbcv_score = np.nan
    if num_clusters > 1:
        try:
            dbcv_score = clusterer.relative_validity_
        except:
            dbcv_score = np.nan
            
    return {
        'eps': eps,
        'min_cluster_size': mcs,
        'DBCV': dbcv_score,
        'noise_ratio': noise_ratio,
        'num_clusters': num_clusters
    }

def grid_search(node_feat_scaled, node_ids, target, eps_range, min_cluster_range, n_jobs=1):
    print(f"Starting grid search for HDBSCAN parameters (n_jobs={n_jobs})...")
    
    param_grid = list(itertools.product(eps_range, min_cluster_range))
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_hdbscan)(eps, mcs, node_feat_scaled) for eps, mcs in tqdm(param_grid, desc="Grid Search")
    )

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{target}_cluster_results_gridsearch.csv", index=False)

    best_params = results_df.nlargest(3, 'DBCV')

    print("Grid search completed. Top 3 candidates for clustering:")
    print(best_params)

    for idx, row in best_params.iterrows():
        eps = row['eps']
        mcs = int(row['min_cluster_size'])
        
        clusterer = hdbscan.HDBSCAN(
            cluster_selection_epsilon=eps,
            min_cluster_size=mcs,
            gen_min_span_tree=True
        )
        
        labels = clusterer.fit_predict(node_feat_scaled)

        pd.DataFrame({
            'node_id': node_ids,
            'cluster_label': labels
        }).to_csv(f"{target}_cluster_labels_eps{eps}_mcs{mcs}.csv", index=False)
        
        print(f"Saved: {target}_cluster_labels_eps{eps}_mcs{mcs}.csv")

    return results_df

def elbow_plot(df, eps_range, metric="DBCV", target="residue"):
    for eps in eps_range:
        plt.figure(figsize=(8, 5))
        eps_df = df[df['eps'] == eps]
        sns.lineplot(data=eps_df, x="min_cluster_size", y=metric, marker="o")
        plt.title(f"Elbow Plot for eps = {eps}")
        plt.savefig(f'{target}_elbow_plot_{metric}_{eps}.png', dpi=600, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":

    args = get_args()
    config = load_yaml(args.config)
    param = load_yaml(args.param)

    if args.target == 'neighbor':
        target_param = param.neighbor
    elif args.target == 'residue':
        target_param = param.residue
    else:
        raise ValueError(f"Invalid target: {args.target}")

    eps_range = target_param['eps']
    min_cluster_range = target_param['min_cluster_size']

    if args.target == 'neighbor':
        print(f"Starting clustering pipeline for target: {args.target}")
        if not os.path.exists(args.ngb_feat_path):
            raise FileNotFoundError(f"Neighbor features not found at {args.ngb_feat_path}")
        else:
            feat_scaled, node_ids = load_data(args.ngb_feat_path, args.target, target_param['used_cols'])
    elif args.target == 'residue':
        print(f"Starting clustering pipeline for target: {args.target}")
        if not os.path.exists(config.DATABASE):
            raise FileNotFoundError(f"Database not found at {config.DATABASE}")
        else:
            DATABASE = f'{config.DATABASE}/merged_feature_data_v041226.csv'
            feat_scaled, node_ids = load_data(DATABASE, args.target, target_param['used_cols'])

    print("Running HDBSCAN grid search...")
    results_df = grid_search(feat_scaled, node_ids, args.target, eps_range, min_cluster_range, args.n_jobs)
    
    print("Generating elbow plots...")
    elbow_plot(results_df, eps_range, metric="DBCV", target=args.target)
    elbow_plot(results_df, eps_range, metric="noise_ratio", target=args.target)
    elbow_plot(results_df, eps_range, metric="num_clusters", target=args.target)
    print("Pipeline completed successfully.")