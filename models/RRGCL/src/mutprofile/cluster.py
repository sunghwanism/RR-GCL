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
from utils.graphfunction import load_graph
from data.augmentation import ss_aug
from data.scaler import sin_cos_transform

import warnings
warnings.filterwarnings('ignore')


ss_cols = ['rel_sasa', 'depth', 'hse_up', 'hse_down', 'dssp_accessibility', 'dssp_TCO', 'dssp_alpha', 'dssp_phi', 'dssp_psi']

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, choices=['residue', 'neighbor'], required=True, help='Target of clustering (residue or neighbor)')
    parser.add_argument('--ngb_feat_path', type=str, default='data/proc_data/ngb_avg_feat.csv', help='Path to neighbor features')
    parser.add_argument("--config", type=str, default='config/RRGCL.yaml', help='Path to config file')
    parser.add_argument("--n_jobs", type=int, default=1, help='Number of jobs for parallel processing')
    parser.add_argument("--param", type=str, default='models/RRGCL/src/mutprofile/param.yaml', help='Path to param file')
    parser.add_argument("--aug_mode", type=str, choices=['pos', 'neighbor'], default='pos', help='Augmentation mode: pos (positional local mean) or neighbor (graph-neighbor mean)')
    parser.add_argument("--graph_path", type=str, default=None, help='Path to graph pickle file (required when --aug_mode=neighbor)')
    return parser.parse_args()

def load_data(DATABASE, target, cluster_cols, aug_mode='pos', graph_path=None):

    print(f"Loading data from {DATABASE}...")
    feat_df = pd.read_csv(DATABASE)

    # Select clustering columns
    print("Selecting clustering columns...")
    cluster_feat_df = feat_df[['node_id'] + cluster_cols].copy()

    # Secondary Structure Augmentation
    if aug_mode == 'neighbor':
        if graph_path is None:
            raise ValueError("--graph_path is required when --aug_mode=neighbor")
        print(f"Augmenting Secondary Structure features (mode=neighbor, graph={graph_path})...")
        graph = load_graph(graph_path)
        cluster_feat_df = ss_aug(cluster_feat_df, ss_cols, mode='neighbor', graph=graph)
    else:
        print("Augmenting Secondary Structure features (mode=pos, num_ref=10)...")
        cluster_feat_df = ss_aug(cluster_feat_df, ss_cols, mode='pos', num_ref=10, )

    # Sin/Cos Transform (keep original angle columns)
    print("Applying sin/cos transforms...")
    for angle_col in ['dssp_phi', 'dssp_psi', 'dssp_alpha']:
        if angle_col in cluster_cols:
            cluster_feat_df = sin_cos_transform(cluster_feat_df, angle_col, 'sin')
            cluster_feat_df = sin_cos_transform(cluster_feat_df, angle_col, 'cos')

    # Correlation Visualization
    print("Generating correlation visualization...")
    corr = cluster_feat_df.select_dtypes(include=['number']).corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(f'[{target}] Clustering Features Correlation Heatmap')
    plt.tight_layout()

    asset_dir = f'asset/{target}'
    os.makedirs(asset_dir, exist_ok=True)
    plt.savefig(f'{asset_dir}/{target}_clustering_feat_corr.png', dpi=600, bbox_inches='tight')
    plt.close()

    # Feature Scaling
    id_cols = ['node_id']
    feature_cols = [col for col in cluster_feat_df.columns if col not in id_cols]

    node_X = cluster_feat_df[feature_cols].values

    print("Scaling features...")
    scaler = StandardScaler()
    node_X_scaled = scaler.fit_transform(node_X)

    cluster_feat_df.to_csv(f'{target}_feat_for_cluster_df.csv', index=False)

    return node_X_scaled, cluster_feat_df['node_id']

def evaluate_hdbscan(eps, mcs, mss, csm, node_feat_scaled):
    clusterer = hdbscan.HDBSCAN(
        cluster_selection_epsilon=eps,
        min_cluster_size=mcs,
        min_samples=mss,
        cluster_selection_method=csm,
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
        'min_sample_size': mss,
        'cluster_selection_method': csm,
        'DBCV': dbcv_score,
        'noise_ratio': noise_ratio,
        'num_clusters': num_clusters
    }

def grid_search(node_feat_scaled, node_ids, target, eps_range, min_cluster_range, min_sample_range, cluster_selection_methods, n_jobs=1):
    print(f"Starting grid search for HDBSCAN parameters (n_jobs={n_jobs})...")

    if isinstance(cluster_selection_methods, str):
        cluster_selection_methods = [cluster_selection_methods]

    # HDBSCAN defaults for empty param lists — effectively excludes that dimension from grid search
    hdbscan_defaults = {
        'eps': [0.0],
        'min_cluster_size': [5],
        'min_sample_size': [None],
        'cluster_selection_method': ['eom'],
    }

    if not eps_range:
        print(f"  eps_range is empty, using default: {hdbscan_defaults['eps']}")
        eps_range = hdbscan_defaults['eps']
    if not min_cluster_range:
        print(f"  min_cluster_range is empty, using default: {hdbscan_defaults['min_cluster_size']}")
        min_cluster_range = hdbscan_defaults['min_cluster_size']
    if not min_sample_range:
        print(f"  min_sample_range is empty, using default: {hdbscan_defaults['min_sample_size']}")
        min_sample_range = hdbscan_defaults['min_sample_size']
    if not cluster_selection_methods:
        print(f"  cluster_selection_methods is empty, using default: {hdbscan_defaults['cluster_selection_method']}")
        cluster_selection_methods = hdbscan_defaults['cluster_selection_method']

    param_grid = list(itertools.product(eps_range, min_cluster_range, min_sample_range, cluster_selection_methods))
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_hdbscan)(eps, mcs, mss, csm, node_feat_scaled) for eps, mcs, mss, csm in tqdm(param_grid, desc="Grid Search")
    )

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{target}_cluster_results_gridsearch.csv", index=False)

    best_params = results_df.nlargest(3, 'DBCV')

    print("Grid search completed. Top 3 candidates for clustering:")
    print(best_params)

    for idx, row in best_params.iterrows():
        eps = float(row['eps'])
        mcs = int(row['min_cluster_size'])
        mss = int(row['min_sample_size'])
        csm = row['cluster_selection_method']
        
        clusterer = hdbscan.HDBSCAN(
            cluster_selection_epsilon=eps,
            min_cluster_size=mcs,
            min_samples=mss,
            cluster_selection_method=csm,
            gen_min_span_tree=True
        )
        
        labels = clusterer.fit_predict(node_feat_scaled)

        pd.DataFrame({
            'node_id': node_ids,
            'cluster_label': labels
        }).to_csv(f"{target}_cluster_labels_eps{eps}_mcs{mcs}_mss{mss}_{csm}.csv", index=False)
        
        print(f"Saved: {target}_cluster_labels_eps{eps}_mcs{mcs}_mss{mss}_{csm}.csv")

    return results_df

def elbow_plot(df, eps_range, cluster_selection_methods, metric="DBCV", target="residue"):
    asset_dir = f'asset/{target}'
    os.makedirs(asset_dir, exist_ok=True)

    if isinstance(cluster_selection_methods, str):
        cluster_selection_methods = [cluster_selection_methods]

    for csm in cluster_selection_methods:
        csm_df = df[df['cluster_selection_method'] == csm]
        for eps in eps_range:
            plt.figure(figsize=(8, 5))
            eps_df = csm_df[csm_df['eps'] == eps]
            sns.lineplot(data=eps_df, x="min_cluster_size", y=metric, hue="min_sample_size", marker="o")
            plt.title(f"Elbow Plot for eps = {eps} (method={csm})")
            plt.savefig(f'{asset_dir}/{csm}_elbow_plot_{metric}_{eps}.png', dpi=600, bbox_inches='tight')
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
    min_sample_range = target_param['min_sample_size']
    cluster_selection_methods = target_param['cluster_selection_method']

    if args.target == 'neighbor':
        print(f"Starting clustering pipeline for target: {args.target}")
        if not os.path.exists(args.ngb_feat_path):
            raise FileNotFoundError(f"Neighbor features not found at {args.ngb_feat_path}")
        else:
            feat_scaled, node_ids = load_data(args.ngb_feat_path, args.target, target_param['cluster_cols'], aug_mode=args.aug_mode, graph_path=args.graph_path)
    elif args.target == 'residue':
        print(f"Starting clustering pipeline for target: {args.target}")
        if not os.path.exists(config.DATABASE):
            raise FileNotFoundError(f"Database not found at {config.DATABASE}")
        else:
            DATABASE = f'{config.DATABASE}/merged_feature_data_v041226.csv'
            feat_scaled, node_ids = load_data(DATABASE, args.target, target_param['cluster_cols'], aug_mode=args.aug_mode, graph_path=args.graph_path)

    print("Running HDBSCAN grid search...")
    results_df = grid_search(feat_scaled, node_ids, args.target, eps_range, min_cluster_range, min_sample_range, cluster_selection_methods, args.n_jobs)
    
    print("Generating elbow plots...")
    elbow_plot(results_df, eps_range, cluster_selection_methods, metric="DBCV", target=args.target)
    elbow_plot(results_df, eps_range, cluster_selection_methods, metric="noise_ratio", target=args.target)
    elbow_plot(results_df, eps_range, cluster_selection_methods, metric="num_clusters", target=args.target)
    print("Pipeline completed successfully.")