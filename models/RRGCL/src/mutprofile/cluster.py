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
from data.scaler import sin_cos_transform, pca_transform

import warnings
warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, choices=['residue', 'neighbor'], required=True, help='Target of clustering (residue or neighbor)')
    parser.add_argument('--feat_path', type=str, default='data/proc_data/aug_ngb_orthogonal_feature_data_for_clustering.csv', help='Path to neighbor features')
    parser.add_argument("--config", type=str, default='config/RRGCL.yaml', help='Path to config file')
    parser.add_argument("--n_jobs", type=int, default=1, help='Number of jobs for parallel processing')
    parser.add_argument("--param", type=str, default='models/RRGCL/src/mutprofile/param.yaml', help='Path to param file')
    parser.add_argument("--grid_sc", action='store_true', help='Perform grid search if set')
    return parser.parse_args()

def load_data(FEAT_PATH, target):

    print(f"Loading data from {FEAT_PATH}...")
    feat_df = pd.read_csv(FEAT_PATH)

    # evol_df = feat_df[['node_id','pssm_entropy', 'hmm_neff']]

    # 1. AA1 PCA
    print("Extracting and PCA transforming AA1 features...")
    aa1_cols = [col for col in feat_df.columns if col.startswith('aa1') and col != 'aa1']
    aa1_df = feat_df[['node_id'] + aa1_cols].copy()
    aa1_df[aa1_cols] = StandardScaler().fit_transform(aa1_df[aa1_cols])
    pca_aa1_df, _ = pca_transform(aa1_df, aa1_cols, 2)
    
    # 2. Secondary Structure
    print("Extracting and PCA transforming Secondary Structure features...")
    ss_cols = ['rel_sasa', 'depth', 'hse_up', 'hse_down', 'dssp_accessibility', 'dssp_TCO', 'dssp_alpha', 'dssp_phi', 'dssp_psi']
    ss_df = feat_df[['node_id'] + ss_cols].copy()
    
    for angle_col in ['dssp_phi', 'dssp_psi', 'dssp_alpha']:
        ss_df = sin_cos_transform(ss_df, angle_col, 'sin')
        ss_df = sin_cos_transform(ss_df, angle_col, 'cos')

    ss_pca_cols1 = ['rel_sasa', 'depth', 'hse_up', 'hse_down', 'dssp_accessibility']
    ss_pca_cols2 = ['dssp_TCO', 'dssp_phi_sin', 'dssp_phi_cos', 'dssp_psi_sin', 'dssp_psi_cos', 'dssp_alpha_sin', 'dssp_alpha_cos']
    all_ss_pca_cols = ss_pca_cols1 + ss_pca_cols2

    ss_df[all_ss_pca_cols] = StandardScaler().fit_transform(ss_df[all_ss_pca_cols])

    pca_ss_df1, _ = pca_transform(ss_df, ss_pca_cols1, 2)
    
    if target == 'neighbor':
        # pca_ss_df2, _ = pca_transform(ss_df, ss_pca_cols2, 4) # for otrhogonal feature
        pca_ss_df2, _ = pca_transform(ss_df, ss_pca_cols2, 2)
    else:
        pca_ss_df2, _ = pca_transform(ss_df, ss_pca_cols2, 2)

    # Combine
    print("Combining engineered features...")
    cluster_feat_df = pd.DataFrame({'node_id': feat_df['node_id']})
    cluster_feat_df = pd.merge(cluster_feat_df, pca_aa1_df, on='node_id', how='left')
    cluster_feat_df = pd.merge(cluster_feat_df, pca_ss_df1, on='node_id', how='left')
    cluster_feat_df = pd.merge(cluster_feat_df, pca_ss_df2, on='node_id', how='left')
    
    # Evolutionary Information
    # cluster_feat_df = pd.merge(cluster_feat_df, evol_df, on='node_id', how='left')
    ## For orthogonal feature
    # if target == 'neighbor':
    #     cluster_feat_df.drop(['hmm_neff'], axis=1, inplace=True)

    # if target == 'neighbor':
    #     ngb_specific_cols = ['ss_helix', 'ss_sheet', 'ss_loop']
    #     cluster_feat_df = pd.merge(cluster_feat_df, feat_df[['node_id'] + ngb_specific_cols], on='node_id', how='left')

    # if target == 'neighbor':
    #     ngb_specific_cols = ['ss_helix', 'ss_sheet', 'ss_loop']
    #     feat_df[ngb_specific_cols] = StandardScaler().fit_transform(feat_df[ngb_specific_cols])
    #     pca_ss_df3, pca_ss3 = pca_transform(feat_df, ngb_specific_cols, 1)
    #     print(f"PCA SS3 explained variance ratio: {pca_ss3.explained_variance_ratio_}")
    #     cluster_feat_df = pd.merge(cluster_feat_df, pca_ss_df3, on='node_id', how='left')

    # Normalization right before clustering
    print("Normalizing features before clustering...")
    feature_cols = [col for col in cluster_feat_df.columns if col != 'node_id']
    cluster_feat_df[feature_cols] = StandardScaler().fit_transform(cluster_feat_df[feature_cols])

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

    node_X_scaled = cluster_feat_df[feature_cols].values
    cluster_feat_df.to_csv(f'{target}_feat_for_cluster_df.csv', index=False)

    return node_X_scaled, cluster_feat_df

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
    print("Parameters for Clustering")
    print(param)

    if args.target == 'neighbor':
        target_param = param.neighbor
        assert 'ngb' in args.feat_path, "Feature path must contain 'ngb' for neighbor clustering"
    elif args.target == 'residue':
        target_param = param.residue
        assert 'res' in args.feat_path, "Feature path must contain 'rs' for residue clustering"
    else:
        raise ValueError(f"Invalid target: {args.target}")

    eps_range = target_param['eps']
    min_cluster_range = target_param['min_cluster_size']
    min_sample_range = target_param['min_sample_size']
    cluster_selection_methods = target_param['cluster_selection_method']

    print(f"Starting clustering pipeline for target: {args.target}")
    if not os.path.exists(args.feat_path):
        raise FileNotFoundError(f"Features not found at {args.feat_path}")
    else:
        feat_scaled, proc_feat_df = load_data(args.feat_path, args.target)

    if not args.grid_sc:
        params_dict = {
            'eps': eps_range,
            'min_cluster_size': min_cluster_range,
            'min_sample_size': min_sample_range,
            'cluster_selection_method': cluster_selection_methods
        }
        
        hdbscan_defaults = {
            'eps': 0.0,
            'min_cluster_size': 5,
            'min_sample_size': None,
            'cluster_selection_method': 'eom',
        }

        final_params = {}
        for param_name, param_val in params_dict.items():
            if isinstance(param_val, list):
                if len(param_val) > 1:
                    raise ValueError(f"Parameter '{param_name}' has multiple values {param_val}, but --grid_sc is not set. Please use --grid_sc to run grid search or provide only one value.")
                elif len(param_val) == 1:
                    final_params[param_name] = param_val[0]
                else:
                    final_params[param_name] = hdbscan_defaults[param_name]
            elif param_val is None:
                final_params[param_name] = hdbscan_defaults[param_name]
            else:
                final_params[param_name] = param_val
                
        print(f"Running HDBSCAN with single parameter set: {final_params}")
        clusterer = hdbscan.HDBSCAN(
            cluster_selection_epsilon=final_params['eps'],
            min_cluster_size=final_params['min_cluster_size'],
            min_samples=final_params['min_sample_size'],
            cluster_selection_method=final_params['cluster_selection_method'],
            gen_min_span_tree=True
        )
        labels = clusterer.fit_predict(feat_scaled)
        save_name = f"{args.target}_cluster_labels_eps{final_params['eps']}_mcs{final_params['min_cluster_size']}_mss{final_params['min_sample_size']}_{final_params['cluster_selection_method']}.csv"
        cluster_results = pd.DataFrame({'node_id': proc_feat_df['node_id'], 'cluster': labels})
        cluster_results = pd.merge(cluster_results, proc_feat_df, on='node_id', how='left')
        cluster_results.to_csv(save_name, index=False)
        print(f"Saved: {save_name}")
        print("Pipeline completed successfully.")
    else:
        print("Running HDBSCAN grid search...")
        results_df = grid_search(feat_scaled, node_ids, args.target, eps_range, min_cluster_range, min_sample_range, cluster_selection_methods, args.n_jobs)
        
        # print("Generating elbow plots...")
        # elbow_plot(results_df, eps_range, cluster_selection_methods, metric="DBCV", target=args.target)
        # elbow_plot(results_df, eps_range, cluster_selection_methods, metric="noise_ratio", target=args.target)
        # elbow_plot(results_df, eps_range, cluster_selection_methods, metric="num_clusters", target=args.target)
        print("Pipeline completed successfully.")