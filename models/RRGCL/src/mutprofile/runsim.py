import os
import argparse

import numpy as np
import pandas as pd

import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from models.RRGCL.src.mutprofile.similarity import sim_pair_node

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_jobs", type=int, default=1, help='Number of jobs for parallel processing')
    parser.add_argument('--n_sample', type=int, default=10_000, help='Number of samples for similarity calculation')
    parser.add_argument('--feat_file', type=str, default='data/proc_data/merged_feature_data_v041226.csv')
    parser.add_argument('--ngb_cluster_file', type=str, default='neighbor_cluster_labels_eps0_mcs9_mss5_eom.csv')
    parser.add_argument('--res_cluster_file', type=str, default='residue_cluster_labels_eps0_mcs19_mss5_eom.csv')
    parser.add_argument('--sim_type', type=str, required=True, help='Choose among [am, pssm, hmm]')
    parser.add_argument('--sim_method', type=str, required=True, help='Choose among [pearson, cosine, entropy_corr, es_corr]')

    return parser.parse_args()


# [Core Optimization 1] Changed from single operation to 'batch operation' function
def calc_batch_sim(node_list, indexed_df, feat_prefix, sim_method, batch_size):
    if len(node_list) < 2:
        return []
    
    sims = []
    # Continuous calculation within a single process for the assigned batch_size (removing overhead)
    for _ in range(batch_size):
        node1, node2 = random.sample(node_list, 2)
        sim = sim_pair_node(indexed_df, node1, node2, feat_prefix=feat_prefix, sim_method=args.sim_method)
        sims.append(sim)
    return sims


def main(args):

    org_res_feat_df = pd.read_csv(args.feat_file)
    ngb_cluster_df = pd.read_csv(args.ngb_cluster_file)
    res_cluster_df = pd.read_csv(args.res_cluster_file)

    res_cluster_df.rename({'cluster_label': 'cluster'}, axis=1, inplace=True)
    ngb_cluster_df.rename({'cluster_label': 'cluster'}, axis=1, inplace=True)

    indexed_df = org_res_feat_df.set_index('node_id')

    # ---------------------------------------------------------
    # Residue Cluster Visualization
    # ---------------------------------------------------------
    res_cluster_idxs = sorted([c for c in res_cluster_df.cluster.unique()])

    fig, axs = plt.subplots(10, 5, figsize=(15, 20))
    axs = axs.ravel()

    last_idx = -1

    for i, cls_idx in enumerate(tqdm(res_cluster_idxs, desc="Processing Res Clusters")):
        if i >= len(axs): break
        
        temp = res_cluster_df[res_cluster_df['cluster'] == cls_idx]
        node_lists = list(temp['node_id'])
        
        if len(node_lists) < 2:
            continue
        
        # [Core Optimization 2] Split total sample count into batches equal to n_jobs
        n_jobs_actual = min(args.n_jobs, args.n_sample) 
        batch_sizes = [args.n_sample // n_jobs_actual] * n_jobs_actual
        batch_sizes[-1] += args.n_sample % n_jobs_actual # Assign remaining samples to the last core
        
        sim_list_nested = Parallel(n_jobs=args.n_jobs)(
            delayed(calc_batch_sim)(node_lists, indexed_df, args.sim_type, args.sim_method, b_size) 
            for b_size in batch_sizes
        )
        
        # Flatten the 2D list (results per batch) into a 1D list
        sim_list = [sim for sublist in sim_list_nested for sim in sublist]
        
        if sim_list:
            axs[i].hist(sim_list, bins=30, alpha=0.7)
            axs[i].set_title(f'Cluster {cls_idx}')

        last_idx = i
    
    for j in range(last_idx + 1, len(axs)):
        axs[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'residue_cluster_sim_{args.sim_type}_{args.sim_method}.png', dpi=600, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # Neighbor Cluster Visualization
    # ---------------------------------------------------------
    nbh_cluster_idxs = sorted([c for c in ngb_cluster_df.cluster.unique()])

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs = axs.ravel()

    last_idx = -1

    for i, cls_idx in enumerate(tqdm(nbh_cluster_idxs, desc="Processing Ngb Clusters")):
        if i >= len(axs): break
        
        temp = ngb_cluster_df[ngb_cluster_df['cluster'] == cls_idx]
        node_lists = list(temp['node_id'])
        
        if len(node_lists) < 2:
            continue
        
        # Split into batches as before
        n_jobs_actual = min(args.n_jobs, args.n_sample)
        batch_sizes = [args.n_sample // n_jobs_actual] * n_jobs_actual
        batch_sizes[-1] += args.n_sample % n_jobs_actual
        
        sim_list_nested = Parallel(n_jobs=args.n_jobs)(
            delayed(calc_batch_sim)(node_lists, indexed_df, args.sim_type, args.sim_method, b_size) 
            for b_size in batch_sizes
        )
        
        sim_list = [sim for sublist in sim_list_nested for sim in sublist]
        
        if sim_list:
            axs[i].hist(sim_list, bins=30, alpha=0.7)
            axs[i].set_title(f'Cluster {cls_idx}')

        last_idx = i
    
    for j in range(last_idx + 1, len(axs)):
        axs[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'neighbor_cluster_sim_{args.sim_type}_{args.sim_method}.png', dpi=600, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    args = get_args()
    main(args)