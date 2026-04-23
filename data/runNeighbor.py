import os
import json
import numpy as np
import pandas as pd

import time
import argparse

from utils.functions import load_yaml, print_time
from utils.graphfunction import load_graph

from data.neighborExtractor import extract_neighbor_features_parallel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config/RRGCL.yaml')
    parser.add_argument('--save_path', type=str, default='data/proc_data/ngb_weighted_avg_feat.csv')
    parser.add_argument('--n_jobs', type=int)
    return parser.parse_args()


def main(args):
    start = time.time()

    # Load Configuration
    config = load_yaml(args.config_path)
    DATABASE = config.DATABASE

    # Load Data
    finalG = load_graph((f"{DATABASE}/cleaned_weighted_graph_041226.pkl"))
    org_res_feat_df = pd.read_csv(f'{DATABASE}/merged_feature_data_v041226.csv')

    # Preprocessing
    try:
        slurm_cpus = os.getenv('SLURM_CPUS_ON_NODE')
        print("SLURM_CPUS_ON_NODE:", slurm_cpus)
        n_jobs = min(int(slurm_cpus), args.n_jobs)
    except:
        print("No allocated CPUS from SLURM")
        n_jobs = os.cpu_count() - 1
    print("--> Run NUM CPUS", n_jobs)

    org_ngb_feat_df, failed_nodes = extract_neighbor_features_parallel(org_res_feat_df, finalG, n_jobs=n_jobs)
    org_ngb_feat_df.to_csv(args.save_path, index=False)

    with open('data/failed_calc_neighbor.py', 'w') as f:
        f.write(f"failed_nodes = {failed_nodes}\n")

    end = time.time()
    finish_time = print_time(end-start)
    print(f"[Finish] Total Processing Time: {finish_time}")

if __name__ == "__main__":
    args = get_args()
    main(args)
    