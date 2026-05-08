import os
import argparse
import time
import random
import numpy as np

import torch
import networkx as nx

from utils.functions import load_yaml, set_seed, formatTime, init_wandb
from models.DGI.dataset import create_dgi_loaders, load_data, augment_connected_components
from models.Classifier.execute import run_downstream

from models.Classifier.FFN import FFNClassifier
from models.Classifier.GATClassifier import GATClassifier
from models.Classifier.SAGEClassifier import GNNClassifier as SAGEClassifier

def get_parser():
    parser = argparse.ArgumentParser(description="Train Downstream Classifier.")

    parser.add_argument('--DATABASE', type=str, help='Path to database')
    parser.add_argument('--FeatFile', type=str, default='merged_feature_data_v041226.csv', help='Path to features file')
    parser.add_argument('--GraphFile', type=str, default='cleaned_weighted_graph_041226.pkl', help='Path to graph file')
    parser.add_argument('--config', type=str, default='config/DGI.yaml', help='Path to config file')
    parser.add_argument('--clf_config', type=str, default='config/clf.yaml', help='Path to classifier config file')
    parser.add_argument('--SAVEPATH', type=str, default=None, help='Path to save results')
    parser.add_argument('--node_att', type=str, nargs='+', default=None, help='List of node attributes to override config')
    
    # Classifier Specific Args
    parser.add_argument('--clf_model', type=str, default='FFN', choices=['FFN', 'GAT', 'SAGE'], help='Classifier model to use')
    parser.add_argument('--load_model', type=str, default='DGI', help='Name of the pretrained model (for embeddings)')
    parser.add_argument('--load_wandb_id', type=str, default='SanityCheck', help='Wandb ID for the pretrained model')
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_coef', type=float, default=0.00001)
    
    parser.add_argument('--seed', type=int, default=42, help='Seed for random number generator')
    
    # wandb args
    parser.add_argument('--nowandb', action='store_true', help='Do not use wandb')
    parser.add_argument('--wandb_key', type=str, default=None, help='wandb API key')
    parser.add_argument('--project_name', type=str, default='RR-GCL-Classifier', help='wandb project name')
    parser.add_argument('--entity_name', type=str, default=None, help='wandb entity name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb run name')
    
    return parser

def main():
    args = get_parser().parse_args()
    config = load_yaml(args.config)
    clf_config = load_yaml(args.clf_config)

    # Merge clf_config into config
    for k, v in vars(clf_config).items():
        setattr(config, k, v)

    # Overwrite with args (highest priority)
    for k, v in vars(args).items():
        if v is not None:
            setattr(config, k, v)
            
    set_seed(config.seed)
    print(config)

    print("================ [Load Dataset] ================")
    start = time.time()
    graph = load_data(config)
    print(f"[Load Dataset] Done! Time: {formatTime(time.time() - start)}")

    start = time.time()
    cc_list = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
    cc_list.sort(key=lambda x: len(x.nodes()), reverse=True)

    aug_anchor_ratio = getattr(config, 'aug_anchor_ratio', 0.10)
    hop_ratios = getattr(config, 'hop_ratios', [0.9, 0.8, 0.5, 0.5, 0.5])
    aug_min_node = getattr(config, 'aug_min_node', 30)

    rng = random.Random(config.seed)
    
    num_total = len(cc_list)
    train_val_idx = rng.sample(range(num_total), min(124, num_total))
    exclude_values = {45, 46}
    train_val_idx = [i for i in train_val_idx if i not in exclude_values]
    test_idx = [i for i in range(num_total) if i not in train_val_idx]
    
    val_ratio = 0.15
    num_val = int(len(train_val_idx) * val_ratio)
    val_idx = rng.sample(train_val_idx, num_val)
    train_idx = [i for i in train_val_idx if i not in val_idx]

    train_idx.sort()
    val_idx.sort()
    test_idx.sort()

    print(f"[Data Augmentation] Augmenting ONLY train graph components...")
    train_data_list = augment_connected_components(cc_list, graph, aug_anchor_ratio, hop_ratios, min_aug_node=aug_min_node, idx_list=train_idx, seed=config.seed)
    
    val_data_list = [cc_list[i] for i in val_idx]
    test_data_list = [cc_list[i] for i in test_idx] 
    
    print("================ [Generate Train/Val/Test DataLoader] ================")
    train_loader = create_dgi_loaders(train_data_list, batch_size=config.batch_size, shuffle=True)
    val_loader = create_dgi_loaders(val_data_list, batch_size=config.batch_size, shuffle=False)
    test_loader = create_dgi_loaders(test_data_list, batch_size=config.batch_size, shuffle=False)
    
    first_batch = next(iter(train_loader))
    in_ft_gnn = first_batch.x.size(1)
    
    in_ft_ffn = config.in_ft_ffn
    n_cls = config.n_cls
    out_ft_list = config.out_ft_list
    activation = config.activation
    drop_prob = config.drop_prob

    print(f"Initializing {config.clf_model} model...")
    if config.clf_model == 'FFN':
        # Check actual .npy shape robustly
        sample_node = None
        for attr in ['node_names', 'node_id']:
            if hasattr(first_batch, attr):
                val = getattr(first_batch, attr)
                if isinstance(val, list) and len(val) > 0:
                    if isinstance(val[0], list) and len(val[0]) > 0:
                        sample_node = val[0][0]
                    elif isinstance(val[0], (str, int)):
                        sample_node = val[0]
                if sample_node: break
        
        if sample_node:
            # Check train/val/test folders for the sample node
            found_npy = False
            for split in ['train', 'val', 'test']:
                npy_path = os.path.join(config.SAVEPATH, config.load_model, config.load_wandb_id, split, f"{sample_node}.npy")
                if os.path.exists(npy_path):
                    sample_npy = np.load(npy_path)
                    actual_dim = sample_npy.shape[0]
                    if actual_dim != in_ft_ffn:
                        print("============================================================")
                        print(f"[WARNING] Config in_ft_ffn ({in_ft_ffn}) is different from actual .npy dimension ({actual_dim}).")
                        print(f"-> Automatically changing in_ft_ffn to {actual_dim} to proceed.")
                        print("============================================================")
                        in_ft_ffn = actual_dim
                    found_npy = True
                    break
            if not found_npy:
                print(f"[INFO] Could not find .npy file for sample node {sample_node} to verify dimension. Proceeding with config value.")
                    
        clf_model = FFNClassifier(in_ft_ffn, out_ft_list, activation, drop_prob, n_cls)
    elif config.clf_model == 'GAT':
        clf_model = GATClassifier(in_ft_gnn, out_ft_list, activation, drop_prob, n_cls)
    elif config.clf_model == 'SAGE':
        clf_model = SAGEClassifier(in_ft_gnn, out_ft_list, activation, drop_prob, n_cls)
    else:
        raise ValueError(f"Unknown clf_model: {config.clf_model}")

    # Initialize wandb
    use_wandb = getattr(config, 'nowandb', False)
    run_wandb = None
    if not use_wandb:
        # Avoid init_wandb trying to resume from DGI's wandb_run_id by unsetting load_pretrained
        # or setting it safely for classifier.
        if hasattr(config, 'load_pretrained'):
            config.load_pretrained = False
        run_wandb = init_wandb(config)

    run_downstream(config, clf_model, train_loader, val_loader, test_loader, run_wandb=run_wandb)

if __name__ == '__main__':
    main()
