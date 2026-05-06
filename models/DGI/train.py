import os
import argparse

import time
import random
import pickle
import networkx as nx

from utils.functions import load_yaml, set_seed, formatTime, init_wandb
from utils.graphfunction import load_graph
from models.DGI.dataset import create_dgi_loaders, cc_splitor, load_data, augment_connected_components
from models.DGI.execute import run_training


def get_parser():
    parser = argparse.ArgumentParser(description="Train DGI model on Inductive graphs.")

    parser.add_argument('--DATABASE', type=str, help='Path to database')
    parser.add_argument('--FeatFile', type=str, default='merged_feature_data_v041226.csv', help='Path to features file')
    parser.add_argument('--GraphFile', type=str, default='cleaned_weighted_graph_041226.pkl', help='Path to graph file')
    parser.add_argument('--config', type=str, default='config/DGI.yaml', help='Path to config file')
    parser.add_argument('--SAVEPATH', type=str, default=None, help='Path to save results')
    parser.add_argument('--load_pretrained', action='store_true', help='Load pretrained model')
    parser.add_argument('--node_att', type=str, nargs='+', default=None, help='List of node attributes to override config')
    
    # Training args overrides
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--l2_coef', type=float, default=0.00001)
    
    # EarlyStopping args overrides
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--min_delta', type=float, default=None)
    
    # LR Scheduler args
    parser.add_argument('--use_scheduler', action='store_true', help='Use ReduceLROnPlateau scheduler')
    parser.add_argument('--lr_patience', type=int, default=10, help='Patience for LR scheduler')
    parser.add_argument('--lr_factor', type=float, default=0.5, help='Factor to reduce LR')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum LR for scheduler')

    # graph augmentation args
    parser.add_argument('--aug_anchor_ratio', type=float, default=0.01, help='Ratio of anchor nodes to total nodes')
    parser.add_argument('--aug_hop_ratios', type=list, default=[0.9, 0.8, 0.5, 0.5, 0.5], help='Ratios of neighbors to select at each hop')
    
    parser.add_argument('--seed', type=int, default=42, help='Seed for random number generator')
    
    # wandb args
    parser.add_argument('--nowandb', action='store_true', help='Do not use wandb')
    parser.add_argument('--wandb_key', type=str, default=None, help='wandb API key')
    parser.add_argument('--project_name', type=str, default='RR-GCL', help='wandb project name')
    parser.add_argument('--entity_name', type=str, default=None, help='wandb entity name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb run name')
    
    return parser

def main():
    print("================ [Load Configuration] ================")
    args = get_parser().parse_args()
    config = load_yaml(args.config)

    # if var is overwritted, then update config (args is priority)
    for k, v in vars(args).items():
        if v is not None:
            setattr(config, k, v)
            
    set_seed(config.seed)
    print(config)

    print("================ [Load Dataset] ================")
    start = time.time()
    # Load graph including node attribute by merging DataFrame and nx.graph object
    graph = load_data(config)
    print(f"[Load Dataset] Done! Time: {formatTime(time.time() - start)}")

    # print("================ [Data Augmentation] ================")
    start = time.time()
    print(f"[Data Augmentation] Augmenting graph with anchor nodes...")
    cc_list = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
    cc_list.sort(key=lambda x: len(x.nodes()), reverse=True)

    # Augment connected components
    aug_anchor_ratio = getattr(config, 'aug_anchor_ratio', 0.10)
    hop_ratios = getattr(config, 'hop_ratios', [0.9, 0.8, 0.5, 0.5, 0.5])
    aug_min_node = getattr(config, 'aug_min_node', 30)

    # split cc_list into training, validation and testing sets
    rng = random.Random(config.seed)
    
    num_total = len(cc_list)
    train_val_idx = rng.sample(range(num_total), min(124, num_total))
    exclude_values = {45, 46}  # incldue cancer driver
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
    
    print(f"[Data Augmentation] Augmented train graph has {len(train_data_list)} connected components. Time: {formatTime(time.time() - start)}")
    
    # Create PyG DataLoaders
    print("================ [Generate Train/Val/Test DataLoader] ================")
    start = time.time() 
    
    batch_size = getattr(config, 'batch_size', 32)
    
    train_loader = create_dgi_loaders(train_data_list, batch_size=batch_size, shuffle=True)
    val_loader = create_dgi_loaders(val_data_list, batch_size=batch_size, shuffle=False)
    test_loader = create_dgi_loaders(test_data_list, batch_size=batch_size, shuffle=False)
    print(f"DataLoaders created. Train: {len(train_data_list)}, Val: {len(val_data_list)}, Test: {len(test_data_list)}. Time: {formatTime(time.time() - start)}")
    
    print("Starting DGI Training...")
    # Initialize wandb
    use_wandb = getattr(config, 'nowandb', False)
    run_wandb = None
    if not use_wandb:
        run_wandb = init_wandb(config)
    
    # Run DGI training function
    run_training(config, train_loader, val_loader, test_loader, run_wandb=run_wandb)

if __name__ == '__main__':
    main()