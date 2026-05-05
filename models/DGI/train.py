import os
import argparse

import time
import random
import pickle
import networkx as nx

from utils.functions import load_yaml, set_seed, formatTime
from utils.graphfunction import load_graph
from models.DGI.dataset import create_dgi_loaders, cc_splitor, load_data, augment_connected_components
from models.DGI.execute import run_training


def get_parser():
    parser = argparse.ArgumentParser(description="Train DGI model on Inductive graphs.")

    parser.add_argument('--DATABASE', type=str, help='Path to database')
    parser.add_argument('--FeatFile', type=str, default='merged_feature_data_v041226.csv', help='Path to features file')
    parser.add_argument('--GraphFile', type=str, default='cleaned_weighted_graph_041226.pkl', help='Path to graph file')
    parser.add_argument('--config', type=str, default='config/DGI.yaml', help='Path to config file')
    
    # Training args overrides
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)

    # EarlyStopping args overrides
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--min_delta', type=float, default=None)

    # graph augmentation args
    parser.add_argument('--aug_anchor_ratio', type=float, default=0.01, help='Ratio of anchor nodes to total nodes')
    parser.add_argument('--aug_hop_ratios', type=list, default=[0.8, 0.4, 0.2], help='Ratios of neighbors to select at each hop')
    
    parser.add_argument('--seed', type=int, default=42, help='Seed for random number generator')
    return parser

def main():
    print("================ [Load Configuration] ================")
    args = get_parser().parse_args()
    config = load_yaml(args.config)

    # if var is overwritted, then update config (args is priority)
    vars(config).update(vars(args))
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

    # split cc_list into training and testing sets
    rng = random.Random(config.seed)
    train_idx = rng.sample(range(135), 124)
    exclude_values = {45, 46}  # incldue cancer driver
    train_idx = [i for i in train_idx if i not in exclude_values]
    test_idx = [i for i in range(135) if i not in train_idx]
    train_idx.sort()
    test_idx.sort()

    augmented_cc_list = augment_connected_components(cc_list, graph, aug_anchor_ratio, hop_ratios, min_aug_node=aug_min_node, idx_list=train_idx, seed=config.seed)
    
    print(f"[Data Augmentation] Augmented graph has {len(augmented_cc_list)} connected components. Time: {formatTime(time.time() - start)}")
    
    # Create PyG DataLoaders
    print("================ [Generate Train/Val/Test DataLoader] ================")
    start = time.time()
    val_ratio = 0.15
    num_val = int(len(augmented_cc_list)*val_ratio)

    aug_val_idx = rng.sample(range(len(augmented_cc_list)), num_val)
    aug_train_idx = [i for i in range(len(augmented_cc_list)) if i not in aug_val_idx]  
    
    aug_val_idx.sort()
    aug_train_idx.sort()

    train_data_list = [augmented_cc_list[i] for i in aug_train_idx]
    val_data_list = [augmented_cc_list[i] for i in aug_val_idx]
    test_data_list = [cc_list[i] for i in test_idx] 
    
    batch_size = getattr(config, 'batch_size', 32)
    
    train_loader = create_dgi_loaders(train_data_list, batch_size=batch_size, shuffle=True)
    val_loader = create_dgi_loaders(val_data_list, batch_size=batch_size, shuffle=False)
    test_loader = create_dgi_loaders(test_data_list, batch_size=batch_size, shuffle=False)
    print(f"DataLoaders created. Train: {len(aug_train_idx)}, Val: {len(aug_val_idx)}, Test: {len(test_idx)}. Time: {formatTime(time.time() - start)}")
    
    # print("Starting DGI Training...")
    # # Run DGI training function
    # run_training(config, train_loader, val_loader, test_loader)

if __name__ == '__main__':
    main()