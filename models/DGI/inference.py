import os
import argparse
import time
import torch
import pandas as pd
import numpy as np
import networkx as nx

from torch_geometric.loader import DataLoader

from utils.functions import load_yaml, set_seed, formatTime
from models.DGI.dataset import load_data, nx_to_pyg_data
from models.DGI.models.dgi import DGI
from data.vocab import attr_mappings

def get_parser():
    parser = argparse.ArgumentParser(description="Inference DGI model to extract node embeddings.")

    parser.add_argument('--DATABASE', type=str, help='Path to database')
    parser.add_argument('--FeatFile', type=str, default='merged_feature_data_v041226.csv', help='Path to features file')
    parser.add_argument('--GraphFile', type=str, default='cleaned_weighted_graph_041226.pkl', help='Path to graph file')
    parser.add_argument('--config', type=str, default='config/DGI.yaml', help='Path to config file')
    parser.add_argument('--SAVEPATH', type=str, default=None, help='Path to save results')
    
    parser.add_argument('--model_path', type=str, default=None, help='Path to the trained model (BestPerformance.pth)')
    parser.add_argument('--output_path', type=str, default='node_embeddings.csv', help='Path to save the output node embeddings')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42, help='Seed for random number generator')
    
    return parser

def extract_all_node_embeddings(model, loader, device, node_ids):
    """
    Extract node embeddings for all graphs in the loader.
    Returns:
        node_ids: List of node IDs
        embeddings: Tensor of node embeddings
    """
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            cat_feats = {}
            for key in model.cat_feat_emb_dict.keys():
                if hasattr(batch, key):
                    cat_feats[key] = getattr(batch, key)
            
            # Extract node embeddings
            embeds, _ = model.embed(batch.x, cat_feats, batch.edge_index, batch.batch)
            all_embeddings.append(embeds.cpu())
            
    embeddings = torch.cat(all_embeddings, dim=0)
    
    return node_ids, embeddings

def main():
    print("================ [Load Configuration] ================")
    args = get_parser().parse_args()
    config = load_yaml(args.config)

    # if var is overwritted, then update config (args is priority)
    vars(config).update(vars(args))
    set_seed(config.seed)
    print(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("============================"*2)
    print(f'Using device: {device}')
    print("============================"*2)

    print("================ [Load Dataset] ================")
    start = time.time()
    # Load graph including node attribute by merging DataFrame and nx.graph object
    graph = load_data(config)
    print(f"[Load Dataset] Done! Time: {formatTime(time.time() - start)}")

    print("================ [Prepare DataLoader] ================")
    start = time.time()
    
    # Process connected components
    cc_list = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
    
    data_list = []
    flat_node_ids = []
    
    for comp in cc_list:
        data = nx_to_pyg_data(comp)
        data_list.append(data)
        flat_node_ids.extend(list(comp.nodes()))
        
    loader = DataLoader(data_list, batch_size=config.batch_size, shuffle=False)
    print(f"DataLoader created. Graphs: {len(data_list)}, Total Nodes: {len(flat_node_ids)}. Time: {formatTime(time.time() - start)}")

    # Initialize model
    print("================ [Initialize Model] ================")
    first_batch = next(iter(loader))
    num_ft_size = first_batch.x.size(1)
    cat_feat_num_dict = {}
    for key in first_batch.keys():
        if key != 'x' and key in config.node_att:
            if hasattr(first_batch, key):
                if key in attr_mappings:
                    cat_feat_num_dict[key] = max(attr_mappings[key].values()) + 1

    hid_units = config.model_param['hidden_dims']
    nonlinearity = config.model_param['activation']
    drop_prob = config.model_param['drop_prob']
    emb_dim = config.model_param['emb_dim']

    model = DGI(num_ft_size, cat_feat_num_dict, emb_dim, hid_units, nonlinearity, drop_prob).to(device)
    
    model_path = config.model_path
    if model_path is None:
        if hasattr(config, 'SAVEPATH') and config.SAVEPATH is not None:
            BASESAVEPATH = os.path.join(config.SAVEPATH, 'DGI')
            model_path = os.path.join(BASESAVEPATH, 'BestPerformance.pth')
        else:
            print("Error: Please provide --model_path or ensure SAVEPATH is set in config.")
            return
            
    if os.path.exists(model_path):
        print(f"Loading best model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Error: Model not found at {model_path}")
        return

    print("================ [Extract Embeddings] ================")
    start = time.time()
    
    node_ids, embeddings = extract_all_node_embeddings(model, loader, device, flat_node_ids)
    
    print(f"Extracted embeddings shape: {embeddings.shape}. Time: {formatTime(time.time() - start)}")
    
    print("================ [Save Results] ================")
    # Save as .npy files
    embeddings_np = embeddings.numpy()
    
    if hasattr(config, 'SAVEPATH') and config.SAVEPATH is not None:
        BASESAVEPATH = os.path.join(config.SAVEPATH, 'DGI')
    else:
        BASESAVEPATH = os.path.dirname(config.output_path) if os.path.dirname(config.output_path) else '.'
        
    test_dir = os.path.join(BASESAVEPATH, 'test')
    os.makedirs(test_dir, exist_ok=True)
    
    for i, node_id in enumerate(node_ids):
        np.save(os.path.join(test_dir, f"{node_id}.npy"), embeddings_np[i])
        
    print(f"Successfully saved {len(node_ids)} node embeddings to {test_dir}")

if __name__ == '__main__':
    main()
