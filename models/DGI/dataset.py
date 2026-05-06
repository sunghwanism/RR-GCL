
import os
import random

import gc
import ast
import pandas as pd
import networkx as nx

import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx

from data.vocab import attr_mappings
from data.scaler import sin_cos_transform, standard_scale

from utils.graphfunction import load_graph


def create_dgi_loaders(cc_list, batch_size=32, shuffle=False):
    """
    Convert all NetworkX graphs in cc_list to PyG Data and
    return a single DataLoader.
    """
    data_list = []
    
    for component in cc_list:
        # Check if logic for converting node list data to
        # tensors is included inside nx_to_pyg_data.
        data = nx_to_pyg_data(component)
        data_list.append(data)

    # Create a single loader for the entire data
    # shuffle=True shuffles the order of graphs, not the order of nodes within a graph, during training.
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)
    
    return loader

def cc_splitor(connected_component, num_anchor, min_aug_node=50, hop_ratios=[0.8, 0.4], seed=None):
    """
    Subsamples a connected component (NetworkX graph) by:
    1. Randomly picking `num_anchor` nodes.
    2. For each anchor, iteratively sampling a given ratio of neighbors 
       for EACH node at the current hop based on `hop_ratios`.
    Returns a list of induced subgraphs, each originating from one anchor node.
    """
    rng = random.Random(seed) if seed is not None else random
    
    nodes = list(connected_component.nodes())
    
    # If the number of nodes is less than or equal to num_anchor, treat all possible nodes as anchors
    actual_num_anchor = min(len(nodes), num_anchor)
    anchors = rng.sample(nodes, actual_num_anchor)
    
    # Manage global visited records to prevent different anchors from picking the same node redundantly
    global_visited = set(anchors) 
    
    result_components = [] # List of subgraphs to be returned as result
    
    # Create independent Connected Component for each anchor node
    for anchor in anchors:
        component_nodes = {anchor} # Nodes belonging to the current anchor
        current_hop_nodes = [anchor]
        
        # Search according to the given hop_ratios
        for ratio in hop_ratios:
            next_hop_nodes = []
            
            # Check neighbors of "each node" in the current hop
            for node in current_hop_nodes:
                # Filter only nodes not in global visited records as candidates
                neighbors = set(connected_component.neighbors(node))
                candidates = list(neighbors - global_visited)
                
                if not candidates:
                    continue
                    
                # Randomly select based on ratio from candidate neighbors
                k_hop = int(len(candidates) * ratio)
                
                if k_hop > 0:
                    selected_hop = rng.sample(candidates, k_hop)
                    next_hop_nodes.extend(selected_hop)
                    
                    # Add to both current component and global visited records
                    component_nodes.update(selected_hop)
                    global_visited.update(selected_hop)
            
            # Update for next hop search
            current_hop_nodes = next_hop_nodes
            
            if not current_hop_nodes:
                break
        
        if len(component_nodes) > min_aug_node:
            # After search, add the subgraph created around the current anchor to the list
            result_components.append(connected_component.subgraph(component_nodes).copy())
        
    return result_components

def augment_connected_components(cc_list, graph, aug_anchor_ratio, hop_ratio, min_aug_node, idx_list, seed):
    aug_cc_list=[]

    nodes_in_idx_list = []
    for idx in idx_list:
        nodes_in_idx_list.extend(cc_list[idx])

    all_nodes = []
    for cc in cc_list:
        all_nodes.extend(cc)

    print(f"Total number of nodes for Train+Validation : {len(nodes_in_idx_list)} ({round(len(nodes_in_idx_list) / len(all_nodes), 4) * 100}%)")

    for idx in idx_list:
        cc = graph.subgraph(cc_list[idx]).copy()
        num_nodes_in_cc = len(cc.nodes())
        num_anchor = int(num_nodes_in_cc* aug_anchor_ratio)

        if num_anchor > 0:
            aug_cc_list.extend(cc_splitor(cc, num_anchor=num_anchor, min_aug_node=min_aug_node, hop_ratios=hop_ratio, seed=seed))
            aug_cc_list.append(cc)
    
    return aug_cc_list

def safe_parse(x):
    if x == 'None':
        return []
    if isinstance(x, str):
        if x.startswith('['):
            return ast.literal_eval(x)
        else:
            return [x] 
    return x

def load_data(config):
    print("[Load Dataset] Loading Graph...")
    graph_path = os.path.join(config.DATABASE, config.GraphFile)
    graph = load_graph(graph_path)
    
    print(f"Graph loaded: {len(graph.nodes)} nodes, {len(graph.edges)} edges.")

    print(f"[Load Dataset] Load Node Features.")
    feat_path = os.path.join(config.DATABASE, config.FeatFile)
    
    required_cols = set(['node_id', 'copy_idx', 'pos'] + config.node_att)
    
    feat_df = pd.read_csv(feat_path, usecols=lambda c: c in required_cols)
    
    valid_nodes = set(graph.nodes())
    feat_df = feat_df[feat_df['node_id'].isin(valid_nodes)].reset_index(drop=True)

    print("[Load Dataset] Preprocessing Node Features...")
    print("1. Applying sin cos transform to dssp_phi, dssp_psi, dssp_alpha")
    
    # 1. Trigonometric transforms
    for angle in ['dssp_phi', 'dssp_psi', 'dssp_alpha']:
        if angle in config.node_att:
            feat_df = sin_cos_transform(feat_df, angle, 'sin')
            feat_df = sin_cos_transform(feat_df, angle, 'cos')
            config.node_att.remove(angle)
            config.node_att.extend([f'{angle}_sin', f'{angle}_cos'])

    print("2. Preprocessing object type (Categorical Padding)")
    obj_cols = list(feat_df.select_dtypes(include=['object']).columns)
    feat_df[obj_cols] = feat_df[obj_cols].fillna('None')

    PAD_VALUE = 0
    cat_cols = [c for c in config.node_att if c in obj_cols]

    # 2. Vectorized Categorical Encoding & Padding
    for col in cat_cols:
        mapping_dict = attr_mappings.get(col, {})
        feat_df[col] = feat_df[col].apply(safe_parse)
        
        max_len = max(feat_df[col].apply(len).max(), 1)

        def fast_encode(val_list):
            mapped = [mapping_dict.get(str(v), PAD_VALUE) for v in val_list]
            return mapped + [PAD_VALUE] * (max_len - len(mapped))

        feat_df[col] = feat_df[col].apply(fast_encode)

    # ---------------------------------------------------------
    # Continuous Column Identification
    cont_cols = [c for c in config.node_att if c not in obj_cols]
    if 'copy_idx' in cont_cols: cont_cols.remove('copy_idx')
    if 'pos' in cont_cols: cont_cols.remove('pos')

    feat_df[cont_cols] = feat_df[cont_cols].fillna(0.0)

    # 3. Normalize Continuous Features
    print("[Load Dataset] Normalize Continuous Features")
    norm_cols = []
    for col in cont_cols:
        if 'sin' not in col and 'cos' not in col:
            feat_df = standard_scale(feat_df, col)
            norm_cols.append(col)

    print("[Load Dataset] Normalized columns: ", norm_cols)

    # ---------------------------------------------------------
    print("3. Organizing features for separate Embedding Layers")
    
    # We already filtered valid nodes, so we just set the index
    feat_df.set_index('node_id', inplace=True)

    # 4. Highly optimized assignment
    # Convert all continuous columns + copy_idx to a single 'x' column
    feat_df['x'] = feat_df[cont_cols + ['copy_idx']].astype(float).values.tolist()

    # Keep only the newly created 'x' column and the categorical columns
    cols_to_keep = ['x'] + cat_cols
    feat_df = feat_df[cols_to_keep]

    # Convert DataFrame to dictionary {node_id: {'x': [...], 'cat_col1': [...], ...}}
    node_attr_dict = feat_df.to_dict(orient='index')
    
    # Apply attributes to graph
    nx.set_node_attributes(graph, node_attr_dict)

    del feat_df, node_attr_dict
    gc.collect()
    # ---------------------------------------------------------

    # 5. Set edge attributes
    edge_att_val = config.model_param.get('edge_att')
    if edge_att_val is None or str(edge_att_val).lower() == 'none':
        for u, v, d in graph.edges(data=True):
            d.clear()

    print(f"[Load Dataset] Final Node Attributes || Use Edge Weight {edge_att_val}")
    print(f"Continuous features (x): {cont_cols}")
    print(f"Categorical features (x_cat): {cat_cols}")

    return graph

def shuffle_node_features(x):
    """
    Shuffles the node features.
    """
    if x is None:
        return None
    idx = torch.randperm(x.size(0))
    return x[idx]
    
def nx_to_pyg_data(G):
    """
    Converts a NetworkX graph to a PyG Data object.
    Extracts 'x' (continuous features) and dynamically converts 
    any categorical list features into torch.long tensors.
    """
    data = from_networkx(G)
    
    if hasattr(data, 'x') and data.x is not None:
        if not isinstance(data.x, torch.Tensor):
            x_val = data.x.tolist() if hasattr(data.x, 'tolist') else data.x
            data.x = torch.tensor(x_val, dtype=torch.float)
        elif data.x.dtype != torch.float:
            data.x = data.x.float()
            
    # Find attributes other than the standard PyG keys (e.g., pssm, sequence_pattern) and convert them to LongTensor
    standard_keys = {'x', 'y', 'edge_index', 'edge_weight', 'edge_attr', 'node_id', 'num_nodes', 'weight', 'cleaned_total_energy'}
    for key in data.keys():
        if key not in standard_keys:
            val = getattr(data, key)
            if val is not None:
                if not isinstance(val, torch.Tensor):
                    # Convert list data to tensor
                    val_list = val.tolist() if hasattr(val, 'tolist') else val
                    setattr(data, key, torch.tensor(val_list, dtype=torch.long))
                elif val.dtype != torch.long:
                    # If it is already a tensor but of a different type such as float, cast it to long (for use in Embedding layers)
                    setattr(data, key, val.long())
                    
    return data