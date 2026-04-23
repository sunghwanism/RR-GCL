import pandas as pd
import numpy as np
import networkx as nx
from joblib import Parallel, delayed
import multiprocessing
import ast
import re
from utils.functions import get_optimized_n_jobs
from utils.graphfunction import get_uniprot_from_nodes

from tqdm import tqdm


def extract_neighbor_features(feat_df, graph, weight_key='weight'):
    """
    Build a neighbor feature table from node features and a graph.

    For each node in feat_df, extract its graph neighbors (same uniprot_id),
    then:
      - float columns: weighted average using edge weights
      - str/object columns: concatenate into a list representation

    Args:
        feat_df: DataFrame with 'node_id' column and feature columns.
        graph: networkx Graph with edge weights.
        weight_key: edge attribute key for weights (default: 'weight').

    Returns:
        DataFrame with the same columns as feat_df, where values are
        neighbor-aggregated.
    """
    # Build a fast lookup: node_id -> row index
    node_id_to_idx = pd.Series(feat_df.index, index=feat_df['node_id'])

    # Identify column types
    float_cols = feat_df.select_dtypes(include=[np.number]).columns.tolist()
    str_cols = feat_df.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'node_id' in str_cols:
        str_cols.remove('node_id')
    if 'node_id' in float_cols:
        float_cols.remove('node_id')

    result_rows = []
    failed_nodes = []

    for _, row in tqdm(feat_df.iterrows(), total=len(feat_df)):
        node_id = row['node_id']
        u_id = get_uniprot_from_nodes(node_id)

        new_row = {'node_id': node_id}

        # Get neighbors from graph
        if node_id not in graph:
            print(f"[Warning] Node {node_id} is not in the graph. Using default/NaN values.")
            failed_nodes.append(node_id)
            # Node not in graph: fill with defaults
            for col in float_cols:
                new_row[col] = np.nan
            for col in str_cols:
                new_row[col] = '[]'
            result_rows.append(new_row)
            continue

        neighbors = list(graph.neighbors(node_id))

        # Filter: same uniprot_id and present in feat_df
        valid_neighbors = []
        valid_weights = []
        for n in neighbors:
            if get_uniprot_from_nodes(n) == u_id and n in node_id_to_idx.index:
                edge_data = graph.edges[node_id, n]
                w = edge_data.get(weight_key, 1.0)
                valid_neighbors.append(n)
                valid_weights.append(w)

        if not valid_neighbors:
            print(f"[Warning] Node {node_id} has no valid neighbors (same uniprot_id). Using default/NaN values.")
            failed_nodes.append(node_id)
            for col in float_cols:
                new_row[col] = np.nan
            for col in str_cols:
                new_row[col] = '[]'
            result_rows.append(new_row)
            continue

        # Get neighbor indices
        neighbor_indices = node_id_to_idx[valid_neighbors].values
        weights = np.array(valid_weights)
        weight_sum = weights.sum()

        # Normalize weights so they sum to 1 (relative importance among neighbors)
        if weight_sum > 0:
            norm_weights = weights / weight_sum
        else:
            norm_weights = np.ones_like(weights) / len(weights)

        # Float columns: weighted average using normalized weights
        for col in float_cols:
            neighbor_values = feat_df.loc[neighbor_indices, col].values.astype(float)
            # Handle NaN in neighbor values
            valid_mask = ~np.isnan(neighbor_values)
            if valid_mask.any():
                valid_vals = neighbor_values[valid_mask]
                valid_w = norm_weights[valid_mask]
                # Re-normalize after masking NaNs
                valid_w_sum = valid_w.sum()
                if valid_w_sum > 0:
                    valid_w = valid_w / valid_w_sum
                    new_row[col] = np.dot(valid_w, valid_vals)
                else:
                    new_row[col] = np.nan
            else:
                new_row[col] = np.nan

        # String columns: extract unique values
        for col in str_cols:
            raw_vals = feat_df.loc[neighbor_indices, col].values
            unique_vals = []
            for v in raw_vals:
                if pd.isna(v): continue
                v_str = str(v).strip()
                if v_str.lower() in ('nan', 'none', ''): continue
                
                if v_str.startswith('[') and v_str.endswith(']'):
                    # Replace unquoted nan with None for ast.literal_eval
                    v_str_clean = re.sub(r'\bnan\b', 'None', v_str)
                    try:
                        parsed = ast.literal_eval(v_str_clean)
                        if isinstance(parsed, list):
                            for x in parsed:
                                x_str = str(x)
                                if x is None or x_str.lower() in ('nan', 'none', ''): continue
                                if x_str not in unique_vals:
                                    unique_vals.append(x_str)
                        else:
                            p_str = str(parsed)
                            if parsed is not None and p_str.lower() not in ('nan', 'none', '') and p_str not in unique_vals:
                                unique_vals.append(p_str)
                    except (ValueError, SyntaxError):
                        if v_str not in unique_vals:
                            unique_vals.append(v_str)
                else:
                    if v_str not in unique_vals:
                        unique_vals.append(v_str)
            
            if not unique_vals:
                new_row[col] = np.nan
            else:
                new_row[col] = str(unique_vals)

        result_rows.append(new_row)

    result_df = pd.DataFrame(result_rows)

    # Reorder columns to match input
    result_df = result_df[feat_df.columns.tolist()]

    return result_df, failed_nodes

def _process_chunk(chunk_df, full_feat_df, graph, node_id_to_idx, float_cols, str_cols, weight_key):
    result_rows = []
    failed_nodes = []
    
    for _, row in chunk_df.iterrows():
        node_id = row['node_id']
        u_id = get_uniprot_from_nodes(node_id)
        new_row = {'node_id': node_id}

        if node_id not in graph:
            print(f"[Warning] Node {node_id} is not in the graph. Using default/NaN values.")
            failed_nodes.append(node_id)
            for col in float_cols: new_row[col] = np.nan
            for col in str_cols: new_row[col] = '[]'
            result_rows.append(new_row)
            continue

        neighbors = list(graph.neighbors(node_id))
        valid_neighbors = []
        valid_weights = []
        
        for n in neighbors:
            if n in node_id_to_idx:
                edge_data = graph.edges[node_id, n]
                valid_neighbors.append(n)
                valid_weights.append(edge_data.get(weight_key, 1.0))

        if not valid_neighbors:
            print(f"[Warning] Node {node_id} has no valid neighbors. Using default/NaN values.")
            failed_nodes.append(node_id)
            for col in float_cols: new_row[col] = np.nan
            for col in str_cols: new_row[col] = '[]'
        else:
            neighbor_indices = node_id_to_idx.loc[valid_neighbors].values
            weights = np.array(valid_weights)
            weight_sum = weights.sum()

            # Normalize weights so they sum to 1 (relative importance among neighbors)
            if weight_sum > 0:
                norm_weights = weights / weight_sum
            else:
                norm_weights = np.ones_like(weights) / len(weights)

            for col in float_cols:
                vals = full_feat_df.loc[neighbor_indices, col].values.astype(float)
                mask = ~np.isnan(vals)
                if mask.any():
                    valid_w = norm_weights[mask]
                    valid_w_sum = valid_w.sum()
                    if valid_w_sum > 0:
                        valid_w = valid_w / valid_w_sum
                        new_row[col] = np.dot(valid_w, vals[mask])
                    else:
                        new_row[col] = np.nan
                else:
                    new_row[col] = np.nan

            for col in str_cols:
                raw_vals = full_feat_df.loc[neighbor_indices, col].values
                unique_vals = []
                for v in raw_vals:
                    if pd.isna(v): continue
                    v_str = str(v).strip()
                    if v_str.lower() in ('nan', 'none', ''): continue
                    
                    if v_str.startswith('[') and v_str.endswith(']'):
                        # Replace unquoted nan with None for ast.literal_eval
                        v_str_clean = re.sub(r'\bnan\b', 'None', v_str)
                        try:
                            parsed = ast.literal_eval(v_str_clean)
                            if isinstance(parsed, list):
                                for x in parsed:
                                    x_str = str(x)
                                    if x is None or x_str.lower() in ('nan', 'none', ''): continue
                                    if x_str not in unique_vals:
                                        unique_vals.append(x_str)
                            else:
                                p_str = str(parsed)
                                if parsed is not None and p_str.lower() not in ('nan', 'none', '') and p_str not in unique_vals:
                                    unique_vals.append(p_str)
                        except (ValueError, SyntaxError):
                            if v_str not in unique_vals:
                                unique_vals.append(v_str)
                    else:
                        if v_str not in unique_vals:
                            unique_vals.append(v_str)
                
                if not unique_vals:
                    new_row[col] = np.nan
                else:
                    new_row[col] = str(unique_vals)

        result_rows.append(new_row)
    
    return result_rows, failed_nodes

def extract_neighbor_features_parallel(feat_df, graph, weight_key='weight', n_jobs=-1, chunk_size=5000):
    """
    Extract neighbor features in parallel. 
    Uses chunk_size to balance load across cores and avoid memory overflow.
    """
    # Pre-calculate metadata once to avoid redundant operations in chunks
    node_id_to_idx = pd.Series(feat_df.index, index=feat_df['node_id'])
    float_cols = feat_df.select_dtypes(include=[np.number]).columns.drop('node_id', errors='ignore').tolist()
    str_cols = feat_df.select_dtypes(include=['object', 'category']).columns.drop('node_id', errors='ignore').tolist()

    # Determine number of workers
    num_cores = get_optimized_n_jobs(n_jobs)
    
    # Split dataframe into manageable chunks for better load balancing
    total_rows = len(feat_df)
    chunks = [feat_df.iloc[i : i + chunk_size] for i in range(0, total_rows, chunk_size)]
    
    print(f"Total Rows: {total_rows} | Chunk Size: {chunk_size} | Total Chunks: {len(chunks)}")

    # Execute parallel processing
    # Using 'loky' backend is recommended for robust process-based parallelism
    results = Parallel(n_jobs=num_cores, backend="loky")(
        delayed(_process_chunk)(
            chunk, feat_df, graph, node_id_to_idx, float_cols, str_cols, weight_key
        ) for chunk in tqdm(chunks, desc="Processing Graph Chunks", total=len(chunks), unit="chunk")
    )

    # Flatten results efficiently
    flattened_results = []
    failed_nodes = []
    
    for res_rows, f_nodes in results:
        if res_rows:
            flattened_results.extend(res_rows)
        if f_nodes:
            failed_nodes.extend(f_nodes)

    # Concatenate results and restore original column order
    result_df = pd.DataFrame(flattened_results)
    
    # Ensure columns match original dataframe and return
    return result_df[feat_df.columns], failed_nodes