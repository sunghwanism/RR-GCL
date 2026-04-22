import pandas as pd
import numpy as np
import networkx as nx

from utils.graphfunction import get_uniprot_from_nodes


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

    for _, row in feat_df.iterrows():
        node_id = row['node_id']
        u_id = get_uniprot_from_nodes(node_id)

        new_row = {'node_id': node_id}

        # Get neighbors from graph
        if node_id not in graph:
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

        # Float columns: weighted average
        for col in float_cols:
            neighbor_values = feat_df.loc[neighbor_indices, col].values.astype(float)
            # Handle NaN in neighbor values
            valid_mask = ~np.isnan(neighbor_values)
            if valid_mask.any() and weight_sum > 0:
                valid_vals = neighbor_values[valid_mask]
                valid_w = weights[valid_mask]
                w_sum = valid_w.sum()
                if w_sum > 0:
                    new_row[col] = np.average(valid_vals, weights=valid_w)
                else:
                    new_row[col] = np.nan
            else:
                new_row[col] = np.nan

        # String columns: concatenate as list
        for col in str_cols:
            neighbor_values = feat_df.loc[neighbor_indices, col].values.tolist()
            new_row[col] = str(neighbor_values)

        result_rows.append(new_row)

    result_df = pd.DataFrame(result_rows)

    # Reorder columns to match input
    result_df = result_df[feat_df.columns.tolist()]

    return result_df
