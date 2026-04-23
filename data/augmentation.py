import pandas as pd
import numpy as np

from utils.graphfunction import get_uniprot_from_nodes


def ss_aug(df, aug_cols, num_ref=10, mode='pos', graph=None):
    """
    Secondary structure augmentation for NaN imputation.

    Args:
        df: DataFrame with 'node_id' column and feature columns.
        aug_cols: list of column names to augment.
        num_ref: window size for positional mode, or not used in neighbor mode.
        mode: 'pos' for positional local mean, 'neighbor' for graph-neighbor mean.
        graph: networkx Graph required when mode='neighbor'.
    """
    aug_df = df.copy()

    if mode == 'neighbor' and graph is None:
        raise ValueError("mode='neighbor' requires a graph argument.")

    if 'uniprot' not in aug_df.columns:
        aug_df['uniprot'] = aug_df['node_id'].apply(get_uniprot_from_nodes)
    if 'pos' not in aug_df.columns:
        aug_df['pos'] = aug_df['node_id'].apply(lambda x: int(x.split('_')[1]))

    for col in aug_cols:
        if col in ('rel_sasa', 'depth', 'hse_up', 'hse_down',
                   'dssp_phi', 'dssp_psi', 'dssp_alpha',
                   'dssp_TCO', 'dssp_accessibility', 'ss_helix', 'ss_sheet', 'ss_loop'):
            if mode == 'pos':
                aug_df = impute_local_mean(aug_df, col, num_ref)
            elif mode == 'neighbor':
                aug_df = impute_neighbor_mean(aug_df, col, graph)
            else:
                raise ValueError(f"Unknown mode: {mode}. Must be 'pos' or 'neighbor'.")
        else:
            raise ValueError(f"Unknown Augmentation Method for column: {col}")

    aug_df.drop(columns=['uniprot', 'pos'], inplace=True)

    return aug_df


def impute_local_mean(df, colname, window_size=10):
    """Impute NaN values using positional local mean within the same uniprot_id."""
    original_values = df[colname].copy()
    nan_indices = df[df[colname].isna()].index

    if len(nan_indices) == 0:
        return df

    for idx in nan_indices:
        u_id = df.loc[idx, 'uniprot']
        pos = df.loc[idx, 'pos']

        mask = (df['uniprot'] == u_id) & (df['pos'].between(pos - window_size, pos + window_size))
        local_mean = original_values[mask].mean()

        if pd.isna(local_mean):
            local_mean = 0 
            
        df.at[idx, colname] = local_mean

    return df


def impute_neighbor_mean(df, colname, graph):
    """
    Impute NaN values using graph-neighbor mean.
    For each NaN node, find its neighbors in the graph,
    filter to those sharing the same uniprot_id,
    and use their mean value for imputation.

    Args:
        df: DataFrame with 'node_id', 'uniprot', and feature columns.
        colname: column name to impute.
        graph: networkx Graph where nodes correspond to node_id values.
    """
    original_values = df[colname].copy()
    nan_indices = df[df[colname].isna()].index

    if len(nan_indices) == 0:
        return df

    # Build a lookup: node_id -> index for fast access
    node_id_to_idx = pd.Series(df.index, index=df['node_id'])

    for idx in nan_indices:
        node_id = df.loc[idx, 'node_id']
        u_id = df.loc[idx, 'uniprot']

        # Get graph neighbors
        if node_id not in graph:
            # Node not in graph — fallback to 0
            df.at[idx, colname] = 0
            continue

        neighbors = list(graph.neighbors(node_id))

        # Filter neighbors: same uniprot_id and present in df
        same_uniprot_neighbors = [
            n for n in neighbors
            if get_uniprot_from_nodes(n) == u_id and n in node_id_to_idx.index
        ]

        if not same_uniprot_neighbors:
            df.at[idx, colname] = 0
            continue

        # Get indices of matching neighbors in df
        neighbor_indices = node_id_to_idx[same_uniprot_neighbors].values
        neighbor_mean = original_values.iloc[neighbor_indices].mean()

        if pd.isna(neighbor_mean):
            neighbor_mean = 0

        df.at[idx, colname] = neighbor_mean

    return df