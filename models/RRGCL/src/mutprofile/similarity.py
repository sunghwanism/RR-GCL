import os
import numpy as np
import pandas as pd
import warnings

from utils.functions import load_yaml
from data.reference import residue3to1
from utils.graphfunction import get_res_from_nodes

class CKACalculator:
    def __init__(self, eps=1e-8):
        self.eps = eps

    def normalize_rows(self, X):
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)
        return (X - mean) / (std + self.eps)

    def _centering(self, K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    def _hsic(self, K, L):
        K_c = self._centering(K)
        L_c = self._centering(L)
        return np.sum(K_c * L_c)

    def score(self, X, Y, perform_normalization=True):
        if perform_normalization:
            X = self.normalize_rows(X)
            Y = self.normalize_rows(Y)

        K = X @ X.T
        L = Y @ Y.T

        hsic_kl = self._hsic(K, L)
        hsic_kk = self._hsic(K, K)
        hsic_ll = self._hsic(L, L)

        return hsic_kl / np.sqrt(hsic_kk * hsic_ll + self.eps)

def RVcoefficient(X, Y):

    X = X - np.mean(X, axis=0)
    Y = Y - np.mean(Y, axis=0)
    
    S_X = np.dot(X, X.T)
    S_Y = np.dot(Y, Y.T)
    
    numerator = np.trace(np.dot(S_X, S_Y))
    
    denominator = np.sqrt(np.trace(np.dot(S_X, S_X)) * np.trace(np.dot(S_Y, S_Y)))
    
    return numerator / denominator

def sim_pair_node(df_indexed, node1, node2, feat_prefix, sim_method, max_val_sum=None, feat_cols=None):
    """
    Extracts feature vectors for two given nodes and calculates similarity using the specified sim_method.
    Designed to be easily called individually by the user.
    
    Args:
        df_indexed (pd.DataFrame): DataFrame with 'node_id' set as the index.
        node1 (str): First node ID.
        node2 (str): Second node ID.
        feat_prefix (str): Prefix of the features to use (e.g., 'hmm', 'pssm').
        sim_method (str): Similarity calculation method ('pearson', 'cosine', 'entropy_corr', 'es_corr').
        max_val_sum (float, optional): Pre-calculated max IC denominator (improves speed in batch processing).
        feat_cols (list, optional): Pre-extracted list of feature columns.
    """
    if df_indexed.index.name != 'node_id':
        raise ValueError("df_indexed must have 'node_id' as its index. Please run df.set_index('node_id') first.")
        
    try:
        row1 = df_indexed.loc[node1]
        row2 = df_indexed.loc[node2]
    except KeyError:
        return None
        
    if feat_cols is None:
        feat_cols = [col for col in df_indexed.columns if col.startswith(feat_prefix)]
        
    if max_val_sum is None:
        if sim_method == 'entropy_corr':
            max_val_sum = df_indexed[f'{feat_prefix}_entropy'].max() * 2
        elif sim_method == 'es_corr':
            max_val_sum = (151 - df_indexed[f'{feat_prefix}_neff'].min()) * 2
        else:
            max_val_sum = 1 # Not used
            
    res1 = residue3to1[get_res_from_nodes(node1).upper()].upper()
    res2 = residue3to1[get_res_from_nodes(node2).upper()].upper()
    
    res1_suffix = f"_{res1}"
    res2_suffix = f"_{res2}"
    
    # Column filtering (protects Transition columns by using exact suffix matching)
    if sim_method == 'entropy_corr':
        mask = [not (col.endswith(res1_suffix) or col.endswith(res2_suffix) or ('entropy' in col.lower())) for col in feat_cols]
    elif sim_method == 'es_corr':
        mask = [not (col.endswith(res1_suffix) or col.endswith(res2_suffix) or ('neff' in col.lower())) for col in feat_cols]
    else:
        mask = [not (col.endswith(res1_suffix) or col.endswith(res2_suffix)) for col in feat_cols]
    
    # O(1) row lookup and masking
    v1 = row1[feat_cols].values.astype(np.float32)[mask]
    v2 = row2[feat_cols].values.astype(np.float32)[mask]
    
    # Calculate similarity
    if sim_method == 'pearson':
        sim = np.corrcoef(v1, v2)[0, 1]
    elif sim_method == 'cosine':
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
        sim = np.dot(v1, v2) / denom if denom != 0 else 0
    elif sim_method == 'entropy_corr':
        sim = np.corrcoef(v1, v2)[0, 1]
        ic1 = row1[f'{feat_prefix}_entropy']
        ic2 = row2[f'{feat_prefix}_entropy']
        sim = sim * ((ic1 + ic2) / max_val_sum)
    elif sim_method == 'es_corr':
        masked_cols = [c for i, c in enumerate(feat_cols) if mask[i]]
        E_idx = [i for i, c in enumerate(masked_cols) if len(c.split('_')[-1]) == 1]
        T_idx = [i for i, c in enumerate(masked_cols) if len(c.split('_')[-1]) == 2]
        
        E1, E2 = v1[E_idx], v2[E_idx]
        T1, T2 = v1[T_idx], v2[T_idx]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            S_E = np.corrcoef(E1, E2)[0, 1]
            S_T = np.corrcoef(T1, T2)[0, 1]
        
        if np.isnan(S_E): S_E = 0
        if np.isnan(S_T): S_T = 0
        
        alpha = 0.75
        S_base = (alpha * S_E) + ((1 - alpha) * S_T)
        
        ic1 = 151 - row1[f'{feat_prefix}_neff']
        ic2 = 151 - row2[f'{feat_prefix}_neff']
        sim = S_base * ((ic1 + ic2) / max_val_sum)
    else:
        raise ValueError(f"Unknown similarity method: {sim_method}")
        
    return sim
