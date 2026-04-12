import pandas as pd
import numpy as np

def ss_aug(df, aug_cols, num_ref=10):
    aug_df = df.copy()

    if 'uniprot' not in aug_df.columns:
        aug_df['uniprot'] = aug_df['node_id'].apply(lambda x: x.split('_')[0])
    if 'pos' not in aug_df.columns:
        aug_df['pos'] = aug_df['node_id'].apply(lambda x: int(x.split('_')[1]))

    for col in aug_cols:
        if col == 'rel_sasa':
            aug_df = impute_local_mean(aug_df, col, num_ref)
        elif col == 'depth':
            aug_df = impute_local_mean(aug_df, col, num_ref)
        elif col =='hse_up':
            aug_df = impute_local_mean(aug_df, col, num_ref)
        elif col == 'hse_down':
            aug_df = impute_local_mean(aug_df, col, num_ref)
        elif col == 'dssp_phi':
            aug_df = impute_local_mean(aug_df, col, num_ref)
        elif col == 'dssp_psi':
            aug_df = impute_local_mean(aug_df, col, num_ref)
        elif col == 'dssp_TCO':
            aug_df = impute_local_mean(aug_df, col, num_ref)

    return aug_df

def impute_local_mean(df, colname, window_size=10):
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