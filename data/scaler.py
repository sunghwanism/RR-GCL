import math
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

def sin_cos_transform(df, colname, rot_type):
    if rot_type == 'sin':
        df[f'{colname}_sin'] = np.sin(df[colname] * np.pi / 180)
    elif rot_type == 'cos':
        df[f'{colname}_cos'] = np.cos(df[colname] * np.pi / 180)
    else:
        raise ValueError("rot_type must be 'sin' or 'cos'")
    return df

def min_max_scale(df, colname, min_val, max_val):
    df[colname] = (df[colname] - min_val) / (max_val - min_val)
    return df

def standard_scale(df, colname):
    df[colname] = (df[colname] - df[colname].mean()) / df[colname].std()
    return df

def all_standard_scale(df):
    for col in df.columns:
        if df[col].dtype == 'float64' or df[col].dtype == 'int64':
            df = standard_scale(df, col)
    return df


def pca_transform(df, cols, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(df[cols])
    pca_features = pca.transform(df[cols])
    pca_cols = [f'{cols[0].split("_")[0]}_PCA{i}' for i in range(pca.n_components_)]
    pca_df = pd.DataFrame(pca_features, columns=pca_cols, index=df.index)
    pca_df['node_id'] = df['node_id']

    return pca_df, pca