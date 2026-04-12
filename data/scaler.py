import math
import pandas as pd
from sklearn.decomposition import PCA

def sin_cos_transform(df, colname, rot_type):
    if rot_type == 'sin':
        df[colname] = np.sin(df[colname] * np.pi / 180)
    elif rot_type == 'cos':
        df[colname] = np.cos(df[colname] * np.pi / 180)
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
    df[cols] = pca.transform(df[cols])
    return df