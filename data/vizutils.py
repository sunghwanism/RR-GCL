import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_subplots(df, col_list, cols_per_row=3, log_scale=False):

    n_cols = len(col_list)
    if n_cols == 0:
        print("Empty Column List")
        return
        
    n_rows = math.ceil(n_cols / cols_per_row)
    
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(cols_per_row * 5, n_rows * 4))
    
    if n_rows > 1 or cols_per_row > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
        
    for i, col in enumerate(col_list):
        ax = axes[i]
        
        if col not in df.columns:
            ax.text(0.5, 0.5, f"'{col}'\nNot Found", ha='center', va='center', fontsize=12, color='red')
            ax.set_title(col)
            continue
            
        if pd.api.types.is_numeric_dtype(df[col]):
            sns.histplot(df[col].dropna(), kde=True, ax=ax, color='steelblue', bins=30)
            ax.set_title(f'Distribution of {col}', fontweight='bold')
            
        else:
            sns.countplot(data=df, x=col, ax=ax, palette='viridis')
            ax.set_title(f'Count of {col}', fontweight='bold')
            ax.tick_params(axis='x', rotation=45) 
        
        if log_scale:
            ax.set_yscale('log')
            
    for j in range(n_cols, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.show()