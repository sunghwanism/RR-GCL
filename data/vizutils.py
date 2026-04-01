import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_distribution_subplots(df, col_list, cols_per_row=3, log_scale=False):

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


def plot_value_counts(df, col_list, cols_per_row=3, top_n=20):
    """
    Plots value counts for a list of columns as subplots.
    If a column contains list-type data, it converts them to strings for visualization.
    """
    n_cols = len(col_list)
    if n_cols == 0:
        print("Empty Column List")
        return
        
    n_rows = math.ceil(n_cols / cols_per_row)
    
    # Adjust figure size based on the number of rows
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(cols_per_row * 6, n_rows * 5))
    
    # Flatten axes if there are multiple subplots
    if n_rows * cols_per_row > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
        
    for i, col in enumerate(col_list):
        ax = axes[i]
        
        if col not in df.columns:
            ax.text(0.5, 0.5, f"'{col}'\nNot Found", ha='center', va='center', fontsize=12, color='red')
            ax.set_title(col)
            continue
            
        # --- Handle List-type Data ---
        # 1. Create a temporary Series for counting
        data_to_plot = df[col].copy()
        
        # 2. Check if the column contains any list objects
        is_list_col = data_to_plot.apply(lambda x: isinstance(x, list)).any()
        
        if is_list_col:
            # Convert list to joined string (e.g., ['H', 'T'] -> "H, T")
            # We use map(str, x) to handle potential non-string elements inside the list
            data_to_plot = data_to_plot.apply(
                lambda x: ', '.join(map(str, x)) if isinstance(x, list) else str(x)
            )
        
        # Calculate counts
        counts = data_to_plot.value_counts().head(top_n)
        
        # Plotting
        sns.barplot(x=counts.index, y=counts.values, ax=ax, palette='viridis')
        ax.set_title(f'Value Counts of {col}', fontweight='bold', fontsize=13)
        ax.tick_params(axis='x', rotation=45) # Rotate labels slightly for list-strings
        ax.set_ylabel('Count')
        
        # Add count labels on top of bars
        for p in ax.patches:
            height = p.get_height()
            if height > 0: # Only annotate bars with values
                ax.annotate(f'{int(height)}', 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='center', 
                            fontsize=10, color='black', 
                            xytext=(0, 7), 
                            textcoords='offset points')
            
    # Remove unused axes
    for j in range(n_cols, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.show()