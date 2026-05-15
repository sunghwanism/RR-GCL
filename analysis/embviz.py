import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_driver_embedding(SAVEPATH, models_dict, method, num_cols=3):
    """
    Plots embeddings for multiple models in a grid layout.
    
    Args:
        models_dict: Dictionary of models {model_name: wandb_id}
        method: Method to use for dimensionality reduction ('tsne' or 'pca' or 'umap')
        num_cols: Number of columns in the grid
    """
    num_models = len(models_dict)
    num_rows = math.ceil(num_models / num_cols)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, num_rows * 6))
    
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, (model_name, wandb_id) in enumerate(models_dict.items()):
        ax = axes[i]
        
        vis_df = pd.read_csv(f"{SAVEPATH}/DGI/{wandb_id}/tSNE_result_driver+am.csv")

        if method == 'tsne':
            if model_name == 'All':
                vis_df = vis_df[(vis_df['tsne_1']<-30)]
            
            elif model_name == 'All+Egy':
                vis_df = vis_df[(vis_df['tsne_1'] < -3)  & (vis_df['tsne_2'] > 20)]

            elif model_name == 'ExLocalGeo':
                vis_df = vis_df[(vis_df['tsne_1']>30)]

            elif model_name == 'ExLocalGeo+Egy':
                vis_df = vis_df[(vis_df['tsne_2']<-20)]

            elif model_name == 'ExSpEnv':
                pass
            
            elif model_name == 'ExSpEnv+Egy':
                vis_df = vis_df[(vis_df['tsne_1'] < -18)]

            elif model_name == 'ExEvol':
                vis_df = vis_df[(vis_df['tsne_1'] > 40)]

            elif model_name == 'ExEvol+Egy':
                vis_df = vis_df[(vis_df['tsne_1'] < -20)]
            else:
                raise ValueError(f"Unknown model name: {model_name}")

        elif method == 'pca':
            if model_name == 'All':
                vis_df = vis_df[(vis_df['pca_1'] < -10)]
            
            elif model_name == 'All+Egy':
                vis_df = vis_df[(vis_df['pca_2'] > 10)]

            elif model_name == 'ExLocalGeo':
                vis_df = vis_df[(vis_df['pca_1'] > 10)]

            elif model_name == 'ExLocalGeo+Egy':
                vis_df = vis_df[(vis_df['pca_2'] < -4)]

            elif model_name == 'ExSpEnv':
                pass
            
            elif model_name == 'ExSpEnv+Egy':
                vis_df = vis_df[(vis_df['pca_1'] < -5)]

            elif model_name == 'ExEvol':
                vis_df = vis_df[(vis_df['pca_1'] > 10)]

            elif model_name == 'ExEvol+Egy':
                vis_df = vis_df[(vis_df['pca_1'] < -5)]
            else:
                raise ValueError(f"Unknown model name: {model_name}")

        elif method == 'umap':
            if model_name == 'All':
                vis_df = vis_df[(vis_df['umap_1']<2)]
            
            elif model_name == 'All+Egy':
                vis_df = vis_df[(vis_df['umap_2']>8)]

            elif model_name == 'ExLocalGeo':
                vis_df = vis_df[(vis_df['umap_1'] < 7)]

            elif model_name == 'ExLocalGeo+Egy':
                vis_df = vis_df[(vis_df['umap_2'] > 10)]

            elif model_name == 'ExSpEnv':
                pass
            
            elif model_name == 'ExSpEnv+Egy':
                vis_df = vis_df[(vis_df['umap_2'] < 0)]

            elif model_name == 'ExEvol':
                vis_df = vis_df[(vis_df['umap_1'] < 2.5)]

            elif model_name == 'ExEvol+Egy':
                vis_df = vis_df[(vis_df['umap_2'] < 0)]
            else:
                raise ValueError(f"Unknown model name: {model_name}")
            
        
        sns.scatterplot(
            x=f'{method}_1', 
            y=f'{method}_2',
            hue='label',
            palette='Set2',
            data=vis_df,
            alpha=0.7,
            ax=ax,
            s=50
        )
        ax.set_title(f'{model_name} (Driver/Passenger)')
        ax.set_xlabel(f'{method} 1')
        ax.set_ylabel(f'{method} 2')
        
    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show() 


def plot_am_embedding(SAVEPATH, models_dict, method, target, num_cols=3, portion=0.3, only_driver=False):

    num_models = len(models_dict)
    num_rows = math.ceil(num_models / num_cols)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, num_rows * 6))
    
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()
    
    scatter = None 
    
    for i, (model_name, wandb_id) in enumerate(models_dict.items()):
        ax = axes[i]
        vis_df = pd.read_csv(f"{SAVEPATH}/DGI/{wandb_id}/tSNE_result_driver+am.csv")
        if only_driver:

            vis_df = vis_df[~vis_df['label'].isna()]

            if method == 'tsne':
                if model_name == 'All':
                    vis_df = vis_df[(vis_df['tsne_1']<-30)]
            
                elif model_name == 'All+Egy':
                    vis_df = vis_df[(vis_df['tsne_1'] < -3)  & (vis_df['tsne_2'] > 20)]

                elif model_name == 'ExLocalGeo':
                    vis_df = vis_df[(vis_df['tsne_1']>30)]

                elif model_name == 'ExLocalGeo+Egy':
                    vis_df = vis_df[(vis_df['tsne_2']<-20)]

                elif model_name == 'ExSpEnv':
                    pass
                
                elif model_name == 'ExSpEnv+Egy':
                    vis_df = vis_df[(vis_df['tsne_1'] < -18)]

                elif model_name == 'ExEvol':
                    vis_df = vis_df[(vis_df['tsne_1'] > 40)]

                elif model_name == 'ExEvol+Egy':
                    vis_df = vis_df[(vis_df['tsne_1'] < -20)]
                else:
                    raise ValueError(f"Unknown model name: {model_name}")

            elif method == 'pca':
                if model_name == 'All':
                    vis_df = vis_df[(vis_df['pca_1'] < -10)]
                
                elif model_name == 'All+Egy':
                    vis_df = vis_df[(vis_df['pca_2'] > 10)]

                elif model_name == 'ExLocalGeo':
                    vis_df = vis_df[(vis_df['pca_1'] > 10)]

                elif model_name == 'ExLocalGeo+Egy':
                    vis_df = vis_df[(vis_df['pca_2'] < -4)]

                elif model_name == 'ExSpEnv':
                    pass
                
                elif model_name == 'ExSpEnv+Egy':
                    vis_df = vis_df[(vis_df['pca_1'] < -5)]

                elif model_name == 'ExEvol':
                    vis_df = vis_df[(vis_df['pca_1'] > 10)]

                elif model_name == 'ExEvol+Egy':
                    vis_df = vis_df[(vis_df['pca_1'] < -5)]
                else:
                    raise ValueError(f"Unknown model name: {model_name}")

            elif method == 'umap':
                if model_name == 'All':
                    vis_df = vis_df[(vis_df['umap_1']<2)]
                
                elif model_name == 'All+Egy':
                    vis_df = vis_df[(vis_df['umap_2']>8)]

                elif model_name == 'ExLocalGeo':
                    vis_df = vis_df[(vis_df['umap_1'] < 7)]

                elif model_name == 'ExLocalGeo+Egy':
                    vis_df = vis_df[(vis_df['umap_2'] > 10)]

                elif model_name == 'ExSpEnv':
                    pass
                
                elif model_name == 'ExSpEnv+Egy':
                    vis_df = vis_df[(vis_df['umap_2'] < 0)]

                elif model_name == 'ExEvol':
                    vis_df = vis_df[(vis_df['umap_1'] < 2.5)]

                elif model_name == 'ExEvol+Egy':
                    vis_df = vis_df[(vis_df['umap_2'] < 0)]
                else:
                    raise ValueError(f"Unknown model name: {model_name}")

        else:
            vis_df = vis_df.sample(int(len(vis_df)*portion))
        
        scatter_obj = ax.scatter(
            vis_df[f'{method}_1'], 
            vis_df[f'{method}_2'],
            c=vis_df[target],
            cmap='viridis_r',
            alpha=0.7 if only_driver else 0.5,
            s=50 if only_driver else 30,
            edgecolors='w',
            linewidth=0.2,
        )
        scatter = scatter_obj
        
        ax.set_title(f'{model_name} ({target})')
        ax.set_xlabel(f'{method} 1')
        ax.set_ylabel(f'{method} 2')
        
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 0.95, 1]) 
    
    if scatter is not None:
        cbar_ax = fig.add_axes([0.96, 0.15, 0.015, 0.7]) 
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.set_label(f'{target}')
        
    plt.show()


def plot_org_embedding(vis_df, method='tsne', label_col='label_x', am_col='max_am'):
    """
    Visualizes dimensionality reduction results (t-SNE, PCA, UMAP) side-by-side.
    Left: Colored by label.
    Right: Colored by AlphaMissense score (max_am).
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    method_label = method.upper() if method in ['tsne', 'pca', 'umap'] else method

    # [Left] Scatter plot by Label
    sns.scatterplot(
        x=f'{method}_1', y=f'{method}_2', hue=label_col, palette='Set2',
        data=vis_df, alpha=0.7, ax=axes[0]
    )
    axes[0].set_title(f'{method_label} of Original Features by Label')
    axes[0].set_xlabel(f'{method_label} 1')
    axes[0].set_ylabel(f'{method_label} 2')

    # [Right] Scatter plot by max_am
    scatter = axes[1].scatter(
        vis_df[f'{method}_1'], vis_df[f'{method}_2'],
        c=vis_df[am_col], cmap='viridis_r', alpha=0.7,
        edgecolors='w', linewidth=0.2, s=30
    )
    axes[1].set_title(f'{method_label} of Original Features by {am_col}')
    axes[1].set_xlabel(f'{method_label} 1')
    axes[1].set_ylabel(f'{method_label} 2')

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=axes[1])
    cbar.set_label(f'{am_col}')
    
    plt.tight_layout()
    plt.show()

def visualize_feat_with_am(
    data, 
    feature='chain_flag', 
    must_incl_val='mixed', 
    num_sample=300, 
    point_size=100, 
    palette=["#001F3F", "#FFBF00"],
    seed=42 # Added seed parameter
):
    """
    Performs UMAP visualization with fixed seed for reproducibility.
    """
    # 0. Fix Seed for all libraries
    random.seed(seed)
    np.random.seed(seed)
    # Note: If you are using UMAP/t-SNE inside this function, 
    # you should also pass the seed to their random_state.

    # 1. Preprocessing and Filtering
    plot_df = data[~data[feature].isna() & ~data['umap_1'].isna()].copy()
    
    # 2. Sampling Logic
    must_include_mask = plot_df[feature] == must_incl_val
    incl_all = plot_df[must_include_mask]
    
    remaining_count = max(0, num_sample - len(incl_all))
    others_df = plot_df[~must_include_mask]
    
    # Use the provided seed in sample()
    actual_sample_n = min(len(others_df), remaining_count)
    if actual_sample_n > 0:
        others_sampled = others_df.sample(n=actual_sample_n, random_state=seed)
        filtered_df = pd.concat([incl_all, others_sampled]).drop_duplicates()
    else:
        filtered_df = incl_all.copy()
    
    driver_data = filtered_df[filtered_df['label'] == 'driver']
    
    # 3. Visualization (Rest of the code remains consistent)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # [Left Plot]
    sns.scatterplot(
        x='umap_1', y='umap_2', hue=feature, palette=palette,
        data=filtered_df, alpha=0.6, ax=axes[0], s=point_size, edgecolor='none'
    )
    sns.scatterplot(
        x='umap_1', y='umap_2', data=driver_data, 
        color='none', edgecolor='red', linewidth=2.0, alpha=1.0,
        ax=axes[0], s=point_size, legend=False, zorder=10
    )
    axes[0].set_title(f'UMAP by {feature} (Seed: {seed})')

    # [Right Plot]
    scatter = axes[1].scatter(
        filtered_df['umap_1'], filtered_df['umap_2'],
        c=filtered_df['avg_am'], cmap='viridis_r', alpha=0.6,
        edgecolors='none', s=point_size
    )
    sns.scatterplot(
        x='umap_1', y='umap_2', data=driver_data, 
        color='none', edgecolor='red', linewidth=2.0, alpha=1.0,
        ax=axes[1], s=point_size, legend=False, zorder=10
    )
    axes[1].set_title('UMAP by Average AlphaMissense Score')

    cbar = fig.colorbar(scatter, ax=axes[1])
    cbar.set_label('Average AlphaMissense Score (avg_am)')

    plt.tight_layout()
    plt.show()