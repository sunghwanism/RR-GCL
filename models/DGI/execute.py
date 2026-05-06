import torch
import torch.nn as nn
import yaml
import os
from torch_geometric.loader import DataLoader

import torch
import torch.nn as nn
import yaml
import os
import numpy as np
from torch_geometric.loader import DataLoader

from models.DGI.models.dgi import DGI
from models.DGI.models.logreg import LogReg
from models.DGI.dataset import shuffle_node_features

from torch_geometric.data import Data
import networkx as nx


def train_dgi_epoch(model, loader, optimizer, criterion, device):
    """Train DGI for one epoch on multiple graphs."""
    model.train()
    total_loss = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        num_nodes = batch.x.size(0)
        shuf_idx = torch.randperm(num_nodes)
        
        # Shuffle features for negative sampling
        shuf_fts = batch.x[shuf_idx]
        
        cat_feats = {}
        shuf_cat_feats = {}
        for key in model.cat_feat_emb_dict.keys():
            if hasattr(batch, key):
                feat = getattr(batch, key)
                cat_feats[key] = feat
                shuf_cat_feats[key] = feat[shuf_idx]
                
        # Create labels: 1 for real, 0 for fake
        lbl_1 = torch.ones(num_nodes, 1, device=device)
        lbl_2 = torch.zeros(num_nodes, 1, device=device)
        lbl = torch.cat((lbl_1, lbl_2), 0)
        
        logits = model(batch.x, cat_feats, shuf_fts, shuf_cat_feats, 
                       batch.edge_index, batch.batch, None, None)
        
        loss = criterion(logits, lbl)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Free memory before next iteration
        del batch, shuf_idx, shuf_fts, cat_feats, shuf_cat_feats, lbl_1, lbl_2, lbl, logits, loss
        
    return total_loss / len(loader)


def extract_embeddings(model, loader, device):
    """Extract embeddings for all graphs in the loader."""
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            cat_feats = {}
            for key in model.cat_feat_emb_dict.keys():
                if hasattr(batch, key):
                    cat_feats[key] = getattr(batch, key)
            embeds, _ = model.embed(batch.x, cat_feats, batch.edge_index, batch.batch)
            
            # Get graph-level embeddings (mean pooling per graph)
            from torch_geometric.nn import global_mean_pool
            graph_embeds = global_mean_pool(embeds, batch.batch)
            
            all_embeddings.append(graph_embeds.cpu())
            if hasattr(batch, 'y'):
                # Assuming y is graph-level label
                all_labels.append(batch.y.cpu() if isinstance(batch.y, torch.Tensor) else batch.y)
            
            # Free memory
            del batch, cat_feats, embeds, graph_embeds
    
    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0) if all_labels else None
    
    return embeddings, labels


def run_training(config, train_loader, val_loader, test_loader, run_wandb=None):
    
    batch_size = config.batch_size
    nb_epochs = config.epoch
    patience = config.patience
    lr = config.lr
    l2_coef = config.l2_coef

    # Model Arguments
    hid_units = config.model_param['hidden_dims']
    nonlinearity = config.model_param['activation']
    drop_prob = config.model_param['drop_prob']
    emb_dim = config.model_param['emb_dim']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("============================"*2)
    print(f'Using device: {device}')
    print("============================"*2)
    
    # Check if loaders are provided
    if not train_loader:
        raise ValueError("train_loader is None")

    from data.vocab import attr_mappings
    # Get feature size from first batch
    first_batch = next(iter(train_loader))
    num_ft_size = first_batch.x.size(1)
    cat_feat_num_dict = {}
    for key in first_batch.keys():
        if key != 'x' and key in config.node_att:
            if hasattr(first_batch, key):
                print(f"Feature: {key}, Shape: {getattr(first_batch, key).shape}")
                if key in attr_mappings:
                    cat_feat_num_dict[key] = max(attr_mappings[key].values()) + 1

    print("Input Feature Shape in DataLoader")
    for key, values in first_batch.items():
        if hasattr(values, 'shape'):
            print(key, ':', values.shape)
        else:
            print(key, ':', type(values), 'len:', len(values) if hasattr(values, '__len__') else 'N/A')
    # print(f"Feature dimension: Numerical: {num_ft_size}")
    # print(f"Categorical: {cat_feat_num_dict}")
    print("============================"*2)
    
    # Free first_batch to save memory during training
    del first_batch

    # Initialize model
    model = DGI(num_ft_size, cat_feat_num_dict, emb_dim, hid_units, nonlinearity, drop_prob).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    print("============================"*2)
    print("TRAINING DGI")
    print("============================"*2)
    
    best_loss = float('inf')
    best_epoch = 0
    cnt_wait = 0
    
    if run_wandb:
        BASESAVEPATH = os.path.join(config.SAVEPATH, 'DGI', run_wandb.id) # train-`1`
    else:
        BASESAVEPATH = os.path.join(config.SAVEPATH, 'DGI')
    
    os.makedirs(BASESAVEPATH, exist_ok=True)
    save_path = os.path.join(BASESAVEPATH, 'BestPerformance.pth')
    
    for epoch in range(nb_epochs):
        train_loss = train_dgi_epoch(model, train_loader, optimizer, criterion, device)
                
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                num_nodes = batch.x.size(0)
                shuf_idx = torch.randperm(num_nodes)

                # Shuffle features
                shuf_fts = batch.x[shuf_idx]
                cat_feats = {}
                shuf_cat_feats = {}
                for key in model.cat_feat_emb_dict.keys():
                    if hasattr(batch, key):
                        feat = getattr(batch, key)
                        cat_feats[key] = feat
                        shuf_cat_feats[key] = feat[shuf_idx]
                lbl_1 = torch.ones(num_nodes, 1, device=device)
                lbl_2 = torch.zeros(num_nodes, 1, device=device)
                lbl = torch.cat((lbl_1, lbl_2), 0)
                
                logits = model(batch.x, cat_feats, shuf_fts, shuf_cat_feats, 
                               batch.edge_index, batch.batch, None, None)
                loss = criterion(logits, lbl)
                val_loss += loss.item()
                
                # Free memory
                del batch, shuf_idx, shuf_fts, cat_feats, shuf_cat_feats, lbl_1, lbl_2, lbl, logits, loss
        
        if len(val_loader) > 0:
            val_loss /= len(val_loader)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            torch.save(model.state_dict(), os.path.join(BASESAVEPATH, f'checkpoint_{epoch}.pth'))
        
        # Early stopping based on validation loss
        if val_loss < best_loss and epoch > 4:
            best_loss = val_loss
            best_epoch = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), save_path)
            if run_wandb:
                run_wandb.summary["best_val_loss"] = best_loss
                run_wandb.summary["best_epoch"] = best_epoch
        else:
            if epoch > 4: # wait for 5 epochs to start early stopping
                cnt_wait += 1

        if run_wandb:
            run_wandb.log({
                "epoch": epoch, 
                "train_loss": train_loss,
                "val_loss": val_loss
            })
        
        if cnt_wait >= patience:
            print("=========="*10)
            print(f'\nEarly stopping at epoch {epoch}')
            print("=========="*10)
            break
        
        
    print("##########"*10)
    print(f'\nBest epoch: {best_epoch}, Best val loss: {best_loss:.4f}')
    print("##########"*10)
    
    # Load best model
    if os.path.exists(save_path):
        print(f"\nLoading best model from {save_path} for testing...")
        model.load_state_dict(torch.load(save_path, map_location=device))
    else:
        print("\nWarning: Best model file not found, using current model state.")

    # Calculate Test Loss and save embeddings
    print("\n" + "="*60)
    print("TESTING AND SAVING EMBEDDINGS")
    print("="*60)
    
    model.eval()
    test_loss = 0
    test_dir = os.path.join(BASESAVEPATH, 'test')
    os.makedirs(test_dir, exist_ok=True)

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            num_nodes = batch.x.size(0)
            shuf_idx = torch.randperm(num_nodes)

            # Shuffle features
            shuf_fts = batch.x[shuf_idx]
            cat_feats = {}
            shuf_cat_feats = {}
            for key in model.cat_feat_emb_dict.keys():
                if hasattr(batch, key):
                    feat = getattr(batch, key)
                    cat_feats[key] = feat
                    shuf_cat_feats[key] = feat[shuf_idx]
            lbl_1 = torch.ones(num_nodes, 1, device=device)
            lbl_2 = torch.zeros(num_nodes, 1, device=device)
            lbl = torch.cat((lbl_1, lbl_2), 0)
            
            logits = model(batch.x, cat_feats, shuf_fts, shuf_cat_feats, 
                           batch.edge_index, batch.batch, None, None)
            loss = criterion(logits, lbl)
            test_loss += loss.item()

            # Now extract and save embeddings
            embeds, _ = model.embed(batch.x, cat_feats, batch.edge_index, batch.batch)
            embeds_np = embeds.cpu().numpy()
            
            # Flatten node_names from batch
            flat_node_ids = []
            if hasattr(batch, 'node_names'):
                for names in batch.node_names:
                    if isinstance(names, list):
                        flat_node_ids.extend(names)
                    else:
                        flat_node_ids.append(names)
            
            # Save each node's embedding as a .npy file
            for i, node_id in enumerate(flat_node_ids):
                np.save(os.path.join(test_dir, f"{node_id}.npy"), embeds_np[i])

    if len(test_loader) > 0:
        test_loss /= len(test_loader)
        
    print(f"Test Loss: {test_loss:.4f}")
    if run_wandb:
        run_wandb.summary["test_loss"] = test_loss
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
