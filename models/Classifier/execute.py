import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader

from models.DGI.models.dgi import DGI
from models.DGI.execute import extract_embeddings as DGI_extract_embeddings

from models.Classifier.FFN import TrainFFN, EvaluateFFN


class NpzFeatureDataset(Dataset):
    def __init__(self, node_list, label_df, npz_paths):
        self.npz_paths = npz_paths if isinstance(npz_paths, list) else [npz_paths]
        
        # Load npz data from all paths
        embed_dict = {}
        for path in self.npz_paths:
            if os.path.exists(path):
                data = np.load(path)
                embed_dict.update(dict(zip(data['node_ids'], data['embeddings'])))
        
        valid_nodes = []
        labels = []
        features = []
        
        # Label mapping: Driver=1, Passenger=0
        label_map = {'Passenger': 0, 'Driver': 1}
        node_to_label = dict(zip(label_df['node_id'], label_df['label']))
        
        for node in node_list:
            if node in node_to_label and node in embed_dict:
                valid_nodes.append(node)
                labels.append(label_map[node_to_label[node]])
                features.append(embed_dict[node])
                    
        self.nodes = valid_nodes
        self.labels = labels
        self.features = features
        
    def __len__(self):
        return len(self.nodes)
        
    def __getitem__(self, idx):
        node = self.nodes[idx]
        x = self.features[idx]
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.long), node


def get_node_list_from_loader(loader):
    node_list = []
    for batch in loader:
        if hasattr(batch, 'node_names'):
            for names in batch.node_names:
                if isinstance(names, list):
                    node_list.extend(names)
                else:
                    node_list.append(names)
    return node_list


def TrainGNN(model, loader, optimizer, criterion, device, label_map):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_targets = []
    
    if len(loader) == 0:
        return 0, 0, 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        logits = model(batch.x, batch.edge_index)
        
        valid_indices = []
        valid_labels = []
        
        flat_node_ids = []
        if hasattr(batch, 'node_names'):
            for names in batch.node_names:
                if isinstance(names, list):
                    flat_node_ids.extend(names)
                else:
                    flat_node_ids.append(names)
                    
        for i, node in enumerate(flat_node_ids):
            if node in label_map:
                valid_indices.append(i)
                valid_labels.append(label_map[node])
                
        if len(valid_indices) == 0:
            continue
            
        valid_indices = torch.tensor(valid_indices, dtype=torch.long, device=device)
        valid_labels = torch.tensor(valid_labels, dtype=torch.long, device=device)
        
        valid_logits = logits[valid_indices]
        loss = criterion(valid_logits, valid_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(valid_logits, dim=1)
        total_correct += (preds == valid_labels).sum().item()
        total_samples += valid_labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(valid_labels.cpu().numpy())
    
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return total_loss / len(loader), total_correct / total_samples, f1

def EvaluateGNN(model, loader, criterion, device, label_map):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_targets = []
    
    if len(loader) == 0:
        return 0, 0, 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index)
            
            valid_indices = []
            valid_labels = []
            
            flat_node_ids = []
            if hasattr(batch, 'node_names'):
                for names in batch.node_names:
                    if isinstance(names, list):
                        flat_node_ids.extend(names)
                    else:
                        flat_node_ids.append(names)
                        
            for i, node in enumerate(flat_node_ids):
                if node in label_map:
                    valid_indices.append(i)
                    valid_labels.append(label_map[node])
                    
            if len(valid_indices) == 0:
                continue
                
            valid_indices = torch.tensor(valid_indices, dtype=torch.long, device=device)
            valid_labels = torch.tensor(valid_labels, dtype=torch.long, device=device)
            
            valid_logits = logits[valid_indices]
            loss = criterion(valid_logits, valid_labels)
            total_loss += loss.item()
            preds = torch.argmax(valid_logits, dim=1)
            total_correct += (preds == valid_labels).sum().item()
            total_samples += valid_labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(valid_labels.cpu().numpy())
    
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return total_loss / len(loader), total_correct / total_samples, f1


def run_downstream(config, clf_model, train_loader, val_loader, test_loader, run_wandb=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Downstream Task
    if config.clf_model == 'FFN':
        clf_model.to(device)
        
        # Load matched cancer driver df
        label_df = pd.read_csv('models/matched_cancer_driver_df.csv')
        
        # Extract node lists
        train_nodes = get_node_list_from_loader(train_loader)
        val_nodes = get_node_list_from_loader(val_loader)
        test_nodes = get_node_list_from_loader(test_loader)
        
        BASESAVEPATH = os.path.join(config.SAVEPATH, config.load_model, config.load_wandb_id)
        
        all_npz_paths = [
            os.path.join(BASESAVEPATH, 'train', 'train_embeddings.npz'),
            os.path.join(BASESAVEPATH, 'val', 'val_embeddings.npz'),
            os.path.join(BASESAVEPATH, 'test', 'test_embeddings.npz')
        ]
        
        # Datasets
        train_dataset = NpzFeatureDataset(train_nodes, label_df, all_npz_paths)
        val_dataset = NpzFeatureDataset(val_nodes, label_df, all_npz_paths)
        test_dataset = NpzFeatureDataset(test_nodes, label_df, all_npz_paths)
        
        # DataLoaders
        ffn_train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        ffn_val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        ffn_test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        
        optimizer = torch.optim.Adam(clf_model.parameters(), lr=config.lr, weight_decay=config.l2_coef)
        criterion = nn.CrossEntropyLoss()
        
        use_scheduler = getattr(config, 'use_scheduler', False)
        if use_scheduler:
            lr_patience = getattr(config, 'lr_patience', 10)
            lr_factor = getattr(config, 'lr_factor', 0.1)
            min_lr = getattr(config, 'min_lr', 1e-6)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=lr_factor, patience=lr_patience, min_lr=min_lr,
            )
        else:
            scheduler = None
        
        best_val_f1 = 0.0
        if run_wandb:
            clf_save_dir = os.path.join(config.SAVEPATH, 'Classifier', run_wandb.id)
        else:
            clf_save_dir = os.path.join(config.SAVEPATH, 'Classifier')
        os.makedirs(clf_save_dir, exist_ok=True)
        save_path = os.path.join(clf_save_dir, 'BestPerformance.pth')
        
        print("============================"*2)
        print("TRAINING FFN CLASSIFIER")
        print("============================"*2)
        
        for epoch in range(config.epoch):
            train_loss, train_acc, train_f1 = TrainFFN(clf_model, ffn_train_loader, optimizer, criterion, device)
            val_loss, val_acc, val_f1 = EvaluateFFN(clf_model, ffn_val_loader, criterion, device)
            test_loss, test_acc, test_f1 = EvaluateFFN(clf_model, ffn_test_loader, criterion, device)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(clf_model.state_dict(), save_path)
                
            if scheduler is not None:
                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']
            else:
                current_lr = optimizer.param_groups[0]['lr']
                
            if epoch % 10 == 0:
                print(f'Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f} | LR: {current_lr:.6f}')
            
            if run_wandb:
                run_wandb.log({
                    "Epoch": epoch,
                    "Train Loss": train_loss,
                    "Val Loss": val_loss,
                    "Test Loss": test_loss,
                    "Train Acc": train_acc,
                    "Val Acc": val_acc,
                    "Test Acc": test_acc,
                    "Train F1": train_f1,
                    "Val F1": val_f1,
                    "Test F1": test_f1,
                    "LR": current_lr,
                })
                
        # Load best model for evaluation
        if os.path.exists(save_path):
            clf_model.load_state_dict(torch.load(save_path, map_location=device))
            
        clf_model.eval()
        
        results = []
        loaders_to_eval = [('train', ffn_train_loader), ('val', ffn_val_loader), ('test', ffn_test_loader)]
        
        print("Evaluating and saving predictions...")
        with torch.no_grad():
            for split_name, loader in loaders_to_eval:
                for x, y, nodes in loader:
                    x = x.to(device)
                    logits = clf_model(x)
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy() # Prob for Driver (class 1)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    
                    for i in range(len(nodes)):
                        results.append({
                            'type': split_name,
                            'node_id': nodes[i],
                            'prob': probs[i],
                            'class': preds[i],
                            'true_class': y[i].item()
                        })
                        
        results_df = pd.DataFrame(results)
        
        # --- Visualization & Quantification ---
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
        
        # Plot ROC Curves
        plt.figure(figsize=(15, 5))
        for idx, split_name in enumerate(['train', 'val', 'test']):
            split_df = results_df[results_df['type'] == split_name]
            if len(split_df) == 0: continue
            
            y_true = split_df['true_class']
            y_prob = split_df['prob']
            
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.subplot(1, 3, idx+1)
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{split_name.capitalize()} ROC Curve')
            plt.legend(loc="lower right")
            
        plt.tight_layout()
        plt.savefig(os.path.join(clf_save_dir, 'roc_curve.png'))
        plt.close()
        
        # Plot Confusion Matrices
        plt.figure(figsize=(15, 5))
        for idx, split_name in enumerate(['train', 'val', 'test']):
            split_df = results_df[results_df['type'] == split_name]
            if len(split_df) == 0: continue
            
            y_true = split_df['true_class']
            y_pred = split_df['class']
            
            cm = confusion_matrix(y_true, y_pred)
            
            plt.subplot(1, 3, idx+1)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Passenger', 'Driver'], yticklabels=['Passenger', 'Driver'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'{split_name.capitalize()} Confusion Matrix')
            
        plt.tight_layout()
        plt.savefig(os.path.join(clf_save_dir, 'confusion_matrix.png'))
        plt.close()
        
        from sklearn.metrics import accuracy_score
        
        # Quantified Results Print
        print("="*60)
        print("PERFORMANCE METRICS")
        print("="*60)
        for split_name in ['train', 'val', 'test']:
            split_df = results_df[results_df['type'] == split_name]
            if len(split_df) == 0: continue
            y_true = split_df['true_class']
            y_prob = split_df['prob']
            y_pred = split_df['class']
            
            acc = accuracy_score(y_true, y_pred)
            print(f"[{split_name.upper()}] Accuracy: {acc:.4f}")
            if run_wandb:
                run_wandb.log({f"{split_name.capitalize()} Accuracy": acc})
            
            try:
                roc_auc = roc_auc_score(y_true, y_prob)
                print(f"[{split_name.upper()}] AUC-ROC: {roc_auc:.4f}")
                if run_wandb:
                    run_wandb.log({f"{split_name.capitalize()} AUC-ROC": roc_auc})
            except ValueError:
                print(f"[{split_name.upper()}] AUC-ROC: Error (Only one class present in y_true)")

        if run_wandb:
            import wandb
            run_wandb.log({
                "ROC Curves": wandb.Image(os.path.join(clf_save_dir, 'roc_curve.png')),
                "Confusion Matrices": wandb.Image(os.path.join(clf_save_dir, 'confusion_matrix.png'))
            })

        # Save the DataFrame with just the columns requested
        save_df = results_df[['type', 'node_id', 'prob', 'class']]
        csv_path = os.path.join(clf_save_dir, 'ffn_predictions.csv')
        save_df.to_csv(csv_path, index=False)
        print(f"Predictions saved to {csv_path}")

    elif config.clf_model in ['GAT', 'SAGE']:
        clf_model.to(device)
        
        # Load matched cancer driver df
        label_df = pd.read_csv('models/matched_cancer_driver_df.csv')
        str_to_int = {'Passenger': 0, 'Driver': 1}
        node_to_label = {row['node_id']: str_to_int[row['label']] for _, row in label_df.iterrows()}
        
        optimizer = torch.optim.Adam(clf_model.parameters(), lr=config.lr, weight_decay=config.l2_coef)
        criterion = nn.CrossEntropyLoss()
        
        use_scheduler = getattr(config, 'use_scheduler', False)
        if use_scheduler:
            lr_patience = getattr(config, 'lr_patience', 10)
            lr_factor = getattr(config, 'lr_factor', 0.1)
            min_lr = getattr(config, 'min_lr', 1e-6)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=lr_factor, patience=lr_patience, min_lr=min_lr,
            )
        else:
            scheduler = None
        
        best_val_f1 = 0.0
        if run_wandb:
            clf_save_dir = os.path.join(config.SAVEPATH, 'Classifier', run_wandb.id)
        else:
            clf_save_dir = os.path.join(config.SAVEPATH, 'Classifier')
        os.makedirs(clf_save_dir, exist_ok=True)
        save_path = os.path.join(clf_save_dir, 'BestPerformance.pth')
        
        print("============================"*2)
        print(f"TRAINING {config.clf_model} CLASSIFIER")
        print("============================"*2)
        
        for epoch in range(config.epoch):
            train_loss, train_acc, train_f1 = TrainGNN(clf_model, train_loader, optimizer, criterion, device, node_to_label)
            val_loss, val_acc, val_f1 = EvaluateGNN(clf_model, val_loader, criterion, device, node_to_label)
            test_loss, test_acc, test_f1 = EvaluateGNN(clf_model, test_loader, criterion, device, node_to_label)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(clf_model.state_dict(), save_path)
                
            if scheduler is not None:
                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']
            else:
                current_lr = optimizer.param_groups[0]['lr']
                
            if epoch % 10 == 0:
                print(f'Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f} | LR: {current_lr:.6f}')
                
            if run_wandb:
                run_wandb.log({
                    "Epoch": epoch,
                    "Train Loss": train_loss,
                    "Val Loss": val_loss,
                    "Test Loss": test_loss,
                    "Train Acc": train_acc,
                    "Val Acc": val_acc,
                    "Test Acc": test_acc,
                    "Train F1": train_f1,
                    "Val F1": val_f1,
                    "Test F1": test_f1,
                    "LR": current_lr,
                })
        # Load best model for evaluation
        if os.path.exists(save_path):
            clf_model.load_state_dict(torch.load(save_path, map_location=device))
            
        clf_model.eval()
        
        results = []
        loaders_to_eval = [('train', train_loader), ('val', val_loader), ('test', test_loader)]
        
        print("Evaluating and saving predictions...")
        with torch.no_grad():
            for split_name, loader in loaders_to_eval:
                for batch in loader:
                    batch = batch.to(device)
                    logits = clf_model(batch.x, batch.edge_index)
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    
                    flat_node_ids = []
                    if hasattr(batch, 'node_names'):
                        for names in batch.node_names:
                            if isinstance(names, list):
                                flat_node_ids.extend(names)
                            else:
                                flat_node_ids.append(names)
                                
                    for i, node in enumerate(flat_node_ids):
                        if node in node_to_label:
                            results.append({
                                'type': split_name,
                                'node_id': node,
                                'prob': probs[i],
                                'class': preds[i],
                                'true_class': node_to_label[node]
                            })
                            
        results_df = pd.DataFrame(results)
        
        # --- Visualization & Quantification ---
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
        
        # Plot ROC Curves
        plt.figure(figsize=(15, 5))
        for idx, split_name in enumerate(['train', 'val', 'test']):
            split_df = results_df[results_df['type'] == split_name]
            if len(split_df) == 0: continue
            
            y_true = split_df['true_class']
            y_prob = split_df['prob']
            
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.subplot(1, 3, idx+1)
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{split_name.capitalize()} ROC Curve')
            plt.legend(loc="lower right")
            
        plt.tight_layout()
        plt.savefig(os.path.join(clf_save_dir, f'{config.clf_model}_roc_curve.png'))
        plt.close()
        
        # Plot Confusion Matrices
        plt.figure(figsize=(15, 5))
        for idx, split_name in enumerate(['train', 'val', 'test']):
            split_df = results_df[results_df['type'] == split_name]
            if len(split_df) == 0: continue
            
            y_true = split_df['true_class']
            y_pred = split_df['class']
            
            cm = confusion_matrix(y_true, y_pred)
            
            plt.subplot(1, 3, idx+1)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Passenger', 'Driver'], yticklabels=['Passenger', 'Driver'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'{split_name.capitalize()} Confusion Matrix')
            
        plt.tight_layout()
        plt.savefig(os.path.join(clf_save_dir, f'{config.clf_model}_confusion_matrix.png'))
        plt.close()
        
        from sklearn.metrics import accuracy_score

        # Quantified Results Print
        print("="*60)
        print("PERFORMANCE METRICS")
        print("="*60)
        for split_name in ['train', 'val', 'test']:
            split_df = results_df[results_df['type'] == split_name]
            if len(split_df) == 0: continue
            y_true = split_df['true_class']
            y_prob = split_df['prob']
            y_pred = split_df['class']
            
            acc = accuracy_score(y_true, y_pred)
            print(f"[{split_name.upper()}] Accuracy: {acc:.4f}")
            if run_wandb:
                run_wandb.log({f"{split_name.capitalize()} Accuracy": acc})
            
            try:
                roc_auc = roc_auc_score(y_true, y_prob)
                print(f"[{split_name.upper()}] AUC-ROC: {roc_auc:.4f}")
                if run_wandb:
                    run_wandb.log({f"{split_name.capitalize()} AUC-ROC": roc_auc})
            except ValueError:
                print(f"[{split_name.upper()}] AUC-ROC: Error (Only one class present in y_true)")

        if run_wandb:
            import wandb
            run_wandb.log({
                "ROC Curves": wandb.Image(os.path.join(clf_save_dir, f'{config.clf_model}_roc_curve.png')),
                "Confusion Matrices": wandb.Image(os.path.join(clf_save_dir, f'{config.clf_model}_confusion_matrix.png'))
            })

        # Save the DataFrame with just the columns requested
        save_df = results_df[['type', 'node_id', 'prob', 'class']]
        csv_path = os.path.join(clf_save_dir, f'{config.clf_model}_predictions.csv')
        save_df.to_csv(csv_path, index=False)
        print(f"Predictions saved to {csv_path}")