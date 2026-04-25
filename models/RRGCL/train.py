import os
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.functions import load_yaml
from models.RRGCL.data.anchor_dataset import (
    AnchorBatchDataset,
    CurriculumConfig,
    SimilarityProvider,
    build_contrastive_masks,
)
from models.RRGCL.data.subgraph_extractor import AnchorSubgraphCollator


class FeatureSimilarityProvider(SimilarityProvider):
    """Computes similarity dynamically using vectorized PyTorch operations.
    """
    def __init__(self, feature_df: pd.DataFrame, feat_prefix: str = "am", metric: str = "pearson"):
        # Filter columns
        feat_cols = [col for col in feature_df.columns if col.startswith(feat_prefix)]
        self.node_ids = feature_df['node_id'].values
        self.id_to_idx = {nid: i for i, nid in enumerate(self.node_ids)}
        
        # Convert to torch tensor for fast computation
        features = torch.tensor(feature_df[feat_cols].values, dtype=torch.float32)
        
        if metric == "pearson":
            # Normalize for Pearson correlation (mean=0, std=1 per row)
            features = features - features.mean(dim=1, keepdim=True)
            features = features / (features.norm(dim=1, keepdim=True) + 1e-8)
            self.features = features
        else:
            self.features = features
            
        self.metric = metric

    def get_candidates_with_scores(self, node_id: str | int) -> tuple[np.ndarray, np.ndarray]:
        idx = self.id_to_idx.get(node_id)
        if idx is None:
            raise KeyError(f"Node {node_id} not found in features.")
            
        anchor_feat = self.features[idx].unsqueeze(0)  # (1, D)
        
        if self.metric == "pearson":
            # Since vectors are normalized, dot product = pearson correlation
            sim_scores = torch.mm(anchor_feat, self.features.T).squeeze(0)  # (N,)
        else:
            # Fallback to cosine similarity
            sim_scores = torch.nn.functional.cosine_similarity(anchor_feat, self.features)
            
        return self.node_ids.copy(), sim_scores.numpy()

    def pairwise_similarity(self, node_id_a: str | int, node_id_b: str | int) -> float:
        idx_a = self.id_to_idx.get(node_id_a)
        idx_b = self.id_to_idx.get(node_id_b)
        
        feat_a = self.features[idx_a]
        feat_b = self.features[idx_b]
        
        if self.metric == "pearson":
            return float(torch.dot(feat_a, feat_b).item())
        else:
            return float(torch.nn.functional.cosine_similarity(feat_a.unsqueeze(0), feat_b.unsqueeze(0)).item())


def load_data(config):
    """Load graph, features, and clusters based on analysis.ipynb logic."""
    print("Loading Graph...")
    graph_path = os.path.join(config["DATABASE"], config["GraphPATH"])
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
        
    print(f"Graph loaded: {len(graph.nodes)} nodes, {len(graph.edges)} edges.")
    
    print("Loading Features...")
    feat_path = os.path.join(config["DATABASE"], config.get("FeatPATH", "merged_feature_data_v041226.csv"))
    res_feat_df = pd.read_csv(feat_path)
    
    print("Loading Clusters...")
    # Adjust this path if needed, using the one from analysis.ipynb
    cluster_path = config.get("ClusterPATH", "residue_cluster_labels_eps0_mcs19_mss5_eom.csv")
    # If the file is inside the database folder, you might want to os.path.join it as well
    if not os.path.exists(cluster_path) and os.path.exists(os.path.join(config["DATABASE"], cluster_path)):
        cluster_path = os.path.join(config["DATABASE"], cluster_path)
        
    cluster_df = pd.read_csv(cluster_path)
    cluster_df.rename(columns={'cluster_label': 'cluster'}, inplace=True)
    
    # Build node -> cluster mapping, filtering out noise (-1) if necessary
    node_to_cluster = {}
    for _, row in cluster_df.iterrows():
        nid = row['node_id']
        cid = int(row['cluster'])
        node_to_cluster[nid] = cid
        
    return graph, res_feat_df, node_to_cluster


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/RRGCL.yaml')
    
    # Training args
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--max_epochs', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--num_hops', type=int, default=None)
    
    # Curriculum args
    parser.add_argument('--n_pos', type=int, default=None)
    parser.add_argument('--n_neg', type=int, default=None)
    parser.add_argument('--warmup_epochs', type=int, default=None)
    parser.add_argument('--top_k_size', type=int, default=None)
    parser.add_argument('--bottom_k_size', type=int, default=None)
    parser.add_argument('--hard_percentile', type=float, default=None)
    return parser


def main():
    args = get_parser().parse_args()
    config = load_yaml(args.config)
    
    # 1. Load Data
    graph, res_feat_df, node_to_cluster = load_data(config)
    
    # 2. Setup Similarity Provider (using AlphaMissense features as example)
    print("Initializing Similarity Provider...")
    sim_provider = FeatureSimilarityProvider(res_feat_df, feat_prefix="am", metric="pearson")
    
    # 3. Setup Curriculum Config
    cur_cfg = config.get("curriculum", {})
    curriculum_cfg = CurriculumConfig(
        n_pos=args.n_pos if args.n_pos is not None else cur_cfg.get("n_pos", 10),
        n_neg=args.n_neg if args.n_neg is not None else cur_cfg.get("n_neg", 10),
        warmup_epochs=args.warmup_epochs if args.warmup_epochs is not None else cur_cfg.get("warmup_epochs", 10),
        top_k_size=args.top_k_size if args.top_k_size is not None else cur_cfg.get("top_k_size", 100),
        bottom_k_size=args.bottom_k_size if args.bottom_k_size is not None else cur_cfg.get("bottom_k_size", 100),
        hard_percentile=args.hard_percentile if args.hard_percentile is not None else cur_cfg.get("hard_percentile", 10.0)
    )
    
    # 4. Initialize Dataset
    print("Initializing AnchorBatchDataset (Building Pools)...")
    dataset = AnchorBatchDataset(
        graphs=[graph], # Assuming single connected graph or we treat it as one
        similarity_provider=sim_provider,
        curriculum_cfg=curriculum_cfg,
        node_to_cluster=node_to_cluster,
        seed=config.get("SEED", 42)
    )
    
    # 5. Initialize Collator
    print("Initializing Subgraph Collator...")
    
    train_cfg = config.get("training", {})
    num_hops = args.num_hops if args.num_hops is not None else train_cfg.get("num_hops", 2)
    batch_size = args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 32)
    num_workers = args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 4)
    max_epochs = args.max_epochs if args.max_epochs is not None else train_cfg.get("max_epochs", 50)
    
    # Define features to extract into the PyG subgraphs
    aa1_cols = [col for col in res_feat_df.columns if col.startswith('aa1') and col != 'aa1']
    node_feat_keys = ['rel_sasa', 'depth', 'hse_up', 'hse_down'] + aa1_cols
    
    collator = AnchorSubgraphCollator(
        graph_index=dataset.graph_index,
        similarity_provider=sim_provider,
        num_hops=num_hops,
        node_feature_keys=node_feat_keys,
        edge_feature_keys=None # Add if you have edge features
    )
    
    # 6. DataLoader
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 7. Mock Training Loop
    print("\nStarting Training Loop...")
    
    for epoch in range(max_epochs):
        dataset.set_epoch(epoch)
        print(f"\n--- Epoch {epoch} (Warmup: {epoch < curriculum_cfg.warmup_epochs}) ---")
        
        for batch_idx, batch in enumerate(tqdm(loader)):
            # 1. Forward pass
            pyg_batch = batch["pyg_batch"]
            # embeddings = model(pyg_batch) 
            
            # 2. Extract center embeddings
            # center_embs = embeddings[batch["center_mask"]]
            
            # 3. Build dense masks for Contrastive Loss
            cluster_ids = batch["cluster_ids"]
            sim_matrix = batch["similarity_matrix"]
            
            masks = build_contrastive_masks(cluster_ids, sim_matrix, curriculum_cfg)
            pos_mask = masks["pos_mask"]
            neg_mask = masks["neg_mask"]
            
            # loss = contrastive_loss(center_embs, pos_mask, neg_mask)
            # loss.backward()
            # optimizer.step()
            
            if batch_idx == 0:
                print(f"B_total nodes encoded: {len(batch['all_node_ids'])}")
                print(f"Mask shapes: {pos_mask.shape}")
                break # Just run one batch for demonstration
                

if __name__ == "__main__":
    main()
