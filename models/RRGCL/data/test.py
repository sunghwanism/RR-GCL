import sys
import os
import networkx as nx
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from models.RRGCL.data.anchor_dataset import (
    AnchorBatchDataset,
    CurriculumConfig,
    DenseMatrixSimilarityProvider,
    build_contrastive_masks,
)
from models.RRGCL.data.subgraph_extractor import AnchorSubgraphCollator


def create_dummy_data():
    """Create dummy graphs and similarity provider for testing."""
    graphs = []
    node_ids = []
    
    # Create 3 disconnected graphs (components)
    for comp_idx in range(3):
        g = nx.Graph()
        for i in range(15):
            node_id = f"C{comp_idx}_N{i}"
            g.add_node(i, node_id=node_id)
            node_ids.append(node_id)
            if i > 0:
                g.add_edge(i, i - 1)
        graphs.append(g)
        
    num_nodes = len(node_ids)
    
    # Create a dummy similarity matrix (random values between 0 and 1)
    np.random.seed(42)
    # Make it symmetric
    sim_matrix = np.random.rand(num_nodes, num_nodes)
    sim_matrix = (sim_matrix + sim_matrix.T) / 2
    np.fill_diagonal(sim_matrix, 1.0)
    
    sim_provider = DenseMatrixSimilarityProvider(node_ids, sim_matrix)
    
    return graphs, sim_provider, node_ids


def test_dataloader():
    print("=== Testing In-Batch Contrastive DataLoader ===")
    graphs, sim_provider, node_ids = create_dummy_data()
    
    # 1. Config
    # We set small values for testing
    cfg = CurriculumConfig(
        n_pos=4, 
        n_neg=4, 
        warmup_epochs=2,
        top_k_size=10,
        bottom_k_size=10,
        hard_percentile=20.0
    )
    
    # 2. Dataset
    print("\n[1] Initializing AnchorBatchDataset...")
    dataset = AnchorBatchDataset(graphs, sim_provider, curriculum_cfg=cfg)
    print(f"Eligible anchors: {len(dataset)}")
    
    # 3. Collator
    print("\n[2] Initializing AnchorSubgraphCollator...")
    collator = AnchorSubgraphCollator(
        graph_index=dataset.graph_index,
        similarity_provider=sim_provider,
        num_hops=1, # 1-hop subgraphs to keep it small
        node_feature_keys=None,
        edge_feature_keys=None
    )
    
    # 4. DataLoader
    batch_size = 2
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    
    # 5. Run Epochs
    for epoch in range(3):
        print(f"\n--- Epoch {epoch} (Warmup: {epoch < cfg.warmup_epochs}) ---")
        dataset.set_epoch(epoch)
        
        for batch_idx, batch in enumerate(loader):
            if batch_idx > 0:
                break # Just test first batch
                
            pyg_batch = batch["pyg_batch"]
            all_node_ids = batch["all_node_ids"]
            cluster_ids = batch["cluster_ids"]
            sim_matrix = batch["similarity_matrix"]
            
            print(f"Batch {batch_idx}:")
            print(f"  PyG Batch: {pyg_batch}")
            print(f"  B_total (unique nodes in batch): {len(all_node_ids)}")
            print(f"  Cluster IDs shape: {cluster_ids.shape}")
            print(f"  Similarity matrix shape: {sim_matrix.shape}")
            
            # Test mask building
            masks = build_contrastive_masks(cluster_ids, sim_matrix, cfg)
            
            pos_mask = masks["pos_mask"]
            neg_mask = masks["neg_mask"]
            ignore_mask = masks["ignore_mask"]
            
            print(f"  Positive mask shape: {pos_mask.shape}, sum: {pos_mask.sum().item()}")
            print(f"  Negative mask shape: {neg_mask.shape}, sum: {neg_mask.sum().item()}")
            print(f"  Ignore mask shape: {ignore_mask.shape}, sum: {ignore_mask.sum().item()}")
            
            # Verify basic mask properties
            assert pos_mask.shape == (len(all_node_ids), len(all_node_ids))
            assert not torch.any(pos_mask & neg_mask), "Overlap between pos and neg masks!"
            
            # Diagonal should be False in pos_mask/neg_mask and True in ignore_mask
            b_total = len(all_node_ids)
            diag = torch.eye(b_total, dtype=torch.bool)
            assert not torch.any(pos_mask & diag), "Diagonal should not be in pos_mask"
            assert not torch.any(neg_mask & diag), "Diagonal should not be in neg_mask"
            assert torch.all(ignore_mask[diag]), "Diagonal should be in ignore_mask"

    print("\n=== Test Completed Successfully! ===")

if __name__ == "__main__":
    test_dataloader()
