# RR-GCL
Graph Contrastive Learning for Residue-Residue Network

## Dependency
- Python >= 3.11
- torch >= 2.7.0
- networkx >= 3.5
```
git clone https://github.com/sunghwanism/RR-GCL.git
cd RR-GCL
pip install -e .
```

## Data Pipeline for Training

The project utilizes an **in-batch contrastive learning pipeline** (`models/RRGCL/data/anchor_dataset.py`) with the following components:

1. **In-Batch Target Expansion**
   - The `AnchorBatchDataset` samples `1 + N_pos + N_neg` candidates per anchor.
   - The `ExpandedBatchCollator` flattens a batch of $B$ anchors into an expanded batch of $B_{total}$ deduplicated nodes, returning PyG subgraphs for all nodes and a $B_{total} \times B_{total}$ pairwise similarity matrix.

2. **Curriculum-Based Pair Sampling (`CurriculumConfig`)**
   - **Warm-up Stage**: Positives are self-positives (70%) and same-cluster soft-positives (30%). Negatives are different-cluster soft-negatives (100%).
   - **Post Warm-up Stage**: Positives include different-cluster hard-positives (20%). Negatives include same-cluster hard-negatives (15%), which are subsampled from similarity-ranked pools.

3. **Dense Mask Construction**
   - The `build_contrastive_masks()` utility dynamically constructs $B_{total} \times B_{total}$ `pos_mask` and `neg_mask` at loss computation time.
   - Masks are determined by absolute rules using the nodes' cluster IDs and their pre-computed similarities.