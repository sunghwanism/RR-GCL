"""N-hop subgraph extraction and PyG-ready batch construction."""

from __future__ import annotations

from typing import Dict, List, Sequence

import networkx as nx
import torch
from torch_geometric.data import Batch, Data

from models.RRGCL.data.anchor_dataset import (
    ExpandedBatchCollator,
    GraphIndex,
    SimilarityProvider,
)


def _resolve_graph_node_key_by_node_id(graph: nx.Graph, center_node_id: str | int) -> object:
    """Resolve a graph-local node key from a global node_id attribute."""
    for graph_node, attrs in graph.nodes(data=True):
        if attrs.get("node_id") == center_node_id:
            return graph_node
    raise KeyError(f"center_node_id not found in graph node attributes: {center_node_id}")


def _gather_node_features(
    ego: nx.Graph,
    ego_nodes: Sequence[object],
    node_feature_keys: Sequence[str],
) -> tuple[torch.Tensor, Dict[str, List[str]]]:
    numeric_keys = set(node_feature_keys)
    string_keys = set()

    # Determine type of each key by checking all nodes.
    # If any node has a value that cannot be cast to float, the key is treated as a string feature.
    for key in node_feature_keys:
        for node in ego_nodes:
            val = ego.nodes[node].get(key)
            if val is not None:
                try:
                    float(val)
                except (ValueError, TypeError):
                    numeric_keys.remove(key)
                    string_keys.add(key)
                    break  # Found a string, no need to check other nodes for this key

    # Convert sets back to lists for stable ordering
    numeric_keys_list = [k for k in node_feature_keys if k in numeric_keys]
    string_keys_list = [k for k in node_feature_keys if k in string_keys]

    rows: List[List[float]] = []
    string_features: Dict[str, List[str]] = {key: [] for key in string_keys_list}

    for node in ego_nodes:
        attrs = ego.nodes[node]
        row: List[float] = []
        for key in numeric_keys_list:
            if key not in attrs:
                raise KeyError(f"Missing node feature key '{key}' for node {attrs.get('node_id', node)}")
            row.append(float(attrs[key]))
        rows.append(row)

        for key in string_keys_list:
            if key not in attrs:
                raise KeyError(f"Missing node feature key '{key}' for node {attrs.get('node_id', node)}")
            string_features[key].append(attrs[key])

    if len(rows) > 0:
        x = torch.tensor(rows, dtype=torch.float)
    else:
        x = torch.empty((len(ego_nodes), 0), dtype=torch.float)

    return x, string_features


def _gather_edge_features(
    ego: nx.Graph,
    local_idx: Dict[object, int],
    edge_feature_keys: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    src_list: List[int] = []
    dst_list: List[int] = []
    edge_rows: List[List[float]] = []

    for src_node, dst_node, edge_attrs in ego.edges(data=True):
        src_local = local_idx[src_node]
        dst_local = local_idx[dst_node]

        feature_row: List[float] = []
        for key in edge_feature_keys:
            if key not in edge_attrs:
                raise KeyError(
                    f"Missing edge feature key '{key}' for edge ({src_node}, {dst_node})"
                )
            feature_row.append(float(edge_attrs[key]))

        # Add both directions for undirected message passing.
        src_list.extend([src_local, dst_local])
        dst_list.extend([dst_local, src_local])
        edge_rows.extend([feature_row, feature_row])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    if len(edge_rows) == 0:
        edge_attr = torch.empty((0, len(edge_feature_keys)), dtype=torch.float)
    else:
        edge_attr = torch.tensor(edge_rows, dtype=torch.float)
    return edge_index, edge_attr


def _build_edge_index_only(ego: nx.Graph, local_idx: Dict[object, int]) -> torch.Tensor:
    src_list: List[int] = []
    dst_list: List[int] = []
    for src_node, dst_node in ego.edges():
        src_local = local_idx[src_node]
        dst_local = local_idx[dst_node]
        src_list.extend([src_local, dst_local])
        dst_list.extend([dst_local, src_local])
    return torch.tensor([src_list, dst_list], dtype=torch.long)


def extract_nhop_subgraph(
    graph: nx.Graph,
    center_node_id: str | int,
    num_hops: int,
    node_feature_keys: Sequence[str] | None = None,
    edge_feature_keys: Sequence[str] | None = None,
) -> Data:
    """Extract an N-hop ego-subgraph centered at a global node_id.

    center_node_id must match a node-level ``node_id`` attribute in graph.
    """
    if num_hops < 1:
        raise ValueError("num_hops must be >= 1")

    # center_graph_key = _resolve_graph_node_key_by_node_id(graph, center_node_id)
    center_graph_key = center_node_id
    ego = nx.ego_graph(graph, center_graph_key, radius=num_hops)

    ego_nodes = list(ego.nodes())
    local_idx: Dict[object, int] = {node: idx for idx, node in enumerate(ego_nodes)}
    num_nodes = len(ego_nodes)

    center_mask = torch.zeros(num_nodes, dtype=torch.bool)
    center_mask[local_idx[center_graph_key]] = True

    data = Data(center_mask=center_mask, num_nodes=num_nodes)

    if edge_feature_keys is not None:
        edge_index, edge_attr = _gather_edge_features(ego, local_idx, edge_feature_keys)
        data.edge_index = edge_index
        data.edge_attr = edge_attr
    else:
        data.edge_index = _build_edge_index_only(ego, local_idx)

    if node_feature_keys is not None:
        x, string_feats = _gather_node_features(ego, ego_nodes, node_feature_keys)
        data.x = x
        for k, v in string_feats.items():
            setattr(data, k, v)

    return data


class AnchorSubgraphCollator:
    """Collator that builds PyG batched subgraphs from the expanded batch.

    Delegates relationship metadata to ``ExpandedBatchCollator`` and adds
    N-hop subgraph extraction for each unique node in the expanded batch.
    """

    def __init__(
        self,
        graph_index: GraphIndex,
        similarity_provider: SimilarityProvider,
        num_hops: int,
        node_feature_keys: Sequence[str] | None = None,
        edge_feature_keys: Sequence[str] | None = None,
    ) -> None:
        if num_hops < 1:
            raise ValueError("num_hops must be >= 1")
        self.graph_index = graph_index
        self.expanded_collator = ExpandedBatchCollator(similarity_provider)
        self.num_hops = int(num_hops)
        self.node_feature_keys = node_feature_keys
        self.edge_feature_keys = edge_feature_keys

    def __call__(self, batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
        # 1) Flatten & build metadata (cluster_ids, sim matrix, etc.)
        expanded = self.expanded_collator(batch)
        all_node_ids: List[str | int] = expanded["all_node_ids"]

        # 2) Extract subgraphs for each unique node
        data_list: List[Data] = []
        for node_id in all_node_ids:
            component_id = self.graph_index.component_id(node_id)
            graph = self.graph_index.graphs[component_id]
            subgraph_data = extract_nhop_subgraph(
                graph=graph,
                center_node_id=node_id,
                num_hops=self.num_hops,
                node_feature_keys=self.node_feature_keys,
                edge_feature_keys=self.edge_feature_keys,
            )
            data_list.append(subgraph_data)

        pyg_batch = Batch.from_data_list(data_list)

        return {
            "pyg_batch": pyg_batch,
            "center_mask": pyg_batch.center_mask,
            "all_node_ids": expanded["all_node_ids"],
            "cluster_ids": expanded["cluster_ids"],
            "similarity_matrix": expanded["similarity_matrix"],
            "anchor_indices": expanded["anchor_indices"],
            "is_self_positive": expanded["is_self_positive"],
        }

