"""Tests for N-hop subgraph extraction and AnchorSubgraphCollator."""

from __future__ import annotations

import unittest

import networkx as nx
import numpy as np

from data.anchor_dataset import AnchorRelationDataset, RelationConfig
from data.subgraph_extractor import AnchorSubgraphCollator, extract_nhop_subgraph


class DictSimilarityProvider:
    """Simple similarity provider used by unit tests."""

    def __init__(self, mapping):
        self.mapping = mapping

    def get_candidates_with_scores(self, node_id):
        candidate_ids, scores = self.mapping[node_id]
        return np.asarray(candidate_ids, dtype=object), np.asarray(scores, dtype=np.float64)


class ExtractNHopSubgraphTest(unittest.TestCase):
    """Tests for the standalone extract_nhop_subgraph function."""

    def _build_chain_graph(self) -> nx.Graph:
        graph = nx.path_graph(5)
        for node in graph.nodes():
            graph.nodes[node]["node_id"] = f"n{node}"
            graph.nodes[node]["feat_a"] = float(node * 10)
        for src, dst in graph.edges():
            graph.edges[src, dst]["w"] = float(src + dst)
        return graph

    def test_nhop_coverage(self):
        graph = self._build_chain_graph()

        hop1 = extract_nhop_subgraph(graph, center_node_id="n2", num_hops=1)
        hop2 = extract_nhop_subgraph(graph, center_node_id="n2", num_hops=2)
        hop3 = extract_nhop_subgraph(graph, center_node_id="n0", num_hops=3)

        self.assertEqual(hop1.num_nodes, 3)
        self.assertEqual(hop2.num_nodes, 5)
        self.assertEqual(hop3.num_nodes, 4)

    def test_center_mask_single_true(self):
        graph = self._build_chain_graph()
        data = extract_nhop_subgraph(graph, center_node_id="n1", num_hops=2)
        self.assertEqual(data.center_mask.sum().item(), 1)
        self.assertEqual(data.center_mask.shape[0], data.num_nodes)

    def test_local_reindex_and_symmetric_edges(self):
        graph = self._build_chain_graph()
        data = extract_nhop_subgraph(graph, center_node_id="n1", num_hops=1)

        self.assertTrue(data.edge_index.min().item() >= 0)
        self.assertTrue(data.edge_index.max().item() < data.num_nodes)

        edge_set = set(
            (int(data.edge_index[0, idx].item()), int(data.edge_index[1, idx].item()))
            for idx in range(data.edge_index.shape[1])
        )
        for src, dst in list(edge_set):
            self.assertIn((dst, src), edge_set)

    def test_feature_fail_fast_on_missing_key(self):
        graph = self._build_chain_graph()
        with self.assertRaises(KeyError):
            extract_nhop_subgraph(graph, center_node_id="n1", num_hops=1, node_feature_keys=["missing"])

        with self.assertRaises(KeyError):
            extract_nhop_subgraph(graph, center_node_id="n1", num_hops=1, edge_feature_keys=["missing"])

    def test_non_positive_hops_raise(self):
        graph = self._build_chain_graph()
        with self.assertRaises(ValueError):
            extract_nhop_subgraph(graph, center_node_id="n1", num_hops=0)


class AnchorSubgraphCollatorTest(unittest.TestCase):
    """Integration tests for AnchorSubgraphCollator."""

    def _build_graphs(self):
        g1 = nx.Graph()
        g1.add_node(0, node_id="a0", feat=1.0)
        g1.add_node(1, node_id="a1", feat=2.0)
        g1.add_node(2, node_id="a2", feat=3.0)
        g1.add_edge(0, 1, edge_feat=1.0)
        g1.add_edge(1, 2, edge_feat=2.0)

        g2 = nx.Graph()
        g2.add_node(0, node_id="b0", feat=4.0)
        g2.add_node(1, node_id="b1", feat=5.0)
        g2.add_node(2, node_id="b2", feat=6.0)
        g2.add_edge(0, 1, edge_feat=3.0)
        g2.add_edge(1, 2, edge_feat=4.0)

        return [g1, g2]

    def _build_provider(self):
        all_ids = ["a0", "a1", "a2", "b0", "b1", "b2"]
        mapping = {}
        for nid in all_ids:
            candidates = list(all_ids)
            scores = []
            for cid in candidates:
                if cid == nid:
                    scores.append(1.0)
                elif cid[0] == nid[0]:
                    scores.append(0.8)
                else:
                    scores.append(0.2)
            mapping[nid] = (candidates, scores)
        return DictSimilarityProvider(mapping)

    def test_collator_outputs_pyg_batch_and_masks(self):
        graphs = self._build_graphs()
        provider = self._build_provider()

        relation_cfg = RelationConfig(
            pos_strategy="topk",
            pos_topk=2,
            neg_strategy="topk",
            neg_topk=2,
            k_pos=1,
            k_neg=1,
        )
        dataset = AnchorRelationDataset(graphs, provider, relation_cfg=relation_cfg, seed=42)

        collator = AnchorSubgraphCollator(
            graph_index=dataset.graph_index,
            num_hops=2,
            node_feature_keys=["feat"],
            edge_feature_keys=["edge_feat"],
        )

        items = [dataset[0], dataset[1]]
        out = collator(items)

        self.assertIn("pyg_batch", out)
        self.assertIn("center_mask", out)
        self.assertEqual(out["pyg_batch"].num_graphs, len(out["batch_candidate_node_ids"]))
        self.assertEqual(out["center_mask"].sum().item(), len(out["batch_candidate_node_ids"]))

        batch_size = len(items)
        candidate_count = len(out["batch_candidate_node_ids"])
        self.assertEqual(out["pos_mask"].shape, (batch_size, candidate_count))
        self.assertEqual(out["neg_mask"].shape, (batch_size, candidate_count))
        self.assertEqual(out["ignore_mask"].shape, (batch_size, candidate_count))
        self.assertEqual(out["neg_weight"].shape, (batch_size, candidate_count))

    def test_ignore_mask_rowwise_rule(self):
        graphs = self._build_graphs()
        provider = self._build_provider()

        relation_cfg = RelationConfig(
            pos_strategy="topk",
            pos_topk=2,
            neg_strategy="topk",
            neg_topk=2,
            k_pos=1,
            k_neg=1,
        )
        dataset = AnchorRelationDataset(graphs, provider, relation_cfg=relation_cfg, seed=7)
        collator = AnchorSubgraphCollator(dataset.graph_index, num_hops=2)

        items = [dataset[0], dataset[1]]
        out = collator(items)

        idx_map = {nid: idx for idx, nid in enumerate(out["batch_candidate_node_ids"])}

        for row, item in enumerate(items):
            anchor_idx = idx_map[item["anchor_node_id"]]
            self.assertFalse(bool(out["ignore_mask"][row, anchor_idx]))
            for pid in item["pos_node_ids"]:
                self.assertFalse(bool(out["ignore_mask"][row, idx_map[pid]]))
            for nid in item["neg_node_ids"]:
                self.assertFalse(bool(out["ignore_mask"][row, idx_map[nid]]))


if __name__ == "__main__":
    unittest.main()
