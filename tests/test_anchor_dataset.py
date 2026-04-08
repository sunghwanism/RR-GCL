from __future__ import annotations

import unittest

import networkx as nx
import numpy as np

from data.anchor_dataset import (
    AnchorRelationDataset,
    NegativeSamplingConfig,
    RelationConfig,
    SimilarityProvider,
    anchor_relation_collate,
)


class DictSimilarityProvider:
    def __init__(self, mapping):
        self.mapping = mapping

    def get_candidates_with_scores(self, node_id):
        candidates, scores = self.mapping[node_id]
        return np.asarray(candidates, dtype=object), np.asarray(scores, dtype=np.float64)


class AnchorDatasetTest(unittest.TestCase):
    def _build_graphs(self):
        g1 = nx.Graph()
        g1.add_node(0, node_id="a0")
        g1.add_node(1, node_id="a1")
        g1.add_edge(0, 1)

        g2 = nx.Graph()
        g2.add_node(0, node_id="b0")
        g2.add_node(1, node_id="b1")
        g2.add_edge(0, 1)

        return [g1, g2]

    def _build_provider(self):
        mapping = {
            "a0": (["a0", "a1", "b0", "b1"], [1.0, 0.9, 0.2, 0.1]),
            "a1": (["a1", "a0", "b0", "b1"], [1.0, 0.95, 0.3, 0.2]),
            "b0": (["b0", "b1", "a0", "a1"], [1.0, 0.8, 0.25, 0.15]),
            "b1": (["b1", "b0", "a0", "a1"], [1.0, 0.85, 0.22, 0.11]),
        }
        return DictSimilarityProvider(mapping)

    def test_duplicate_node_id_fail_fast(self):
        g1 = nx.Graph()
        g1.add_node(0, node_id="x")
        g2 = nx.Graph()
        g2.add_node(0, node_id="x")

        provider = DictSimilarityProvider({"x": (["x"], [1.0])})
        with self.assertRaises(ValueError):
            AnchorRelationDataset([g1, g2], provider)

    def test_provider_shape_validation(self):
        class BadProvider:
            def get_candidates_with_scores(self, node_id):
                return np.asarray(["a0", "a1"]), np.asarray([[1.0, 0.5]])

        graphs = self._build_graphs()
        with self.assertRaises(TypeError):
            AnchorRelationDataset(graphs, BadProvider())

    def test_pos_neg_disjoint_and_self_exclusion(self):
        graphs = self._build_graphs()
        provider = self._build_provider()

        relation_cfg = RelationConfig(
            pos_strategy="topk",
            pos_topk=1,
            neg_strategy="topk",
            neg_topk=2,
            k_pos=1,
            k_neg=2,
        )
        dataset = AnchorRelationDataset(graphs, provider, relation_cfg=relation_cfg, seed=123)
        sample = dataset[0]

        anchor = sample["anchor_node_id"]
        self.assertNotIn(anchor, sample["pos_node_ids"])
        self.assertNotIn(anchor, sample["neg_node_ids"])
        self.assertEqual(set(sample["pos_node_ids"]).intersection(set(sample["neg_node_ids"])), set())

    def test_component_aware_rounding_and_fallback(self):
        graphs = self._build_graphs()
        provider = self._build_provider()

        relation_cfg = RelationConfig(
            pos_strategy="topk",
            pos_topk=1,
            neg_strategy="topk",
            neg_topk=3,
            k_pos=1,
            k_neg=3,
        )
        neg_cfg = NegativeSamplingConfig(
            same_cc_target_ratio=0.7,
            cross_cc_target_ratio=0.3,
            strict=False,
        )
        dataset = AnchorRelationDataset(graphs, provider, relation_cfg=relation_cfg, neg_sampling_cfg=neg_cfg, seed=7)
        sample = dataset[0]

        self.assertGreaterEqual(len(sample["neg_node_ids"]), 1)
        self.assertEqual(len(sample["neg_node_ids"]), len(sample["neg_hardness"]))

    def test_deterministic_sampling_with_epoch(self):
        graphs = self._build_graphs()
        provider = self._build_provider()

        relation_cfg = RelationConfig(
            pos_strategy="topk",
            pos_topk=2,
            neg_strategy="topk",
            neg_topk=2,
            k_pos=2,
            k_neg=2,
        )
        dataset = AnchorRelationDataset(graphs, provider, relation_cfg=relation_cfg, seed=99)

        dataset.set_epoch(3)
        a = dataset[1]
        b = dataset[1]
        self.assertEqual(a["pos_node_ids"], b["pos_node_ids"])
        self.assertEqual(a["neg_node_ids"], b["neg_node_ids"])

        dataset.set_epoch(4)
        c = dataset[1]
        # Different epoch should allow different stochastic outcomes.
        self.assertTrue(a["pos_node_ids"] != c["pos_node_ids"] or a["neg_node_ids"] != c["neg_node_ids"])

    def test_collate_ignore_rule(self):
        batch = [
            {
                "anchor_node_id": "a0",
                "component_id": 0,
                "pos_node_ids": ["a1"],
                "neg_node_ids": ["b0"],
                "neg_hardness": [0.8],
            },
            {
                "anchor_node_id": "b0",
                "component_id": 1,
                "pos_node_ids": ["b1"],
                "neg_node_ids": ["a0"],
                "neg_hardness": [0.2],
            },
        ]
        out = anchor_relation_collate(batch)
        candidates = out["batch_candidate_node_ids"]
        idx_map = {nid: i for i, nid in enumerate(candidates)}

        row0_ignore = out["ignore_mask"][0]
        self.assertFalse(bool(row0_ignore[idx_map["a0"]]))
        self.assertFalse(bool(row0_ignore[idx_map["a1"]]))
        self.assertFalse(bool(row0_ignore[idx_map["b0"]]))


if __name__ == "__main__":
    unittest.main()
