from __future__ import annotations

import unittest

import networkx as nx
import numpy as np

from data.anchor_dataset import (
    AnchorRelationDataset,
    NegativeSamplingConfig,
    RelationConfig,
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
        # Changed epoch still must satisfy invariants.
        anchor = c["anchor_node_id"]
        self.assertNotIn(anchor, c["pos_node_ids"])
        self.assertNotIn(anchor, c["neg_node_ids"])
        self.assertEqual(set(c["pos_node_ids"]).intersection(set(c["neg_node_ids"])), set())
        self.assertEqual(len(c["neg_node_ids"]), len(c["neg_hardness"]))
        for hardness in c["neg_hardness"]:
            self.assertGreaterEqual(hardness, 0.0)
            self.assertLessEqual(hardness, 1.0)

    def test_sampling_diversity_across_epochs(self):
        g1 = nx.Graph()
        for idx in range(10):
            g1.add_node(idx, node_id=f"a{idx}")
        for idx in range(9):
            g1.add_edge(idx, idx + 1)

        g2 = nx.Graph()
        for idx in range(10):
            g2.add_node(idx, node_id=f"b{idx}")
        for idx in range(9):
            g2.add_edge(idx, idx + 1)

        graphs = [g1, g2]

        all_ids = [f"a{i}" for i in range(10)] + [f"b{i}" for i in range(10)]
        mapping = {}
        for node_id in all_ids:
            candidates = list(all_ids)
            scores = []
            node_is_a = node_id.startswith("a")
            for cand in candidates:
                if cand == node_id:
                    scores.append(1.0)
                elif cand.startswith("a") == node_is_a:
                    scores.append(0.7)
                else:
                    scores.append(0.2)
            mapping[node_id] = (candidates, scores)

        provider = DictSimilarityProvider(mapping)
        relation_cfg = RelationConfig(
            pos_strategy="topk",
            pos_topk=8,
            neg_strategy="topk",
            neg_topk=8,
            k_pos=3,
            k_neg=4,
        )
        dataset = AnchorRelationDataset(graphs, provider, relation_cfg=relation_cfg, seed=1234)

        anchor_index = dataset.eligible_anchor_ids.index("a0")
        seen = set()
        for epoch in range(8):
            dataset.set_epoch(epoch)
            sample = dataset[anchor_index]
            signature = (tuple(sample["pos_node_ids"]), tuple(sample["neg_node_ids"]))
            seen.add(signature)

        self.assertGreaterEqual(len(seen), 2)

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
