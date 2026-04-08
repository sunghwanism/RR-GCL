from __future__ import annotations

import tempfile
import unittest

from data.config_loader import (
    load_dataloader_config_bundle,
    load_negative_sampling_config,
    load_relation_config,
    load_subgraph_config,
)


class ConfigLoaderTest(unittest.TestCase):
    def _write_yaml(self, text: str) -> str:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        tmp.write(text)
        tmp.flush()
        tmp.close()
        return tmp.name

    def test_load_relation_and_defaults(self):
        path = self._write_yaml("pos_strategy: topk\npos_topk: 5\n")
        cfg = load_relation_config(path)
        self.assertEqual(cfg.pos_strategy, "topk")
        self.assertEqual(cfg.pos_topk, 5)
        self.assertEqual(cfg.neg_strategy, "topk")

    def test_load_negative_sampling(self):
        path = self._write_yaml(
            "mode: component_aware\nsame_cc_target_ratio: 0.6\ncross_cc_target_ratio: 0.4\nfallback: fill_from_other_pool\n"
        )
        cfg = load_negative_sampling_config(path)
        self.assertEqual(cfg.mode, "component_aware")
        self.assertAlmostEqual(cfg.same_cc_target_ratio + cfg.cross_cc_target_ratio, 1.0)

    def test_subgraph_non_positive_hops_raise(self):
        path = self._write_yaml("num_hops: 0\n")
        with self.assertRaises(ValueError):
            load_subgraph_config(path)

    def test_subgraph_defaults_and_types(self):
        path = self._write_yaml("num_hops: 3\nnode_feature_keys: [a, b]\n")
        cfg = load_subgraph_config(path)
        self.assertEqual(cfg.num_hops, 3)
        self.assertEqual(cfg.node_feature_keys, ("a", "b"))
        self.assertIsNone(cfg.edge_feature_keys)

    def test_invalid_strategy_typo_raises(self):
        path = self._write_yaml("pos_strategy: topkk\n")
        with self.assertRaises(ValueError):
            load_relation_config(path)

    def test_ratio_sum_mismatch_raises(self):
        path = self._write_yaml(
            "mode: component_aware\nsame_cc_target_ratio: 0.8\ncross_cc_target_ratio: 0.3\nfallback: fill_from_other_pool\n"
        )
        with self.assertRaises(ValueError):
            load_negative_sampling_config(path)

    def test_load_dataloader_bundle(self):
        path = self._write_yaml(
            """
relation:
  pos_strategy: topk
  pos_topk: 3
neg_sampling:
  mode: component_aware
  same_cc_target_ratio: 0.7
  cross_cc_target_ratio: 0.3
  fallback: fill_from_other_pool
subgraph:
  num_hops: 2
"""
        )
        relation_cfg, neg_cfg, subgraph_cfg = load_dataloader_config_bundle(path)
        self.assertEqual(relation_cfg.pos_topk, 3)
        self.assertEqual(neg_cfg.mode, "component_aware")
        self.assertEqual(subgraph_cfg.num_hops, 2)


if __name__ == "__main__":
    unittest.main()
