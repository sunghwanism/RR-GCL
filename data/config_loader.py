from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Sequence
import math

import yaml

from data.anchor_dataset import NegativeSamplingConfig, RelationConfig


@dataclass(frozen=True)
class SubgraphConfig:
    """Configuration for N-hop subgraph extraction and feature gathering."""

    num_hops: int = 2
    node_feature_keys: tuple[str, ...] | None = None
    edge_feature_keys: tuple[str, ...] | None = None


def _read_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML payload must be a mapping: {path}")
    return payload


def _filter_dataclass_fields(payload: Dict[str, Any], cls: type) -> Dict[str, Any]:
    valid = {item.name for item in fields(cls)}
    return {k: v for k, v in payload.items() if k in valid}


def _to_tuple_or_none(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    raise ValueError("Feature key fields must be a sequence of strings or null")


def load_relation_config(path: str | Path) -> RelationConfig:
    payload = _read_yaml(path)
    config = RelationConfig(**_filter_dataclass_fields(payload, RelationConfig))
    validate_relation_config(config)
    return config


def load_negative_sampling_config(path: str | Path) -> NegativeSamplingConfig:
    payload = _read_yaml(path)
    config = NegativeSamplingConfig(**_filter_dataclass_fields(payload, NegativeSamplingConfig))
    validate_negative_sampling_config(config)
    return config


def load_subgraph_config(path: str | Path) -> SubgraphConfig:
    payload = _read_yaml(path)
    data = _filter_dataclass_fields(payload, SubgraphConfig)

    if "node_feature_keys" in data:
        data["node_feature_keys"] = _to_tuple_or_none(data["node_feature_keys"])
    if "edge_feature_keys" in data:
        data["edge_feature_keys"] = _to_tuple_or_none(data["edge_feature_keys"])

    config = SubgraphConfig(**data)
    validate_subgraph_config(config)
    return config


def validate_subgraph_config(config: SubgraphConfig) -> None:
    if config.num_hops < 1:
        raise ValueError("num_hops must be >= 1")


def validate_relation_config(config: RelationConfig) -> None:
    if config.pos_strategy not in {"topk", "percentile"}:
        raise ValueError("pos_strategy must be one of {'topk','percentile'}")
    if config.neg_strategy not in {"topk", "percentile"}:
        raise ValueError("neg_strategy must be one of {'topk','percentile'}")
    if config.k_pos <= 0:
        raise ValueError("k_pos must be positive")
    if config.k_neg <= 0:
        raise ValueError("k_neg must be positive")


def validate_negative_sampling_config(config: NegativeSamplingConfig) -> None:
    if config.mode != "component_aware":
        raise ValueError("mode must be 'component_aware'")
    ratio_sum = config.same_cc_target_ratio + config.cross_cc_target_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("same_cc_target_ratio + cross_cc_target_ratio must equal 1.0")
    if config.fallback != "fill_from_other_pool":
        raise ValueError("fallback must be 'fill_from_other_pool'")


def load_dataloader_config_bundle(path: str | Path) -> tuple[RelationConfig, NegativeSamplingConfig, SubgraphConfig]:
    """Load an optional merged dataloader YAML bundle.

    Expected shape:
      relation: {...}
      neg_sampling: {...}
      subgraph: {...}
    """

    payload = _read_yaml(path)
    relation_payload = payload.get("relation", {})
    neg_payload = payload.get("neg_sampling", {})
    subgraph_payload = payload.get("subgraph", {})

    if not isinstance(relation_payload, dict) or not isinstance(neg_payload, dict) or not isinstance(subgraph_payload, dict):
        raise ValueError("dataloader bundle must contain mapping blocks: relation, neg_sampling, subgraph")

    relation_cfg = RelationConfig(**_filter_dataclass_fields(relation_payload, RelationConfig))
    neg_cfg = NegativeSamplingConfig(**_filter_dataclass_fields(neg_payload, NegativeSamplingConfig))

    if "node_feature_keys" in subgraph_payload:
        subgraph_payload["node_feature_keys"] = _to_tuple_or_none(subgraph_payload["node_feature_keys"])
    if "edge_feature_keys" in subgraph_payload:
        subgraph_payload["edge_feature_keys"] = _to_tuple_or_none(subgraph_payload["edge_feature_keys"])

    subgraph_cfg = SubgraphConfig(**_filter_dataclass_fields(subgraph_payload, SubgraphConfig))
    validate_relation_config(relation_cfg)
    validate_negative_sampling_config(neg_cfg)
    validate_subgraph_config(subgraph_cfg)

    return relation_cfg, neg_cfg, subgraph_cfg
