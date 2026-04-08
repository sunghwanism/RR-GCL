from __future__ import annotations

import hashlib
import logging
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Protocol, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)


class SimilarityProvider(Protocol):
    """Protocol for candidate similarity lookup."""

    def get_candidates_with_scores(self, node_id: str | int) -> Tuple[np.ndarray, np.ndarray]:
        """Return aligned candidate IDs and similarity scores for one anchor."""


@dataclass(frozen=True)
class RelationConfig:
    """Configuration for relation assignment and anchor eligibility."""

    pos_strategy: str = "topk"
    pos_topk: int = 10
    pos_percentile: float = 95.0

    neg_strategy: str = "topk"
    neg_topk: int = 100
    neg_percentile: float = 10.0

    k_pos: int = 10
    k_neg: int = 100

    invalid_anchor_behavior: str = "drop"
    coverage_strict: bool = False

    min_cross_component_rate: float = 0.0
    min_k_neg_feasible_rate: float = 0.0
    min_median_candidate_pool: float = 0.0


@dataclass(frozen=True)
class NegativeSamplingConfig:
    """Configuration for component-aware negative sampling."""

    mode: str = "component_aware"
    same_cc_target_ratio: float = 0.7
    cross_cc_target_ratio: float = 0.3
    fallback: str = "fill_from_other_pool"
    strict: bool = False


@dataclass
class CoverageStats:
    """Startup validation statistics for candidate coverage."""

    num_anchors: int
    cross_component_candidate_rate: float
    k_neg_feasible_rate: float
    candidate_pool_size_min: int
    candidate_pool_size_median: float
    candidate_pool_size_max: int


class GraphIndex:
    """Global node/component index from connected component graphs."""

    def __init__(self, graphs: Sequence[nx.Graph]) -> None:
        if len(graphs) == 0:
            raise ValueError("graphs must contain at least one connected component graph")

        self.graphs: List[nx.Graph] = list(graphs)
        self.node_to_component: Dict[str | int, int] = {}
        self.node_to_graph_node: Dict[str | int, object] = {}
        self.anchor_node_ids: List[str | int] = []

        for component_id, graph in enumerate(self.graphs):
            for graph_node, attrs in graph.nodes(data=True):
                if "node_id" not in attrs:
                    raise ValueError("Every node must contain a globally unique 'node_id' attribute")
                node_id = attrs["node_id"]
                if node_id in self.node_to_component:
                    raise ValueError(f"Duplicate node_id detected across components: {node_id}")
                self.node_to_component[node_id] = component_id
                self.node_to_graph_node[node_id] = graph_node
                self.anchor_node_ids.append(node_id)

    def component_id(self, node_id: str | int) -> int:
        return self.node_to_component[node_id]


class AnchorRelationDataset(Dataset):
    """Anchor-centric relation dataset with explicit similarity-driven supervision."""

    def __init__(
        self,
        graphs: Sequence[nx.Graph],
        similarity_provider: SimilarityProvider,
        relation_cfg: RelationConfig | None = None,
        neg_sampling_cfg: NegativeSamplingConfig | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.graph_index = GraphIndex(graphs)
        self.similarity_provider = similarity_provider
        self.relation_cfg = relation_cfg or RelationConfig()
        self.neg_sampling_cfg = neg_sampling_cfg or NegativeSamplingConfig()
        self.base_seed = int(seed)
        self.epoch = 0

        self._validate_configs()

        self.relation_cache: Dict[str | int, Dict[str, object]] = {}
        self.eligible_anchor_ids: List[str | int] = []
        self.invalid_anchor_reasons: Dict[str | int, str] = {}
        self._candidate_pool_size: Dict[str | int, int] = {}

        self.coverage_stats = self._build_relation_cache()
        self._validate_coverage(self.coverage_stats)

    def _validate_configs(self) -> None:
        cfg = self.relation_cfg
        neg_cfg = self.neg_sampling_cfg

        if cfg.pos_strategy not in {"topk", "percentile"}:
            raise ValueError("pos_strategy must be one of {'topk','percentile'}")
        if cfg.neg_strategy not in {"topk", "percentile"}:
            raise ValueError("neg_strategy must be one of {'topk','percentile'}")
        if cfg.k_pos <= 0:
            raise ValueError("k_pos must be positive")
        if cfg.k_neg <= 0:
            raise ValueError("k_neg must be positive")
        if cfg.invalid_anchor_behavior not in {"drop", "raise"}:
            raise ValueError("invalid_anchor_behavior must be one of {'drop','raise'}")

        if neg_cfg.mode != "component_aware":
            raise ValueError("Only component_aware mode is supported")
        ratio_sum = neg_cfg.same_cc_target_ratio + neg_cfg.cross_cc_target_ratio
        if not math.isclose(ratio_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("same_cc_target_ratio + cross_cc_target_ratio must equal 1.0")
        if neg_cfg.fallback != "fill_from_other_pool":
            raise ValueError("Only fill_from_other_pool fallback is supported")

    def _build_relation_cache(self) -> CoverageStats:
        cross_component_flags: List[int] = []
        k_neg_feasible_flags: List[int] = []
        pool_sizes: List[int] = []

        for anchor_id in self.graph_index.anchor_node_ids:
            candidate_ids, candidate_scores = self._provider_candidates(anchor_id)
            candidate_ids, candidate_scores = self._sanitize_candidates(anchor_id, candidate_ids, candidate_scores)

            candidate_size = int(candidate_ids.size)
            self._candidate_pool_size[anchor_id] = candidate_size
            pool_sizes.append(candidate_size)

            anchor_cc = self.graph_index.component_id(anchor_id)
            cross_component_flags.append(int(np.any(np.array([self.graph_index.component_id(x) != anchor_cc for x in candidate_ids]))))

            pos_ids = self._select_positive(candidate_ids, candidate_scores)
            neg_candidate_ids, neg_candidate_scores = self._select_negative(candidate_ids, candidate_scores, pos_ids)
            neg_hardness_map = self._build_negative_hardness(neg_candidate_ids, neg_candidate_scores)

            if len(pos_ids) == 0:
                self._register_invalid_anchor(anchor_id, "zero_positive")
                continue
            if len(neg_candidate_ids) == 0:
                self._register_invalid_anchor(anchor_id, "zero_negative")
                continue

            k_neg_feasible_flags.append(int(len(neg_candidate_ids) >= self.relation_cfg.k_neg))

            self.relation_cache[anchor_id] = {
                "component_id": anchor_cc,
                "pos_pool": pos_ids,
                "neg_pool": neg_candidate_ids,
                "neg_hardness_map": neg_hardness_map,
            }
            self.eligible_anchor_ids.append(anchor_id)

        if len(self.eligible_anchor_ids) == 0:
            raise ValueError("No eligible anchors remain after relation assignment")

        pool_sizes_np = np.asarray(pool_sizes, dtype=np.int64)
        coverage = CoverageStats(
            num_anchors=len(self.graph_index.anchor_node_ids),
            cross_component_candidate_rate=float(np.mean(cross_component_flags)) if cross_component_flags else 0.0,
            k_neg_feasible_rate=float(np.mean(k_neg_feasible_flags)) if k_neg_feasible_flags else 0.0,
            candidate_pool_size_min=int(pool_sizes_np.min()) if pool_sizes_np.size else 0,
            candidate_pool_size_median=float(np.median(pool_sizes_np)) if pool_sizes_np.size else 0.0,
            candidate_pool_size_max=int(pool_sizes_np.max()) if pool_sizes_np.size else 0,
        )
        return coverage

    def _provider_candidates(self, anchor_id: str | int) -> Tuple[np.ndarray, np.ndarray]:
        payload = self.similarity_provider.get_candidates_with_scores(anchor_id)
        if not isinstance(payload, tuple) or len(payload) != 2:
            raise TypeError("SimilarityProvider must return Tuple[np.ndarray, np.ndarray]")
        candidate_ids, candidate_scores = payload
        if not isinstance(candidate_ids, np.ndarray) or not isinstance(candidate_scores, np.ndarray):
            raise TypeError("SimilarityProvider output must be numpy arrays")
        if candidate_ids.ndim != 1 or candidate_scores.ndim != 1:
            raise TypeError("SimilarityProvider output arrays must be 1-dimensional")
        if candidate_ids.shape[0] != candidate_scores.shape[0]:
            raise ValueError("Candidate IDs and scores must have the same length")
        return candidate_ids, candidate_scores.astype(np.float64, copy=False)

    def _sanitize_candidates(
        self,
        anchor_id: str | int,
        candidate_ids: np.ndarray,
        candidate_scores: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        valid_ids: List[str | int] = []
        valid_scores: List[float] = []
        seen = set()

        for cid, score in zip(candidate_ids.tolist(), candidate_scores.tolist()):
            if cid == anchor_id:
                continue
            if cid in seen:
                raise ValueError(f"Duplicate candidate node_id for anchor {anchor_id}: {cid}")
            if cid not in self.graph_index.node_to_component:
                raise ValueError(f"Unknown candidate node_id from SimilarityProvider: {cid}")
            seen.add(cid)
            valid_ids.append(cid)
            valid_scores.append(float(score))

        return np.asarray(valid_ids, dtype=object), np.asarray(valid_scores, dtype=np.float64)

    def _rank_indices(self, candidate_ids: np.ndarray, scores: np.ndarray, descending: bool) -> np.ndarray:
        if candidate_ids.size == 0:
            return np.empty((0,), dtype=np.int64)
        order = np.lexsort((candidate_ids.astype(str), -scores if descending else scores))
        return order

    def _select_positive(self, candidate_ids: np.ndarray, scores: np.ndarray) -> List[str | int]:
        if candidate_ids.size == 0:
            return []

        cfg = self.relation_cfg
        if cfg.pos_strategy == "topk":
            order = self._rank_indices(candidate_ids, scores, descending=True)
            k = min(cfg.pos_topk, order.size)
            selected = candidate_ids[order[:k]].tolist()
            return selected

        threshold = np.percentile(scores, cfg.pos_percentile)
        mask = scores >= threshold
        ids = candidate_ids[mask]
        scr = scores[mask]
        order = self._rank_indices(ids, scr, descending=True)
        return ids[order].tolist()

    def _select_negative(
        self,
        candidate_ids: np.ndarray,
        scores: np.ndarray,
        pos_ids: Sequence[str | int],
    ) -> Tuple[List[str | int], List[float]]:
        if candidate_ids.size == 0:
            return [], []

        pos_set = set(pos_ids)
        remain_idx = [idx for idx, cid in enumerate(candidate_ids.tolist()) if cid not in pos_set]
        if len(remain_idx) == 0:
            return [], []

        remain_ids = candidate_ids[remain_idx]
        remain_scores = scores[remain_idx]
        cfg = self.relation_cfg

        if cfg.neg_strategy == "topk":
            order = self._rank_indices(remain_ids, remain_scores, descending=False)
            k = min(cfg.neg_topk, order.size)
            ids = remain_ids[order[:k]].tolist()
            vals = remain_scores[order[:k]].tolist()
            return ids, vals

        threshold = np.percentile(remain_scores, cfg.neg_percentile)
        mask = remain_scores <= threshold
        ids = remain_ids[mask]
        vals = remain_scores[mask]
        order = self._rank_indices(ids, vals, descending=False)
        return ids[order].tolist(), vals[order].tolist()

    def _build_negative_hardness(
        self,
        neg_ids: Sequence[str | int],
        neg_scores: Sequence[float],
    ) -> Dict[str | int, float]:
        if len(neg_ids) == 0:
            return {}

        ids = np.asarray(neg_ids, dtype=object)
        scores = np.asarray(neg_scores, dtype=np.float64)
        order = self._rank_indices(ids, scores, descending=False)

        hardness_map: Dict[str | int, float] = {}
        if order.size == 1:
            hardness_map[ids[order[0]]] = 1.0
            return hardness_map

        denom = float(order.size - 1)
        for rank, idx in enumerate(order.tolist()):
            hardness_map[ids[idx]] = rank / denom
        return hardness_map

    def _register_invalid_anchor(self, anchor_id: str | int, reason: str) -> None:
        if self.relation_cfg.invalid_anchor_behavior == "raise":
            raise ValueError(f"Invalid anchor {anchor_id}: {reason}")
        self.invalid_anchor_reasons[anchor_id] = reason

    def _validate_coverage(self, stats: CoverageStats) -> None:
        messages: List[str] = []
        cfg = self.relation_cfg

        if stats.cross_component_candidate_rate < cfg.min_cross_component_rate:
            messages.append(
                f"cross_component_candidate_rate={stats.cross_component_candidate_rate:.4f} "
                f"is below min_cross_component_rate={cfg.min_cross_component_rate:.4f}"
            )
        if stats.k_neg_feasible_rate < cfg.min_k_neg_feasible_rate:
            messages.append(
                f"k_neg_feasible_rate={stats.k_neg_feasible_rate:.4f} "
                f"is below min_k_neg_feasible_rate={cfg.min_k_neg_feasible_rate:.4f}"
            )
        if stats.candidate_pool_size_median < cfg.min_median_candidate_pool:
            messages.append(
                f"candidate_pool_size_median={stats.candidate_pool_size_median:.2f} "
                f"is below min_median_candidate_pool={cfg.min_median_candidate_pool:.2f}"
            )

        if not messages:
            return

        text = "Coverage validation warning: " + "; ".join(messages)
        if cfg.coverage_strict:
            raise ValueError(text)
        warnings.warn(text)
        LOGGER.warning(text)

    def set_epoch(self, epoch: int) -> None:
        """Set epoch value for deterministic sampling changes across epochs."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.eligible_anchor_ids)

    def __getitem__(self, index: int) -> Dict[str, object]:
        anchor_id = self.eligible_anchor_ids[index]
        relation = self.relation_cache[anchor_id]

        rng = np.random.default_rng(self._anchor_seed(anchor_id))

        pos_pool = relation["pos_pool"]
        neg_pool = relation["neg_pool"]
        neg_hardness_map = relation["neg_hardness_map"]

        pos_count = min(self.relation_cfg.k_pos, len(pos_pool))
        if pos_count == 0:
            raise RuntimeError("eligible anchors must always have at least one positive")
        pos_ids = self._sample_without_replacement(pos_pool, pos_count, rng)

        neg_ids = self._sample_component_aware_negatives(anchor_id, neg_pool, rng)
        neg_hardness = [float(neg_hardness_map[nid]) for nid in neg_ids]

        return {
            "anchor_node_id": anchor_id,
            "component_id": relation["component_id"],
            "pos_node_ids": pos_ids,
            "neg_node_ids": neg_ids,
            "neg_hardness": neg_hardness,
        }

    def _sample_without_replacement(
        self,
        pool: Sequence[str | int],
        k: int,
        rng: np.random.Generator,
    ) -> List[str | int]:
        if k >= len(pool):
            # Keep deterministic ordering when full pool is returned.
            return list(pool)
        selected_idx = rng.choice(len(pool), size=k, replace=False)
        return [pool[int(i)] for i in selected_idx.tolist()]

    def _sample_component_aware_negatives(
        self,
        anchor_id: str | int,
        neg_pool: Sequence[str | int],
        rng: np.random.Generator,
    ) -> List[str | int]:
        cfg = self.neg_sampling_cfg
        anchor_cc = self.graph_index.component_id(anchor_id)

        same_pool = [nid for nid in neg_pool if self.graph_index.component_id(nid) == anchor_cc]
        cross_pool = [nid for nid in neg_pool if self.graph_index.component_id(nid) != anchor_cc]

        same_target = math.floor(self.relation_cfg.k_neg * cfg.same_cc_target_ratio)
        cross_target = self.relation_cfg.k_neg - same_target

        same_selected = self._sample_without_replacement(same_pool, min(same_target, len(same_pool)), rng)
        cross_selected = self._sample_without_replacement(cross_pool, min(cross_target, len(cross_pool)), rng)

        selected = list(same_selected) + list(cross_selected)
        shortfall = self.relation_cfg.k_neg - len(selected)

        if shortfall > 0:
            if cfg.fallback != "fill_from_other_pool":
                raise ValueError("Unsupported fallback policy")

            same_remaining = [nid for nid in same_pool if nid not in same_selected]
            cross_remaining = [nid for nid in cross_pool if nid not in cross_selected]

            if len(same_selected) < same_target:
                fallback_pool = cross_remaining
            else:
                fallback_pool = same_remaining

            add_count = min(shortfall, len(fallback_pool))
            if add_count > 0:
                selected.extend(self._sample_without_replacement(fallback_pool, add_count, rng))
                shortfall = self.relation_cfg.k_neg - len(selected)

            if shortfall > 0:
                other_pool = same_remaining if fallback_pool is cross_remaining else cross_remaining
                add_count = min(shortfall, len(other_pool))
                if add_count > 0:
                    selected.extend(self._sample_without_replacement(other_pool, add_count, rng))
                    shortfall = self.relation_cfg.k_neg - len(selected)

        if shortfall > 0 and cfg.strict:
            raise ValueError(
                f"Negative pool shortage for anchor {anchor_id}: required={self.relation_cfg.k_neg}, available={len(selected)}"
            )

        return selected

    def _anchor_seed(self, anchor_id: str | int) -> int:
        return stable_hash_seed(self.base_seed, self.epoch, anchor_id)


class DenseMatrixSimilarityProvider:
    """Simple in-memory similarity provider based on a dense matrix."""

    def __init__(self, node_ids: Sequence[str | int], similarity_matrix: np.ndarray) -> None:
        node_ids_arr = np.asarray(node_ids, dtype=object)
        if node_ids_arr.ndim != 1:
            raise ValueError("node_ids must be 1-dimensional")
        if similarity_matrix.ndim != 2:
            raise ValueError("similarity_matrix must be 2-dimensional")
        if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
            raise ValueError("similarity_matrix must be square")
        if similarity_matrix.shape[0] != node_ids_arr.shape[0]:
            raise ValueError("similarity_matrix size must match node_ids length")

        self.node_ids = node_ids_arr
        self.matrix = similarity_matrix.astype(np.float64, copy=False)
        self.id_to_idx = {nid: idx for idx, nid in enumerate(self.node_ids.tolist())}

    def get_candidates_with_scores(self, node_id: str | int) -> Tuple[np.ndarray, np.ndarray]:
        idx = self.id_to_idx[node_id]
        return self.node_ids.copy(), self.matrix[idx].copy()


def stable_hash_seed(*values: object) -> int:
    """Build a stable 32-bit seed from arbitrary values."""
    payload = "|".join(str(v) for v in values).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**32)


def anchor_relation_collate(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Build row-wise relation masks for a batch of sampled anchors."""
    if len(batch) == 0:
        raise ValueError("Batch must not be empty")

    anchor_node_ids = [item["anchor_node_id"] for item in batch]

    candidate_ids: List[str | int] = []
    seen = set()

    for anchor_id in anchor_node_ids:
        if anchor_id not in seen:
            seen.add(anchor_id)
            candidate_ids.append(anchor_id)

    for item in batch:
        for key in ("pos_node_ids", "neg_node_ids"):
            for node_id in item[key]:
                if node_id not in seen:
                    seen.add(node_id)
                    candidate_ids.append(node_id)

    idx_map = {node_id: idx for idx, node_id in enumerate(candidate_ids)}
    batch_size = len(batch)
    num_candidates = len(candidate_ids)

    pos_mask = torch.zeros((batch_size, num_candidates), dtype=torch.bool)
    neg_mask = torch.zeros((batch_size, num_candidates), dtype=torch.bool)
    ignore_mask = torch.zeros((batch_size, num_candidates), dtype=torch.bool)
    neg_weight = torch.zeros((batch_size, num_candidates), dtype=torch.float)

    for row, item in enumerate(batch):
        anchor_id = item["anchor_node_id"]
        anchor_idx = idx_map[anchor_id]

        pos_indices = [idx_map[nid] for nid in item["pos_node_ids"]]
        neg_indices = [idx_map[nid] for nid in item["neg_node_ids"]]

        if pos_indices:
            pos_mask[row, pos_indices] = True
        if neg_indices:
            neg_mask[row, neg_indices] = True

        for col_idx, hardness in zip(neg_indices, item["neg_hardness"]):
            neg_weight[row, col_idx] = float(hardness)

        ignore_mask[row, :] = True
        ignore_mask[row, anchor_idx] = False
        if pos_indices:
            ignore_mask[row, pos_indices] = False
        if neg_indices:
            ignore_mask[row, neg_indices] = False

    return {
        "anchor_node_ids": anchor_node_ids,
        "batch_candidate_node_ids": candidate_ids,
        "pos_mask": pos_mask,
        "neg_mask": neg_mask,
        "ignore_mask": ignore_mask,
        "neg_weight": neg_weight,
    }
