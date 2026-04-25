"""In-batch contrastive learning pipeline with curriculum-based sampling.

DataLoader expands each anchor with N_pos + N_neg candidates.
Loss function dynamically builds B_total x B_total dense positive/negative
masks using cluster IDs and pre-computed similarity thresholds.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Protocol, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

class SimilarityProvider(Protocol):
    """Protocol for candidate similarity lookup."""

    def get_candidates_with_scores(
        self, node_id: str | int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return aligned candidate IDs and similarity scores for one anchor."""

    def pairwise_similarity(
        self, node_id_a: str | int, node_id_b: str | int
    ) -> float:
        """Return the pre-computed similarity between two nodes."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CurriculumConfig:
    """Configuration for curriculum-based in-batch contrastive learning."""

    # Fixed sampling budget per anchor
    n_pos: int = 10
    n_neg: int = 10

    # Curriculum transition
    warmup_epochs: int = 10

    # Pool sizing for pre-computation
    top_k_size: int = 100
    bottom_k_size: int = 100
    hard_percentile: float = 10.0  # top/bottom X% for hard pairs

    # --- Warm-up stage ratios (within N_pos / N_neg) ---
    warmup_pos_self_ratio: float = 0.70
    warmup_pos_same_soft_ratio: float = 0.30
    # warmup negatives: 100% cross-cluster soft
    warmup_neg_cross_soft_ratio: float = 1.00

    # --- Post warm-up stage ratios ---
    post_pos_self_ratio: float = 0.30
    post_pos_same_soft_ratio: float = 0.50
    post_pos_cross_hard_ratio: float = 0.20

    post_neg_cross_soft_ratio: float = 0.85
    post_neg_same_hard_ratio: float = 0.15

    # --- Similarity thresholds for dense mask construction ---
    soft_pos_threshold: float = 0.7   # same-cluster positive if sim >= this
    hard_pos_threshold: float = 0.8   # cross-cluster positive if sim >= this
    soft_neg_threshold: float = 0.3   # cross-cluster negative if sim <= this
    hard_neg_threshold: float = 0.2   # same-cluster negative if sim <= this

    invalid_anchor_behavior: str = "drop"


# ---------------------------------------------------------------------------
# Graph Index (unchanged)
# ---------------------------------------------------------------------------

class GraphIndex:
    """Global node/component index from connected component graphs."""

    def __init__(self, graphs: Sequence[nx.Graph]) -> None:
        if len(graphs) == 0:
            raise ValueError(
                "graphs must contain at least one connected component graph"
            )

        self.graphs: List[nx.Graph] = list(graphs)
        self.node_to_component: Dict[str | int, int] = {}
        self.node_to_graph_node: Dict[str | int, object] = {}
        self.anchor_node_ids: List[str | int] = []

        for component_id, graph in enumerate(self.graphs):
            for graph_node, attrs in graph.nodes(data=True):
                if "node_id" not in attrs:
                    raise ValueError(
                        "Every node must contain a globally unique 'node_id' attribute"
                    )
                node_id = attrs["node_id"]
                if node_id in self.node_to_component:
                    raise ValueError(
                        f"Duplicate node_id detected across components: {node_id}"
                    )
                self.node_to_component[node_id] = component_id
                self.node_to_graph_node[node_id] = graph_node
                self.anchor_node_ids.append(node_id)

    def component_id(self, node_id: str | int) -> int:
        return self.node_to_component[node_id]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AnchorBatchDataset(Dataset):
    """Anchor-centric dataset that expands each anchor into a targeted pool.

    Each ``__getitem__`` returns:
      - ``anchor_id``
      - ``cluster_id``: component ID of the anchor
      - ``candidate_node_ids``: list of (1 + N_pos + N_neg) node IDs
      - ``candidate_cluster_ids``: cluster ID for each candidate
      - ``candidate_similarities``: similarity of each candidate to anchor
      - ``is_self_positive``: bool list marking self-positive entries
    """

    def __init__(
        self,
        graphs: Sequence[nx.Graph],
        similarity_provider: SimilarityProvider,
        curriculum_cfg: CurriculumConfig | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.graph_index = GraphIndex(graphs)
        self.similarity_provider = similarity_provider
        self.cfg = curriculum_cfg or CurriculumConfig()
        self.base_seed = int(seed)
        self.epoch = 0

        self.pool_cache: Dict[str | int, Dict[str, object]] = {}
        self.eligible_anchor_ids: List[str | int] = []
        self.invalid_anchor_reasons: Dict[str | int, str] = {}

        self._build_pools()

    # ---- Pool construction (offline, once) ----

    def _build_pools(self) -> None:
        """Pre-compute the 4 candidate pools for each anchor node."""
        for anchor_id in self.graph_index.anchor_node_ids:
            candidate_ids, candidate_scores = self._fetch_candidates(anchor_id)
            candidate_ids, candidate_scores = self._sanitize_candidates(
                anchor_id, candidate_ids, candidate_scores
            )

            if candidate_ids.size == 0:
                self._register_invalid(anchor_id, "no_candidates")
                continue

            anchor_cc = self.graph_index.component_id(anchor_id)

            # Split by cluster membership
            same_mask = np.array(
                [self.graph_index.component_id(c) == anchor_cc for c in candidate_ids]
            )
            cross_mask = ~same_mask

            same_ids, same_scores = candidate_ids[same_mask], candidate_scores[same_mask]
            cross_ids, cross_scores = candidate_ids[cross_mask], candidate_scores[cross_mask]

            # 1) Same-cluster Top-K (soft positives)
            same_top_k = self._topk(same_ids, same_scores, self.cfg.top_k_size, descending=True)

            # 2) Cross-cluster Top-K → hard positives (top hard_percentile %)
            cross_top_k = self._topk(cross_ids, cross_scores, self.cfg.top_k_size, descending=True)
            k_hard_pos = max(1, int(len(cross_top_k) * (self.cfg.hard_percentile / 100.0)))
            cross_top_k_hard = cross_top_k[:k_hard_pos]

            # 3) Cross-cluster Bottom-K (soft negatives)
            cross_bottom_k = self._topk(cross_ids, cross_scores, self.cfg.bottom_k_size, descending=False)

            # 4) Same-cluster Bottom-K → hard negatives (bottom hard_percentile %)
            same_bottom_k = self._topk(same_ids, same_scores, self.cfg.bottom_k_size, descending=False)
            k_hard_neg = max(1, int(len(same_bottom_k) * (self.cfg.hard_percentile / 100.0)))
            same_bottom_k_hard = same_bottom_k[:k_hard_neg]

            # Validity check: need at least some positives and negatives
            if len(same_top_k) == 0 and len(cross_top_k_hard) == 0:
                self._register_invalid(anchor_id, "zero_positive_pool")
                continue
            if len(cross_bottom_k) == 0:
                self._register_invalid(anchor_id, "zero_negative_pool")
                continue

            self.pool_cache[anchor_id] = {
                "component_id": anchor_cc,
                "same_cluster_top_k": same_top_k,
                "cross_cluster_top_k_hard": cross_top_k_hard,
                "cross_cluster_bottom_k": cross_bottom_k,
                "same_cluster_bottom_k_hard": same_bottom_k_hard,
            }
            self.eligible_anchor_ids.append(anchor_id)

        if len(self.eligible_anchor_ids) == 0:
            raise ValueError("No eligible anchors remain after pool construction")

        LOGGER.info(
            "AnchorBatchDataset: %d / %d anchors eligible",
            len(self.eligible_anchor_ids),
            len(self.graph_index.anchor_node_ids),
        )

    def _topk(
        self,
        ids: np.ndarray,
        scores: np.ndarray,
        k: int,
        descending: bool,
    ) -> List[str | int]:
        if ids.size == 0:
            return []
        order = np.lexsort(
            (ids.astype(str), -scores if descending else scores)
        )
        return ids[order[: min(k, len(order))]].tolist()

    # ---- Candidate fetching / sanitization ----

    def _fetch_candidates(
        self, anchor_id: str | int
    ) -> Tuple[np.ndarray, np.ndarray]:
        payload = self.similarity_provider.get_candidates_with_scores(anchor_id)
        if not isinstance(payload, tuple) or len(payload) != 2:
            raise TypeError(
                "SimilarityProvider must return Tuple[np.ndarray, np.ndarray]"
            )
        cids, cscores = payload
        if not isinstance(cids, np.ndarray) or not isinstance(cscores, np.ndarray):
            raise TypeError("SimilarityProvider output must be numpy arrays")
        if cids.ndim != 1 or cscores.ndim != 1:
            raise TypeError("SimilarityProvider output arrays must be 1-D")
        if cids.shape[0] != cscores.shape[0]:
            raise ValueError("Candidate IDs and scores must have the same length")
        return cids, cscores.astype(np.float64, copy=False)

    def _sanitize_candidates(
        self,
        anchor_id: str | int,
        candidate_ids: np.ndarray,
        candidate_scores: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        valid_ids: List[str | int] = []
        valid_scores: List[float] = []
        seen: set = set()

        for cid, score in zip(candidate_ids.tolist(), candidate_scores.tolist()):
            if cid == anchor_id:
                continue
            if cid in seen:
                raise ValueError(
                    f"Duplicate candidate node_id for anchor {anchor_id}: {cid}"
                )
            if cid not in self.graph_index.node_to_component:
                raise ValueError(
                    f"Unknown candidate node_id from SimilarityProvider: {cid}"
                )
            seen.add(cid)
            valid_ids.append(cid)
            valid_scores.append(float(score))

        return (
            np.asarray(valid_ids, dtype=object),
            np.asarray(valid_scores, dtype=np.float64),
        )

    def _register_invalid(self, anchor_id: str | int, reason: str) -> None:
        if self.cfg.invalid_anchor_behavior == "raise":
            raise ValueError(f"Invalid anchor {anchor_id}: {reason}")
        self.invalid_anchor_reasons[anchor_id] = reason

    # ---- Epoch & length ----

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for curriculum stage transitions and seed variation."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.eligible_anchor_ids)

    # ---- Sampling helpers ----

    def _sample_exact(
        self,
        pool: List[str | int],
        k: int,
        rng: np.random.Generator,
    ) -> List[str | int]:
        """Sample exactly k items from pool (with replacement if needed)."""
        if k == 0 or len(pool) == 0:
            return []
        replace = len(pool) < k
        idx = rng.choice(len(pool), size=k, replace=replace)
        return [pool[int(i)] for i in idx]

    # ---- Core: __getitem__ ----

    def __getitem__(self, index: int) -> Dict[str, object]:
        anchor_id = self.eligible_anchor_ids[index]
        pools = self.pool_cache[anchor_id]
        anchor_cc = pools["component_id"]
        rng = np.random.default_rng(self._anchor_seed(anchor_id))

        cfg = self.cfg
        n_pos = cfg.n_pos
        n_neg = cfg.n_neg

        # --- Determine counts based on curriculum stage ---
        if self.epoch < cfg.warmup_epochs:
            n_self = int(n_pos * cfg.warmup_pos_self_ratio)
            n_same_soft = n_pos - n_self
            n_cross_hard_pos = 0

            n_cross_soft_neg = int(n_neg * cfg.warmup_neg_cross_soft_ratio)
            n_same_hard_neg = n_neg - n_cross_soft_neg
        else:
            n_self = int(n_pos * cfg.post_pos_self_ratio)
            n_cross_hard_pos = int(n_pos * cfg.post_pos_cross_hard_ratio)
            n_same_soft = n_pos - n_self - n_cross_hard_pos

            n_same_hard_neg = int(n_neg * cfg.post_neg_same_hard_ratio)
            n_cross_soft_neg = n_neg - n_same_hard_neg

        # --- Assemble positives ---
        pos_ids: List[str | int] = []
        is_self: List[bool] = []

        # Self-positives (anchor itself, to be augmented downstream)
        pos_ids.extend([anchor_id] * n_self)
        is_self.extend([True] * n_self)

        # Same-cluster soft positives
        sampled = self._sample_exact(pools["same_cluster_top_k"], n_same_soft, rng)
        pos_ids.extend(sampled)
        is_self.extend([False] * len(sampled))

        # Cross-cluster hard positives (post warm-up only)
        if n_cross_hard_pos > 0:
            pool = pools["cross_cluster_top_k_hard"]
            if len(pool) > 0:
                sampled = self._sample_exact(pool, n_cross_hard_pos, rng)
            else:
                # Fallback to same-cluster soft
                sampled = self._sample_exact(pools["same_cluster_top_k"], n_cross_hard_pos, rng)
            pos_ids.extend(sampled)
            is_self.extend([False] * len(sampled))

        # Pad positives to budget if needed
        while len(pos_ids) < n_pos:
            pos_ids.append(anchor_id)
            is_self.append(True)

        # --- Assemble negatives ---
        neg_ids: List[str | int] = []

        # Cross-cluster soft negatives
        sampled = self._sample_exact(pools["cross_cluster_bottom_k"], n_cross_soft_neg, rng)
        neg_ids.extend(sampled)

        # Same-cluster hard negatives (post warm-up only)
        if n_same_hard_neg > 0:
            pool = pools["same_cluster_bottom_k_hard"]
            if len(pool) > 0:
                sampled = self._sample_exact(pool, n_same_hard_neg, rng)
            else:
                sampled = self._sample_exact(pools["cross_cluster_bottom_k"], n_same_hard_neg, rng)
            neg_ids.extend(sampled)

        # Pad negatives to budget if needed
        while len(neg_ids) < n_neg:
            if pools["cross_cluster_bottom_k"]:
                neg_ids.append(rng.choice(pools["cross_cluster_bottom_k"]))
            else:
                break

        # --- Build candidate list: anchor + positives + negatives ---
        candidate_ids: List[str | int] = [anchor_id] + pos_ids + neg_ids
        candidate_is_self: List[bool] = [False] + is_self + [False] * len(neg_ids)

        # Look up cluster IDs and similarities to anchor
        candidate_cluster_ids: List[int] = []
        candidate_similarities: List[float] = []
        for cid in candidate_ids:
            candidate_cluster_ids.append(self.graph_index.component_id(cid))
            if cid == anchor_id:
                candidate_similarities.append(1.0)  # self-similarity
            else:
                candidate_similarities.append(
                    float(self.similarity_provider.pairwise_similarity(anchor_id, cid))
                )

        return {
            "anchor_id": anchor_id,
            "cluster_id": anchor_cc,
            "candidate_node_ids": candidate_ids,
            "candidate_cluster_ids": candidate_cluster_ids,
            "candidate_similarities": candidate_similarities,
            "is_self_positive": candidate_is_self,
        }

    def _anchor_seed(self, anchor_id: str | int) -> int:
        return stable_hash_seed(self.base_seed, self.epoch, anchor_id)


# ---------------------------------------------------------------------------
# Expanded Batch Collator
# ---------------------------------------------------------------------------

class ExpandedBatchCollator:
    """Collate function that flattens B anchor items into one expanded batch.

    Requires a reference to the SimilarityProvider to build the pairwise
    similarity matrix across the entire expanded batch.

    Returns a dict with:
      - ``all_node_ids``: deduplicated list of node IDs (length B_total)
      - ``cluster_ids``: LongTensor (B_total,)
      - ``similarity_matrix``: FloatTensor (B_total, B_total)
      - ``anchor_indices``: LongTensor (B,) — position of each anchor
      - ``is_self_positive``: BoolTensor (B_total,)
    """

    def __init__(self, similarity_provider: SimilarityProvider) -> None:
        self.similarity_provider = similarity_provider

    def __call__(self, batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
        if len(batch) == 0:
            raise ValueError("Batch must not be empty")

        # --- Flatten & deduplicate node IDs across all items ---
        all_node_ids: List[str | int] = []
        id_to_idx: Dict[str | int, int] = {}
        # Track self-positive status per global index
        self_positive_flags: Dict[int, bool] = {}

        anchor_indices: List[int] = []

        for item in batch:
            for local_pos, (nid, is_self) in enumerate(
                zip(item["candidate_node_ids"], item["is_self_positive"])
            ):
                if nid not in id_to_idx:
                    idx = len(all_node_ids)
                    id_to_idx[nid] = idx
                    all_node_ids.append(nid)
                    self_positive_flags[idx] = bool(is_self)
                else:
                    # If any item marks this node as self-positive, keep that
                    if is_self:
                        self_positive_flags[id_to_idx[nid]] = True

            anchor_indices.append(id_to_idx[item["anchor_id"]])

        b_total = len(all_node_ids)

        # --- Cluster IDs ---
        cluster_ids_list: List[int] = [
            item["candidate_cluster_ids"][
                item["candidate_node_ids"].index(nid)
            ]
            if nid in item["candidate_node_ids"]
            else 0
            for item in batch
            for nid in [all_node_ids[0]]  # placeholder; overridden below
        ]
        # Actually build from per-item data already available
        # More efficient: use the first item that mentions each node
        cluster_ids_map: Dict[str | int, int] = {}
        for item in batch:
            for nid, cid in zip(
                item["candidate_node_ids"], item["candidate_cluster_ids"]
            ):
                if nid not in cluster_ids_map:
                    cluster_ids_map[nid] = cid

        cluster_ids = torch.zeros(b_total, dtype=torch.long)
        for nid, idx in id_to_idx.items():
            cluster_ids[idx] = cluster_ids_map[nid]

        # --- Pairwise similarity matrix ---
        sim_matrix = torch.zeros(b_total, b_total, dtype=torch.float)
        for i in range(b_total):
            sim_matrix[i, i] = 1.0  # self-similarity
            for j in range(i + 1, b_total):
                s = float(
                    self.similarity_provider.pairwise_similarity(
                        all_node_ids[i], all_node_ids[j]
                    )
                )
                sim_matrix[i, j] = s
                sim_matrix[j, i] = s

        # --- Self-positive mask ---
        is_self_positive = torch.zeros(b_total, dtype=torch.bool)
        for idx, flag in self_positive_flags.items():
            is_self_positive[idx] = flag

        return {
            "all_node_ids": all_node_ids,
            "cluster_ids": cluster_ids,
            "similarity_matrix": sim_matrix,
            "anchor_indices": torch.tensor(anchor_indices, dtype=torch.long),
            "is_self_positive": is_self_positive,
        }


# ---------------------------------------------------------------------------
# Dense Mask Construction (for loss function)
# ---------------------------------------------------------------------------

def build_contrastive_masks(
    cluster_ids: torch.Tensor,
    similarity_matrix: torch.Tensor,
    cfg: CurriculumConfig,
) -> Dict[str, torch.Tensor]:
    """Build dense B_total x B_total contrastive masks.

    Args:
        cluster_ids: LongTensor of shape (B_total,)
        similarity_matrix: FloatTensor of shape (B_total, B_total)
        cfg: CurriculumConfig with threshold values

    Returns:
        Dict with keys ``pos_mask``, ``neg_mask``, ``ignore_mask``.
        All are BoolTensors of shape (B_total, B_total).
    """
    b = cluster_ids.size(0)

    # Pairwise cluster comparison
    same_cluster = cluster_ids.unsqueeze(0) == cluster_ids.unsqueeze(1)  # (B, B)
    diff_cluster = ~same_cluster

    # --- Positive mask ---
    # Self-positive: diagonal (same node)
    self_pos = torch.eye(b, dtype=torch.bool, device=cluster_ids.device)

    # Same cluster AND sim >= soft_pos_threshold
    same_cluster_pos = same_cluster & (similarity_matrix >= cfg.soft_pos_threshold)

    # Different cluster AND sim >= hard_pos_threshold
    cross_cluster_pos = diff_cluster & (similarity_matrix >= cfg.hard_pos_threshold)

    pos_mask = self_pos | same_cluster_pos | cross_cluster_pos

    # --- Negative mask ---
    # Different cluster AND sim <= soft_neg_threshold
    cross_cluster_neg = diff_cluster & (similarity_matrix <= cfg.soft_neg_threshold)

    # Same cluster AND sim <= hard_neg_threshold
    same_cluster_neg = same_cluster & (similarity_matrix <= cfg.hard_neg_threshold)

    neg_mask = cross_cluster_neg | same_cluster_neg

    # --- Ensure no overlap; negative takes precedence if both fire ---
    pos_mask = pos_mask & ~neg_mask

    # --- Ignore mask: everything that is neither positive nor negative ---
    # Also ignore diagonal (self) for standard contrastive losses
    ignore_mask = ~(pos_mask | neg_mask)

    # Remove diagonal from pos/neg (handled separately by most losses)
    diag = torch.eye(b, dtype=torch.bool, device=cluster_ids.device)
    pos_mask = pos_mask & ~diag
    neg_mask = neg_mask & ~diag
    ignore_mask = ignore_mask | diag

    return {
        "pos_mask": pos_mask,
        "neg_mask": neg_mask,
        "ignore_mask": ignore_mask,
    }


# ---------------------------------------------------------------------------
# Similarity Providers
# ---------------------------------------------------------------------------

class DenseMatrixSimilarityProvider:
    """Simple in-memory similarity provider based on a dense matrix."""

    def __init__(
        self,
        node_ids: Sequence[str | int],
        similarity_matrix: np.ndarray,
    ) -> None:
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
        self.id_to_idx: Dict[str | int, int] = {
            nid: idx for idx, nid in enumerate(self.node_ids.tolist())
        }

    def get_candidates_with_scores(
        self, node_id: str | int
    ) -> Tuple[np.ndarray, np.ndarray]:
        idx = self.id_to_idx[node_id]
        return self.node_ids.copy(), self.matrix[idx].copy()

    def pairwise_similarity(
        self, node_id_a: str | int, node_id_b: str | int
    ) -> float:
        idx_a = self.id_to_idx[node_id_a]
        idx_b = self.id_to_idx[node_id_b]
        return float(self.matrix[idx_a, idx_b])


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def stable_hash_seed(*values: object) -> int:
    """Build a stable 32-bit seed from arbitrary values."""
    payload = "|".join(str(v) for v in values).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**32)
