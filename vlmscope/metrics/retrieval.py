"""Image-text retrieval metrics.

These operate on a dense similarity matrix ``scores`` of shape
``(num_queries, num_candidates)`` together with the ground-truth positives for
each query. ``positives[i]`` is either a single candidate index or an iterable
of acceptable indices.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Union

import numpy as np

#: A query's ground truth: a single candidate index or a set of indices.
Positive = Union[int, Iterable[int]]


def _as_index_set(positive: Positive) -> set[int]:
    if isinstance(positive, (int, np.integer)):
        return {int(positive)}
    return {int(p) for p in positive}


def cosine_similarity(queries: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two 2-D arrays.

    Returns a ``(len(queries), len(candidates))`` matrix.
    """
    q = np.asarray(queries, dtype=np.float64)
    c = np.asarray(candidates, dtype=np.float64)
    if q.ndim != 2 or c.ndim != 2:
        raise ValueError("queries and candidates must both be 2-D arrays")
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    c = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12)
    return q @ c.T


def recall_at_k(
    scores: np.ndarray, positives: Sequence[Positive], k: int
) -> float:
    """Fraction of queries whose top-``k`` candidates include a positive."""
    scores = np.asarray(scores, dtype=np.float64)
    num_queries = scores.shape[0]
    k = min(k, scores.shape[1])
    topk = np.argsort(-scores, axis=1)[:, :k]
    hits = 0
    for i in range(num_queries):
        wanted = _as_index_set(positives[i])
        if wanted & set(topk[i].tolist()):
            hits += 1
    return hits / num_queries


def ranks(scores: np.ndarray, positives: Sequence[Positive]) -> np.ndarray:
    """1-based rank of the best-scoring positive for each query."""
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, axis=1)
    out = np.empty(scores.shape[0], dtype=np.int64)
    for i in range(scores.shape[0]):
        wanted = _as_index_set(positives[i])
        row = order[i].tolist()
        out[i] = min(row.index(p) for p in wanted) + 1
    return out


def median_rank(scores: np.ndarray, positives: Sequence[Positive]) -> float:
    return float(np.median(ranks(scores, positives)))


def mean_rank(scores: np.ndarray, positives: Sequence[Positive]) -> float:
    return float(np.mean(ranks(scores, positives)))


def retrieval_report(
    scores: np.ndarray,
    positives: Sequence[Positive],
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """Convenience bundle: recall at several ``k`` plus rank statistics."""
    report = {f"recall@{k}": recall_at_k(scores, positives, k) for k in ks}
    report["median_rank"] = median_rank(scores, positives)
    report["mean_rank"] = mean_rank(scores, positives)
    return report
