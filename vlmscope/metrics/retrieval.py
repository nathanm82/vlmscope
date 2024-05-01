"""Image-text retrieval metrics.

These operate on a dense similarity matrix ``scores`` of shape
``(num_queries, num_candidates)`` together with the ground-truth positives for
each query. ``positives[i]`` is either a single candidate index or an iterable
of acceptable indices.
"""

from __future__ import annotations

import numpy as np


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
