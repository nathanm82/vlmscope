"""Aggregation helpers for metric values across samples or repeated runs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()) if arr.size else 0.0


def stddev(values: Sequence[float], ddof: int = 1) -> float:
    """Sample standard deviation; 0.0 when there are too few values."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= ddof:
        return 0.0
    return float(arr.std(ddof=ddof))


def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    num_resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = np.empty(num_resamples)
    for i in range(num_resamples):
        idx = rng.integers(0, arr.size, arr.size)
        means[i] = arr[idx].mean()
    lo = float(np.percentile(means, 100 * (1 - confidence) / 2))
    hi = float(np.percentile(means, 100 * (1 + confidence) / 2))
    return (lo, hi)


def summarize(values: Sequence[float]) -> dict[str, float]:
    """A compact mean/std/n summary, handy for reports."""
    return {
        "n": float(len(values)),
        "mean": mean(values),
        "std": stddev(values),
    }
