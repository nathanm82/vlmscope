from __future__ import annotations

from vlmscope.aggregate import bootstrap_ci, mean, stddev, summarize


def test_mean() -> None:
    assert mean([1.0, 2.0, 3.0]) == 2.0
    assert mean([]) == 0.0


def test_stddev_constant_is_zero() -> None:
    assert stddev([5.0, 5.0, 5.0]) == 0.0
    assert stddev([1.0]) == 0.0


def test_bootstrap_ci_brackets_the_mean() -> None:
    values = [0.0] * 50 + [1.0] * 50
    lo, hi = bootstrap_ci(values, seed=0)
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0


def test_bootstrap_ci_empty() -> None:
    assert bootstrap_ci([]) == (0.0, 0.0)


def test_summarize() -> None:
    s = summarize([1.0, 3.0])
    assert s["n"] == 2.0
    assert s["mean"] == 2.0
