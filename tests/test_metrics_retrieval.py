from __future__ import annotations

import numpy as np
from vlmscope.metrics.retrieval import (
    cosine_similarity,
    mean_rank,
    median_rank,
    ranks,
    recall_at_k,
    retrieval_report,
)


def test_cosine_identity() -> None:
    x = np.array([[1.0, 0.0], [0.0, 1.0]])
    sim = cosine_similarity(x, x)
    assert np.allclose(np.diag(sim), [1.0, 1.0])


def test_recall_at_k_perfect() -> None:
    scores = np.eye(3)
    assert recall_at_k(scores, [0, 1, 2], 1) == 1.0


def test_recall_at_k_partial() -> None:
    scores = np.array([[0.9, 0.1, 0.0], [0.2, 0.1, 0.7]])
    assert recall_at_k(scores, [0, 0], 1) == 0.5
    assert recall_at_k(scores, [0, 0], 3) == 1.0


def test_recall_at_k_accepts_multiple_positives() -> None:
    scores = np.array([[0.1, 0.9, 0.0]])
    assert recall_at_k(scores, [{0, 1}], 1) == 1.0


def test_ranks_are_one_based() -> None:
    scores = np.array([[0.9, 0.1, 0.0], [0.2, 0.1, 0.7]])
    assert list(ranks(scores, [0, 2])) == [1, 1]
    assert list(ranks(scores, [2, 1])) == [3, 3]


def test_median_and_mean_rank() -> None:
    scores = np.array([[0.9, 0.1, 0.0], [0.2, 0.1, 0.7]])
    assert median_rank(scores, [0, 2]) == 1.0
    assert mean_rank(scores, [0, 2]) == 1.0


def test_retrieval_report_keys() -> None:
    rep = retrieval_report(np.eye(4), [0, 1, 2, 3], ks=(1, 2))
    assert rep["recall@1"] == 1.0
    assert "recall@2" in rep
    assert "median_rank" in rep
    assert "mean_rank" in rep


def test_empty_inputs_do_not_crash() -> None:
    empty = np.zeros((0, 0))
    assert recall_at_k(empty, [], 1) == 0.0
    assert median_rank(empty, []) == 0.0
    assert mean_rank(empty, []) == 0.0
