from __future__ import annotations

import numpy as np
import pytest
from vlmscope.tasks.retrieval import RetrievalTask


def test_perfect_diagonal_retrieval() -> None:
    emb = np.eye(4)
    res = RetrievalTask(ks=(1, 2)).evaluate(emb, emb)
    assert res.task == "retrieval"
    assert res.num_samples == 4
    assert res.metrics["recall@1"] == 1.0
    assert res.metrics["median_rank"] == 1.0


def test_direction_is_recorded() -> None:
    emb = np.eye(3)
    res = RetrievalTask(ks=(1,)).evaluate(emb, emb, direction="image_to_text")
    assert res.extra["direction"] == "image_to_text"


def test_invalid_direction_raises() -> None:
    with pytest.raises(ValueError):
        RetrievalTask().evaluate(np.eye(2), np.eye(2), direction="bogus")


def test_metric_names() -> None:
    names = RetrievalTask(ks=(1, 5)).metric_names()
    assert names == ("recall@1", "recall@5", "median_rank", "mean_rank")
