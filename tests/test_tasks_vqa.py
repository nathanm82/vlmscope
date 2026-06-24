from __future__ import annotations

import pytest

from vlmscope.tasks.vqa import VQATask
from vlmscope.types import Prediction, Sample


def _data() -> tuple[list[Sample], list[Prediction]]:
    samples = [
        Sample("1", references=("cat", "cat", "kitten")),
        Sample("2", references=("two", "2")),
    ]
    preds = [Prediction("1", text="cat"), Prediction("2", text="two")]
    return samples, preds


def test_vqa_scores_are_reported() -> None:
    samples, preds = _data()
    res = VQATask().evaluate(samples, preds)
    assert res.task == "vqa"
    assert res.num_samples == 2
    assert res.metrics["exact_match"] == 1.0
    assert res.metrics["vqa_accuracy"] > 0.0


def test_predictions_align_by_uid_not_order() -> None:
    samples, preds = _data()
    res = VQATask().evaluate(samples, list(reversed(preds)))
    assert res.metrics["exact_match"] == 1.0


def test_missing_prediction_raises() -> None:
    samples, _ = _data()
    with pytest.raises(ValueError, match="missing predictions"):
        VQATask().evaluate(samples, [Prediction("1", text="cat")])


def test_custom_metric_subset() -> None:
    task = VQATask(metrics=("exact_match",))
    assert task.metric_names() == ("exact_match",)
    res = task.evaluate(*_data())
    assert set(res.metrics) == {"exact_match"}
