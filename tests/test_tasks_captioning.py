from __future__ import annotations

from vlmscope.tasks.captioning import CaptioningTask
from vlmscope.types import Prediction, Sample


def test_captioning_reports_all_metrics() -> None:
    samples = [
        Sample("1", references=("a cat on a mat", "a cat on the mat")),
        Sample("2", references=("a dog in a park", "a dog at the park")),
    ]
    preds = [
        Prediction("1", text="a cat on a mat"),
        Prediction("2", text="a dog in a park"),
    ]
    res = CaptioningTask().evaluate(samples, preds)
    assert set(res.metrics) == {"bleu", "rouge_l", "cider"}
    assert res.metrics["rouge_l"] > 0.5
    assert res.metrics["bleu"] >= 0.0


def test_empty_prediction_lowers_scores() -> None:
    samples = [Sample("1", references=("a cat on a mat",))]
    good = CaptioningTask(metrics=("rouge_l",)).evaluate(
        samples, [Prediction("1", text="a cat on a mat")]
    )
    bad = CaptioningTask(metrics=("rouge_l",)).evaluate(
        samples, [Prediction("1", text="something else entirely")]
    )
    assert good.metrics["rouge_l"] > bad.metrics["rouge_l"]
