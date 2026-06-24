from __future__ import annotations

import pytest

from vlmscope.data.toy import toy_captions, toy_retrieval, toy_vqa
from vlmscope.models.dummy import (
    HashingEmbeddingModel,
    LookupCaptionModel,
    LookupVQAModel,
)
from vlmscope.runner import evaluate
from vlmscope.types import Prediction


def test_evaluate_vqa_with_predictions() -> None:
    ds = toy_vqa()
    preds = [Prediction(s.uid, text=s.references[0]) for s in ds]
    res = evaluate("vqa", ds, predictions=preds)
    assert res.task == "vqa"
    assert res.metrics["exact_match"] == 1.0


def test_evaluate_vqa_with_model() -> None:
    ds = toy_vqa()
    answers = {s.image_id or "": s.references[0] for s in ds}
    res = evaluate("vqa", ds, model=LookupVQAModel(answers))
    assert res.metrics["vqa_accuracy"] > 0.0


def test_evaluate_captioning_with_model() -> None:
    ds = toy_captions()
    captions = {s.image_id or "": s.references[0] for s in ds}
    res = evaluate("captioning", ds, model=LookupCaptionModel(captions))
    assert res.metrics["rouge_l"] > 0.9


def test_evaluate_retrieval_with_embeddings() -> None:
    res = evaluate("retrieval", toy_retrieval(), model=HashingEmbeddingModel(dim=128))
    assert res.metrics["recall@1"] >= 0.0
    assert "median_rank" in res.metrics


def test_requires_model_or_predictions() -> None:
    with pytest.raises(ValueError, match="model or predictions"):
        evaluate("vqa", toy_vqa())


def test_limit_truncates_samples() -> None:
    ds = toy_vqa()
    preds = [Prediction(s.uid, text="x") for s in ds]
    res = evaluate("vqa", ds, predictions=preds, limit=2)
    assert res.num_samples == 2
