from __future__ import annotations

import numpy as np

from vlmscope.models.base import CaptionModel, EmbeddingModel, VQAModel
from vlmscope.models.dummy import (
    HashingEmbeddingModel,
    LookupCaptionModel,
    LookupVQAModel,
)


def test_lookup_vqa_default_and_hit() -> None:
    model = LookupVQAModel({"img1": "cat"}, default="no")
    assert model.answer("img1", "what is it?") == "cat"
    assert model.answer("imgX", "what is it?") == "no"


def test_lookup_caption() -> None:
    model = LookupCaptionModel({"img1": "a dog"})
    assert model.caption("img1") == "a dog"
    assert model.caption("imgX") == "a photo"


def test_dummies_satisfy_protocols() -> None:
    assert isinstance(LookupVQAModel(), VQAModel)
    assert isinstance(LookupCaptionModel(), CaptionModel)
    assert isinstance(HashingEmbeddingModel(), EmbeddingModel)


def test_hashing_embeddings_are_deterministic() -> None:
    model = HashingEmbeddingModel(dim=16)
    a = model.encode_texts(["a red car"])
    b = model.encode_texts(["a red car"])
    assert np.allclose(a, b)
    assert a.shape == (1, 16)


def test_shared_words_increase_similarity() -> None:
    model = HashingEmbeddingModel(dim=128)
    base = model.encode_texts(["a red car"])[0]
    close = model.encode_texts(["a red car on a road"])[0]
    far = model.encode_texts(["blue sky over mountains"])[0]

    def cos(x: np.ndarray, y: np.ndarray) -> float:
        return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))

    assert cos(base, close) > cos(base, far)
