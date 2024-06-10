"""Deterministic dummy models.

These exist so the examples, tests and ``vlmscope run`` smoke paths work with
no downloads and no randomness between runs. They are not meant to be good --
just predictable.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from vlmscope.metrics.text import tokenize


class LookupVQAModel:
    """Answers from a fixed ``image_id -> answer`` table, else a default."""

    def __init__(self, answers: dict[str, str] | None = None, default: str = "yes"):
        self._answers = dict(answers or {})
        self._default = default

    def answer(self, image_id: str, question: str) -> str:
        return self._answers.get(image_id, self._default)


class LookupCaptionModel:
    """Captions from a fixed ``image_id -> caption`` table, else a default."""

    def __init__(self, captions: dict[str, str] | None = None, default: str = "a photo"):
        self._captions = dict(captions or {})
        self._default = default

    def caption(self, image_id: str) -> str:
        return self._captions.get(image_id, self._default)


class HashingEmbeddingModel:
    """A bag-of-words hashing encoder.

    Each token is hashed to a fixed unit vector; a string embeds to the mean of
    its token vectors. Strings that share words land close together, so an image
    described by similar words to its caption retrieves well -- enough to make
    retrieval smoke tests meaningful without a real encoder.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _token_vector(self, token: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim)
        return vec / (np.linalg.norm(vec) + 1e-12)

    def _embed(self, text: str) -> np.ndarray:
        tokens = tokenize(text)
        if not tokens:
            return np.zeros(self.dim)
        return np.mean([self._token_vector(t) for t in tokens], axis=0)

    def encode_images(self, image_ids: Sequence[str]) -> np.ndarray:
        return np.stack([self._embed(i) for i in image_ids])

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._embed(t) for t in texts])
