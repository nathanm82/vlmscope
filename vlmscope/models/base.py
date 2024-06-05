"""Model adapter protocols.

vlmscope never loads a model itself -- callers wrap whatever they already have
(a HF pipeline, an API client, a CLIP encoder) in something that satisfies one
of these protocols. Each protocol is intentionally tiny so adapters stay thin.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VQAModel(Protocol):
    """Answers a question about an image, returning a short string."""

    def answer(self, image_id: str, question: str) -> str: ...


@runtime_checkable
class CaptionModel(Protocol):
    """Produces a caption for an image."""

    def caption(self, image_id: str) -> str: ...


@runtime_checkable
class EmbeddingModel(Protocol):
    """Encodes images and texts into a shared embedding space.

    Both methods return a 2-D array of shape ``(len(inputs), dim)``.
    """

    def encode_images(self, image_ids: Sequence[str]) -> np.ndarray: ...

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...
