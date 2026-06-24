"""Model adapter protocols.

vlmscope never loads a model itself -- callers wrap whatever they already have
(a HF pipeline, an API client, a CLIP encoder) in something that satisfies one
of these protocols. The protocols are intentionally tiny.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class VQAModel(Protocol):
    """Answers a question about an image, returning a short string."""

    def answer(self, image_id: str, question: str) -> str: ...
