"""Core data structures shared across tasks and metrics.

The containers here are deliberately generic so the same ``Sample`` works for
VQA, captioning and retrieval. Not every field is meaningful for every task --
e.g. ``question`` is unused for captioning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class Sample:
    """A single evaluation example.

    Args:
        uid: Stable identifier for the example.
        image_id: Identifier of the associated image, when relevant.
        question: The question text (VQA only).
        references: Ground-truth answers or reference captions.
        metadata: Free-form extra fields carried through untouched.
    """

    uid: str
    image_id: Optional[str] = None
    question: Optional[str] = None
    references: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Prediction:
    """A model output for a single sample.

    ``text`` holds a generated answer or caption; ``score`` is reserved for
    tasks that emit a scalar (e.g. a ranking score).
    """

    uid: str
    text: Optional[str] = None
    score: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)
