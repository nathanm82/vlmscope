"""Core data structures shared across tasks and metrics.

The containers here are deliberately generic so the same ``Sample`` works for
VQA, captioning and retrieval. Not every field is meaningful for every task --
e.g. ``question`` is unused for captioning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
    image_id: str | None = None
    question: str | None = None
    references: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Prediction:
    """A model output for a single sample.

    ``text`` holds a generated answer or caption; ``score`` is reserved for
    tasks that emit a scalar (e.g. a ranking score).
    """

    uid: str
    text: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricResult:
    """The value of a single metric plus optional per-sample detail."""

    name: str
    value: float
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Aggregated result of evaluating one task with one or more metrics."""

    task: str
    metrics: dict[str, float] = field(default_factory=dict)
    num_samples: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def add(self, result: MetricResult) -> None:
        """Record a metric value (overwrites any earlier value of same name)."""
        self.metrics[result.name] = result.value

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "num_samples": self.num_samples,
            "metrics": dict(self.metrics),
            "extra": dict(self.extra),
        }
