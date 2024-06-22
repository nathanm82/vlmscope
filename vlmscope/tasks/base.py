"""Task base classes.

``Task`` is the common marker (name + metric list). ``GenerationTask`` covers
VQA and captioning, which share the exact same flow: align predictions to
samples by ``uid``, then apply each registered generation metric to the
hypotheses and references. Retrieval is different enough to live on its own
(see :mod:`vlmscope.tasks.retrieval`).
"""

from __future__ import annotations

import abc
from collections.abc import Sequence

from vlmscope.metrics import metric_registry
from vlmscope.types import EvalResult, Prediction, Sample


class Task(abc.ABC):
    """Common base: a name and the metrics the task reports."""

    name: str = ""
    default_metrics: tuple[str, ...] = ()

    def metric_names(self) -> tuple[str, ...]:
        return self.default_metrics


class GenerationTask(Task):
    """Compares generated text against references using generation metrics."""

    def __init__(
        self, name: str | None = None, metrics: Sequence[str] | None = None
    ) -> None:
        if name is not None:
            self.name = name
        self._metrics = tuple(metrics) if metrics is not None else self.default_metrics

    def metric_names(self) -> tuple[str, ...]:
        return self._metrics

    @staticmethod
    def _align(
        samples: Sequence[Sample], predictions: Sequence[Prediction]
    ) -> list[Prediction]:
        by_uid = {p.uid: p for p in predictions}
        missing = [s.uid for s in samples if s.uid not in by_uid]
        if missing:
            raise ValueError(
                f"missing predictions for {len(missing)} sample(s); "
                f"first few: {missing[:3]}"
            )
        return [by_uid[s.uid] for s in samples]

    def evaluate(
        self, samples: Sequence[Sample], predictions: Sequence[Prediction]
    ) -> EvalResult:
        ordered = self._align(samples, predictions)
        hypotheses = [p.text or "" for p in ordered]
        references = [list(s.references) for s in samples]
        result = EvalResult(task=self.name, num_samples=len(samples))
        for name in self._metrics:
            metric = metric_registry.get(name)
            result.metrics[name] = metric(hypotheses, references)
        return result
