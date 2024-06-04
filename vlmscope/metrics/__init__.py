"""Generation-metric registry.

A *generation metric* maps aligned hypotheses and per-hypothesis references to
a single scalar (an accuracy in ``[0, 1]`` or a BLEU/CIDEr-style score):

    metric(hypotheses, references) -> float

where ``references[i]`` is the list of acceptable strings for ``hypotheses[i]``.
Retrieval metrics have a different shape (they consume a similarity matrix) and
live in :mod:`vlmscope.metrics.retrieval`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

from vlmscope.registry import Registry

GenerationMetric = Callable[[Sequence[str], Sequence[Sequence[str]]], float]

metric_registry: Registry[GenerationMetric] = Registry("metric")

from vlmscope.metrics.bleu import corpus_bleu  # noqa: E402
from vlmscope.metrics.cider import cider  # noqa: E402
from vlmscope.metrics.rouge import rouge_l  # noqa: E402
from vlmscope.metrics.vqa import exact_match, vqa_accuracy  # noqa: E402

metric_registry.register("vqa_accuracy", vqa_accuracy)
metric_registry.register("exact_match", exact_match)
metric_registry.register("bleu", corpus_bleu)
metric_registry.register("rouge_l", rouge_l)
metric_registry.register("cider", cider)

__all__ = [
    "GenerationMetric",
    "metric_registry",
    "vqa_accuracy",
    "exact_match",
    "corpus_bleu",
    "rouge_l",
    "cider",
]
