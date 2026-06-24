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
