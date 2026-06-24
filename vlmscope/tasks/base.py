"""Base class for evaluation tasks."""

from __future__ import annotations

import abc
from collections.abc import Sequence

from vlmscope.types import EvalResult, Prediction, Sample


class Task(abc.ABC):
    """Turns aligned samples and predictions into an :class:`EvalResult`."""

    #: Short identifier used by the registry and CLI.
    name: str = ""

    @abc.abstractmethod
    def evaluate(
        self,
        samples: Sequence[Sample],
        predictions: Sequence[Prediction],
    ) -> EvalResult:
        """Compute metrics for this task. Subclasses must implement."""
        raise NotImplementedError
