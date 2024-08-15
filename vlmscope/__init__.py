"""vlmscope: an evaluation harness for vision-language models.

The public surface is intentionally small::

    import vlmscope
    result = vlmscope.evaluate("vqa", dataset, model=my_model)
    print(vlmscope.render(result, "table"))
"""

from __future__ import annotations

from vlmscope.__about__ import __version__
from vlmscope.config import RunConfig
from vlmscope.metrics import metric_registry
from vlmscope.report import render
from vlmscope.runner import evaluate
from vlmscope.tasks import (
    CaptioningTask,
    RetrievalTask,
    Task,
    VQATask,
    task_registry,
)
from vlmscope.types import EvalResult, Prediction, Sample

__all__ = [
    "CaptioningTask",
    "EvalResult",
    "Prediction",
    "RetrievalTask",
    "RunConfig",
    "Sample",
    "Task",
    "VQATask",
    "__version__",
    "evaluate",
    "metric_registry",
    "render",
    "task_registry",
]
