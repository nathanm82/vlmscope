"""Visual question answering task."""

from __future__ import annotations

from vlmscope.tasks.base import GenerationTask


class VQATask(GenerationTask):
    """Scores short answers with VQA soft accuracy and exact match."""

    name = "vqa"
    default_metrics = ("vqa_accuracy", "exact_match")
