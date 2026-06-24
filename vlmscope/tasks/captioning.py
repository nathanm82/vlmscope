"""Image captioning task."""

from __future__ import annotations

from vlmscope.tasks.base import GenerationTask


class CaptioningTask(GenerationTask):
    """Scores captions with BLEU, ROUGE-L and CIDEr."""

    name = "captioning"
    default_metrics = ("bleu", "rouge_l", "cider")
