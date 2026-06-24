"""Image-text retrieval task.

Unlike the generation tasks, retrieval works on dense embeddings: given image
and text embeddings (plus the ground-truth pairing) it builds a similarity
matrix and reports recall@k and rank statistics for one retrieval direction.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from vlmscope.metrics.retrieval import Positive, cosine_similarity, retrieval_report
from vlmscope.tasks.base import Task
from vlmscope.types import EvalResult

DIRECTIONS = ("text_to_image", "image_to_text")


class RetrievalTask(Task):
    """Recall@k / rank metrics over an embedding similarity matrix."""

    name = "retrieval"

    def __init__(self, ks: Sequence[int] = (1, 5, 10)) -> None:
        self.ks = tuple(ks)

    def metric_names(self) -> tuple[str, ...]:
        return tuple(f"recall@{k}" for k in self.ks) + ("median_rank", "mean_rank")

    def evaluate(
        self,
        image_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        positives: Sequence[Positive] | None = None,
        direction: str = "text_to_image",
    ) -> EvalResult:
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}")
        images = np.asarray(image_embeddings, dtype=np.float64)
        texts = np.asarray(text_embeddings, dtype=np.float64)

        if direction == "text_to_image":
            scores = cosine_similarity(texts, images)
            num_queries = texts.shape[0]
        else:
            scores = cosine_similarity(images, texts)
            num_queries = images.shape[0]

        if positives is None:
            positives = list(range(num_queries))

        report = retrieval_report(scores, positives, ks=self.ks)
        result = EvalResult(task=self.name, num_samples=num_queries)
        result.metrics.update(report)
        result.extra["direction"] = direction
        return result
