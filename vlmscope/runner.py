"""High-level evaluation entry points.

``evaluate`` is the one function most users call: point it at a task, a dataset
and either a model (to generate predictions) or a set of precomputed
predictions, and get back an :class:`~vlmscope.types.EvalResult`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from vlmscope.tasks import Task, task_registry
from vlmscope.tasks.retrieval import RetrievalTask
from vlmscope.types import EvalResult, Prediction, Sample


def _resolve_task(task: Task | str) -> Task:
    return task if isinstance(task, Task) else task_registry.get(task)


def _generate(task: Task, samples: Sequence[Sample], model: Any) -> list[Prediction]:
    predictions: list[Prediction] = []
    for sample in samples:
        if task.name == "vqa":
            text = model.answer(sample.image_id or "", sample.question or "")
        elif task.name == "captioning":
            text = model.caption(sample.image_id or "")
        else:
            raise ValueError(f"cannot generate predictions for task {task.name!r}")
        predictions.append(Prediction(uid=sample.uid, text=text))
    return predictions


def _evaluate_retrieval(
    task: RetrievalTask,
    samples: Sequence[Sample],
    model: Any,
    direction: str,
) -> EvalResult:
    image_ids = [s.image_id or s.uid for s in samples]
    texts = [s.references[0] if s.references else "" for s in samples]
    image_embeddings = model.encode_images(image_ids)
    text_embeddings = model.encode_texts(texts)
    return task.evaluate(image_embeddings, text_embeddings, direction=direction)


def evaluate(
    task: Task | str,
    dataset: Sequence[Sample],
    *,
    model: Any = None,
    predictions: Sequence[Prediction] | None = None,
    limit: int | None = None,
    direction: str = "text_to_image",
) -> EvalResult:
    """Run an evaluation and return its result.

    For VQA/captioning supply either ``predictions`` (already produced) or a
    ``model`` to generate them. Retrieval always needs an embedding ``model``.
    """
    resolved = _resolve_task(task)
    samples = list(dataset)
    if limit is not None:
        samples = samples[:limit]

    if isinstance(resolved, RetrievalTask):
        if model is None:
            raise ValueError("retrieval evaluation requires an embedding model")
        return _evaluate_retrieval(resolved, samples, model, direction)

    if predictions is None:
        if model is None:
            raise ValueError("generation evaluation requires a model or predictions")
        predictions = _generate(resolved, samples, model)
    return resolved.evaluate(samples, predictions)
