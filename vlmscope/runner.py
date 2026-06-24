"""High-level evaluation entry points.

``evaluate`` is the one function most users call: point it at a task, a dataset
and either a model (to generate predictions) or a set of precomputed
predictions, and get back an :class:`~vlmscope.types.EvalResult`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from vlmscope.config import RunConfig
from vlmscope.data.loaders import Dataset, load_csv, load_jsonl
from vlmscope.data.toy import TOY_DATASETS
from vlmscope.tasks import GenerationTask, Task, task_registry
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
    dataset: Iterable[Sample],
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

    if not isinstance(resolved, GenerationTask):
        raise ValueError(f"task {resolved.name!r} does not support prediction-based evaluation")
    if predictions is None:
        if model is None:
            raise ValueError("generation evaluation requires a model or predictions")
        predictions = _generate(resolved, samples, model)
    return resolved.evaluate(samples, predictions)


def resolve_dataset(spec: str) -> Dataset:
    """Resolve a dataset spec: ``toy:<name>`` or a ``.jsonl`` / ``.csv`` path."""
    if spec.startswith("toy:"):
        name = spec.split(":", 1)[1]
        if name not in TOY_DATASETS:
            raise ValueError(f"unknown toy dataset {name!r}; choose from {sorted(TOY_DATASETS)}")
        return TOY_DATASETS[name]()
    path = Path(spec)
    if path.suffix == ".jsonl":
        return load_jsonl(path)
    if path.suffix == ".csv":
        return load_csv(path)
    raise ValueError(f"unsupported dataset {spec!r}; use a .jsonl/.csv path or toy:<name>")


def run(
    config: RunConfig,
    *,
    model: Any = None,
    predictions: Sequence[Prediction] | None = None,
) -> EvalResult:
    """Run an evaluation described by a :class:`RunConfig`."""
    if config.dataset is None:
        raise ValueError("config.dataset is required to run")
    dataset = resolve_dataset(config.dataset)
    task = task_registry.get(config.task)
    if config.metrics and isinstance(task, GenerationTask):
        task = type(task)(metrics=config.metrics)
    return evaluate(task, dataset, model=model, predictions=predictions, limit=config.limit)
