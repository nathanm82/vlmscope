"""Command-line interface for vlmscope."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from vlmscope.__about__ import __version__
from vlmscope.metrics import metric_registry
from vlmscope.models.dummy import (
    HashingEmbeddingModel,
    LookupCaptionModel,
    LookupVQAModel,
)
from vlmscope.report import render
from vlmscope.runner import evaluate, resolve_dataset
from vlmscope.tasks import task_registry
from vlmscope.types import Prediction


def _cmd_list_tasks(args: argparse.Namespace) -> int:
    for name in task_registry.names():
        task = task_registry.get(name)
        print(f"{name}\t{', '.join(task.metric_names())}")
    return 0


def _cmd_list_metrics(args: argparse.Namespace) -> int:
    for name in metric_registry.names():
        print(name)
    return 0


def _add_list_commands(subparsers: argparse._SubParsersAction) -> None:
    p_tasks = subparsers.add_parser("list-tasks", help="List available tasks.")
    p_tasks.set_defaults(func=_cmd_list_tasks)

    p_metrics = subparsers.add_parser("list-metrics", help="List generation metrics.")
    p_metrics.set_defaults(func=_cmd_list_metrics)


def _load_predictions(path: str) -> list[Prediction]:
    predictions: list[Prediction] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            predictions.append(
                Prediction(
                    uid=str(record["uid"]),
                    text=record.get("text"),
                    score=record.get("score"),
                )
            )
    return predictions


def _demo_model(task: str) -> object:
    # The CLI ships dummy models so `run` works with zero setup; real
    # evaluations pass a model (or a predictions file) through the Python API.
    if task == "vqa":
        return LookupVQAModel()
    if task == "captioning":
        return LookupCaptionModel()
    return HashingEmbeddingModel()


def _cmd_run(args: argparse.Namespace) -> int:
    dataset = resolve_dataset(args.dataset)
    predictions = _load_predictions(args.predictions) if args.predictions else None
    model = None if predictions is not None else _demo_model(args.task)
    result = evaluate(
        args.task,
        dataset,
        model=model,
        predictions=predictions,
        limit=args.limit,
    )
    text = render(result, args.format)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


def _add_run_command(subparsers: argparse._SubParsersAction) -> None:
    run = subparsers.add_parser("run", help="Run an evaluation and print the result.")
    run.add_argument("--task", required=True, help="Task name (see list-tasks).")
    run.add_argument("--dataset", required=True, help="Path to .jsonl/.csv or toy:<name>.")
    run.add_argument("--predictions", help="JSONL of {uid, text} predictions to score.")
    run.add_argument("--format", default="table", choices=("table", "json", "markdown"))
    run.add_argument("--limit", type=int, default=None, help="Evaluate at most N samples.")
    run.add_argument("--output", help="Write the report to a file instead of stdout.")
    run.set_defaults(func=_cmd_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vlmscope",
        description="Evaluate vision-language models on VQA, captioning and retrieval.",
    )
    parser.add_argument("--version", action="version", version=f"vlmscope {__version__}")
    parser.set_defaults(func=None)
    subparsers = parser.add_subparsers(dest="command")
    _add_list_commands(subparsers)
    _add_run_command(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        parser.print_help()
        return 0
    try:
        return int(args.func(args))
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
