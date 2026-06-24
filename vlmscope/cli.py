"""Command-line interface for vlmscope."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from vlmscope.__about__ import __version__
from vlmscope.metrics import metric_registry
from vlmscope.tasks import task_registry


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vlmscope",
        description="Evaluate vision-language models on VQA, captioning and retrieval.",
    )
    parser.add_argument(
        "--version", action="version", version=f"vlmscope {__version__}"
    )
    parser.set_defaults(func=None)
    subparsers = parser.add_subparsers(dest="command")
    _add_list_commands(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
