"""Render evaluation results for humans and machines."""

from __future__ import annotations

import json

from vlmscope.types import EvalResult


def format_table(result: EvalResult) -> str:
    """A compact aligned text table, suitable for a terminal."""
    header = f"task: {result.task}  (n={result.num_samples})"
    if not result.metrics:
        return header
    width = max(len(name) for name in result.metrics)
    lines = [header]
    for name, value in result.metrics.items():
        lines.append(f"  {name:<{width}}  {value:.4f}")
    return "\n".join(lines)


def format_markdown(result: EvalResult) -> str:
    """A GitHub-flavoured Markdown table of the metrics."""
    lines = [
        f"### {result.task} (n={result.num_samples})",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for name, value in result.metrics.items():
        lines.append(f"| {name} | {value:.4f} |")
    return "\n".join(lines)


def format_json(result: EvalResult, indent: int = 2) -> str:
    """Stable JSON, with metric keys sorted for reproducible diffs."""
    return json.dumps(result.as_dict(), indent=indent, sort_keys=True)


def format_csv(result: EvalResult) -> str:
    """Two columns -- ``metric,value`` -- with a header row."""
    lines = ["metric,value"]
    for name, value in result.metrics.items():
        lines.append(f"{name},{value:.6f}")
    return "\n".join(lines)


_RENDERERS = {
    "table": format_table,
    "markdown": format_markdown,
    "json": format_json,
    "csv": format_csv,
}


def render(result: EvalResult, fmt: str = "table") -> str:
    """Render ``result`` in one of the supported formats."""
    try:
        renderer = _RENDERERS[fmt]
    except KeyError:
        raise ValueError(f"unknown format {fmt!r}; choose from {sorted(_RENDERERS)}") from None
    return renderer(result)
