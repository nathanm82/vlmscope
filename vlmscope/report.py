"""Render evaluation results for humans and machines."""

from __future__ import annotations

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


_RENDERERS = {"table": format_table}


def render(result: EvalResult, fmt: str = "table") -> str:
    """Render ``result`` in one of the supported formats."""
    try:
        renderer = _RENDERERS[fmt]
    except KeyError:
        raise ValueError(
            f"unknown format {fmt!r}; choose from {sorted(_RENDERERS)}"
        ) from None
    return renderer(result)
