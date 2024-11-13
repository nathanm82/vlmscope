from __future__ import annotations

import json

import pytest
from vlmscope.report import (
    format_csv,
    format_json,
    format_markdown,
    format_table,
    render,
)
from vlmscope.types import EvalResult


def _result() -> EvalResult:
    result = EvalResult(task="vqa", num_samples=3)
    result.metrics["vqa_accuracy"] = 0.6667
    result.metrics["exact_match"] = 1.0
    return result


def test_table_contains_task_and_metrics() -> None:
    out = format_table(_result())
    assert "vqa" in out
    assert "vqa_accuracy" in out
    assert "1.0000" in out


def test_table_handles_no_metrics() -> None:
    out = format_table(EvalResult(task="empty"))
    assert "empty" in out


def test_markdown_is_a_table() -> None:
    out = format_markdown(_result())
    assert out.startswith("### vqa")
    assert "| metric | value |" in out


def test_json_roundtrips() -> None:
    data = json.loads(format_json(_result()))
    assert data["task"] == "vqa"
    assert data["metrics"]["exact_match"] == 1.0


def test_csv_has_header_and_rows() -> None:
    out = format_csv(_result()).splitlines()
    assert out[0] == "metric,value"
    assert any(line.startswith("exact_match,") for line in out[1:])


def test_render_dispatch_and_unknown_format() -> None:
    assert render(_result(), "table")
    assert render(_result(), "json")
    assert render(_result(), "csv")
    with pytest.raises(ValueError, match="unknown format"):
        render(_result(), "bogus")
