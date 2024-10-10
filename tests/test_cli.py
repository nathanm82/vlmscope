from __future__ import annotations

import json
from pathlib import Path

import pytest
from vlmscope.cli import main


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "vlmscope" in capsys.readouterr().out


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_list_tasks(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list-tasks"]) == 0
    out = capsys.readouterr().out
    assert "vqa" in out
    assert "captioning" in out
    assert "retrieval" in out


def test_list_metrics(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list-metrics"]) == 0
    assert "bleu" in capsys.readouterr().out


def test_run_toy_vqa_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--task", "vqa", "--dataset", "toy:vqa", "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["task"] == "vqa"
    assert data["num_samples"] == 6


def test_run_with_predictions(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    preds = tmp_path / "preds.jsonl"
    preds.write_text(
        "\n".join(json.dumps({"uid": f"q{i}", "text": "cat"}) for i in range(1, 7)),
        encoding="utf-8",
    )
    code = main(
        [
            "run",
            "--task",
            "vqa",
            "--dataset",
            "toy:vqa",
            "--predictions",
            str(preds),
            "--format",
            "json",
        ]
    )
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert "vqa_accuracy" in data["metrics"]


def test_unknown_task_exits_with_code_2(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["run", "--task", "nope", "--dataset", "toy:vqa"])
    assert code == 2
    assert "error" in capsys.readouterr().err.lower()


def test_unknown_dataset_exits_with_code_2() -> None:
    assert main(["run", "--task", "vqa", "--dataset", "toy:nope"]) == 2


def test_run_writes_output_file(tmp_path: Path) -> None:
    out = tmp_path / "report.txt"
    code = main(["run", "--task", "retrieval", "--dataset", "toy:retrieval", "--output", str(out)])
    assert code == 0
    assert out.read_text(encoding="utf-8").strip()
