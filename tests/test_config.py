from __future__ import annotations

from pathlib import Path

import pytest

from vlmscope.config import RunConfig
from vlmscope.models.dummy import LookupVQAModel
from vlmscope.runner import run


def test_from_dict_ignores_unknown_keys() -> None:
    cfg = RunConfig.from_dict({"task": "vqa", "dataset": "toy:vqa", "bogus": 1})
    assert cfg.task == "vqa"
    assert cfg.dataset == "toy:vqa"


def test_invalid_format_raises() -> None:
    with pytest.raises(ValueError):
        RunConfig(task="vqa", output_format="xml")


def test_invalid_limit_raises() -> None:
    with pytest.raises(ValueError):
        RunConfig(task="vqa", limit=0)


def test_dict_roundtrip() -> None:
    cfg = RunConfig(task="vqa", dataset="d.jsonl", metrics=["exact_match"])
    data = cfg.to_dict()
    assert data["task"] == "vqa"
    assert data["metrics"] == ["exact_match"]


def test_from_yaml(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    p = tmp_path / "config.yaml"
    p.write_text("task: captioning\ndataset: toy:captions\nlimit: 3\n", encoding="utf-8")
    cfg = RunConfig.from_yaml(p)
    assert cfg.task == "captioning"
    assert cfg.limit == 3


def test_run_via_config() -> None:
    cfg = RunConfig(task="vqa", dataset="toy:vqa")
    res = run(cfg, model=LookupVQAModel())
    assert res.task == "vqa"
    assert res.num_samples == 6


def test_run_requires_dataset() -> None:
    with pytest.raises(ValueError, match="dataset is required"):
        run(RunConfig(task="vqa"), model=LookupVQAModel())
