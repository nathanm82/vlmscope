from __future__ import annotations

import json
from pathlib import Path

from vlmscope.data.loaders import Dataset, load_csv, load_jsonl
from vlmscope.types import Sample


def test_load_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "data.jsonl"
    lines = [
        json.dumps(
            {
                "uid": "a",
                "image_id": "i1",
                "question": "what?",
                "answers": ["yes", "no"],
                "split": "val",
            }
        ),
        json.dumps({"id": 2, "captions": ["a cat"]}),
        "",  # blank lines are skipped
    ]
    p.write_text("\n".join(lines), encoding="utf-8")

    ds = load_jsonl(p)
    assert len(ds) == 2
    assert ds.samples[0].uid == "a"
    assert ds.samples[0].references == ("yes", "no")
    assert ds.samples[0].metadata["split"] == "val"
    assert ds.samples[1].uid == "2"
    assert ds.samples[1].references == ("a cat",)


def test_load_csv(tmp_path: Path) -> None:
    p = tmp_path / "data.csv"
    p.write_text(
        "uid,image_id,question,references\n1,i1,what?,yes;no\n2,i2,where?,home\n",
        encoding="utf-8",
    )
    ds = load_csv(p)
    assert len(ds) == 2
    assert ds.samples[0].references == ("yes", "no")
    assert ds.samples[1].references == ("home",)


def test_dataset_take() -> None:
    ds = Dataset("x", [Sample("1"), Sample("2"), Sample("3")])
    assert len(ds.take(2)) == 2
    assert len(ds) == 3
