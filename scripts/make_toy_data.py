"""Dump the built-in toy datasets to JSONL so you can see the on-disk format.

python scripts/make_toy_data.py --out toy_data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlmscope.data.loaders import Dataset
from vlmscope.data.toy import toy_captions, toy_retrieval, toy_vqa


def _dump(dataset: Dataset, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for sample in dataset:
            record = {
                "uid": sample.uid,
                "image_id": sample.image_id,
                "question": sample.question,
                "references": list(sample.references),
            }
            fh.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="toy_data", help="Output directory.")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _dump(toy_vqa(), out / "vqa.jsonl")
    _dump(toy_captions(), out / "captions.jsonl")
    _dump(toy_retrieval(), out / "retrieval.jsonl")
    print(f"wrote toy datasets to {out}/")


if __name__ == "__main__":
    main()
