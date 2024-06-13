"""Dataset container and file loaders.

A :class:`Dataset` is just a named list of :class:`~vlmscope.types.Sample`.
Loaders accept forgiving field names (``answers``/``captions``/``references``
all populate ``Sample.references``) so the same reader works across tasks.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

from vlmscope.types import Sample

PathLike = Union[str, Path]

# Record keys consumed into typed Sample fields; everything else -> metadata.
_RESERVED = {
    "uid",
    "id",
    "image_id",
    "image",
    "question",
    "references",
    "answers",
    "captions",
}


@dataclass
class Dataset:
    """A named collection of samples."""

    name: str
    samples: list[Sample]

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    def take(self, n: int) -> Dataset:
        """Return a new dataset with at most ``n`` samples."""
        return Dataset(self.name, self.samples[:n])


def _references(record: dict[str, Any]) -> tuple[str, ...]:
    refs = record.get("references")
    if refs is None:
        refs = record.get("answers")
    if refs is None:
        refs = record.get("captions")
    if refs is None:
        return ()
    if isinstance(refs, str):
        return (refs,)
    return tuple(str(r) for r in refs)


def _sample_from_record(record: dict[str, Any], index: int) -> Sample:
    uid = str(record.get("uid", record.get("id", index)))
    metadata = {k: v for k, v in record.items() if k not in _RESERVED}
    return Sample(
        uid=uid,
        image_id=record.get("image_id") or record.get("image"),
        question=record.get("question"),
        references=_references(record),
        metadata=metadata,
    )


def load_jsonl(path: PathLike, name: str | None = None) -> Dataset:
    """Load a dataset from a JSON-lines file (one record per line)."""
    p = Path(path)
    samples: list[Sample] = []
    with p.open(encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            samples.append(_sample_from_record(json.loads(line), index))
    return Dataset(name or p.stem, samples)
