"""Run configuration.

A :class:`RunConfig` fully describes one evaluation: which task to run, where
the data comes from, which metrics to report and how to render the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Output renderers understood by the CLI / reporter.
OUTPUT_FORMATS = ("table", "json", "markdown", "csv")


@dataclass
class RunConfig:
    """Parameters for a single evaluation run."""

    task: str
    dataset: str | None = None
    metrics: list[str] | None = None
    limit: int | None = None
    seed: int = 0
    output_format: str = "table"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.output_format not in OUTPUT_FORMATS:
            raise ValueError(
                f"output_format must be one of {OUTPUT_FORMATS}, got {self.output_format!r}"
            )
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be a positive integer or None")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunConfig:
        """Build a config from a plain dict, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        """Load a config from a YAML file (requires the ``yaml`` extra)."""
        try:
            import yaml
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "PyYAML is required for YAML configs; install vlmscope[yaml]"
            ) from exc
        with Path(path).open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "dataset": self.dataset,
            "metrics": list(self.metrics) if self.metrics is not None else None,
            "limit": self.limit,
            "seed": self.seed,
            "output_format": self.output_format,
            "extra": dict(self.extra),
        }
