# Contributing

Thanks for taking the time to help out. vlmscope is small on purpose, so most
contributions are a single focused metric, task, loader or bug fix.

## Getting set up

```bash
git clone https://github.com/nathanm82/vlmscope.git
cd vlmscope
pip install -e ".[dev]"
```

## Before you open a PR

Run the same checks CI runs:

```bash
./scripts/check.sh
```

or individually:

```bash
ruff check .
ruff format .
mypy
pytest
```

New behaviour needs a test. The metrics in particular are tested against
hand-computed values for small inputs — please follow that pattern so the
numbers stay verifiable.

## Adding a metric

1. Add a `(hypotheses, references) -> float` function under `vlmscope/metrics/`.
2. Register it in `vlmscope/metrics/__init__.py`.
3. Add a test with at least one example whose expected value you worked out by
   hand, plus an edge case (empty input, length mismatch).

## Adding a task

Subclass `GenerationTask` (set `name` and `default_metrics`) for text tasks, or
`Task` for something with a different shape, then register an instance in
`vlmscope/tasks/__init__.py`.

## Style

- Type hints on public functions; `from __future__ import annotations` at the
  top of every module.
- Keep dependencies to NumPy. If you need something heavier, it probably belongs
  behind an optional extra.
- Commit messages: a short imperative summary is plenty.

## Reporting bugs

Open an issue with a minimal reproduction — the smaller the snippet, the faster
it gets fixed.
