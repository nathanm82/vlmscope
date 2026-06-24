#!/usr/bin/env bash
# Run the same checks CI runs: lint, format, type-check and tests.
set -euo pipefail

echo "==> ruff check"
ruff check .

echo "==> ruff format --check"
ruff format --check .

echo "==> mypy"
mypy

echo "==> pytest"
pytest --cov=vlmscope --cov-report=term-missing
