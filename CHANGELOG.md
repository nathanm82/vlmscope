# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/).

## [0.3.0]

### Added

- Full documentation set under `docs/` (architecture, usage, metrics, design
  notes, API reference) and a release workflow.

### Changed

- Reworked the task base so VQA and captioning share one generation flow.
- Runner now logs at debug level and accepts any iterable of samples.

## [0.2.0]

### Added

- CSV export for evaluation reports: `render(result, "csv")` and
  `vlmscope run --format csv`.

## [0.1.0]

Initial release.

### Added

- VQA metrics: soft accuracy and exact match with VQA-style answer
  normalization.
- Captioning metrics: BLEU, ROUGE-L and CIDEr, pure-Python implementations.
- Retrieval metrics: Recall@k, median and mean rank over a similarity matrix.
- Task definitions for VQA, captioning and retrieval, behind a registry.
- Model adapter protocols and deterministic dummy models.
- JSONL/CSV loaders and small built-in toy datasets.
- `evaluate()` entry point and table / JSON / Markdown reporting.
- `vlmscope` command-line interface (`list-tasks`, `list-metrics`, `run`).

[0.3.0]: https://github.com/nathanm82/vlmscope/releases/tag/v0.3.0
[0.2.0]: https://github.com/nathanm82/vlmscope/releases/tag/v0.2.0
[0.1.0]: https://github.com/nathanm82/vlmscope/releases/tag/v0.1.0
