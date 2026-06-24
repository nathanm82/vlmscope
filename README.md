# vlmscope

[![CI](https://github.com/nathanm82/vlmscope/actions/workflows/ci.yml/badge.svg)](https://github.com/nathanm82/vlmscope/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)

A small, dependency-light evaluation harness for **vision-language models**. It
covers the three benchmark families that keep coming up in practice and that I
got tired of re-implementing by hand:

- **VQA** — visual question answering, scored with the standard soft accuracy
  (`min(1, matches / 3)`) and exact match.
- **Captioning** — BLEU, ROUGE-L and CIDEr, implemented from scratch in pure
  Python so there's nothing to compile.
- **Image-text retrieval** — Recall@k plus median / mean rank over a similarity
  matrix.

The only runtime dependency is NumPy. Everything runs on built-in toy datasets
out of the box, so you can try it without downloading anything.

## Install

vlmscope isn't on PyPI yet; install from source:

```bash
git clone https://github.com/nathanm82/vlmscope.git
cd vlmscope
pip install -e .
```

## Quickstart

### Python

```python
import vlmscope
from vlmscope.data.toy import toy_vqa
from vlmscope.models.dummy import LookupVQAModel

dataset = toy_vqa()
model = LookupVQAModel({s.image_id: s.references[0] for s in dataset})

result = vlmscope.evaluate("vqa", dataset, model=model)
print(vlmscope.render(result, "table"))
```

Your model only needs to satisfy a tiny protocol — for VQA, an
`answer(image_id, question) -> str` method. See `vlmscope/models/base.py`.

Already have model outputs? Score them directly instead of wiring up a model:

```python
from vlmscope.types import Prediction

preds = [Prediction(uid=s.uid, text="...") for s in dataset]
result = vlmscope.evaluate("vqa", dataset, predictions=preds)
```

### Command line

```bash
# list what's available
vlmscope list-tasks
vlmscope list-metrics

# evaluate on a built-in toy dataset
vlmscope run --task captioning --dataset toy:captions --format markdown

# score your own predictions file against a dataset
vlmscope run --task vqa --dataset data.jsonl --predictions preds.jsonl
```

## Tasks and metrics

| Task | Metrics |
| --- | --- |
| `vqa` | `vqa_accuracy`, `exact_match` |
| `captioning` | `bleu`, `rouge_l`, `cider` |
| `retrieval` | `recall@k`, `median_rank`, `mean_rank` |

## Project layout

```
vlmscope/
  metrics/    BLEU, ROUGE-L, CIDEr, VQA accuracy, retrieval recall
  tasks/      VQA, captioning, retrieval task definitions
  models/     adapter protocols + deterministic dummy models
  data/       loaders (JSONL/CSV) and small toy datasets
  runner.py   the evaluate() entry point
  report.py   table / JSON / Markdown rendering
  cli.py      the vlmscope command
```

## Documentation

- [Usage guide](docs/usage.md)
- [Architecture](docs/architecture.md)
- [Metrics reference](docs/metrics.md)
- [Design notes](docs/design-notes.md)
- [API reference](docs/api-reference.md)

## Contributing

Bug reports and metric/task contributions are welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
