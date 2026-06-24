# Examples

Small, self-contained scripts that run against the built-in toy datasets, so
they work with no downloads and finish instantly.

| Script | What it shows |
| --- | --- |
| `vqa_eval.py` | Scoring VQA answers (soft accuracy + exact match) |
| `captioning_eval.py` | BLEU / ROUGE-L / CIDEr on captions, rendered as Markdown |
| `retrieval_eval.py` | Recall@k and rank stats in both retrieval directions |
| `compare_runs.py` | Evaluating two models and printing a mini leaderboard |

Run any of them from the repo root:

```bash
python examples/vqa_eval.py
```

Each script uses a stand-in model from `vlmscope.models.dummy`. To evaluate a
real model, implement the matching protocol from `vlmscope.models.base` and pass
it to `vlmscope.evaluate(...)` instead.
