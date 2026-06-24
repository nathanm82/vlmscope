# Architecture

vlmscope is deliberately small. Five pieces fit together:

```
data ──► Sample ──┐
                  ├─► Task.evaluate ──► EvalResult ──► report.render
model ─► Prediction┘
```

## Building blocks

- **`types.py`** — the data carried between layers: `Sample` (one example),
  `Prediction` (one model output) and `EvalResult` (metrics for a run). They are
  plain dataclasses with no behaviour beyond a little bookkeeping.

- **`metrics/`** — pure functions. The generation metrics (`vqa_accuracy`,
  `exact_match`, `corpus_bleu`, `rouge_l`, `cider`) all share the signature
  `(hypotheses, references) -> float`, which is what lets them sit behind one
  registry. Retrieval metrics work on a similarity matrix and live apart.

- **`tasks/`** — a task knows *which* metrics to compute and *how* to feed them.
  `GenerationTask` handles VQA and captioning (they only differ in their default
  metrics). `RetrievalTask` is separate because it consumes embeddings.

- **`models/`** — only protocols (`VQAModel`, `CaptionModel`, `EmbeddingModel`)
  plus deterministic dummies. vlmscope never loads a model itself; you wrap your
  own and pass it in.

- **`runner.py`** — `evaluate()` ties it together: resolve the task, gather
  samples, get predictions (from a model or supplied directly), score.

## Why registries

Metrics, tasks and toy datasets are all looked up by string name from the CLI
and from config files. A single `Registry` (`registry.py`) backs all three, so
adding a metric is "write a function, register it" — no central `if/elif`.

## Extending

- **New metric** — add a `(hypotheses, references) -> float` function under
  `metrics/` and register it in `metrics/__init__.py`.
- **New task** — subclass `GenerationTask` (set `name` and `default_metrics`) or
  `Task` for something with a different shape, then register an instance.
