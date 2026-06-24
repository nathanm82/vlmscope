# API reference

The public surface is what `import vlmscope` exposes; everything else is an
implementation detail and may change.

## Top level

```python
vlmscope.evaluate(task, dataset, *, model=None, predictions=None,
                  limit=None, direction="text_to_image") -> EvalResult
vlmscope.render(result, fmt="table") -> str        # "table" | "json" | "markdown"
vlmscope.__version__
```

Re-exported for convenience: `Sample`, `Prediction`, `EvalResult`, `RunConfig`,
`Task`, `VQATask`, `CaptioningTask`, `RetrievalTask`, `task_registry`,
`metric_registry`.

## Types (`vlmscope.types`)

- `Sample(uid, image_id=None, question=None, references=(), metadata={})`
- `Prediction(uid, text=None, score=None, metadata={})`
- `EvalResult(task, metrics={}, num_samples=0, extra={})` with `.as_dict()`

## Metrics (`vlmscope.metrics`)

```python
vqa_accuracy(hypotheses, references) -> float
exact_match(hypotheses, references) -> float
corpus_bleu(hypotheses, references, max_n=4) -> float
rouge_l(hypotheses, references, beta=1.2) -> float
cider(hypotheses, references, n=4, sigma=6.0) -> float
```

Retrieval (`vlmscope.metrics.retrieval`):

```python
cosine_similarity(queries, candidates) -> ndarray
recall_at_k(scores, positives, k) -> float
median_rank(scores, positives) -> float
mean_rank(scores, positives) -> float
retrieval_report(scores, positives, ks=(1, 5, 10)) -> dict
```

## Models (`vlmscope.models.base`)

Protocols you implement:

```python
class VQAModel(Protocol):
    def answer(self, image_id: str, question: str) -> str: ...

class CaptionModel(Protocol):
    def caption(self, image_id: str) -> str: ...

class EmbeddingModel(Protocol):
    def encode_images(self, image_ids) -> ndarray: ...
    def encode_texts(self, texts) -> ndarray: ...
```

## Data (`vlmscope.data`)

```python
load_jsonl(path, name=None) -> Dataset
load_csv(path, name=None, *, references_sep=";") -> Dataset
toy_vqa(), toy_captions(), toy_retrieval() -> Dataset
```

## Config + runner

```python
RunConfig(task, dataset=None, metrics=None, limit=None, seed=0,
          output_format="table", extra={})
RunConfig.from_dict(data) / RunConfig.from_yaml(path)
vlmscope.runner.run(config, *, model=None, predictions=None) -> EvalResult
vlmscope.runner.resolve_dataset(spec) -> Dataset   # "toy:<name>" | "*.jsonl" | "*.csv"
```
