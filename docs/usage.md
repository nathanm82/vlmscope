# Usage

## Data format

Datasets are lists of `Sample`. On disk, JSONL is the most convenient:

```json
{"uid": "q1", "image_id": "img1", "question": "What animal is this?", "answers": ["cat", "cat", "kitten"]}
{"uid": "q2", "image_id": "img2", "question": "How many?", "answers": ["two", "2"]}
```

The loader is forgiving about field names: `references`, `answers` and
`captions` all populate `Sample.references`, and `id` works as well as `uid`.
Unknown fields are kept under `Sample.metadata`.

CSV works too; a `references` column may hold several values separated by `;`:

```csv
uid,image_id,question,references
q1,img1,What animal is this?,cat;kitten
```

```python
from vlmscope.data.loaders import load_jsonl, load_csv

dataset = load_jsonl("vqa.jsonl")
```

## Running an evaluation

```python
import vlmscope

# (a) you have a model
result = vlmscope.evaluate("vqa", dataset, model=my_model)

# (b) you already have predictions
result = vlmscope.evaluate("vqa", dataset, predictions=my_predictions)

# limit the number of samples while iterating
result = vlmscope.evaluate("captioning", dataset, model=m, limit=100)
```

### Retrieval

Retrieval needs an embedding model exposing `encode_images` and `encode_texts`:

```python
result = vlmscope.evaluate("retrieval", dataset, model=clip_adapter,
                           direction="text_to_image")
```

## Reporting

```python
print(vlmscope.render(result, "table"))     # aligned text
print(vlmscope.render(result, "markdown"))  # GitHub table
print(vlmscope.render(result, "json"))      # stable, sorted JSON
```

## Config-driven runs

```python
from vlmscope.config import RunConfig
from vlmscope.runner import run

config = RunConfig(task="captioning", dataset="toy:captions", limit=50)
result = run(config, model=my_model)
```

Configs can be loaded from YAML (`pip install vlmscope[yaml]`):

```yaml
task: captioning
dataset: data/captions.jsonl
metrics: [bleu, cider]
limit: 500
```

```python
config = RunConfig.from_yaml("run.yaml")
```

## CLI

```bash
vlmscope run --task vqa --dataset toy:vqa --format json
vlmscope run --task vqa --dataset data.jsonl --predictions preds.jsonl --output out.txt
```

The CLI ships dummy models so `run` works with zero setup; for real numbers,
pass a predictions file or use the Python API with your own model.
