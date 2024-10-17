# Design notes

A few decisions worth writing down, mostly so I remember why later.

## Pure Python + NumPy only

The captioning metrics (BLEU/ROUGE-L/CIDEr) are usually pulled in via
`pycocoevalcap`, which drags along a Java METEOR jar and a Stanford tokenizer.
That's a lot of weight for "score some strings". Reimplementing them in a few
hundred lines keeps install trivial and the behaviour inspectable. The trade-off
is that METEOR is intentionally **not** included — it needs stemming and synonym
resources that don't belong in a lightweight package.

## One signature for generation metrics

VQA and captioning metrics all take `(hypotheses, references)` and return a
float. That uniformity is what makes the registry and the `GenerationTask` flow
possible — VQA and captioning end up being the same task with different default
metrics. Retrieval genuinely doesn't fit (it needs the full embedding matrix),
so it gets its own task rather than being forced into the mold.

## Models are protocols, not base classes

vlmscope never imports torch or transformers. Callers wrap whatever they have in
something that satisfies a `Protocol`; `isinstance` still works because the
protocols are `runtime_checkable`. This keeps the dependency surface at exactly
"NumPy".

## Predictions can be supplied directly

A lot of real evaluation is offline: you already have a file of model outputs
and just want a score. So `evaluate(..., predictions=...)` skips the model
entirely. The `--predictions` CLI flag is the same idea.

## Known limitations

- CIDEr document frequencies are computed from the evaluation set itself, not a
  held-out corpus (this matches common practice but is worth knowing).
- N-gram counts are recomputed per sentence; fine for evaluation-sized data, not
  optimized for very large corpora.
