# Benchmarks

Timing helpers for the metric implementations. These are not a model
leaderboard -- they only measure how fast the pure-Python metrics run, so you
can catch performance regressions.

```bash
python benchmarks/bench_metrics.py --num 1000
```

Example output:

```
benchmarking on 1000 samples

vqa_accuracy   score= 0.0030      12.4 ms
bleu           score= 0.0461     180.7 ms
rouge_l        score= 0.1989     145.2 ms
cider          score= 0.0007     402.5 ms
```

Numbers depend heavily on sentence length and vocabulary size; treat them as
relative, not absolute.
