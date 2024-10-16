# Metrics reference

All text metrics tokenize the same way: lowercase, strip punctuation, split on
whitespace (`metrics/text.py`). VQA answers get extra normalization.

## VQA

- **`vqa_accuracy`** — the official VQA v2 soft score. For each example,
  `min(1, n / 3)` where `n` is the number of human answers matching the
  prediction; averaged over examples. Answer normalization lowercases, removes
  punctuation, maps number words (`two -> 2`), drops articles (`a/an/the`) and
  expands a handful of contractions.
- **`exact_match`** — fraction of predictions equal to any normalized reference.

## Captioning

- **`bleu`** — corpus BLEU with `n = 1..4`, clipped n-gram precision combined as
  a geometric mean and scaled by the brevity penalty. The effective reference
  length per sentence is the one closest to the hypothesis length.
- **`rouge_l`** — longest-common-subsequence F-measure with `beta = 1.2` (recall
  weighted a little above precision); the best-scoring reference is kept.
- **`cider`** — CIDEr-D: TF-IDF weighted n-gram cosine similarity for `n = 1..4`
  with a Gaussian length penalty (`sigma = 6`). Document frequencies come from
  the reference corpus, so CIDEr is only meaningful across a set of images — a
  single-image CIDEr is always 0.

## Retrieval

Given a similarity matrix `scores` of shape `(queries, candidates)` and the
ground-truth positives per query:

- **`recall@k`** — fraction of queries whose top-`k` candidates contain a
  positive.
- **`median_rank` / `mean_rank`** — 1-based rank of the best positive per query.

`retrieval_report(scores, positives, ks=(1, 5, 10))` returns all of these in one
dict.

## A note on exact values

These are clean-room implementations. They follow the standard definitions and
agree with reference implementations on small cases (see the test-suite), but
tokenization differences mean absolute numbers may differ slightly from other
toolkits. Use them for relative comparison within a single run configuration.
