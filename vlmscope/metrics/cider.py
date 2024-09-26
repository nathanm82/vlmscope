"""CIDEr: consensus-based image description evaluation.

Implements CIDEr-D: TF-IDF weighted n-gram cosine similarity for ``n = 1..4``
with a Gaussian length penalty. Document frequencies are computed over the
provided reference corpus, so -- like the original metric -- CIDEr is only
meaningful across a set of images, not a single one.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np

from vlmscope.metrics.text import ngram_counts, tokenize


def _count_ngrams(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    """Count all n-grams of order ``1..n`` in a single token list."""
    # TODO: these counts are recomputed for the hypothesis and every reference;
    # caching by sentence would help on large corpora.
    counts: Counter[tuple[str, ...]] = Counter()
    for order in range(1, n + 1):
        counts.update(ngram_counts(tokens, order))
    return counts


def _document_frequencies(
    references: Sequence[Sequence[str]], n: int
) -> dict[tuple[str, ...], float]:
    df: dict[tuple[str, ...], float] = defaultdict(float)
    for refs in references:
        seen: set[tuple[str, ...]] = set()
        for ref in refs:
            seen.update(_count_ngrams(tokenize(ref), n).keys())
        for gram in seen:
            df[gram] += 1.0
    return df


def _tfidf_vector(
    tokens: Sequence[str],
    df: dict[tuple[str, ...], float],
    log_num_images: float,
    n: int,
) -> tuple[list[dict[tuple[str, ...], float]], list[float]]:
    vecs: list[dict[tuple[str, ...], float]] = [{} for _ in range(n)]
    norms = [0.0] * n
    for gram, tf in _count_ngrams(tokens, n).items():
        order = len(gram) - 1
        idf = log_num_images - np.log(max(df.get(gram, 0.0), 1.0))
        weight = tf * idf
        vecs[order][gram] = weight
        norms[order] += weight * weight
    return vecs, [float(np.sqrt(x)) for x in norms]


def cider(
    hypotheses: Sequence[str],
    references: Sequence[Sequence[str]],
    n: int = 4,
    sigma: float = 6.0,
) -> float:
    """Mean CIDEr-D score (scaled by 10, as is conventional)."""
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must have equal length")
    num_images = len(hypotheses)
    if num_images == 0:
        return 0.0

    df = _document_frequencies(references, n)
    log_num_images = float(np.log(max(num_images, 1)))

    total = 0.0
    for hyp, refs in zip(hypotheses, references):
        h_tokens = tokenize(hyp)
        hv, hn = _tfidf_vector(h_tokens, df, log_num_images, n)
        h_len = len(h_tokens)

        per_order = np.zeros(n)
        for ref in refs:
            r_tokens = tokenize(ref)
            rv, rn = _tfidf_vector(r_tokens, df, log_num_images, n)
            delta = h_len - len(r_tokens)
            penalty = float(np.exp(-(delta**2) / (2 * sigma**2)))
            for order in range(n):
                num = 0.0
                for gram, hw in hv[order].items():
                    rw = rv[order].get(gram, 0.0)
                    num += min(hw, rw) * rw
                if hn[order] > 0 and rn[order] > 0:
                    num /= hn[order] * rn[order]
                per_order[order] += num * penalty
        if refs:
            per_order /= len(refs)
        total += float(np.mean(per_order)) * 10.0

    return total / num_images
