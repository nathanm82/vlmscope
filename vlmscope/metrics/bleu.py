"""Corpus and sentence BLEU.

A faithful, dependency-free BLEU: clipped n-gram precision for ``n = 1..max_n``,
combined as a geometric mean and scaled by the brevity penalty. The effective
reference length per sentence is the one *closest* to the hypothesis length,
matching the common caption-evaluation convention.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from vlmscope.metrics.text import ngram_counts, tokenize


def _closest_ref_len(hyp_len: int, ref_lens: Sequence[int]) -> int:
    # Tie-break towards the shorter reference, like sacrebleu / NLTK.
    return min(ref_lens, key=lambda rl: (abs(rl - hyp_len), rl))


def _brevity_penalty(hyp_len: int, ref_len: int) -> float:
    if hyp_len > ref_len:
        return 1.0
    return math.exp(1.0 - ref_len / hyp_len)


def corpus_bleu(
    hypotheses: Sequence[str],
    references: Sequence[Sequence[str]],
    max_n: int = 4,
) -> float:
    """Corpus-level BLEU in ``[0, 1]``."""
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must have equal length")

    clipped = [0] * max_n
    total = [0] * max_n
    hyp_len_total = 0
    ref_len_total = 0

    for hyp, refs in zip(hypotheses, references):
        h_tokens = tokenize(hyp)
        ref_tokens = [tokenize(r) for r in refs]
        hyp_len_total += len(h_tokens)
        ref_lens = [len(r) for r in ref_tokens] or [0]
        ref_len_total += _closest_ref_len(len(h_tokens), ref_lens)

        for n in range(1, max_n + 1):
            h_counts = ngram_counts(h_tokens, n)
            max_ref: Counter[tuple[str, ...]] = Counter()
            for rt in ref_tokens:
                for gram, count in ngram_counts(rt, n).items():
                    if count > max_ref[gram]:
                        max_ref[gram] = count
            for gram, count in h_counts.items():
                clipped[n - 1] += min(count, max_ref[gram])
                total[n - 1] += count

    precisions = [clipped[i] / total[i] if total[i] > 0 else 0.0 for i in range(max_n)]
    if min(precisions) <= 0.0:
        return 0.0
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    return _brevity_penalty(hyp_len_total, ref_len_total) * geo_mean


def sentence_bleu(hypothesis: str, references: Sequence[str], max_n: int = 4) -> float:
    """BLEU for a single hypothesis against its references."""
    return corpus_bleu([hypothesis], [references], max_n=max_n)
