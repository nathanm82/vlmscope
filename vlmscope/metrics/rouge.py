"""ROUGE-L: longest-common-subsequence F-measure.

Precision uses the hypothesis length, recall uses the reference length, and the
two are combined with an F-measure weighted by ``beta`` (1.2 by convention, so
recall counts a little more than precision). When several references are given,
the best-scoring one is kept.
"""

from __future__ import annotations

from collections.abc import Sequence

from vlmscope.metrics.text import tokenize


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        ai = a[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = prev[j] if prev[j] >= curr[j - 1] else curr[j - 1]
        prev = curr
    return prev[n]


def rouge_l_sentence(
    hypothesis: str, references: Sequence[str], beta: float = 1.2
) -> float:
    """ROUGE-L F-measure for a single hypothesis."""
    h = tokenize(hypothesis)
    if not h:
        return 0.0
    best = 0.0
    b2 = beta * beta
    for ref in references:
        r = tokenize(ref)
        if not r:
            continue
        lcs = _lcs_length(h, r)
        if lcs == 0:
            continue
        precision = lcs / len(h)
        recall = lcs / len(r)
        f = ((1 + b2) * precision * recall) / (recall + b2 * precision)
        best = max(best, f)
    return best


def rouge_l(
    hypotheses: Sequence[str],
    references: Sequence[Sequence[str]],
    beta: float = 1.2,
) -> float:
    """Corpus ROUGE-L: the mean sentence-level F-measure."""
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must have equal length")
    if not hypotheses:
        return 0.0
    scores = [
        rouge_l_sentence(h, r, beta) for h, r in zip(hypotheses, references)
    ]
    return sum(scores) / len(scores)
