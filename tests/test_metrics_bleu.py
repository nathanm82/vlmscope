from __future__ import annotations

import math

import pytest
from vlmscope.metrics.bleu import corpus_bleu, sentence_bleu


def test_perfect_match_is_one() -> None:
    assert corpus_bleu(["the cat sat"], [["the cat sat"]], max_n=2) == 1.0


def test_clipped_precision() -> None:
    # "the" is over-generated; clipping caps it at the reference count.
    val = corpus_bleu(["the the cat"], [["the cat"]], max_n=1)
    assert val == pytest.approx(2 / 3)


def test_brevity_penalty_applied() -> None:
    val = corpus_bleu(["the cat"], [["the cat sat down"]], max_n=1)
    assert val == pytest.approx(math.exp(-1.0))


def test_sentence_bleu_matches_corpus() -> None:
    s = sentence_bleu("a b c", ["a b c"], max_n=2)
    c = corpus_bleu(["a b c"], [["a b c"]], max_n=2)
    assert s == c


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        corpus_bleu(["a"], [["a"], ["b"]])
