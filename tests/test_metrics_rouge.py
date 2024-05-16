from __future__ import annotations

from vlmscope.metrics.rouge import rouge_l, rouge_l_sentence


def test_identity_is_one() -> None:
    assert rouge_l_sentence("the cat sat", ["the cat sat"]) == 1.0


def test_lcs_is_not_contiguous() -> None:
    # "a b c d" is a subsequence of the reference despite the gaps.
    assert rouge_l_sentence("a b c d", ["a x b x c x d"]) > 0.5


def test_empty_hypothesis_is_zero() -> None:
    assert rouge_l_sentence("", ["something"]) == 0.0


def test_corpus_mean() -> None:
    val = rouge_l(["the cat", "a dog"], [["the cat"], ["a dog"]])
    assert val == 1.0


def test_partial_overlap_between_zero_and_one() -> None:
    val = rouge_l_sentence("the cat", ["the cat sat down"])
    assert 0.0 < val < 1.0
