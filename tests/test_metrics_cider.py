from __future__ import annotations

import pytest

from vlmscope.metrics.cider import cider


def test_cider_is_nonnegative() -> None:
    hyps = ["a cat on a mat", "a dog in the park", "blue sky"]
    refs = [["a cat on a mat"], ["a dog in the park"], ["clear blue sky"]]
    assert cider(hyps, refs) >= 0.0


def test_matching_beats_mismatching() -> None:
    refs = [
        ["a cat sat on the mat"],
        ["a dog ran across the park"],
        ["the bright sun warms the field"],
    ]
    good = [r[0] for r in refs]
    bad = [
        "totally unrelated phrase",
        "another random unrelated string",
        "nothing in common here",
    ]
    assert cider(good, refs) > cider(bad, refs)


def test_empty_corpus_is_zero() -> None:
    assert cider([], []) == 0.0


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        cider(["a"], [["a"], ["b"]])
