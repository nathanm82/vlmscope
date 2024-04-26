from __future__ import annotations

import pytest

from vlmscope.metrics.vqa import exact_match, normalize_answer, vqa_accuracy


def test_normalize_drops_articles_and_case() -> None:
    assert normalize_answer("The Red Car") == "red car"


def test_normalize_maps_number_words() -> None:
    assert normalize_answer("two") == "2"


def test_normalize_strips_trailing_period() -> None:
    assert normalize_answer("yes.") == "yes"


def test_exact_match() -> None:
    assert exact_match(["cat"], [["a cat", "cat"]]) == 1.0
    assert exact_match(["dog"], [["cat"]]) == 0.0


def test_vqa_soft_accuracy_full_credit() -> None:
    refs = [["yes", "yes", "yes", "no"]]
    assert vqa_accuracy(["yes"], refs) == 1.0


def test_vqa_soft_accuracy_partial_credit() -> None:
    refs = [["yes", "no", "no"]]
    assert vqa_accuracy(["yes"], refs) == pytest.approx(1 / 3)


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        vqa_accuracy(["a"], [["a"], ["b"]])
