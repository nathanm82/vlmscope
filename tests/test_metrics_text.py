from __future__ import annotations

import pytest
from vlmscope.metrics.text import ngram_counts, ngrams, normalize_text, tokenize


def test_normalize_lowercases_and_strips_punctuation() -> None:
    assert normalize_text("Hello, World!") == "hello world"


def test_tokenize_collapses_whitespace() -> None:
    assert tokenize("A   quick, brown\tfox.") == ["a", "quick", "brown", "fox"]


def test_tokenize_empty_is_empty_list() -> None:
    assert tokenize("   ") == []


def test_ngrams_basic() -> None:
    assert ngrams(["a", "b", "c"], 2) == [("a", "b"), ("b", "c")]


def test_ngrams_shorter_than_n() -> None:
    assert ngrams(["a"], 2) == []


def test_ngram_counts() -> None:
    counts = ngram_counts(["a", "b", "a", "b"], 1)
    assert counts[("a",)] == 2
    assert counts[("b",)] == 2


def test_ngrams_invalid_n() -> None:
    with pytest.raises(ValueError):
        ngrams(["a"], 0)
