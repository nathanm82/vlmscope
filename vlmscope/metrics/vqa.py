"""VQA answer normalization and accuracy.

The normalization mirrors the official VQA v2 evaluation: lowercase, strip
punctuation, map number words to digits, drop articles and expand a handful of
common contractions. ``vqa_accuracy`` then uses the standard soft score,
``min(1, matches / 3)``, averaged over examples.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_ARTICLES = {"a", "an", "the"}

_NUMBER_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

# A small, representative slice of the official contraction table.
_CONTRACTIONS = {
    "dont": "don't",
    "isnt": "isn't",
    "arent": "aren't",
    "cant": "can't",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "wasnt": "wasn't",
    "werent": "weren't",
    "wont": "won't",
    "wouldnt": "wouldn't",
    "its": "it's",
    "thats": "that's",
    "whats": "what's",
}

_PUNCT = list(";/[]\"{}()=+\\_-><@`,?!")
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")


def _strip_punctuation(text: str) -> str:
    out = text
    for p in _PUNCT:
        if (p + " " in text or " " + p in text) or _COMMA_STRIP.search(text):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    out = _PERIOD_STRIP.sub("", out)
    return out


def _process_words(text: str) -> str:
    words = []
    for word in text.split():
        word = _NUMBER_MAP.get(word, word)
        if word in _ARTICLES:
            continue
        word = _CONTRACTIONS.get(word, word)
        words.append(word)
    return " ".join(words)


def normalize_answer(answer: str) -> str:
    """Normalize a single VQA answer to its canonical comparison form."""
    text = answer.replace("\n", " ").replace("\t", " ").strip().lower()
    text = _strip_punctuation(text)
    text = _process_words(text)
    return text.strip()
