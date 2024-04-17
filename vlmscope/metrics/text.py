"""Text normalization shared by the generation metrics.

Caption-style metrics (BLEU, ROUGE-L, CIDEr) all expect lowercased,
punctuation-stripped, whitespace-tokenized text. Keeping that in one place
means every metric tokenizes identically.
"""

from __future__ import annotations

import re

# Punctuation removed before tokenizing. Apostrophes are dropped too, which is
# fine for caption metrics; VQA answer normalization handles contractions
# separately (see :mod:`vlmscope.metrics.vqa`).
_PUNCT_RE = re.compile(r"""[.,?!:;"'`()\[\]{}<>/\\|@#~_*=+\-]+""")
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase ``text``, drop punctuation and collapse whitespace."""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Whitespace tokenization applied after :func:`normalize_text`."""
    norm = normalize_text(text)
    return norm.split() if norm else []
