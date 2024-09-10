"""Evaluate image-text retrieval on the built-in toy pairs.

Run it with::

    python examples/retrieval_eval.py

Uses the hashing embedding stand-in; replace it with a CLIP-style encoder that
exposes ``encode_images`` and ``encode_texts``.
"""

from __future__ import annotations

from vlmscope import evaluate, render
from vlmscope.data.toy import toy_retrieval
from vlmscope.models.dummy import HashingEmbeddingModel


def main() -> None:
    dataset = toy_retrieval()
    model = HashingEmbeddingModel(dim=128)

    for direction in ("text_to_image", "image_to_text"):
        result = evaluate("retrieval", dataset, model=model, direction=direction)
        print(f"# {direction}")
        print(render(result, "table"))
        print()


if __name__ == "__main__":
    main()
