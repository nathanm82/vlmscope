"""Evaluate a VQA model on the built-in toy dataset.

Run it with::

    python examples/vqa_eval.py

Swap ``LookupVQAModel`` for an adapter around your own model -- anything with
an ``answer(image_id, question) -> str`` method satisfies the protocol.
"""

from __future__ import annotations

from vlmscope import evaluate, render
from vlmscope.data.toy import toy_vqa
from vlmscope.models.dummy import LookupVQAModel


def main() -> None:
    dataset = toy_vqa()
    # Stand-in "model": answer each image with its first reference answer.
    answers = {s.image_id or s.uid: s.references[0] for s in dataset}
    model = LookupVQAModel(answers)

    result = evaluate("vqa", dataset, model=model)
    print(render(result, "table"))


if __name__ == "__main__":
    main()
