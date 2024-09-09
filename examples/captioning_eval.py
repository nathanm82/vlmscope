"""Evaluate a captioning model on the built-in toy dataset.

Run it with::

    python examples/captioning_eval.py

The reported numbers are BLEU, ROUGE-L and CIDEr.
"""

from __future__ import annotations

from vlmscope import evaluate, render
from vlmscope.data.toy import toy_captions
from vlmscope.models.dummy import LookupCaptionModel


def main() -> None:
    dataset = toy_captions()
    # Pretend the model produces the first reference caption for each image.
    captions = {s.image_id or s.uid: s.references[0] for s in dataset}
    model = LookupCaptionModel(captions)

    result = evaluate("captioning", dataset, model=model)
    print(render(result, "markdown"))


if __name__ == "__main__":
    main()
