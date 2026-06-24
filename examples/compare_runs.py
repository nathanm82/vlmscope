"""Compare two VQA "models" side by side.

Demonstrates evaluating multiple systems on the same dataset and printing a
small leaderboard. Run with::

    python examples/compare_runs.py
"""

from __future__ import annotations

from vlmscope import evaluate
from vlmscope.data.toy import toy_vqa
from vlmscope.models.dummy import LookupVQAModel


def main() -> None:
    dataset = toy_vqa()

    # A "strong" model that answers with the gold answer, and a constant
    # baseline that always says "yes".
    strong = LookupVQAModel({s.image_id or s.uid: s.references[0] for s in dataset})
    baseline = LookupVQAModel(default="yes")

    runs = {
        "lookup": evaluate("vqa", dataset, model=strong),
        "always-yes": evaluate("vqa", dataset, model=baseline),
    }

    print("| model | vqa_accuracy | exact_match |")
    print("| --- | --- | --- |")
    for name, result in runs.items():
        acc = result.metrics["vqa_accuracy"]
        em = result.metrics["exact_match"]
        print(f"| {name} | {acc:.4f} | {em:.4f} |")


if __name__ == "__main__":
    main()
