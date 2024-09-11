"""Micro-benchmark for the generation metrics.

Generates synthetic hypotheses and references and times each metric so you can
spot regressions. Run with::

    python benchmarks/bench_metrics.py --num 500
"""

from __future__ import annotations

import argparse
import random
import time
from collections.abc import Callable, Sequence

from vlmscope.metrics import cider, corpus_bleu, rouge_l, vqa_accuracy

_VOCAB = [
    "the",
    "a",
    "an",
    "cat",
    "dog",
    "runs",
    "jumps",
    "over",
    "lazy",
    "fox",
    "red",
    "blue",
    "car",
    "fast",
    "slow",
    "boat",
]


def _synthetic(num: int, seed: int = 0) -> tuple[list[str], list[list[str]]]:
    rng = random.Random(seed)

    def sentence() -> str:
        return " ".join(rng.choice(_VOCAB) for _ in range(rng.randint(4, 12)))

    hypotheses = [sentence() for _ in range(num)]
    references = [[sentence(), sentence()] for _ in range(num)]
    return hypotheses, references


def _time(
    name: str,
    fn: Callable[[Sequence[str], Sequence[Sequence[str]]], float],
    hyps: Sequence[str],
    refs: Sequence[Sequence[str]],
) -> None:
    start = time.perf_counter()
    score = fn(hyps, refs)
    elapsed = time.perf_counter() - start
    print(f"{name:<14} score={score:7.4f}  {elapsed * 1000:8.1f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num", type=int, default=500, help="Number of samples.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    hyps, refs = _synthetic(args.num, args.seed)
    print(f"benchmarking on {args.num} samples\n")
    _time("vqa_accuracy", vqa_accuracy, hyps, refs)
    _time("bleu", corpus_bleu, hyps, refs)
    _time("rouge_l", rouge_l, hyps, refs)
    _time("cider", cider, hyps, refs)


if __name__ == "__main__":
    main()
