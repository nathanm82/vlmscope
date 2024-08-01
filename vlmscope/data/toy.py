"""Tiny built-in datasets.

Small enough to read at a glance, big enough to exercise every metric. Used by
the examples, the test-suite and ``vlmscope run --dataset toy:<task>``.
"""

from __future__ import annotations

from vlmscope.data.loaders import Dataset
from vlmscope.types import Sample


def toy_vqa() -> Dataset:
    """Six VQA examples with multiple human answers each."""
    samples = [
        Sample(
            "q1",
            image_id="img1",
            question="What animal is this?",
            references=("cat", "cat", "kitten"),
        ),
        Sample(
            "q2",
            image_id="img2",
            question="How many people are there?",
            references=("two", "2", "two people"),
        ),
        Sample(
            "q3",
            image_id="img3",
            question="What color is the car?",
            references=("red", "red", "dark red"),
        ),
        Sample(
            "q4",
            image_id="img4",
            question="Is it raining?",
            references=("no", "no", "no it is not"),
        ),
        Sample(
            "q5",
            image_id="img5",
            question="What is the person doing?",
            references=("surfing", "surfing", "riding a wave"),
        ),
        Sample(
            "q6",
            image_id="img6",
            question="Where is this?",
            references=("kitchen", "in a kitchen", "kitchen"),
        ),
    ]
    return Dataset("toy-vqa", samples)


def toy_captions() -> Dataset:
    """Five captioning examples with several reference captions."""
    samples = [
        Sample(
            "c1",
            image_id="img1",
            references=("a cat sitting on a sofa", "a cat resting on a couch"),
        ),
        Sample(
            "c2",
            image_id="img2",
            references=("two people walking on a beach", "a couple walks along the shore"),
        ),
        Sample(
            "c3",
            image_id="img3",
            references=("a red car parked on the street", "a red car on the road"),
        ),
        Sample(
            "c4",
            image_id="img4",
            references=("a plate of food on a table", "a meal served on a white plate"),
        ),
        Sample(
            "c5",
            image_id="img5",
            references=("a surfer riding a large wave", "a person surfing in the ocean"),
        ),
    ]
    return Dataset("toy-captions", samples)


def toy_retrieval() -> Dataset:
    """Six image-text pairs; the image id doubles as a short visual descriptor."""
    pairs = [
        ("a fluffy cat on a sofa", "a cat sitting on a couch"),
        ("two people on a sandy beach", "people walking along the beach"),
        ("a red sports car on a road", "a red car driving down the street"),
        ("a plate of pasta on a table", "a meal of pasta served on a plate"),
        ("a surfer on a big ocean wave", "a person surfing a large wave"),
        ("a snowy mountain under blue sky", "a mountain covered in snow"),
    ]
    samples = [
        Sample(f"r{i}", image_id=desc, references=(caption,))
        for i, (desc, caption) in enumerate(pairs)
    ]
    return Dataset("toy-retrieval", samples)


#: Lookup used by the CLI's ``--dataset toy:<name>`` shorthand.
TOY_DATASETS = {
    "vqa": toy_vqa,
    "captions": toy_captions,
    "captioning": toy_captions,
    "retrieval": toy_retrieval,
}
