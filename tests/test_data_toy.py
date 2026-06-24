from __future__ import annotations

from vlmscope.data.toy import TOY_DATASETS, toy_captions, toy_retrieval, toy_vqa


def test_toy_vqa_shape() -> None:
    ds = toy_vqa()
    assert len(ds) == 6
    assert all(s.question for s in ds)
    assert all(s.references for s in ds)


def test_toy_captions_have_multiple_refs() -> None:
    ds = toy_captions()
    assert all(len(s.references) >= 2 for s in ds)


def test_toy_retrieval_pairs() -> None:
    ds = toy_retrieval()
    assert len(ds) == 6
    assert all(s.image_id and s.references for s in ds)


def test_toy_registry_keys() -> None:
    assert {"vqa", "captions", "retrieval"} <= set(TOY_DATASETS)
