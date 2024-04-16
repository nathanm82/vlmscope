from __future__ import annotations

import pytest

from vlmscope.registry import Registry


def test_direct_registration_and_lookup() -> None:
    reg: Registry[int] = Registry("widget")
    reg.register("a", 1)
    assert reg.get("a") == 1
    assert "a" in reg
    assert reg.names() == ["a"]
    assert len(reg) == 1


def test_decorator_registration() -> None:
    reg: Registry[object] = Registry("thing")

    @reg.register("greet")
    def greet() -> str:
        return "hi"

    assert reg.get("greet") is greet


def test_duplicate_name_raises() -> None:
    reg: Registry[int] = Registry("widget")
    reg.register("a", 1)
    with pytest.raises(KeyError, match="already registered"):
        reg.register("a", 2)


def test_unknown_name_lists_available() -> None:
    reg: Registry[int] = Registry("widget")
    reg.register("a", 1)
    reg.register("b", 2)
    with pytest.raises(KeyError, match="available"):
        reg.get("missing")


def test_iter_is_sorted() -> None:
    reg: Registry[int] = Registry("widget")
    for name in ("c", "a", "b"):
        reg.register(name, 0)
    assert list(reg) == ["a", "b", "c"]
