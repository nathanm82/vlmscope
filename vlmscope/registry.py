"""A tiny string-keyed registry.

Metrics, tasks and model adapters are all looked up by name from the CLI and
config files, so they share this minimal registry instead of three bespoke
dispatch tables.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps names to objects of a single ``kind`` (used only in messages)."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str, obj: T | None = None) -> T | Callable[[T], T]:
        """Register ``obj`` under ``name``.

        Usable directly (``reg.register("x", obj)``) or as a decorator
        (``@reg.register("x")``).
        """
        if obj is None:

            def decorator(target: T) -> T:
                self._add(name, target)
                return target

            return decorator

        self._add(name, obj)
        return obj

    def _add(self, name: str, obj: T) -> None:
        if name in self._items:
            raise KeyError(f"{self._kind} {name!r} is already registered")
        self._items[name] = obj

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(f"unknown {self._kind} {name!r}; available: {self.names()}") from None

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._items)
