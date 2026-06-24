"""Task registry and built-in tasks."""

from __future__ import annotations

from vlmscope.registry import Registry
from vlmscope.tasks.base import Task

#: Registry of task instances keyed by ``Task.name``.
task_registry: Registry[Task] = Registry("task")

__all__ = ["Task", "task_registry"]
