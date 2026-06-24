"""Task registry and built-in tasks."""

from __future__ import annotations

from vlmscope.registry import Registry
from vlmscope.tasks.base import GenerationTask, Task
from vlmscope.tasks.captioning import CaptioningTask
from vlmscope.tasks.retrieval import RetrievalTask
from vlmscope.tasks.vqa import VQATask

#: Registry of task instances keyed by ``Task.name``.
task_registry: Registry[Task] = Registry("task")
task_registry.register("vqa", VQATask())
task_registry.register("captioning", CaptioningTask())
task_registry.register("retrieval", RetrievalTask())

__all__ = [
    "CaptioningTask",
    "GenerationTask",
    "RetrievalTask",
    "Task",
    "VQATask",
    "task_registry",
]
