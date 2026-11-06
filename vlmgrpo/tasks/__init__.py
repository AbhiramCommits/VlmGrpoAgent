"""Task suite: deterministic multimodal agentic tasks and their verifiers."""

from .families import FAMILIES, TaskItem, generate_task

__all__ = ["FAMILIES", "TaskItem", "generate_task"]
