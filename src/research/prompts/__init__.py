# src/research/prompts/__init__.py

"""Prompts for Morning Research Agent."""

from src.research.prompts.templates import (
    SYSTEM_PROMPT,
    build_context,
    TASK_PROMPT,
)

__all__ = ["SYSTEM_PROMPT", "build_context", "TASK_PROMPT"]
