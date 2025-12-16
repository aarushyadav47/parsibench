"""Base task generator interfaces and helpers."""

from __future__ import annotations

from typing import List, Protocol

from parsibench.utils.schema import Task


class BaseGenerator(Protocol):
    def generate(self, n: int, seed: int) -> List[Task]:
        ...


def normalize_text(s: str) -> str:
    """Normalize text for comparison."""
    return " ".join(s.strip().lower().split())
