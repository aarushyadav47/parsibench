"""Deterministic RNG helpers."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def stable_hash_str(s: str) -> int:
    digest = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def rng_from(*parts: Any) -> np.random.Generator:
    key = "::".join(map(str, parts))
    seed = stable_hash_str(key)
    return np.random.default_rng(seed)

