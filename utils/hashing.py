"""Hashing helpers."""

from __future__ import annotations

import hashlib


def stable_hash_str(s: str) -> int:
    digest = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)

