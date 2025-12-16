"""JSONL I/O utilities."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, List, Union

JsonLikePath = Union[str, Path]


def read_jsonl(path: JsonLikePath) -> List[dict]:
    file_path = Path(path)
    records: List[dict] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: JsonLikePath, iterable_of_dicts: Iterable[dict]) -> None:
    lines = [json.dumps(obj, ensure_ascii=False) for obj in iterable_of_dicts]
    payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    atomic_write(path, payload)


def atomic_write(path: JsonLikePath, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=target.parent, prefix=target.name, suffix=".tmp") as tmp:
        tmp.write(payload)
        tmp.flush()
        os.fsync(tmp.fileno())
        temp_name = tmp.name
    os.replace(temp_name, target)

