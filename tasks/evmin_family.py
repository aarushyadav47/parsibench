"""Evidence-minimal retrieval task family."""

from __future__ import annotations

from typing import Dict, List, Tuple

from parsibench.tasks.base import normalize_text
from parsibench.utils.rng import rng_from
from parsibench.utils.schema import Task


def _make_required_doc(slot: str, value: str, doc_id: str) -> Tuple[Dict, Dict]:
    text = (
        f"This document contains the key evidence for {slot}. "
        f"FACT(slot={slot}): {value}. "
        f"Use the exact value."
    )
    start = text.index(value)
    span = {"slot": slot, "doc_id": doc_id, "start": start, "end": start + len(value)}
    return {"doc_id": doc_id, "text": text}, span


def _make_distractor(doc_id: str, seed: int, slots: List[str]) -> Dict:
    rng = rng_from("evmin_distractor", doc_id, seed)
    slot = rng.choice(slots)
    wrong_value = f"alt_{rng.integers(1000, 9999)}"
    text = (
        f"Background info near {slot}. "
        f"FACT(slot={slot}): {wrong_value}. "
        f"This is a distractor."
    )
    return {"doc_id": doc_id, "text": text}


def generate_evmin(n: int, seed: int) -> List[Task]:
    rng = rng_from("evmin_tasks", seed)
    tasks: List[Task] = []

    for idx in range(n):
        task_seed = int(rng.integers(0, 2**32 - 1))
        local_rng = rng_from("evmin_task_rng", task_seed)

        m = int(local_rng.integers(2, 4))
        slots = [f"s{j+1}" for j in range(m)]

        required_docs: List[Dict] = []
        required_spans: List[Dict] = []
        values: List[str] = []

        for slot_idx, slot in enumerate(slots):
            value = f"value_{idx}_{slot_idx}_{local_rng.integers(1000, 9999)}"
            doc_id = f"task{idx}_req_{slot}"
            doc, span = _make_required_doc(slot, value, doc_id)
            required_docs.append(doc)
            required_spans.append(span)
            values.append(value)

        distractor_count = int(local_rng.integers(10, 21))
        distractors = [
            _make_distractor(f"task{idx}_d{i}", int(local_rng.integers(0, 2**32 - 1)), slots)
            for i in range(distractor_count)
        ]

        kb_docs = required_docs + distractors
        prompt = (
            "Retrieve each required slot value from the KB. "
            "Use kb_search then kb_fetch to extract exact spans. "
            f"Slots: {', '.join(slots)}. "
            "Return: <value_s1> | <value_s2> | ..."
        )

        tasks.append(
            Task(
                task_id=f"evmin_{idx}",
                family="evmin",
                prompt=prompt,
                gold={"answer": " | ".join(normalize_text(v) for v in values)},
                cert={"required_spans": required_spans},
                tool_state={"kb_docs": kb_docs},
            )
        )

    return tasks

