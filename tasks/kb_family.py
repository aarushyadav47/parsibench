"""Knowledge-boundary task family."""

from __future__ import annotations

from typing import Dict, List

from parsibench.tasks.base import normalize_text
from parsibench.utils.rng import rng_from
from parsibench.utils.schema import Task


FACT_FILLERS = [
    "It is often cited in recent studies.",
    "Scholars discuss it frequently.",
    "The detail is recorded in the archives.",
    "Documentation mentions this clearly.",
    "A brief memo summarizes it.",
]


def make_kb_doc(doc_id: str, seed: int) -> Dict:
    rng = rng_from("kb_doc", doc_id, seed)
    topic = f"topic_{rng.integers(1000, 9999)}"
    place = rng.choice(["Paris", "Cairo", "Seoul", "Lima", "Oslo"])
    access_code = f"code-{rng.integers(10000, 99999)}"
    filler = rng.choice(FACT_FILLERS)
    text = (
        f"{topic} is studied in {place}. "
        f"The access code is {access_code}. "
        f"{filler}"
    )
    return {"doc_id": doc_id, "text": text, "topic": topic, "access_code": access_code}


def _build_incontext_prompt(entity: str, answer: str, rng_key: int) -> str:
    rng = rng_from("kb_in_context", rng_key)
    other_attrs = ["origin", "field", "date", "notable researcher", "archive"]
    facts = []
    facts.append(f"{entity} has access code {answer}.")
    for _ in range(rng.integers(2, 4)):
        attr = rng.choice(other_attrs)
        value = f"detail_{rng.integers(100, 999)}"
        facts.append(f"{entity} {attr} is {value}.")
    rng.shuffle(facts)
    prompt_lines = [f"- {fact}" for fact in facts]
    question = f"What is the access code for {entity}?"
    return "Context:\n" + "\n".join(prompt_lines) + f"\nQuestion: {question}"


def generate_kb(n: int, seed: int) -> List[Task]:
    docs = [make_kb_doc(f"doc_{i}", seed + i) for i in range(50)]
    rng = rng_from("kb_tasks", seed)
    tasks: List[Task] = []

    for idx in range(n):
        requires_tool = idx % 2 == 1  # enforce ~50/50 split
        task_seed = int(rng.integers(0, 2**32 - 1))

        if requires_tool:
            doc = docs[int(rng.integers(0, len(docs)))]
            text = doc["text"]
            answer = doc["access_code"]
            start = text.index(answer)
            prompt = (
                "Use the KB tools if the answer is not in the prompt.\n"
                f"Question: What is the access code associated with {doc['topic']} studied in {doc['text'].split(' is studied in ')[1].split('.')[0]}?"
            )
            cert = {
                "requires_tool": True,
                "required_doc_id": doc["doc_id"],
                "required_span": {
                    "doc_id": doc["doc_id"],
                    "start": start,
                    "end": start + len(answer),
                },
            }
        else:
            entity = f"subject_{task_seed % 10000}"
            answer = f"code-{(task_seed % 90000) + 10000}"
            prompt = _build_incontext_prompt(entity, answer, task_seed)
            cert = {"requires_tool": False, "required_doc_id": None}

        tasks.append(
            Task(
                task_id=f"kb_{idx}",
                family="kb",
                prompt=prompt,
                gold={"answer": normalize_text(answer)},
                cert=cert,
                tool_state={"kb_docs": docs},
            )
        )

    return tasks

