"""Convert labeled episodes into PRM training rows."""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

import pandas as pd

from parsibench.tasks.base import normalize_text
from parsibench.utils.io import read_jsonl
from parsibench.utils.schema import Episode, Task

LABEL_MAP = {"necessary": 0, "avoidable": 1, "redundant": 2}
FAMILY_MAP = {"kb": 0, "evmin": 1, "sample95": 2}
TOOL_MAP = {"kb_search": 0, "kb_fetch": 1, "sim_sample": 2}


def load_tasks(task_paths: List[str]) -> Dict[str, Task]:
    tasks: Dict[str, Task] = {}
    for path in task_paths:
        for row in read_jsonl(path):
            task = Task(**row)
            tasks[task.task_id] = task
    return tasks


def _build_rows_for_episode(ep: Episode, task: Task) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    search_count = 0
    fetch_count = 0
    n_total_A = 0
    n_total_B = 0
    unique_docs = set()
    last_tool_code = -1

    calls = [s.tool_call for s in ep.steps if s.tool_call]
    for idx, call in enumerate(calls):
        label_name = (ep.steps[idx].labels or {}).get("tool_call_label") if idx < len(ep.steps) else None
        if not label_name:
            continue
        features: Dict[str, Any] = {
            "task_id": ep.task_id,
            "family": FAMILY_MAP.get(ep.family, -1),
            "tool_name": TOOL_MAP.get(call.tool_name, -1),
            "step_index": call.step_index,
            "tools_used_count": idx,
            "last_tool_name": last_tool_code,
        }

        if ep.family == "kb":
            if call.tool_name == "kb_search":
                search_count += 1
            if call.tool_name == "kb_fetch":
                fetch_count += 1
            features.update(
                {
                    "search_count": search_count,
                    "fetch_count": fetch_count,
                }
            )
        elif ep.family == "evmin":
            if call.tool_name == "kb_fetch":
                doc_id = normalize_text(call.arguments.get("doc_id", ""))
                unique_docs.add(doc_id)
            features.update(
                {
                    "required_slots_total": len(task.cert.get("required_spans", [])) if isinstance(task.cert, dict) else 0,
                    "unique_doc_ids_fetched_count": len(unique_docs),
                }
            )
        elif ep.family == "sample95":
            if call.tool_name == "sim_sample":
                stream = call.arguments.get("stream")
                n = int(call.arguments.get("n", 0))
                if stream == "A":
                    n_total_A += n
                elif stream == "B":
                    n_total_B += n
            features.update(
                {
                    "n_total_A": n_total_A,
                    "n_total_B": n_total_B,
                }
            )

        last_tool_code = features["tool_name"]
        features["label"] = LABEL_MAP[label_name]
        rows.append(features)
    return rows


def make_prm_dataset(run_paths: List[str], task_paths: List[str], out_path: str) -> None:
    tasks_by_id = load_tasks(task_paths)
    rows: List[Dict[str, Any]] = []

    for run_path in run_paths:
        for ep_row in read_jsonl(run_path):
            ep = Episode(**ep_row)
            task = tasks_by_id.get(ep.task_id)
            if task is None:
                continue
            rows.extend(_build_rows_for_episode(ep, task))

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Make PRM dataset CSV from labeled runs.")
    parser.add_argument("--runs", nargs="+", required=True, help="Paths to labeled run jsonl files.")
    parser.add_argument("--tasks", nargs="+", required=True, help="Paths to task jsonl files.")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    args = parser.parse_args()
    make_prm_dataset(args.runs, args.tasks, args.out)


if __name__ == "__main__":
    main()

