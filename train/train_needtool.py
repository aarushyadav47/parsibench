"""Train NeedTool classifier."""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from parsibench.eval.labels import _overlaps
from parsibench.utils.io import read_jsonl
from parsibench.utils.rng import stable_hash_str
from parsibench.utils.schema import Episode, Task

FAMILY_MAP = {"kb": 0, "evmin": 1, "sample95": 2}
TOOL_MAP = {"kb_search": 0, "kb_fetch": 1, "sim_sample": 2}


def load_tasks(task_paths: List[str]) -> Dict[str, Task]:
    tasks: Dict[str, Task] = {}
    for path in task_paths:
        for row in read_jsonl(path):
            task = Task(**row)
            tasks[task.task_id] = task
    return tasks


def _state_label(task: Task, remaining_spans: List[Dict[str, Any]], n_total: int) -> int:
    if task.family == "kb":
        return 1 if task.cert.get("requires_tool", False) else 0
    if task.family == "evmin":
        return 1 if remaining_spans else 0
    if task.family == "sample95":
        return 1 if n_total < task.cert.get("n_min", 0) else 0
    return 1


def _build_rows(ep: Episode, task: Task) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    remaining_spans = task.cert.get("required_spans", [])[:] if isinstance(task.cert, dict) else []
    n_total = 0
    tools_used = 0

    for step in ep.steps:
        features: Dict[str, Any] = {
            "task_id": ep.task_id,
            "family": FAMILY_MAP.get(ep.family, -1),
            "step_index": step.step_index,
            "tools_used_count": tools_used,
            "last_tool_name": TOOL_MAP.get(step.tool_call.tool_name, -1) if step.tool_call else -1,
            "n_total": n_total,
            "remaining_spans": len(remaining_spans),
        }
        label = _state_label(task, remaining_spans, n_total)
        features["label"] = label
        rows.append(features)

        if step.tool_call:
            tools_used += 1
            if task.family == "sample95" and step.tool_call.tool_name == "sim_sample":
                n_total += int(step.tool_call.arguments.get("n", 0))
            if task.family == "evmin" and step.tool_call.tool_name == "kb_fetch":
                for span in list(remaining_spans):
                    if step.tool_call.arguments.get("doc_id") == span.get("doc_id") and _overlaps(
                        span, step.tool_call.arguments.get("start", 0), step.tool_call.arguments.get("end", 0)
                    ):
                        remaining_spans.remove(span)
                        break

    return rows


def build_dataset(run_paths: List[str], task_paths: List[str]) -> pd.DataFrame:
    tasks_by_id = load_tasks(task_paths)
    rows: List[Dict[str, Any]] = []
    for run_path in run_paths:
        for ep_row in read_jsonl(run_path):
            ep = Episode(**ep_row)
            task = tasks_by_id.get(ep.task_id)
            if task is None:
                continue
            rows.extend(_build_rows(ep, task))
    return pd.DataFrame(rows)


def _train_test_split(df: pd.DataFrame):
    test_mask = df["task_id"].apply(lambda x: stable_hash_str(str(x)) % 5 == 0)
    return df[~test_mask], df[test_mask]


def train_needtool(run_paths: List[str], task_paths: List[str], out_path: str) -> None:
    df = build_dataset(run_paths, task_paths)
    train_df, test_df = _train_test_split(df)

    feature_cols = [c for c in df.columns if c not in {"label", "task_id"}]
    X_train = train_df[feature_cols].fillna(0).to_numpy()
    y_train = train_df["label"].to_numpy()
    X_test = test_df[feature_cols].fillna(0).to_numpy()
    y_test = test_df["label"].to_numpy()

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("NeedTool classification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, out_path)


def main():
    parser = argparse.ArgumentParser(description="Train NeedTool classifier.")
    parser.add_argument("--runs", nargs="+", required=True, help="Paths to labeled run jsonl files.")
    parser.add_argument("--tasks", nargs="+", required=True, help="Paths to task jsonl files.")
    parser.add_argument("--out", required=True, help="Output pickle path.")
    args = parser.parse_args()
    train_needtool(args.runs, args.tasks, args.out)


if __name__ == "__main__":
    main()

