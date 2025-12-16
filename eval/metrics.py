"""Metric computations for episodes."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from parsibench.tasks.base import normalize_text
from parsibench.utils.schema import Episode, Task, ToolCall


def _canonical_args(call: ToolCall) -> Tuple[str, Tuple[Any, ...]]:
    args = call.arguments or {}
    if call.tool_name == "kb_search":
        query = str(args.get("query", "")).strip().lower()
        k = int(args.get("k", 5))
        return call.tool_name, (query, k)
    if call.tool_name == "kb_fetch":
        return call.tool_name, (
            str(args.get("doc_id", "")),
            int(args.get("start", 0)),
            int(args.get("end", 0)),
        )
    if call.tool_name == "sim_sample":
        return call.tool_name, (
            str(args.get("stream", "")).upper(),
            int(args.get("n", 0)),
        )
    return call.tool_name, tuple(sorted(args.items()))


def _extract_tool_calls(episode: Episode) -> List[ToolCall]:
    calls: List[ToolCall] = []
    for step in episode.steps:
        if step.tool_call:
            calls.append(step.tool_call)
    return calls


def _final_answer_text(episode: Episode) -> str:
    ans = episode.final_answer or {}
    if isinstance(ans, dict):
        return str(ans.get("final_answer", ans.get("answer", "")))
    return str(ans)


def _evmin_expected_map(task: Task) -> Dict[str, str]:
    expected: Dict[str, str] = {}
    spans = task.cert.get("required_spans", []) if isinstance(task.cert, dict) else []
    doc_map = {d["doc_id"]: d.get("text", "") for d in task.tool_state.get("kb_docs", [])}
    for span in spans:
        slot = span.get("slot")
        doc_id = span.get("doc_id")
        start = span.get("start", 0)
        end = span.get("end", 0)
        text = doc_map.get(doc_id, "")
        val = text[start:end]
        if slot:
            expected[slot] = normalize_text(val)
    return expected


def compute_episode_metrics(episode: Episode, task: Task) -> Dict[str, Any]:
    tool_calls = _extract_tool_calls(episode)
    seen = set()
    redundancy_count = 0
    for call in tool_calls:
        key = _canonical_args(call)
        if key in seen:
            redundancy_count += 1
        seen.add(key)

    success = False
    if task.family == "kb":
        success = normalize_text(_final_answer_text(episode)) == normalize_text(task.gold.get("answer", ""))
    elif task.family == "evmin":
        ans = episode.final_answer or {}
        slot_map = ans.get("slot_values") if isinstance(ans, dict) else None
        expected_map = _evmin_expected_map(task)
        if slot_map and expected_map:
            norm_ans = {k: normalize_text(v) for k, v in slot_map.items()}
            success = all(norm_ans.get(k) == v for k, v in expected_map.items())
        else:
            success = normalize_text(_final_answer_text(episode)) == normalize_text(task.gold.get("answer", ""))
    elif task.family == "sample95":
        success = (_final_answer_text(episode).strip() == task.gold.get("winner"))

    return {
        "tool_calls_total": len(tool_calls),
        "redundancy_count": redundancy_count,
        "success": success,
        "avoidable_calls_total": 0,
        "necessary_calls_total": 0,
    }
