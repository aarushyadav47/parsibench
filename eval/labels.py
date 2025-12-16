"""Certified labeling for tool calls."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from parsibench.eval.metrics import _canonical_args, _extract_tool_calls
from parsibench.utils.schema import Episode, Task, ToolCall


def _overlaps(span: Dict[str, Any], start: int, end: int) -> bool:
    return not (end <= span["start"] or start >= span["end"])


def _label_kb_call(call: ToolCall, cert: Dict[str, Any], fetched_required: bool, redundant: bool) -> str:
    if redundant:
        return "redundant"

    requires_tool = bool(cert.get("requires_tool", False))
    if not requires_tool:
        return "avoidable"

    required_span = cert.get("required_span")
    required_doc = cert.get("required_doc_id")

    if fetched_required:
        return "avoidable"

    if call.tool_name == "kb_search":
        return "necessary"

    if call.tool_name == "kb_fetch":
        if required_doc and call.arguments.get("doc_id") != required_doc:
            return "avoidable"
        if required_span:
            if _overlaps(required_span, call.arguments.get("start", 0), call.arguments.get("end", 0)):
                return "necessary"
        else:
            return "necessary"
    return "avoidable"


def _label_evmin_call(call: ToolCall, remaining_spans: List[Dict[str, Any]], redundant: bool) -> str:
    if redundant:
        return "redundant"
    if call.tool_name != "kb_fetch":
        return "avoidable"

    for span in list(remaining_spans):
        if call.arguments.get("doc_id") == span["doc_id"] and _overlaps(span, call.arguments.get("start", 0), call.arguments.get("end", 0)):
            remaining_spans.remove(span)
            return "necessary"
    return "avoidable"


def _label_sample95_call(call: ToolCall, n_total: int, n_min: int, redundant: bool) -> Tuple[str, int]:
    n_total += int(call.arguments.get("n", 0))
    if redundant:
        return "redundant", n_total
    if n_total <= n_min:
        return "necessary", n_total
    return "avoidable", n_total


def label_episode(episode: Episode, task: Task) -> Episode:
    tool_calls = _extract_tool_calls(episode)
    seen = set()
    counts = {"necessary": 0, "avoidable": 0, "redundant": 0}

    fetched_required = False
    remaining_spans = task.cert.get("required_spans", [])[:] if isinstance(task.cert, dict) else []
    n_total = 0
    n_min = int(task.cert.get("n_min", 0)) if isinstance(task.cert, dict) else 0

    for step in episode.steps:
        call = step.tool_call
        if not call:
            continue
        canonical = _canonical_args(call)
        redundant = canonical in seen
        seen.add(canonical)

        label = "avoidable"
        if task.family == "kb":
            label = _label_kb_call(call, task.cert, fetched_required, redundant)
            if label == "necessary" and call.tool_name == "kb_fetch":
                fetched_required = True
        elif task.family == "evmin":
            label = _label_evmin_call(call, remaining_spans, redundant)
        elif task.family == "sample95":
            label, n_total = _label_sample95_call(call, n_total, n_min, redundant)
        if label == "redundant":
            counts["redundant"] += 1
            counts["avoidable"] += 1
        elif label == "necessary":
            counts["necessary"] += 1
        else:
            counts["avoidable"] += 1

        step.labels = {"tool_call_label": label}

    episode.metrics["avoidable_calls_total"] = counts["avoidable"]
    episode.metrics["necessary_calls_total"] = counts["necessary"]
    episode.metrics["redundancy_count"] = counts["redundant"]
    return episode

