"""LLM core utilities and tool-calling loop."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Sequence

from openai import OpenAI

from pathlib import Path
import re
import threading

from parsibench.utils.rng import stable_hash_str
from parsibench.utils.schema import Episode, StepTrace, Task, ToolCall, ToolResult

_client_local = threading.local()


def _get_client() -> OpenAI:
    if not hasattr(_client_local, "client"):
        _client_local.client = OpenAI()
    return _client_local.client

MODEL_NAME = "gpt-4.1-mini"


def _log_error(message: str) -> None:
    try:
        log_path = Path("/tmp/parsibench_errors.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.write(f"[{ts}] {message}\n")
    except Exception:
        # swallow logging errors
        pass


def _parse_output_text(raw: str) -> Dict[str, Any]:
    """Robustly parse output_text that sometimes arrives as back-to-back JSON objects."""
    if not raw:
        return {}
    # First try direct
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Try wrapping split between objects
    try:
        combo = "[" + raw.replace("}{", "},{") + "]"
        arr = json.loads(combo)
        if isinstance(arr, list) and arr:
            return arr[-1]  # take last object
    except Exception:
        pass
    # Try extracting all brace-delimited JSON blobs and parse last one
    try:
        matches = re.findall(r"\{.*?\}", raw)
        for candidate in reversed(matches):
            try:
                return json.loads(candidate)
            except Exception:
                continue
    except Exception:
        pass
    # If all fails, raise
    raise json.JSONDecodeError("Unable to parse output_text", raw, 0)


def _ensure_call_outputs(input_items: List[Dict[str, Any]]) -> None:
    """Make sure every function_call has a corresponding function_call_output."""
    call_ids = []
    output_ids = set()
    for item in input_items:
        item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        call_id = item.get("call_id") if isinstance(item, dict) else getattr(item, "call_id", None)
        if item_type == "function_call" and call_id:
            call_ids.append(call_id)
        if item_type == "function_call_output" and call_id:
            output_ids.add(call_id)
    missing = [cid for cid in call_ids if cid not in output_ids]
    for cid in missing:
        _log_error(f"Missing function_call_output for {cid}; inserting stub.")
        input_items.append(
            {
                "type": "function_call_output",
                "call_id": cid,
                "output": json.dumps({"error": "missing_output_stub"}),
            }
        )

ANSWER_SCHEMA = {
    "type": "json_schema",
    "name": "parsibench_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "final_answer": {"type": "string"},
            "evidence_refs": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "stop_reason": {
                "type": "string",
                "enum": ["enough_evidence", "budget_exhausted", "tool_failed", "guess"],
            },
        },
        "required": ["final_answer", "evidence_refs", "confidence", "stop_reason"],
        "additionalProperties": False,
    },
}

BASELINE_SYSTEM_INSTRUCTIONS = (
    "You are an evaluation agent in a sandbox. You may call tools to get information. "
    "Call at most one tool per turn. When you answer, you must output JSON with:\n"
    "- final_answer (string)\n- evidence_refs (list of strings)\n- confidence (0..1)\n"
    "- stop_reason\nRules for final_answer:\n"
    "- For kb: final_answer must be ONLY the exact answer string (no extra words).\n"
    "- For evmin: output JSON with slot_values mapping (s1, s2, s3...), evidence_refs of doc_ids, confidence, stop_reason.\n"
    "- For sample95: final_answer must be \"A\" or \"B\" only.\n"
    "Do not fabricate evidence. Only cite doc ids returned by tools or sim stream ids you sampled."
)


def build_tools() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": "kb_search",
            "description": "Search KB docs by keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "k": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "kb_fetch",
            "description": "Fetch an exact span from a KB doc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "start": {"type": "integer", "minimum": 0},
                    "end": {"type": "integer", "minimum": 1},
                },
                "required": ["doc_id", "start", "end"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "sim_sample",
            "description": "Sample n observations from stream A or B.",
            "parameters": {
                "type": "object",
                "properties": {
                    "stream": {"type": "string", "enum": ["A", "B"]},
                    "n": {"type": "integer", "minimum": 1, "maximum": 200},
                },
                "required": ["stream", "n"],
                "additionalProperties": False,
            },
        },
    ]


def _format_text_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    return {"format": schema}


def call_model(
    input_items: Sequence[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    instructions: str,
    *,
    model: str = MODEL_NAME,
    text_schema: Optional[Dict[str, Any]] = None,
    allowed_tools: Optional[List[str]] = None,
):
    client = _get_client()
    payload: Dict[str, Any] = {
        "model": model,
        "input": list(input_items),
        "instructions": instructions,
        "store": False,
    }
    if tools is not None:
        payload["tools"] = (
            [t for t in tools if t.get("name") in allowed_tools]
            if allowed_tools is not None
            else tools
        )
    if text_schema is not None:
        payload["text"] = _format_text_schema(text_schema)
    return client.responses.create(**payload)


def _item_to_dict(item: Any) -> Dict[str, Any]:
    if hasattr(item, "model_dump"):
        d = item.model_dump()
        if not d.get("type") and hasattr(item, "type"):
            d["type"] = getattr(item, "type")
        if d.get("type") is None and {"name", "arguments", "call_id"} <= set(d.keys()):
            d["type"] = "function_call"
        return d
    if isinstance(item, dict):
        return item
    try:
        return dict(item)
    except Exception:
        return {"value": str(item)}


def _extract_function_calls(output_items: List[Any]) -> List[Any]:
    calls = []
    for item in output_items:
        item_type = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else None)
        if item_type == "function_call":
            calls.append(item)
    return calls


def _function_call_fields(call: Any) -> Dict[str, Any]:
    def _parse_args(raw: Any):
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except Exception:
                return {}
        return raw or {}

    if isinstance(call, dict):
        return {
            "name": call.get("name"),
            "arguments": _parse_args(call.get("arguments", {})),
            "call_id": call.get("call_id"),
        }
    return {
        "name": getattr(call, "name", None),
        "arguments": _parse_args(getattr(call, "arguments", {})),
        "call_id": getattr(call, "call_id", None),
    }


def force_final_answer_call(
    input_items: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    instructions: str,
    *,
    model: str = MODEL_NAME,
    family: str | None = None,
) -> Dict[str, Any]:
    text_schema = ANSWER_SCHEMA
    if family == "evmin":
        try:
            from parsibench.agents.evmin_schema import EV_MIN_SCHEMA
            text_schema = EV_MIN_SCHEMA
        except Exception:
            text_schema = ANSWER_SCHEMA
    response = call_model(
        input_items,
        tools=[],
        instructions=(
            instructions
            + "\nReturn final answer JSON now. Do not call tools. "
            "final_answer must be ONLY the concise answer string (no extra words)."
        ),
        model=model,
        text_schema=text_schema,
    )
    if getattr(response, "output_text", None):
        try:
            return _parse_output_text(response.output_text)
        except Exception as e:
            _log_error(f"force_final_answer_call JSON decode failed: {e}; raw={response.output_text}")
            raise
    return {}


def run_tool_loop(
    task: Task,
    tool_impls: Dict[str, Any],
    max_steps: int,
    budget: int,
    *,
    instructions: str = BASELINE_SYSTEM_INSTRUCTIONS,
    model: str = MODEL_NAME,
    mode: str = "react",
    allowed_tools: Optional[List[str]] = None,
) -> Episode:
    tools = build_tools()
    input_items: List[Dict[str, Any]] = [{"role": "user", "content": task.prompt}]
    steps: List[StepTrace] = []
    remaining_budget = budget
    final_answer: Optional[Dict[str, Any]] = None
    start_ts_ms = int(time.time() * 1000)

    for step_index in range(max_steps):
        _ensure_call_outputs(input_items)
        try:
            response = call_model(
                input_items=input_items,
                tools=tools,
                instructions=instructions,
                model=model,
                allowed_tools=allowed_tools,
            )
        except Exception as e:
            _log_error(f"call_model failed at step {step_index}: {e}")
            break
        raw_output_items = list(getattr(response, "output", []))
        raw_output_items_dicts = [_item_to_dict(i) for i in raw_output_items]
        input_items.extend(raw_output_items_dicts)

        function_calls = _extract_function_calls(raw_output_items_dicts)
        if not function_calls:
            if getattr(response, "output_text", None):
                try:
                    final_answer = _parse_output_text(response.output_text)
                except Exception as e:
                    _log_error(
                        f"run_tool_loop final_answer decode failed: {e}; raw={response.output_text}"
                    )
                    final_answer = force_final_answer_call(
                        input_items, tools, instructions, model=model, family=task.family
                    )
            steps.append(
                StepTrace(
                    step_index=step_index,
                    model_response_id=getattr(response, "id", None),
                    raw_output_items=[_item_to_dict(i) for i in raw_output_items],
                    chosen_action={"type": "answer"},
                    tool_call=None,
                    tool_result=None,
                    answer=final_answer,
                    labels=None,
                    meta={"mode": mode},
                )
            )
            break

        for fc in function_calls:
            fc_fields = _function_call_fields(fc)
            tool_name = fc_fields["name"]
            args = fc_fields["arguments"] or {}
            call_id = fc_fields["call_id"] or f"call_{step_index}"
            now_ms = int(time.time() * 1000)
            try:
                tool_output = tool_impls[tool_name](**args)
            except Exception as e:
                _log_error(f"tool execution failed for {tool_name} with args={args}: {e}")
                tool_output = {"error": str(e)}
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(tool_output),
                }
            )
            steps.append(
                StepTrace(
                    step_index=step_index,
                    model_response_id=getattr(response, "id", None),
                    raw_output_items=raw_output_items_dicts,
                    chosen_action={"type": "function_call", "name": tool_name, "arguments": args},
                    tool_call=ToolCall(
                        call_id=call_id,
                        tool_name=tool_name,
                        arguments=args,
                        step_index=step_index,
                        timestamp_ms=now_ms,
                    ),
                    tool_result=ToolResult(
                        call_id=call_id,
                        tool_name=tool_name,
                        output=json.dumps(tool_output),
                        step_index=step_index,
                    ),
                    answer=None,
                    labels=None,
                    meta={"mode": mode},
                )
            )
            remaining_budget -= 1

            if remaining_budget <= 0:
                final_answer = force_final_answer_call(
                    input_items, tools, instructions, model=model
                )
                steps.append(
                    StepTrace(
                        step_index=step_index + 1,
                        model_response_id=None,
                        raw_output_items=[],
                        chosen_action={"type": "forced_answer"},
                        tool_call=None,
                        tool_result=None,
                        answer=final_answer,
                        labels=None,
                        meta={"mode": mode, "reason": "budget_exhausted"},
                    )
                )
                break

        if final_answer is not None:
            break

    end_ts_ms = int(time.time() * 1000)
    success = bool(final_answer)
    prompt_hash = str(stable_hash_str(task.prompt))

    return Episode(
        task_id=task.task_id,
        family=task.family,
        prompt_hash=prompt_hash,
        start_ts_ms=start_ts_ms,
        end_ts_ms=end_ts_ms,
        budget=budget,
        steps=steps,
        final_answer=final_answer,
        success=success,
        metrics={},
    )

