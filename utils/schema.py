"""Core data schemas for ParsiBench."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel

Family = Literal["kb", "evmin", "sample95"]
ToolName = Literal["kb_search", "kb_fetch", "sim_sample"]


class _JsonMixin(BaseModel):
    """Adds stable JSON serialization."""

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), sort_keys=True)


class Task(_JsonMixin):
    task_id: str
    family: Family
    prompt: str
    gold: Dict[str, Any]
    cert: Dict[str, Any]
    tool_state: Dict[str, Any]


class ToolCall(_JsonMixin):
    call_id: str
    tool_name: ToolName
    arguments: Dict[str, Any]
    step_index: int
    timestamp_ms: int


class ToolResult(_JsonMixin):
    call_id: str
    tool_name: ToolName
    output: str
    step_index: int


class StepTrace(_JsonMixin):
    step_index: int
    model_response_id: Optional[str]
    raw_output_items: List[Dict[str, Any]]
    chosen_action: Dict[str, Any]
    tool_call: Optional[ToolCall]
    tool_result: Optional[ToolResult]
    answer: Optional[Dict[str, Any]]
    labels: Optional[Dict[str, Any]]
    meta: Dict[str, Any]


class Episode(_JsonMixin):
    task_id: str
    family: str
    prompt_hash: str
    start_ts_ms: int
    end_ts_ms: int
    budget: int
    steps: List[StepTrace]
    final_answer: Optional[Dict[str, Any]]
    success: bool
    metrics: Dict[str, Any]

