"""Reactive baseline agent."""

from __future__ import annotations

from typing import Dict, Any

from parsibench.agents.llm_core import (
    BASELINE_SYSTEM_INSTRUCTIONS,
    MODEL_NAME,
    run_tool_loop,
)
from parsibench.agents.evmin_schema import EV_MIN_SCHEMA
from parsibench.utils.schema import Episode, Task


class ReactAgent:
    def __init__(self, model: str = MODEL_NAME, max_steps: int = 10):
        self.model = model
        self.max_steps = max_steps

    def solve(self, task: Task, tools_runtime: Dict[str, Any], budget: int) -> Episode:
        instructions = BASELINE_SYSTEM_INSTRUCTIONS
        # Use structured schema for evmin final answer
        if task.family == "evmin":
            instructions = instructions + "\nFor evmin, output slot_values JSON."
            # we pass text_schema in run_tool_loop via instructions change handled in llm_core force_final_answer_call
        return run_tool_loop(
            task,
            tool_impls=tools_runtime,
            max_steps=self.max_steps,
            budget=budget,
            instructions=instructions,
            model=self.model,
            mode="react",
        )

