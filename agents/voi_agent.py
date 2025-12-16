"""Value-of-Information agent with simple NeedTool heuristic."""

from __future__ import annotations

import time
from typing import Any, Dict

from parsibench.agents.llm_core import (
    BASELINE_SYSTEM_INSTRUCTIONS,
    MODEL_NAME,
    force_final_answer_call,
    run_tool_loop,
)
from parsibench.utils.rng import stable_hash_str
from parsibench.utils.schema import Episode, StepTrace, Task


class NeedToolHeuristic:
    def predict(self, task: Task) -> bool:
        if task.family == "kb":
            return bool(task.cert.get("requires_tool", True))
        if task.family in {"evmin", "sample95"}:
            return True
        return True


def _forced_answer_episode(task: Task, instructions: str, model: str, mode: str) -> Episode:
    start_ts = int(time.time() * 1000)
    answer = force_final_answer_call(
        input_items=[{"role": "user", "content": task.prompt}],
        tools=[],
        instructions=instructions,
        model=model,
    )
    end_ts = int(time.time() * 1000)
    step = StepTrace(
        step_index=0,
        model_response_id=None,
        raw_output_items=[],
        chosen_action={"type": "forced_answer"},
        tool_call=None,
        tool_result=None,
        answer=answer,
        labels=None,
        meta={"mode": mode, "reason": "gate_stop"},
    )
    return Episode(
        task_id=task.task_id,
        family=task.family,
        prompt_hash=str(stable_hash_str(task.prompt)),
        start_ts_ms=start_ts,
        end_ts_ms=end_ts,
        budget=0,
        steps=[step],
        final_answer=answer,
        success=bool(answer),
        metrics={},
    )


class VoiAgent:
    def __init__(self, model: str = MODEL_NAME, max_steps: int = 10):
        self.model = model
        self.max_steps = max_steps
        self.needtool = NeedToolHeuristic()

    def solve(self, task: Task, tools_runtime: Dict[str, Any], budget: int) -> Episode:
        if not self.needtool.predict(task):
            return _forced_answer_episode(task, BASELINE_SYSTEM_INSTRUCTIONS, self.model, mode="voi")

        return run_tool_loop(
            task,
            tool_impls=tools_runtime,
            max_steps=self.max_steps,
            budget=budget,
            instructions=BASELINE_SYSTEM_INSTRUCTIONS,
            model=self.model,
            mode="voi",
        )

