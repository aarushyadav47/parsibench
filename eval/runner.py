"""Dataset runner utilities."""

from __future__ import annotations

import json
from typing import Iterable, List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pathlib import Path

from parsibench.eval.metrics import compute_episode_metrics
from parsibench.tools import KB, Sim
from parsibench.utils.io import write_jsonl
from parsibench.utils.schema import Episode, Task


def _build_tool_runtime(task: Task) -> Dict[str, Any]:
    runtime: Dict[str, Any] = {}

    if "kb_docs" in task.tool_state:
        kb = KB(task.tool_state["kb_docs"])
        runtime["kb_search"] = kb.search
        runtime["kb_fetch"] = kb.fetch

    if "streams" in task.tool_state:
        sim = Sim(task.tool_state["streams"])
        counters = {"A": 0, "B": 0}

        def sim_sample(stream: str, n: int) -> Dict[str, Any]:
            call_idx = counters.get(stream, 0)
            result = sim.sample(stream=stream, n=n, call_index=call_idx)
            counters[stream] = call_idx + 1
            return result

        runtime["sim_sample"] = sim_sample

    return runtime


def run_dataset(
    agent: Any,
    tasks: Iterable[Task],
    budget: int,
    out_path: str | None = None,
    verbose: bool = False,
    incremental: bool = True,
    workers: int = 1,
    resume: bool = False,
) -> List[Episode]:
    episodes: List[Episode] = []
    task_list = list(tasks)
    completed_ids = set()
    if resume and out_path and Path(out_path).exists():
        try:
            with Path(out_path).open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        tid = obj.get("task_id")
                        if tid:
                            completed_ids.add(tid)
                    except Exception:
                        continue
        except Exception:
            if verbose:
                print("[run_dataset] warning: could not load existing out file for resume", flush=True)

    total = len(task_list)
    out_fh = None
    if out_path and incremental:
        out_fh = open(out_path, "a", encoding="utf-8")
    lock = threading.Lock()

    def _run_one(index_task: Tuple[int, Task]) -> Tuple[int, Episode]:
        idx, task = index_task
        if task.task_id in completed_ids:
            if verbose:
                print(f"[run_dataset] skipping completed task {idx+1}/{total} ({task.task_id})", flush=True)
            return idx, None  # placeholder
        if verbose:
            print(f"[run_dataset] starting task {idx+1}/{total} ({task.task_id})", flush=True)
        tools_runtime = _build_tool_runtime(task)
        episode: Episode = agent.solve(task, tools_runtime, budget)
        metrics = compute_episode_metrics(episode, task)
        episode.metrics = metrics
        episode.success = bool(metrics.get("success", False))
        if out_fh:
            with lock:
                out_fh.write(json.dumps(episode.model_dump()) + "\n")
                out_fh.flush()
        if verbose:
            print(
                f"[run_dataset] finished task {idx+1}/{total} "
                f"calls={metrics.get('tool_calls_total', 0)} success={metrics.get('success', False)}",
                flush=True,
            )
        return idx, episode

    if workers <= 1:
        for pair in map(_run_one, enumerate(task_list)):
            if pair[1] is not None:
                episodes.append(pair[1])
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_run_one, item) for item in enumerate(task_list)]
            for fut in as_completed(futures):
                idx, ep = fut.result()
                if ep is not None:
                    episodes.append(ep)

    episodes.sort(key=lambda ep: task_list.index(next(t for t in task_list if t.task_id == ep.task_id)))

    if out_path and not incremental:
        write_jsonl(out_path, [ep.model_dump() for ep in episodes])

    if out_fh:
        out_fh.close()

    return episodes

