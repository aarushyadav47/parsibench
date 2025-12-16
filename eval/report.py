"""Reporting utilities for runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import pandas as pd

from parsibench.utils.io import read_jsonl


def _aggregate_run(path: str) -> Dict[str, Any]:
    rows = read_jsonl(path)
    if not rows:
        return {}
    base = os.path.basename(path)
    parts = base.split("_")
    agent = parts[0]
    family = parts[1] if len(parts) > 1 else "unknown"

    success_rate = sum(1 for r in rows if r.get("metrics", {}).get("success")) / len(rows)
    avg_tool_calls = sum(r.get("metrics", {}).get("tool_calls_total", 0) for r in rows) / len(rows)
    avg_avoidable = sum(r.get("metrics", {}).get("avoidable_calls_total", 0) for r in rows) / len(rows)

    return {
        "agent": agent,
        "family": family,
        "path": path,
        "success_rate": success_rate,
        "avg_tool_calls": avg_tool_calls,
        "avg_avoidable": avg_avoidable,
    }


def generate_report(run_paths: List[str], out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    aggregates = [_aggregate_run(p) for p in run_paths]
    aggregates = [a for a in aggregates if a]
    if not aggregates:
        return

    df = pd.DataFrame(aggregates)

    plt.figure(figsize=(6, 4))
    for _, row in df.iterrows():
        plt.scatter(row["avg_tool_calls"], row["success_rate"], label=f"{row['agent']}-{row['family']}")
    plt.xlabel("Avg tool calls")
    plt.ylabel("Success rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "success_vs_toolcalls.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    for _, row in df.iterrows():
        plt.scatter(row["avg_avoidable"], row["success_rate"], label=f"{row['agent']}-{row['family']}")
    plt.xlabel("Avg avoidable calls")
    plt.ylabel("Success rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "success_vs_avoidable.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    df_group = df.groupby("agent")["avg_avoidable"].mean()
    df_group.plot(kind="bar")
    plt.ylabel("Avg avoidable calls")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "avoidable_by_agent.png")
    plt.close()

    summary_lines = ["# Run Summary"]
    for _, row in df.iterrows():
        summary_lines.append(
            f"- {row['agent']} / {row['family']}: success={row['success_rate']:.2f}, "
            f"avg_tool_calls={row['avg_tool_calls']:.2f}, avg_avoidable={row['avg_avoidable']:.2f}"
        )
    (Path(out_dir) / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

