"""ParsiBench CLI."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import List, Tuple

import typer

from parsibench.agents.react_agent import ReactAgent
from parsibench.agents.voi_agent import VoiAgent
from parsibench.agents.voi_prm_agent import VoiPrmAgent
from parsibench.eval.labels import label_episode
from parsibench.eval.metrics import compute_episode_metrics
from parsibench.eval.report import generate_report
from parsibench.eval.runner import run_dataset
from parsibench.tasks import generate_evmin, generate_kb, generate_sample95
from parsibench.train.make_prm_data import make_prm_dataset
from parsibench.train.train_needtool import train_needtool
from parsibench.train.train_prm import train_prm
from parsibench.utils.io import read_jsonl, write_jsonl
from parsibench.utils.schema import Episode, Task

app = typer.Typer(add_completion=False)


def _load_tasks(path: str, family: str | None = None) -> List[Task]:
    tasks = [Task(**row) for row in read_jsonl(path)]
    if family:
        tasks = [t for t in tasks if t.family == family]
    return tasks


def _expand_paths(paths: List[str]) -> List[str]:
    expanded: List[str] = []
    for p in paths:
        expanded.extend(glob.glob(p))
    return expanded


@app.command()
def gen(family: str, n: int, seed: int = 0, out: str = typer.Option(..., help="Output jsonl path")):
    if family == "kb":
        tasks = generate_kb(n, seed)
    elif family == "evmin":
        tasks = generate_evmin(n, seed)
    elif family == "sample95":
        tasks = generate_sample95(n, seed)
    else:
        raise typer.BadParameter(f"Unknown family {family}")
    write_jsonl(out, [t.model_dump() for t in tasks])
    typer.echo(f"Wrote {len(tasks)} tasks to {out}")


@app.command()
def eval(
    agent: str,
    family: str,
    data: str,
    budget: int = 6,
    out: str = typer.Option(..., help="Output run path"),
    limit: int = typer.Option(None, help="Optional limit of tasks to run"),
    verbose: bool = typer.Option(False, help="Print progress logging"),
    incremental: bool = typer.Option(True, help="Append results after each task"),
    workers: int = typer.Option(1, help="Number of parallel workers"),
    resume: bool = typer.Option(False, help="Skip tasks already present in out file"),
):
    tasks = _load_tasks(data, family=family)
    if limit is not None:
        tasks = tasks[:limit]
    if agent == "react":
        agent_obj = ReactAgent()
    elif agent == "voi":
        agent_obj = VoiAgent()
    elif agent == "voi_prm":
        agent_obj = VoiPrmAgent()
    else:
        raise typer.BadParameter(f"Unknown agent {agent}")
    episodes = run_dataset(
        agent_obj,
        tasks,
        budget,
        out_path=out,
        verbose=verbose,
        incremental=incremental,
        workers=workers,
        resume=resume,
    )
    typer.echo(f"Wrote {len(episodes)} episodes to {out}")


@app.command()
def label(data: str, run_in: str, out: str = typer.Option(..., help="Output labeled run path")):
    tasks = {t.task_id: t for t in _load_tasks(data)}
    labeled_eps = []
    for row in read_jsonl(run_in):
        ep = Episode(**row)
        task = tasks.get(ep.task_id)
        if task is None:
            continue
        ep.metrics = compute_episode_metrics(ep, task)
        ep = label_episode(ep, task)
        labeled_eps.append(ep.model_dump())
    write_jsonl(out, labeled_eps)
    typer.echo(f"Wrote labeled episodes to {out}")


@app.command("train-prm")
def train_prm_cmd(
    runs: List[str] = typer.Option(
        ...,
        "--runs",
        "-r",
        help="Glob or paths to labeled run jsonl files (repeatable)",
        show_default=False,
    ),
    tasks: List[str] = typer.Option(
        ...,
        "--tasks",
        "-t",
        help="Task jsonl files (repeatable, one per family)",
        show_default=False,
    ),
    out: str = typer.Option(..., help="Output prm pickle path"),
):
    run_paths = _expand_paths(runs)
    task_paths = _expand_paths(tasks)
    tmp_csv = Path(out).with_suffix(".csv")
    make_prm_dataset(run_paths, task_paths, str(tmp_csv))
    train_prm(str(tmp_csv), out)
    typer.echo(f"Saved PRM model to {out}")


@app.command("train-needtool")
def train_needtool_cmd(
    runs: List[str] = typer.Option(
        ...,
        "--runs",
        "-r",
        help="Glob or paths to labeled run jsonl files (repeatable)",
        show_default=False,
    ),
    tasks: List[str] = typer.Option(
        ...,
        "--tasks",
        "-t",
        help="Task jsonl files (repeatable, one per family)",
        show_default=False,
    ),
    out: str = typer.Option(..., help="Output needtool pickle path"),
):
    run_paths = _expand_paths(runs)
    task_paths = _expand_paths(tasks)
    train_needtool(run_paths, task_paths, out)
    typer.echo(f"Saved NeedTool model to {out}")


@app.command()
def report(runs: List[str], out: str = typer.Option(..., help="Output directory for report")):
    run_paths = _expand_paths(runs)
    generate_report(run_paths, out)
    typer.echo(f"Report written to {out}")


def main():
    app()


if __name__ == "__main__":
    main()
