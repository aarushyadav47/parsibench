"""Sample95 noisy sampling task family."""

from __future__ import annotations

import math
from typing import Dict, List

from parsibench.utils.rng import rng_from
from parsibench.utils.schema import Task


def _compute_n_min(delta: float, sigma: float, gap: float, alpha: float = 0.05) -> int:
    z = 1.96  # approx z_{1-alpha/2}
    denom = max(gap - delta, 1e-6)
    return max(10, min(200, math.ceil((z * sigma / denom) ** 2)))


def generate_sample95(n: int, seed: int) -> List[Task]:
    rng = rng_from("sample95_tasks", seed)
    tasks: List[Task] = []

    for idx in range(n):
        task_seed = int(rng.integers(0, 2**32 - 1))
        local_rng = rng_from("sample95_local", task_seed)

        delta = float(local_rng.choice([0.5, 0.75, 1.0]))
        sigma = float(local_rng.choice([1.0, 2.0]))
        gap = float(delta + local_rng.uniform(0.1, 1.0))
        winner_is_a = bool(local_rng.integers(0, 2))
        base = float(local_rng.uniform(-1.0, 1.0))

        if winner_is_a:
            mu_a = base + gap / 2
            mu_b = base - gap / 2
            winner = "A"
        else:
            mu_a = base - gap / 2
            mu_b = base + gap / 2
            winner = "B"

        n_min = _compute_n_min(delta=delta, sigma=sigma, gap=abs(mu_a - mu_b))
        seed_a = int(local_rng.integers(0, 2**32 - 1))
        seed_b = int(local_rng.integers(0, 2**32 - 1))

        prompt = (
            "Decide whether A or B has higher mean by at least delta with 95% confidence. "
            "You may sample using sim_sample. Stop once confident. "
            "You cannot infer the answer without calling sim_sample. If you do not sample, your answer will be wrong. "
            "Procedure: sample A and B with n=10, compute the mean difference and standard error, then keep sampling in increments of 5 until confident, and stop as soon as confident."
        )

        tasks.append(
            Task(
                task_id=f"sample95_{idx}",
                family="sample95",
                prompt=prompt,
                gold={"winner": winner, "mu_a": mu_a, "mu_b": mu_b},
                cert={
                    "delta": delta,
                    "alpha": 0.05,
                    "n_min": n_min,
                    "mu_a": mu_a,
                    "mu_b": mu_b,
                    "sigma": sigma,
                    "winner": winner,
                },
                tool_state={
                    "streams": {
                        "A": {"mu": mu_a, "sigma": sigma, "seed": seed_a},
                        "B": {"mu": mu_b, "sigma": sigma, "seed": seed_b},
                    }
                },
            )
        )

    return tasks

