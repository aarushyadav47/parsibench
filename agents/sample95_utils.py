"""Heuristics for sample95 stopping."""

from __future__ import annotations

import math
from typing import Dict


def should_stop_after(samples_a: Dict, samples_b: Dict, delta: float, alpha: float = 0.05) -> bool:
    # simple z-test heuristic: stop if means differ by > delta with CI gap
    mean_a = samples_a.get("mean", 0.0)
    mean_b = samples_b.get("mean", 0.0)
    n_a = samples_a.get("n", 1)
    n_b = samples_b.get("n", 1)
    var_a = (samples_a.get("std", 1.0) ** 2) or 1.0
    var_b = (samples_b.get("std", 1.0) ** 2) or 1.0
    se = math.sqrt(var_a / n_a + var_b / n_b)
    gap = abs(mean_a - mean_b)
    z = 1.96  # approximate
    return gap - delta > z * se

