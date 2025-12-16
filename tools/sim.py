"""Deterministic sampling simulator."""

from __future__ import annotations

from typing import Dict

import numpy as np

from parsibench.utils.rng import rng_from


class Sim:
    def __init__(self, streams: Dict):
        self.streams = streams

    def sample(self, stream: str, n: int, call_index: int) -> Dict:
        if stream not in self.streams:
            raise ValueError(f"Unknown stream {stream}")

        cfg = self.streams[stream]
        rng = rng_from(cfg["seed"], stream, call_index)
        n_int = int(n)
        samples = rng.normal(loc=cfg["mu"], scale=cfg["sigma"], size=n_int)
        mean = float(np.mean(samples))
        std = float(np.std(samples, ddof=0))

        shown_samples = samples[:10] if n_int > 50 else samples
        return {
            "stream": stream,
            "n": n_int,
            "samples": shown_samples.tolist(),
            "mean": mean,
            "std": std,
        }

