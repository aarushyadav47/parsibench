"""Tool-call PRM wrapper."""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional

import joblib


class PRMModel:
    def __init__(
        self,
        model_path: str = "artifacts/prm.pkl",
        feature_spec_path: str = "artifacts/prm_feature_spec.json",
    ):
        self.model = self._load_model(model_path)
        self.feature_names = self._load_feature_spec(feature_spec_path)

    @staticmethod
    def _load_model(path: str):
        if not os.path.exists(path):
            return None
        try:
            return joblib.load(path)
        except Exception:
            return None

    @staticmethod
    def _load_feature_spec(path: str) -> Optional[List[str]]:
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("feature_names") or data
        except Exception:
            return None

    def score(self, features: Any) -> float:
        if self.model is None:
            return 0.0

        vector = [features] if isinstance(features, (int, float)) else features
        if isinstance(features, dict) and self.feature_names:
            vector = [features.get(name, 0) for name in self.feature_names]

        try:
            proba = self.model.predict_proba([vector])[0]
            return float(proba[0]) if hasattr(self.model, "classes_") else float(proba)
        except Exception:
            return 0.0

