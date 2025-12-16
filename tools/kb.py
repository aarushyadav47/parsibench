"""Deterministic knowledge base search and fetch."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class KB:
    def __init__(self, docs: List[Dict]):
        self.docs = docs
        self.doc_map = {doc["doc_id"]: doc for doc in docs}
        texts = [doc.get("text", "") for doc in docs]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=5000,
        )
        self.tfidf_matrix = (
            self.vectorizer.fit_transform(texts) if texts else None
        )

    def search(self, query: str, k: int = 5) -> Dict:
        if not self.docs:
            return {"hits": []}

        k = int(k)
        k = max(1, min(k, 10, len(self.docs)))
        query_vec = self.vectorizer.transform([query])
        scores = (self.tfidf_matrix @ query_vec.T).toarray().ravel()

        ranked = sorted(
            zip(scores, self.docs),
            key=lambda pair: (-pair[0], pair[1]["doc_id"]),
        )
        hits = []
        # consider top 3 for better hit coverage
        for score, doc in ranked[: min(3, len(self.docs))]:
            text = doc.get("text", "")
            idx = self._estimate_hit_position(query, text)
            snippet_start = max(0, idx - 80)
            snippet_end = min(len(text), idx + 80)
            suggested_fetch = {
                "start": max(0, idx - 200),
                "end": min(len(text), idx + 400),
            }
            hits.append(
                {
                    "doc_id": doc["doc_id"],
                    "score": float(score),
                    "snippet_start": snippet_start,
                    "snippet_end": snippet_end,
                    "snippet": text[snippet_start:snippet_end],
                    "suggested_fetch": suggested_fetch,
                }
            )
        return {"hits": hits}

    def fetch(self, doc_id: str, start: int, end: int) -> Dict:
        doc = self.doc_map.get(doc_id)
        if doc is None:
            return {"error": "doc_not_found"}

        text = doc.get("text", "")
        start_clamped = max(0, min(int(start), len(text)))
        end_clamped = max(start_clamped, min(int(end), len(text)))
        return {
            "doc_id": doc_id,
            "start": start_clamped,
            "end": end_clamped,
            "text": text[start_clamped:end_clamped],
        }

    def _estimate_hit_position(self, query: str, text: str) -> int:
        terms = [t.lower() for t in query.split() if t]
        lower = text.lower()
        for term in terms:
            pos = lower.find(term)
            if pos != -1:
                return pos
        return 0

