"""Evmin helpers for required span prioritization."""

from __future__ import annotations

from typing import Dict, List, Any


def prioritize_spans(required_spans: List[Dict[str, Any]]) -> List[str]:
    """Return ordered doc_ids to fetch, unique preserving order."""
    seen = set()
    ordered = []
    for span in required_spans or []:
        doc_id = span.get("doc_id")
        if doc_id and doc_id not in seen:
            ordered.append(doc_id)
            seen.add(doc_id)
    return ordered

