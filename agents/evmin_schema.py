"""Structured output schema for evmin final answers."""

EV_MIN_SCHEMA = {
    "type": "json_schema",
    "name": "evmin_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "slot_values": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "evidence_refs": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "stop_reason": {
                "type": "string",
                "enum": ["enough_evidence", "budget_exhausted", "tool_failed", "guess"],
            },
        },
        "required": ["slot_values", "evidence_refs", "confidence", "stop_reason"],
        "additionalProperties": False,
    },
}

