"""Local query telemetry and feedback for the ClauseLens demo."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

QUERY_METRICS_PATH = Path("data/processed/query_metrics.jsonl")


def percentile(values: list[float], pct: float) -> float:
    """Return a linearly interpolated percentile."""

    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def citation_indexes(answer: str) -> list[int]:
    """Extract bracketed evidence citations such as [1] and [2]."""

    return [int(value) for value in re.findall(r"\[(\d+)\]", answer)]


def evaluate_live_response(response: dict[str, object]) -> dict[str, bool]:
    """Run checks that do not require human-labeled ground truth."""

    results = list(response.get("results", []))
    answer = str(response.get("answer", ""))
    citations = citation_indexes(answer)
    abstained = bool(response.get("abstained", False))
    resolved_clause_type = response.get("resolved_clause_type")

    citation_valid = (
        not citations
        if abstained or not results
        else bool(citations) and all(1 <= index <= len(results) for index in citations)
    )
    route_consistent = all(
        not resolved_clause_type
        or not isinstance(result, dict)
        or result.get("clause_type") == resolved_clause_type
        for result in results
    )
    return {
        "evidence_present": bool(results) or abstained,
        "citation_valid": citation_valid,
        "route_consistent": route_consistent,
    }


def append_query_metric(
    response: dict[str, object],
    *,
    client_total_latency_ms: float,
    client_first_visible_ms: float,
    path: Path = QUERY_METRICS_PATH,
) -> dict[str, object]:
    """Append one completed query record to a local JSONL log."""

    checks = evaluate_live_response(response)
    results = list(response.get("results", []))
    record: dict[str, object] = {
        "record_type": "query",
        "timestamp": datetime.now(UTC).isoformat(),
        "turn_id": response.get("turn_id"),
        "question": response.get("question"),
        "standalone_query": response.get("standalone_query"),
        "resolved_clause_type": response.get("resolved_clause_type"),
        "abstained": bool(response.get("abstained", False)),
        "reranking_applied": bool(response.get("reranking_applied", False)),
        "rerank_reason": response.get("rerank_reason"),
        "result_count": len(results),
        "evidence_ids": [
            result.get("document_id")
            for result in results
            if isinstance(result, dict)
        ],
        "scores": [
            result.get("score") for result in results if isinstance(result, dict)
        ],
        "timings": response.get("timings", {}),
        "generation": response.get("generation", {}),
        "model": dict(response.get("generation", {})).get("model"),
        "requested_service_tier": dict(response.get("generation", {})).get(
            "requested_service_tier"
        ),
        "response_service_tier": dict(response.get("generation", {})).get(
            "response_service_tier"
        ),
        "request_id": dict(response.get("generation", {})).get("request_id"),
        "client_total_latency_ms": round(client_total_latency_ms, 3),
        "client_first_visible_ms": round(client_first_visible_ms, 3),
        **checks,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=True) + "\n")
    return record


def append_feedback(
    turn_id: str,
    rating: str,
    *,
    path: Path = QUERY_METRICS_PATH,
) -> dict[str, object]:
    """Append explicit user feedback for a completed turn."""

    if rating not in {"up", "down"}:
        raise ValueError("rating must be 'up' or 'down'")
    record: dict[str, object] = {
        "record_type": "feedback",
        "timestamp": datetime.now(UTC).isoformat(),
        "turn_id": turn_id,
        "rating": rating,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=True) + "\n")
    return record


def load_query_metrics(path: Path = QUERY_METRICS_PATH) -> list[dict[str, Any]]:
    """Load valid telemetry records, ignoring incomplete lines."""

    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def summarize_query_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """Summarize live checks, latency, and user feedback."""

    queries = [row for row in rows if row.get("record_type") == "query"]
    feedback = [row for row in rows if row.get("record_type") == "feedback"]
    positive = sum(row.get("rating") == "up" for row in feedback)
    latency_values = [
        float(row.get("client_total_latency_ms", 0.0) or 0.0) for row in queries
    ]
    first_visible_values = [
        float(row.get("client_first_visible_ms", 0.0) or 0.0) for row in queries
    ]
    valid_checks = sum(
        bool(row.get("evidence_present"))
        and bool(row.get("citation_valid"))
        and bool(row.get("route_consistent"))
        for row in queries
    )
    return {
        "queries": len(queries),
        "live_check_pass_rate": valid_checks / len(queries) if queries else 0.0,
        "average_latency_ms": (
            sum(latency_values) / len(latency_values) if latency_values else 0.0
        ),
        "p50_latency_ms": percentile(latency_values, 0.50),
        "p95_latency_ms": percentile(latency_values, 0.95),
        "p50_first_visible_ms": percentile(first_visible_values, 0.50),
        "p95_first_visible_ms": percentile(first_visible_values, 0.95),
        "feedback_count": len(feedback),
        "positive_feedback_rate": positive / len(feedback) if feedback else 0.0,
    }
