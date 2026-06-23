from pathlib import Path

from app.telemetry import (
    append_feedback,
    append_query_metric,
    evaluate_live_response,
    load_query_metrics,
    summarize_query_metrics,
)


def sample_response() -> dict[str, object]:
    return {
        "turn_id": "turn-1",
        "question": "What audit rights exist?",
        "standalone_query": "audit rights",
        "resolved_clause_type": "Audit Rights",
        "answer": "The customer may audit records. [1]",
        "results": [
            {
                "document_id": "doc-1",
                "clause_type": "Audit Rights",
                "score": 0.9,
            }
        ],
        "timings": {"total_latency_ms": 1200.0},
        "generation": {
            "prompt_chars": 2400,
            "evidence_chars": 1600,
            "estimated_input_tokens": 600,
        },
        "reranking_applied": False,
        "rerank_reason": "adaptive vector search",
    }


def test_evaluate_live_response_validates_citations_and_route() -> None:
    assert evaluate_live_response(sample_response()) == {
        "evidence_present": True,
        "citation_valid": True,
        "route_consistent": True,
    }


def test_query_metrics_and_feedback_are_persisted(tmp_path: Path) -> None:
    path = tmp_path / "query_metrics.jsonl"
    append_query_metric(
        sample_response(),
        client_total_latency_ms=1300.0,
        client_first_visible_ms=900.0,
        path=path,
    )
    append_feedback("turn-1", "up", path=path)

    rows = load_query_metrics(path)
    summary = summarize_query_metrics(rows)

    assert len(rows) == 2
    assert rows[0]["generation"]["prompt_chars"] == 2400
    assert summary["queries"] == 1
    assert summary["live_check_pass_rate"] == 1.0
    assert summary["positive_feedback_rate"] == 1.0
    assert summary["p95_latency_ms"] == 1300.0
    assert summary["p95_first_visible_ms"] == 900.0
