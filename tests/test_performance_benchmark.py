from evaluation.performance_benchmark import (
    PerformanceRow,
    acceptance_failures,
    summarize,
)


def row(
    *,
    category: str = "vector",
    total: float = 1000,
    first: float = 500,
    critical: bool = False,
    deterministic: bool = True,
    citation: bool = True,
    mode: bool = True,
) -> PerformanceRow:
    return PerformanceRow(
        configuration="test",
        case_id=f"{category}-{total}",
        category=category,
        critical=critical,
        total_latency_ms=total,
        first_token_latency_ms=first,
        retrieval_latency_ms=50,
        reranking_latency_ms=0,
        answer_latency_ms=900,
        deterministic_passed=deterministic,
        citation_valid=citation,
        answer_mode_valid=mode,
        reranking_applied=category == "reranked",
        requested_service_tier=None,
        response_service_tier=None,
        model="model",
        request_id="request",
        estimated_output_tokens=40,
        answer="Supported. [1]",
        deterministic_failures=[] if deterministic else ["failed"],
    )


def test_acceptance_gates_pass_target_workload() -> None:
    rows = [
        row(category="vector"),
        row(category="reranked", total=1800, first=600),
        row(category="follow_up", total=1500, first=550),
        row(category="abstention", total=10, first=5),
    ]

    summary = summarize(rows)

    assert acceptance_failures(summary) == []
    assert summary["citation_valid_rate"] == 1.0


def test_acceptance_gates_report_latency_and_quality_failures() -> None:
    rows = [
        row(
            category="vector",
            total=6000,
            first=4000,
            critical=True,
            deterministic=False,
            citation=False,
            mode=False,
        )
    ]

    failures = acceptance_failures(summarize(rows))

    assert any("P95 total" in failure for failure in failures)
    assert any("critical failures" in failure for failure in failures)
    assert any("citation validity" in failure for failure in failures)
