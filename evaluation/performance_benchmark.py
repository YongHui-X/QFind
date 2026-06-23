"""Run repeatable ClauseLens latency and deterministic-quality acceptance tests."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chat import ChatMessage, ChatRequest, stream_chat_turn  # noqa: E402
from evaluation.answer_cases import AnswerTestCase, load_answer_tests  # noqa: E402
from evaluation.answer_eval import evaluate_deterministically  # noqa: E402
from evaluation.chat_benchmark import build_engine, percentile  # noqa: E402

DEFAULT_OUTPUT = Path("data/processed/performance_benchmark.json")
MODEL_CANDIDATES = (
    ("gpt-4.1-mini-2025-04-14", None),
    ("gpt-4.1-mini-2025-04-14", "priority"),
    ("gpt-5.4-mini-2026-03-17", None),
    ("gpt-5.4-mini-2026-03-17", "priority"),
)


@dataclass(frozen=True)
class PerformanceRow:
    configuration: str
    case_id: str
    category: str
    critical: bool
    total_latency_ms: float
    first_token_latency_ms: float
    retrieval_latency_ms: float
    reranking_latency_ms: float
    answer_latency_ms: float
    deterministic_passed: bool
    citation_valid: bool
    answer_mode_valid: bool
    reranking_applied: bool
    requested_service_tier: str | None
    response_service_tier: str | None
    model: str | None
    request_id: str | None
    estimated_output_tokens: int
    answer: str
    deterministic_failures: list[str]


def classify_category(case: AnswerTestCase, data: dict[str, Any]) -> str:
    if case.answer_mode == "abstain":
        return "abstention"
    if sum(message.role == "user" for message in case.messages) > 1:
        return "follow_up"
    if data.get("reranking_applied"):
        return "reranked"
    return "vector"


def run_case(
    *,
    engine: Any,
    case: AnswerTestCase,
    configuration: str,
) -> PerformanceRow:
    request = ChatRequest(
        messages=[
            ChatMessage(role=message.role, content=message.content)
            for message in case.messages
        ],
        limit=5,
        rerank_mode="auto",
    )
    started = time.perf_counter()
    events = [
        json.loads(line)
        for line in stream_chat_turn(engine=engine, request=request)
    ]
    wall_clock_ms = (time.perf_counter() - started) * 1000
    final = next(event for event in reversed(events) if event.get("event") == "final")
    data = dict(final["data"])
    timings = dict(data.get("timings", {}))
    generation = dict(data.get("generation", {}))
    checks = evaluate_deterministically(case, data)
    return PerformanceRow(
        configuration=configuration,
        case_id=case.case_id,
        category=classify_category(case, data),
        critical=case.critical,
        total_latency_ms=wall_clock_ms,
        first_token_latency_ms=float(
            timings.get("first_token_latency_ms", 0.0) or 0.0
        ),
        retrieval_latency_ms=float(
            timings.get("retrieval_latency_ms", 0.0) or 0.0
        ),
        reranking_latency_ms=float(
            timings.get("reranking_latency_ms", 0.0) or 0.0
        ),
        answer_latency_ms=float(timings.get("answer_latency_ms", 0.0) or 0.0),
        deterministic_passed=checks.passed,
        citation_valid=checks.citation_valid,
        answer_mode_valid=checks.answer_mode_valid,
        reranking_applied=bool(data.get("reranking_applied", False)),
        requested_service_tier=generation.get("requested_service_tier"),
        response_service_tier=generation.get("response_service_tier"),
        model=generation.get("model"),
        request_id=generation.get("request_id"),
        estimated_output_tokens=int(
            generation.get("estimated_output_tokens", 0) or 0
        ),
        answer=str(data.get("answer", "")),
        deterministic_failures=checks.failures,
    )


def metric_summary(rows: list[PerformanceRow]) -> dict[str, float | int]:
    total = [row.total_latency_ms for row in rows]
    first = [row.first_token_latency_ms for row in rows]
    return {
        "requests": len(rows),
        "p50_total_ms": round(percentile(total, 0.50), 3),
        "p95_total_ms": round(percentile(total, 0.95), 3),
        "p99_total_ms": round(percentile(total, 0.99), 3),
        "p95_first_token_ms": round(percentile(first, 0.95), 3),
        "mean_total_ms": round(statistics.mean(total), 3) if total else 0.0,
    }


def summarize(rows: list[PerformanceRow]) -> dict[str, Any]:
    categories = sorted({row.category for row in rows})
    return {
        "overall": metric_summary(rows),
        "categories": {
            category: metric_summary(
                [row for row in rows if row.category == category]
            )
            for category in categories
        },
        "deterministic_pass_rate": (
            sum(row.deterministic_passed for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "citation_valid_rate": (
            sum(row.citation_valid for row in rows) / len(rows) if rows else 0.0
        ),
        "answer_mode_consistency": (
            sum(row.answer_mode_valid for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "critical_failures": sorted(
            {
                row.case_id
                for row in rows
                if row.critical and not row.deterministic_passed
            }
        ),
        "priority_response_rate": (
            sum(row.response_service_tier == "priority" for row in rows)
            / len(rows)
            if rows and any(row.requested_service_tier == "priority" for row in rows)
            else None
        ),
    }


def acceptance_failures(summary: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    overall = summary["overall"]
    if overall["p95_total_ms"] >= 2000:
        failures.append("P95 total latency is not under 2 seconds")
    if overall["p99_total_ms"] >= 10000:
        failures.append("P99 total latency is not under 10 seconds")
    if overall["p95_first_token_ms"] >= 700:
        failures.append("P95 first token is not under 700 ms")
    category_limits = {
        "vector": 2000,
        "reranked": 2500,
        "follow_up": 2500,
        "abstention": 250,
    }
    for category, limit in category_limits.items():
        values = summary["categories"].get(category)
        if values and values["p95_total_ms"] >= limit:
            failures.append(f"{category} P95 total is not under {limit} ms")
    if summary["critical_failures"]:
        failures.append(f"critical failures: {summary['critical_failures']}")
    if summary["citation_valid_rate"] < 1.0:
        failures.append("citation validity is below 100%")
    if summary["answer_mode_consistency"] < 0.95:
        failures.append("answer mode consistency is below 95%")
    priority_rate = summary.get("priority_response_rate")
    if priority_rate is not None and priority_rate < 0.95:
        failures.append("Priority tier response rate is below 95%")
    return failures


def run_configuration(
    *,
    cases: list[AnswerTestCase],
    model: str,
    service_tier: str | None,
    repeats: int,
    seed: int,
    candidate_limit: int,
    qdrant_url: str,
) -> tuple[list[PerformanceRow], dict[str, Any]]:
    label = f"{model}:{service_tier or 'standard'}:candidates={candidate_limit}"
    engine = build_engine(
        rerank_mode="auto",
        qdrant_mode="server",
        qdrant_url=qdrant_url,
        model=model,
        service_tier=service_tier,
        reasoning_effort="none" if model.startswith("gpt-5") else None,
        candidate_limit=candidate_limit,
    )
    workload = cases * repeats
    random.Random(seed).shuffle(workload)
    # Warm every local model and hosted path, but exclude the turn from results.
    run_case(engine=engine, case=cases[0], configuration=label)
    rows = [
        run_case(engine=engine, case=case, configuration=label)
        for case in workload
    ]
    return rows, summarize(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ClauseLens model/tier performance acceptance workloads."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--model", default="gpt-4.1-mini-2025-04-14")
    parser.add_argument("--service-tier", choices=["standard", "priority"])
    parser.add_argument("--candidate-limit", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--case-id",
        action="append",
        help="Run only the named answer-quality case; may be supplied more than once.",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Screen the two pinned model snapshots on Standard and Priority.",
    )
    parser.add_argument(
        "--enforce-gates",
        action="store_true",
        help="Exit unsuccessfully when acceptance thresholds are missed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_answer_tests()
    if args.case_id:
        selected = set(args.case_id)
        cases = [case for case in cases if case.case_id in selected]
        missing = selected - {case.case_id for case in cases}
        if missing:
            raise ValueError(f"unknown case IDs: {sorted(missing)}")
    configurations = (
        MODEL_CANDIDATES
        if args.matrix
        else ((args.model, args.service_tier),)
    )
    repeats = 3 if args.matrix and args.repeats == 10 else args.repeats
    report: dict[str, Any] = {"configurations": []}
    all_failures: list[str] = []
    for model, service_tier in configurations:
        rows, summary = run_configuration(
            cases=cases,
            model=model,
            service_tier=service_tier,
            repeats=repeats,
            seed=args.seed,
            candidate_limit=args.candidate_limit,
            qdrant_url=args.qdrant_url,
        )
        failures = acceptance_failures(summary)
        report["configurations"].append(
            {
                "model": model,
                "service_tier": service_tier or "standard",
                "candidate_limit": args.candidate_limit,
                "summary": summary,
                "gate_failures": failures,
                "rows": [asdict(row) for row in rows],
            }
        )
        all_failures.extend(
            f"{model}:{service_tier or 'standard'}: {failure}"
            for failure in failures
        )
        print(model, service_tier or "standard", summary["overall"], failures)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote performance report to {args.output}")
    if args.enforce_gates and all_failures:
        raise SystemExit("\n".join(all_failures))


if __name__ == "__main__":
    main()
