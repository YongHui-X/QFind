"""Benchmark end-to-end chat latency for ClauseLens."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api import SearchEngine  # noqa: E402
from app.chat import (  # noqa: E402
    ChatEngine,
    ChatMessage,
    ChatRequest,
    create_openai_chat_client,
    stream_chat_turn,
)
from app.rag import (  # noqa: E402
    EMBEDDING_MODEL,
    QDRANT_PATH,
    QDRANT_URL,
    RERANKER_MODEL,
    create_qdrant_client,
    load_embedding_model,
    load_lexical_index,
    load_reranker_model,
)
from evaluation.cases import (  # noqa: E402
    DEFAULT_TEST_FILE,
    RetrievalTestCase,
    load_tests,
)

RESULT_COLUMNS = [
    "question",
    "expected_clause_type",
    "category",
    "reranking_enabled",
    "limit",
    "result_count",
    "resolved_clause_type",
    "abstained",
    "wall_clock_latency_ms",
    "reported_total_latency_ms",
    "first_token_latency_ms",
    "rewrite_latency_ms",
    "contextualization_latency_ms",
    "retrieval_latency_ms",
    "embedding_latency_ms",
    "vector_search_latency_ms",
    "reranker_loading_latency_ms",
    "reranking_latency_ms",
    "answer_latency_ms",
    "generation_first_token_latency_ms",
    "prompt_chars",
    "evidence_chars",
    "estimated_input_tokens",
    "output_chars",
    "estimated_output_tokens",
    "model",
    "requested_service_tier",
    "response_service_tier",
    "request_id",
]


@dataclass(frozen=True)
class ChatBenchmarkRow:
    """One measured chat turn."""

    question: str
    expected_clause_type: str
    category: str
    reranking_enabled: bool
    limit: int
    result_count: int
    resolved_clause_type: str | None
    abstained: bool
    wall_clock_latency_ms: float
    reported_total_latency_ms: float
    first_token_latency_ms: float
    rewrite_latency_ms: float
    contextualization_latency_ms: float
    retrieval_latency_ms: float
    embedding_latency_ms: float
    vector_search_latency_ms: float
    reranker_loading_latency_ms: float
    reranking_latency_ms: float
    answer_latency_ms: float
    generation_first_token_latency_ms: float
    prompt_chars: int
    evidence_chars: int
    estimated_input_tokens: int
    output_chars: int = 0
    estimated_output_tokens: int = 0
    model: str | None = None
    requested_service_tier: str | None = None
    response_service_tier: str | None = None
    request_id: str | None = None


def percentile(values: list[float], pct: float) -> float:
    """Return a simple linear-interpolated percentile."""

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


def build_engine(
    *,
    rerank_mode: str,
    qdrant_mode: str = "embedded",
    qdrant_path: Path = QDRANT_PATH,
    qdrant_url: str = QDRANT_URL,
    model: str | None = None,
    service_tier: str | None = None,
    reasoning_effort: str | None = None,
    candidate_limit: int = 5,
) -> ChatEngine:
    """Create the chat engine used by the benchmark."""

    llm = create_openai_chat_client(
        model=model,
        service_tier=service_tier,
        reasoning_effort=reasoning_effort,
    )
    search_engine = SearchEngine(
        client=(
            create_qdrant_client(url=qdrant_url)
            if qdrant_mode == "server"
            else create_qdrant_client(path=qdrant_path)
        ),
        model=load_embedding_model(EMBEDDING_MODEL),
        reranker=(
            load_reranker_model(RERANKER_MODEL)
            if rerank_mode in {"auto", "always"}
            else None
        ),
        reranking_enabled=rerank_mode != "off",
        lexical_index=load_lexical_index(),
    )
    return ChatEngine(
        search_engine=search_engine,
        llm=llm,
        model_name=llm.model,
        rerank_candidate_limit=candidate_limit,
    )


def benchmark_case(
    *,
    engine: ChatEngine,
    test_case: RetrievalTestCase,
    rerank_mode: str,
) -> ChatBenchmarkRow:
    """Measure one chat turn and capture the returned timings."""

    request = ChatRequest(
        messages=[ChatMessage(role="user", content=test_case.question)],
        limit=5,
        rerank_mode=rerank_mode,
    )
    started = time.perf_counter()
    events = [json.loads(line) for line in stream_chat_turn(engine=engine, request=request)]
    wall_clock_latency_ms = (time.perf_counter() - started) * 1000
    final_event = next(
        (event for event in reversed(events) if event.get("event") == "final"),
        None,
    )
    if final_event is None:
        raise ValueError(f"No final event returned for question: {test_case.question}")

    data = dict(final_event.get("data", {}))
    timings = dict(data.get("timings", {}))
    generation = dict(data.get("generation", {}))
    return ChatBenchmarkRow(
        question=test_case.question,
        expected_clause_type=test_case.expected_clause_type,
        category=test_case.category,
        reranking_enabled=engine.search_engine.reranking_enabled,
        limit=request.limit,
        result_count=int(data.get("result_count", 0) or 0),
        resolved_clause_type=(str(data["resolved_clause_type"]) if data.get("resolved_clause_type") is not None else None),
        abstained=bool(data.get("abstained", False)),
        wall_clock_latency_ms=wall_clock_latency_ms,
        reported_total_latency_ms=float(timings.get("total_latency_ms", 0.0) or 0.0),
        first_token_latency_ms=float(timings.get("first_token_latency_ms", 0.0) or 0.0),
        rewrite_latency_ms=float(timings.get("rewrite_latency_ms", 0.0) or 0.0),
        contextualization_latency_ms=float(
            timings.get("contextualization_latency_ms", 0.0) or 0.0
        ),
        retrieval_latency_ms=float(timings.get("retrieval_latency_ms", 0.0) or 0.0),
        embedding_latency_ms=float(
            timings.get("embedding_latency_ms", 0.0) or 0.0
        ),
        vector_search_latency_ms=float(
            timings.get("vector_search_latency_ms", 0.0) or 0.0
        ),
        reranker_loading_latency_ms=float(
            timings.get("reranker_loading_latency_ms", 0.0) or 0.0
        ),
        reranking_latency_ms=float(timings.get("reranking_latency_ms", 0.0) or 0.0),
        answer_latency_ms=float(timings.get("answer_latency_ms", 0.0) or 0.0),
        generation_first_token_latency_ms=float(
            timings.get("generation_first_token_latency_ms", 0.0) or 0.0
        ),
        prompt_chars=int(generation.get("prompt_chars", 0) or 0),
        evidence_chars=int(generation.get("evidence_chars", 0) or 0),
        estimated_input_tokens=int(
            generation.get("estimated_input_tokens", 0) or 0
        ),
        output_chars=int(generation.get("output_chars", 0) or 0),
        estimated_output_tokens=int(
            generation.get("estimated_output_tokens", 0) or 0
        ),
        model=(str(generation["model"]) if generation.get("model") else None),
        requested_service_tier=(
            str(generation["requested_service_tier"])
            if generation.get("requested_service_tier")
            else None
        ),
        response_service_tier=(
            str(generation["response_service_tier"])
            if generation.get("response_service_tier")
            else None
        ),
        request_id=(
            str(generation["request_id"]) if generation.get("request_id") else None
        ),
    )


def build_rows(
    *,
    tests: list[RetrievalTestCase],
    rerank_mode: str,
    warmup_runs: int,
    repeats: int = 1,
    seed: int = 42,
    qdrant_mode: str = "embedded",
    qdrant_path: Path = QDRANT_PATH,
    qdrant_url: str = QDRANT_URL,
    model: str | None = None,
    service_tier: str | None = None,
    reasoning_effort: str | None = None,
    candidate_limit: int = 5,
) -> list[ChatBenchmarkRow]:
    """Run a benchmark pass and return rows."""

    engine = build_engine(
        rerank_mode=rerank_mode,
        qdrant_mode=qdrant_mode,
        qdrant_path=qdrant_path,
        qdrant_url=qdrant_url,
        model=model,
        service_tier=service_tier,
        reasoning_effort=reasoning_effort,
        candidate_limit=candidate_limit,
    )

    for warmup_index in range(min(warmup_runs, len(tests))):
        benchmark_case(
            engine=engine,
            test_case=tests[warmup_index],
            rerank_mode=rerank_mode,
        )

    workload = tests * repeats
    random.Random(seed).shuffle(workload)
    return [
        benchmark_case(
            engine=engine,
            test_case=test_case,
            rerank_mode=rerank_mode,
        )
        for test_case in workload
    ]


def result_rows(results: list[ChatBenchmarkRow]) -> list[dict[str, Any]]:
    """Convert benchmark rows into a file-friendly representation."""

    return [
        {
            "question": result.question,
            "expected_clause_type": result.expected_clause_type,
            "category": result.category,
            "reranking_enabled": result.reranking_enabled,
            "limit": result.limit,
            "result_count": result.result_count,
            "resolved_clause_type": result.resolved_clause_type,
            "abstained": result.abstained,
            "wall_clock_latency_ms": round(result.wall_clock_latency_ms, 3),
            "reported_total_latency_ms": round(result.reported_total_latency_ms, 3),
            "first_token_latency_ms": round(result.first_token_latency_ms, 3),
            "rewrite_latency_ms": round(result.rewrite_latency_ms, 3),
            "contextualization_latency_ms": round(
                result.contextualization_latency_ms, 3
            ),
            "retrieval_latency_ms": round(result.retrieval_latency_ms, 3),
            "embedding_latency_ms": round(result.embedding_latency_ms, 3),
            "vector_search_latency_ms": round(
                result.vector_search_latency_ms, 3
            ),
            "reranker_loading_latency_ms": round(
                result.reranker_loading_latency_ms, 3
            ),
            "reranking_latency_ms": round(result.reranking_latency_ms, 3),
            "answer_latency_ms": round(result.answer_latency_ms, 3),
            "generation_first_token_latency_ms": round(
                result.generation_first_token_latency_ms, 3
            ),
            "prompt_chars": result.prompt_chars,
            "evidence_chars": result.evidence_chars,
            "estimated_input_tokens": result.estimated_input_tokens,
            "output_chars": result.output_chars,
            "estimated_output_tokens": result.estimated_output_tokens,
            "model": result.model,
            "requested_service_tier": result.requested_service_tier,
            "response_service_tier": result.response_service_tier,
            "request_id": result.request_id,
        }
        for result in results
    ]


def write_results(path: Path, results: list[ChatBenchmarkRow]) -> None:
    """Write benchmark rows to JSON or CSV."""

    rows = result_rows(results)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return

    if path.suffix.lower() != ".csv":
        raise ValueError("output path must end in .json or .csv")

    import csv

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(results: list[ChatBenchmarkRow], *, label: str) -> None:
    """Print a compact benchmark summary."""

    if not results:
        print(f"{label}: no benchmark rows.")
        return

    wall_clock = [result.wall_clock_latency_ms for result in results]
    reported_total = [result.reported_total_latency_ms for result in results]
    first_token = [result.first_token_latency_ms for result in results]
    rewrite = [result.rewrite_latency_ms for result in results]
    contextualization = [
        result.contextualization_latency_ms for result in results
    ]
    retrieval = [result.retrieval_latency_ms for result in results]
    embedding = [result.embedding_latency_ms for result in results]
    vector_search = [result.vector_search_latency_ms for result in results]
    reranker_loading = [result.reranker_loading_latency_ms for result in results]
    reranking = [result.reranking_latency_ms for result in results]
    answer = [result.answer_latency_ms for result in results]
    generation_first_token = [
        result.generation_first_token_latency_ms for result in results
    ]
    prompt_chars = [result.prompt_chars for result in results]
    estimated_input_tokens = [result.estimated_input_tokens for result in results]
    abstained = sum(1 for result in results if result.abstained)

    print(f"{label}")
    print("=" * len(label))
    print(f"Cases: {len(results)}")
    print(f"Abstained: {abstained}/{len(results)}")
    print(f"Wall clock mean: {statistics.mean(wall_clock):.1f} ms")
    print(f"Wall clock p50: {percentile(wall_clock, 0.50):.1f} ms")
    print(f"Wall clock p95: {percentile(wall_clock, 0.95):.1f} ms")
    print(f"Reported total mean: {statistics.mean(reported_total):.1f} ms")
    print(f"First token mean: {statistics.mean(first_token):.1f} ms")
    print(f"First token p95: {percentile(first_token, 0.95):.1f} ms")
    print(f"Rewrite mean: {statistics.mean(rewrite):.1f} ms")
    print(
        "Contextualization mean: "
        f"{statistics.mean(contextualization):.1f} ms"
    )
    print(f"Retrieval mean: {statistics.mean(retrieval):.1f} ms")
    print(f"Embedding mean: {statistics.mean(embedding):.1f} ms")
    print(f"Vector search mean: {statistics.mean(vector_search):.1f} ms")
    print(f"Reranker loading mean: {statistics.mean(reranker_loading):.1f} ms")
    print(f"Reranking mean: {statistics.mean(reranking):.1f} ms")
    print(f"Answer mean: {statistics.mean(answer):.1f} ms")
    print(
        "Generation first token mean: "
        f"{statistics.mean(generation_first_token):.1f} ms"
    )
    print(f"Prompt chars mean: {statistics.mean(prompt_chars):.1f}")
    print(
        "Estimated input tokens mean: "
        f"{statistics.mean(estimated_input_tokens):.1f}"
    )


def print_comparison(
    baseline: list[ChatBenchmarkRow],
    reranked: list[ChatBenchmarkRow],
) -> None:
    """Print a side-by-side reranking comparison."""

    def avg(rows: list[ChatBenchmarkRow], attr: str) -> float:
        return statistics.mean(float(getattr(row, attr)) for row in rows)

    print("Reranking comparison")
    print("====================")
    print(f"Baseline wall clock mean: {avg(baseline, 'wall_clock_latency_ms'):.1f} ms")
    print(f"Rerank wall clock mean:   {avg(reranked, 'wall_clock_latency_ms'):.1f} ms")
    print(
        "Delta wall clock:         "
        f"{avg(reranked, 'wall_clock_latency_ms') - avg(baseline, 'wall_clock_latency_ms'):.1f} ms"
    )
    print(f"Baseline first token mean: {avg(baseline, 'first_token_latency_ms'):.1f} ms")
    print(f"Rerank first token mean:   {avg(reranked, 'first_token_latency_ms'):.1f} ms")
    print(
        "Delta first token:        "
        f"{avg(reranked, 'first_token_latency_ms') - avg(baseline, 'first_token_latency_ms'):.1f} ms"
    )
    print(f"Baseline retrieval mean:   {avg(baseline, 'retrieval_latency_ms'):.1f} ms")
    print(f"Rerank retrieval mean:     {avg(reranked, 'retrieval_latency_ms'):.1f} ms")
    print(f"Baseline rerank mean:      {avg(baseline, 'reranking_latency_ms'):.1f} ms")
    print(f"Rerank rerank mean:        {avg(reranked, 'reranking_latency_ms'):.1f} ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ClauseLens chat latency against JSONL test cases."
    )
    parser.add_argument("--tests", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument(
        "--qdrant-mode",
        choices=["server", "embedded"],
        default="server",
    )
    parser.add_argument("--qdrant-path", type=Path, default=QDRANT_PATH)
    parser.add_argument("--qdrant-url", default=QDRANT_URL)
    parser.add_argument("--model")
    parser.add_argument("--service-tier", choices=["standard", "priority"])
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--candidate-limit", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument(
        "--rerank-mode",
        choices=["off", "auto", "always"],
        default="auto",
    )
    parser.add_argument(
        "--compare-rerank",
        action="store_true",
        help="Run the benchmark once with reranking off and once with reranking on.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for detailed results as JSON or CSV. Comparison mode writes baseline and rerank files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tests = load_tests(args.tests)
    if not tests:
        raise ValueError("no benchmark cases found")

    if args.compare_rerank:
        baseline = build_rows(
            tests=tests,
            rerank_mode="off",
            warmup_runs=args.warmup_runs,
            repeats=args.repeats,
            seed=args.seed,
            qdrant_mode=args.qdrant_mode,
            qdrant_path=args.qdrant_path,
            qdrant_url=args.qdrant_url,
            model=args.model,
            service_tier=args.service_tier,
            reasoning_effort=args.reasoning_effort,
            candidate_limit=args.candidate_limit,
        )
        reranked = build_rows(
            tests=tests,
            rerank_mode="always",
            warmup_runs=args.warmup_runs,
            repeats=args.repeats,
            seed=args.seed,
            qdrant_mode=args.qdrant_mode,
            qdrant_path=args.qdrant_path,
            qdrant_url=args.qdrant_url,
            model=args.model,
            service_tier=args.service_tier,
            reasoning_effort=args.reasoning_effort,
            candidate_limit=args.candidate_limit,
        )
        print_summary(baseline, label="Chat latency benchmark (baseline)")
        print()
        print_summary(reranked, label="Chat latency benchmark (reranked)")
        print()
        print_comparison(baseline, reranked)
        if args.output:
            baseline_path = args.output
            rerank_path = baseline_path.with_name(
                f"{baseline_path.stem}_rerank{baseline_path.suffix}"
            )
            write_results(baseline_path, baseline)
            write_results(rerank_path, reranked)
            print(f"Wrote baseline results to {baseline_path}")
            print(f"Wrote rerank results to {rerank_path}")
        return

    results = build_rows(
        tests=tests,
        rerank_mode=args.rerank_mode,
        warmup_runs=args.warmup_runs,
        repeats=args.repeats,
        seed=args.seed,
        qdrant_mode=args.qdrant_mode,
        qdrant_path=args.qdrant_path,
        qdrant_url=args.qdrant_url,
        model=args.model,
        service_tier=args.service_tier,
        reasoning_effort=args.reasoning_effort,
        candidate_limit=args.candidate_limit,
    )
    print_summary(
        results,
        label=f"Chat latency benchmark ({args.rerank_mode})",
    )
    if args.output:
        write_results(args.output, results)
        print(f"Wrote detailed results to {args.output}")


if __name__ == "__main__":
    main()
