"""Candidate-limit ablation for QFind retrieval evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag import (  # noqa: E402
    COLLECTION,
    EMBEDDING_MODEL,
    QDRANT_PATH,
    QDRANT_URL,
    RERANKER_MODEL,
)
from evaluation.cases import DEFAULT_TEST_FILE  # noqa: E402
from evaluation.eval import RetrievalEvalResult, evaluate_all  # noqa: E402

SWEEP_COLUMNS = [
    "candidate_limit",
    "top_k",
    "case_count",
    "deduplicate_documents",
    "max_passages_per_document",
    "recall_at_k",
    "hit_rate_at_k",
    "context_precision",
    "mrr",
    "ndcg",
    "keyword_hit_rate",
    "average_candidate_count",
    "average_retrieval_latency_ms",
    "average_reranking_latency_ms",
]


@dataclass(frozen=True)
class CandidateSweepResult:
    """Aggregate retrieval metrics for one candidate-limit setting."""

    candidate_limit: int
    top_k: int
    case_count: int
    deduplicate_documents: bool
    max_passages_per_document: int | None
    recall_at_k: float
    hit_rate_at_k: float
    context_precision: float
    mrr: float
    ndcg: float
    keyword_hit_rate: float
    average_candidate_count: float
    average_retrieval_latency_ms: float
    average_reranking_latency_ms: float


def summarize_results(
    results: list[RetrievalEvalResult],
    *,
    candidate_limit: int,
    deduplicate_documents: bool,
    max_passages_per_document: int | None,
) -> CandidateSweepResult:
    """Aggregate per-case metrics for one candidate-limit sweep row."""

    if not results:
        raise ValueError("cannot summarize an empty evaluation result set")

    case_count = len(results)
    return CandidateSweepResult(
        candidate_limit=candidate_limit,
        top_k=results[0].top_k,
        case_count=case_count,
        deduplicate_documents=deduplicate_documents,
        max_passages_per_document=max_passages_per_document
        if deduplicate_documents
        else None,
        recall_at_k=sum(result.recall_at_k for result in results) / case_count,
        hit_rate_at_k=sum(result.hit_rate_at_k for result in results) / case_count,
        context_precision=sum(result.context_precision for result in results)
        / case_count,
        mrr=sum(result.mrr for result in results) / case_count,
        ndcg=sum(result.ndcg for result in results) / case_count,
        keyword_hit_rate=sum(result.keyword_hit_rate for result in results)
        / case_count,
        average_candidate_count=sum(result.candidate_count for result in results)
        / case_count,
        average_retrieval_latency_ms=sum(
            result.retrieval_latency_ms for result in results
        )
        / case_count,
        average_reranking_latency_ms=sum(
            result.reranking_latency_ms for result in results
        )
        / case_count,
    )


def sweep_rows(results: list[CandidateSweepResult]) -> list[dict[str, Any]]:
    """Convert sweep results to stable JSON/CSV rows."""

    return [
        {
            "candidate_limit": result.candidate_limit,
            "top_k": result.top_k,
            "case_count": result.case_count,
            "deduplicate_documents": result.deduplicate_documents,
            "max_passages_per_document": result.max_passages_per_document,
            "recall_at_k": round(result.recall_at_k, 4),
            "hit_rate_at_k": round(result.hit_rate_at_k, 4),
            "context_precision": round(result.context_precision, 4),
            "mrr": round(result.mrr, 4),
            "ndcg": round(result.ndcg, 4),
            "keyword_hit_rate": round(result.keyword_hit_rate, 4),
            "average_candidate_count": round(result.average_candidate_count, 3),
            "average_retrieval_latency_ms": round(
                result.average_retrieval_latency_ms,
                3,
            ),
            "average_reranking_latency_ms": round(
                result.average_reranking_latency_ms,
                3,
            ),
        }
        for result in results
    ]


def write_sweep_results(path: Path, results: list[CandidateSweepResult]) -> None:
    """Write candidate-limit sweep summary rows to JSON or CSV."""

    rows = sweep_rows(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return

    if path.suffix.lower() != ".csv":
        raise ValueError("output path must end in .json or .csv")

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SWEEP_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def run_candidate_sweep(
    *,
    candidate_limits: list[int],
    tests_path: Path = DEFAULT_TEST_FILE,
    qdrant_path: Path = QDRANT_PATH,
    qdrant_mode: str = "server",
    qdrant_url: str = QDRANT_URL,
    collection_name: str = COLLECTION,
    model_name: str = EMBEDDING_MODEL,
    reranker_model_name: str = RERANKER_MODEL,
    top_k: int = 5,
    rerank: bool = False,
    rerank_mode: str | None = None,
    deduplicate_documents: bool = True,
    max_passages_per_document: int | None = 1,
    require_gold_record_ids: bool = False,
) -> list[CandidateSweepResult]:
    """Run retrieval evaluation across candidate-limit values."""

    if not candidate_limits:
        raise ValueError("at least one candidate limit is required")
    if any(candidate_limit < 1 for candidate_limit in candidate_limits):
        raise ValueError("candidate limits must be at least 1")

    sweep_results: list[CandidateSweepResult] = []
    for candidate_limit in candidate_limits:
        case_results = evaluate_all(
            tests_path=tests_path,
            qdrant_path=qdrant_path,
            qdrant_mode=qdrant_mode,
            qdrant_url=qdrant_url,
            collection_name=collection_name,
            model_name=model_name,
            reranker_model_name=reranker_model_name,
            top_k=top_k,
            rerank=rerank,
            rerank_mode=rerank_mode,
            candidate_limit=candidate_limit,
            deduplicate_documents=deduplicate_documents,
            max_passages_per_document=max_passages_per_document,
            require_gold_record_ids=require_gold_record_ids,
        )
        effective_document_limit = (
            max_passages_per_document if deduplicate_documents else None
        )
        sweep_results.append(
            summarize_results(
                case_results,
                candidate_limit=candidate_limit,
                deduplicate_documents=deduplicate_documents,
                max_passages_per_document=effective_document_limit,
            )
        )
    return sweep_results


def print_sweep_summary(results: list[CandidateSweepResult]) -> None:
    """Print a compact candidate-limit comparison table."""

    if not results:
        print("No candidate sweep results found.")
        return

    print("QFind Candidate Limit Sweep")
    print("=" * 92)
    print(
        "candidate_limit | recall@k | hit_rate@k | context_precision | "
        "mrr | ndcg | candidates | retrieval_ms | rerank_ms"
    )
    print("-" * 92)
    for result in results:
        print(
            f"{result.candidate_limit:15d} | "
            f"{result.recall_at_k:8.3f} | "
            f"{result.hit_rate_at_k:10.1%} | "
            f"{result.context_precision:17.3f} | "
            f"{result.mrr:3.3f} | "
            f"{result.ndcg:4.3f} | "
            f"{result.average_candidate_count:10.1f} | "
            f"{result.average_retrieval_latency_ms:12.1f} | "
            f"{result.average_reranking_latency_ms:9.1f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep QFind retrieval candidate limits and report recall plus "
            "context precision."
        )
    )
    parser.add_argument("--tests", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--qdrant-path", type=Path, default=QDRANT_PATH)
    parser.add_argument("--qdrant-url", default=QDRANT_URL)
    parser.add_argument(
        "--qdrant-mode",
        choices=["server", "embedded"],
        default="server",
    )
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--model", default=EMBEDDING_MODEL)
    parser.add_argument("--reranker-model", default=RERANKER_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--candidate-limits",
        type=int,
        nargs="+",
        default=[3, 5, 10, 20, 50],
        help="Candidate limits to evaluate, for example: 3 5 10 20 50.",
    )
    parser.add_argument(
        "--max-passages-per-document",
        type=int,
        default=1,
        help="Maximum passages to keep from each document when dedup is enabled.",
    )
    parser.add_argument(
        "--deduplicate-documents",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable document-level passage limiting.",
    )
    parser.add_argument(
        "--require-gold-record-ids",
        action="store_true",
        help="Fail if any test case lacks expected_record_ids for true recall.",
    )
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable cross-encoder reranking.",
    )
    parser.add_argument(
        "--rerank-mode",
        choices=["off", "auto", "always"],
        help="Override --rerank with per-query off, adaptive, or always behavior.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for sweep summary as JSON or CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rerank = args.rerank
    if args.rerank_mode:
        rerank = args.rerank_mode != "off"
    try:
        results = run_candidate_sweep(
            candidate_limits=args.candidate_limits,
            tests_path=args.tests,
            qdrant_path=args.qdrant_path,
            qdrant_mode=args.qdrant_mode,
            qdrant_url=args.qdrant_url,
            collection_name=args.collection,
            model_name=args.model,
            reranker_model_name=args.reranker_model,
            top_k=args.top_k,
            rerank=rerank,
            rerank_mode=args.rerank_mode,
            deduplicate_documents=args.deduplicate_documents,
            max_passages_per_document=args.max_passages_per_document,
            require_gold_record_ids=args.require_gold_record_ids,
        )
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from None

    print_sweep_summary(results)
    if args.output:
        write_sweep_results(args.output, results)
        print(f"Wrote candidate sweep results to {args.output}")


if __name__ == "__main__":
    main()
