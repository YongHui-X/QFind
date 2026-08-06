from pathlib import Path

import pytest

from evaluation.candidate_sweep import (
    run_candidate_sweep,
    summarize_results,
    sweep_rows,
    write_sweep_results,
)
from evaluation.eval import RetrievalEvalResult


def make_result(
    *,
    recall_at_k: float,
    hit_rate_at_k: float,
    context_precision: float,
    candidate_count: int = 5,
) -> RetrievalEvalResult:
    return RetrievalEvalResult(
        question="Find selected audit passages.",
        expected_clause_type="Audit Rights",
        category="true_recall",
        top_k=5,
        result_count=5,
        expected_clause_type_rank=1,
        clause_type_mrr=1.0,
        first_relevant_rank=1 if hit_rate_at_k else None,
        mrr=1.0 if hit_rate_at_k else 0.0,
        recall_at_k=recall_at_k,
        hit_rate_at_k=hit_rate_at_k,
        expected_record_ids=("gold-1",),
        retrieved_record_ids=("gold-1",) if hit_rate_at_k else ("miss-1",),
        gold_record_ids_found=("gold-1",) if hit_rate_at_k else (),
        missing_gold_record_ids=() if hit_rate_at_k else ("gold-1",),
        context_precision=context_precision,
        top1_clause_hit=True,
        topk_clause_hit=True,
        keyword_hit_rate=1.0,
        keywords_found=1,
        total_keywords=1,
        ndcg=1.0 if hit_rate_at_k else 0.0,
        candidate_count=candidate_count,
        max_passages_per_document=2,
        retrieval_latency_ms=10.0,
        reranking_latency_ms=2.0,
    )


def test_summarize_results_reports_context_precision() -> None:
    summary = summarize_results(
        [
            make_result(recall_at_k=1.0, hit_rate_at_k=1.0, context_precision=0.4),
            make_result(recall_at_k=0.5, hit_rate_at_k=1.0, context_precision=0.2),
        ],
        candidate_limit=10,
        deduplicate_documents=True,
        max_passages_per_document=2,
    )

    assert summary.candidate_limit == 10
    assert summary.recall_at_k == 0.75
    assert summary.hit_rate_at_k == 1.0
    assert summary.context_precision == pytest.approx(0.3)
    assert summary.average_candidate_count == 5.0
    assert summary.max_passages_per_document == 2


def test_sweep_rows_rounds_summary_metrics() -> None:
    summary = summarize_results(
        [
            make_result(
                recall_at_k=2 / 3,
                hit_rate_at_k=1.0,
                context_precision=1 / 3,
            )
        ],
        candidate_limit=3,
        deduplicate_documents=False,
        max_passages_per_document=None,
    )

    [row] = sweep_rows([summary])

    assert row["candidate_limit"] == 3
    assert row["recall_at_k"] == 0.6667
    assert row["context_precision"] == 0.3333
    assert row["max_passages_per_document"] is None


def test_write_sweep_results_exports_json(tmp_path: Path) -> None:
    summary = summarize_results(
        [make_result(recall_at_k=1.0, hit_rate_at_k=1.0, context_precision=0.2)],
        candidate_limit=5,
        deduplicate_documents=True,
        max_passages_per_document=1,
    )
    output = tmp_path / "candidate_sweep.json"

    write_sweep_results(output, [summary])

    data = output.read_text(encoding="utf-8")
    assert '"candidate_limit": 5' in data
    assert '"context_precision": 0.2' in data


def test_run_candidate_sweep_forwards_candidate_limits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_limits: list[int] = []

    def fake_evaluate_all(**kwargs: object) -> list[RetrievalEvalResult]:
        candidate_limit = int(kwargs["candidate_limit"])
        captured_limits.append(candidate_limit)
        return [
            make_result(
                recall_at_k=0.5 + candidate_limit / 100,
                hit_rate_at_k=1.0,
                context_precision=0.2,
                candidate_count=candidate_limit,
            )
        ]

    monkeypatch.setattr("evaluation.candidate_sweep.evaluate_all", fake_evaluate_all)

    summaries = run_candidate_sweep(
        candidate_limits=[3, 10],
        tests_path=tmp_path / "cases.jsonl",
        top_k=5,
        rerank_mode="auto",
        deduplicate_documents=True,
        max_passages_per_document=2,
        require_gold_record_ids=True,
    )

    assert captured_limits == [3, 10]
    assert [summary.candidate_limit for summary in summaries] == [3, 10]
    assert summaries[0].recall_at_k == 0.53
    assert summaries[1].average_candidate_count == 10


def test_run_candidate_sweep_rejects_empty_candidate_limits() -> None:
    with pytest.raises(ValueError, match="at least one candidate limit"):
        run_candidate_sweep(candidate_limits=[])
