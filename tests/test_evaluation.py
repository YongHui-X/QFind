from pathlib import Path
from types import SimpleNamespace

import pytest

from app.rag import ClauseSearchResult
from evaluation.cases import RetrievalTestCase, load_tests
from evaluation.eval import (
    evaluate_all,
    evaluate_case,
    parse_args,
    result_rows,
    validate_gold_record_ids,
    write_results,
)


def test_load_tests_reads_jsonl_cases() -> None:
    tests = load_tests()

    assert tests
    assert tests[0].question == "Does the contract restrict assignment?"
    assert tests[0].expected_clause_type == "Anti-Assignment"


def test_evaluate_case_scores_expected_clause_and_keywords() -> None:
    test_case = RetrievalTestCase(
        question="What audit rights does the customer have?",
        expected_clause_type="Audit Rights",
        keywords=["audit", "records"],
        expected_record_ids=["audit-record"],
        category="clause_type",
    )
    results = [
        ClauseSearchResult(
            score=0.9,
            payload={
                "clause_type": "License Grant",
                "source_pdf": "Example.pdf",
                "text": "A license is granted.",
            },
        ),
        ClauseSearchResult(
            score=0.8,
            payload={
                "clause_type": "Audit Rights",
                "source_pdf": "Example.pdf",
                "id": "audit-record",
                "text": "Customer may audit records during business hours.",
            },
        ),
    ]

    result = evaluate_case(test_case, results, top_k=2)

    assert result.expected_clause_type_rank == 2
    assert result.clause_type_mrr == 0.5
    assert result.first_relevant_rank == 2
    assert result.mrr == 0.5
    assert result.recall_at_k == 1.0
    assert result.hit_rate_at_k == 1.0
    assert result.expected_record_ids == ("audit-record",)
    assert result.retrieved_record_ids == ("audit-record",)
    assert result.gold_record_ids_found == ("audit-record",)
    assert result.missing_gold_record_ids == ()
    assert result.context_precision == 0.5
    assert result.top1_clause_hit is False
    assert result.topk_clause_hit is True
    assert result.keywords_found == 2
    assert result.keyword_coverage == 1.0
    assert result.passed is False
    assert result.reranking_enabled is False


def test_write_results_exports_json(tmp_path) -> None:
    test_case = RetrievalTestCase(
        question="What audit rights does the customer have?",
        expected_clause_type="Audit Rights",
        keywords=["audit", "records"],
        expected_record_ids=["audit-record"],
        category="clause_type",
    )
    result = evaluate_case(
        test_case,
        [
            ClauseSearchResult(
                score=0.9,
                payload={
                    "clause_type": "Audit Rights",
                    "source_pdf": "Example.pdf",
                    "id": "audit-record",
                    "text": "Customer may audit records during business hours.",
                },
            )
        ],
        top_k=1,
    )

    output = tmp_path / "eval.json"
    write_results(output, [result])

    data = output.read_text(encoding="utf-8")
    assert '"question": "What audit rights does the customer have?"' in data
    assert '"hit_rate_at_k": 1.0' in data
    assert '"expected_record_ids": [' in data
    assert '"retrieved_record_ids": [' in data
    assert '"gold_record_ids_found": [' in data
    assert '"missing_gold_record_ids": []' in data
    assert '"top1_clause_hit": true' in data
    assert '"reranking_latency_ms": 0.0' in data


def test_evaluate_case_calculates_true_recall_from_gold_ids() -> None:
    test_case = RetrievalTestCase(
        question="Find selected audit passages.",
        expected_clause_type="Audit Rights",
        keywords=[],
        expected_record_ids=["gold-1", "gold-2", "gold-3", "gold-4"],
        category="true_recall",
    )
    results = [
        ClauseSearchResult(score=0.9, payload={"id": "gold-1", "clause_type": "Audit Rights"}),
        ClauseSearchResult(score=0.8, payload={"id": "miss-1", "clause_type": "Audit Rights"}),
        ClauseSearchResult(score=0.7, payload={"id": "gold-2", "clause_type": "Audit Rights"}),
        ClauseSearchResult(score=0.6, payload={"id": "miss-2", "clause_type": "Audit Rights"}),
        ClauseSearchResult(score=0.5, payload={"id": "miss-3", "clause_type": "Audit Rights"}),
    ]

    result = evaluate_case(test_case, results, top_k=5)

    assert result.recall_at_k == 0.5
    assert result.hit_rate_at_k == 1.0
    assert result.gold_record_ids_found == ("gold-1", "gold-2")
    assert result.missing_gold_record_ids == ("gold-3", "gold-4")
    assert result.context_precision == 0.4


def test_result_rows_include_gold_id_diagnostics() -> None:
    test_case = RetrievalTestCase(
        question="Find selected audit passages.",
        expected_clause_type="Audit Rights",
        keywords=[],
        expected_record_ids=["gold-1", "gold-2"],
        category="true_recall",
    )
    result = evaluate_case(
        test_case,
        [
            ClauseSearchResult(score=0.9, payload={"id": "gold-1"}),
            ClauseSearchResult(score=0.8, payload={"id": "miss-1"}),
        ],
        top_k=2,
        candidate_count=4,
        max_passages_per_document=2,
        rerank_reason="dense and lexical rankings agree",
    )

    [row] = result_rows([result])

    assert row["expected_record_ids"] == ["gold-1", "gold-2"]
    assert row["retrieved_record_ids"] == ["gold-1", "miss-1"]
    assert row["gold_record_ids_found"] == ["gold-1"]
    assert row["missing_gold_record_ids"] == ["gold-2"]
    assert row["candidate_count"] == 4
    assert row["max_passages_per_document"] == 2
    assert row["rerank_reason"] == "dense and lexical rankings agree"


def test_evaluate_case_scores_zero_when_no_gold_ids_are_retrieved() -> None:
    test_case = RetrievalTestCase(
        question="Find selected audit passages.",
        expected_clause_type="Audit Rights",
        keywords=[],
        expected_record_ids=["gold-1", "gold-2"],
        category="true_recall",
    )
    results = [
        ClauseSearchResult(score=0.9, payload={"id": "miss-1", "clause_type": "Audit Rights"}),
        ClauseSearchResult(score=0.8, payload={"id": "miss-2", "clause_type": "Audit Rights"}),
    ]

    result = evaluate_case(test_case, results, top_k=5)

    assert result.recall_at_k == 0.0
    assert result.hit_rate_at_k == 0.0


def test_require_gold_record_ids_fails_for_empty_gold_sets() -> None:
    test_cases = [
        RetrievalTestCase(
            question="Does the contract restrict assignment?",
            expected_clause_type="Anti-Assignment",
            keywords=["assign"],
            category="clause_type",
        )
    ]

    with pytest.raises(ValueError, match="requires non-empty expected_record_ids"):
        validate_gold_record_ids(test_cases, Path("evaluation/tests.jsonl"))


def test_parse_args_accepts_document_dedup_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "eval.py",
            "--no-deduplicate-documents",
            "--max-passages-per-document",
            "2",
        ],
    )

    args = parse_args()

    assert args.deduplicate_documents is False
    assert args.max_passages_per_document == 2


def test_evaluate_all_forwards_deduplicate_documents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_args: list[tuple[bool, int | None]] = []
    test_case = RetrievalTestCase(
        question="Find selected audit passages.",
        expected_clause_type="Audit Rights",
        keywords=[],
        expected_record_ids=["gold-1"],
        category="true_recall",
    )

    class FakeEvalModel:
        def encode(self, query: str, *, normalize_embeddings: bool) -> list[float]:
            return [0.1, 0.2, 0.3]

    class FakeEvalClient:
        def query_points(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(points=[])

    def fake_search_clause_evidence(**kwargs: object) -> list[ClauseSearchResult]:
        captured_args.append(
            (
                bool(kwargs["deduplicate_documents"]),
                kwargs["max_passages_per_document"],  # type: ignore[arg-type]
            )
        )
        return [ClauseSearchResult(score=0.9, payload={"id": "gold-1"})]

    monkeypatch.setattr("evaluation.eval.load_tests", lambda path: [test_case])
    monkeypatch.setattr(
        "evaluation.eval.create_qdrant_client",
        lambda **kwargs: FakeEvalClient(),
    )
    monkeypatch.setattr(
        "evaluation.eval.load_embedding_model",
        lambda model_name: FakeEvalModel(),
    )
    monkeypatch.setattr("evaluation.eval.load_lexical_index", lambda: None)
    monkeypatch.setattr(
        "evaluation.eval.search_clause_evidence",
        fake_search_clause_evidence,
    )

    evaluate_all(
        tests_path=tmp_path / "cases.jsonl",
        deduplicate_documents=False,
        max_passages_per_document=2,
    )

    assert captured_args == [(False, 2)]


def test_evaluate_all_rejects_non_positive_document_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="max_passages_per_document"):
        evaluate_all(
            tests_path=tmp_path / "cases.jsonl",
            max_passages_per_document=0,
        )
