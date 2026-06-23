from app.rag import ClauseSearchResult
from evaluation.cases import RetrievalTestCase, load_tests
from evaluation.eval import evaluate_case, write_results


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
    assert '"top1_clause_hit": true' in data
    assert '"reranking_latency_ms": 0.0' in data
