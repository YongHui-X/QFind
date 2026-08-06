import json
from pathlib import Path

from app.routing import choose_reranking, infer_clause_type


def test_infer_clause_type_handles_supported_paraphrases() -> None:
    assert (
        infer_clause_type("Does the agreement grant a right to use intellectual property?")
        == "License Grant"
    )
    assert (
        infer_clause_type("Can the agreement be transferred to another party?")
        == "Anti-Assignment"
    )
    assert infer_clause_type("What law governs the agreement?") is None


def test_choose_reranking_keeps_adaptive_decisions() -> None:
    assert choose_reranking(
        mode="auto",
        query="What provision defines rights granted for intellectual property use?",
        resolved_clause_type="License Grant",
    ) == (True, "adaptive intellectual-property paraphrase")
    assert choose_reranking(
        mode="auto",
        query="What license rights are granted?",
        resolved_clause_type="License Grant",
    ) == (False, "adaptive vector search")


def test_gold_recall_questions_are_supported_by_router() -> None:
    """Ensure gold retrieval prompts can also be tried in the chat app."""

    path = Path("evaluation/tests_recall_gold.jsonl")
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        case = json.loads(line)
        assert infer_clause_type(case["question"]) == case["expected_clause_type"], (
            f"line {line_number} should route to {case['expected_clause_type']}"
        )
