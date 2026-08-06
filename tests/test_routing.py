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
