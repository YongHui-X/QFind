from types import SimpleNamespace

import pytest

from app.rag import (
    LexicalIndex,
    SearchDiagnostics,
    make_clause_type_filter,
    search_clause_evidence,
)


class FakeVector:
    def tolist(self) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def encode(self, query: str, *, normalize_embeddings: bool) -> FakeVector:
        self.calls.append(
            {
                "query": query,
                "normalize_embeddings": normalize_embeddings,
            }
        )
        return FakeVector()


class FakeClient:
    def __init__(self, points: list[SimpleNamespace] | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.points = points or [
            SimpleNamespace(
                score=0.91,
                payload={
                    "clause_type": "Audit Rights",
                    "source_pdf": "Example.pdf",
                    "text": "Audit evidence",
                },
            )
        ]

    def query_points(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(points=self.points[: int(kwargs["limit"])])


class FakeReranker:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls.append(pairs)
        return self.scores[: len(pairs)]


def test_make_clause_type_filter_returns_none_without_clause_type() -> None:
    assert make_clause_type_filter(None) is None
    assert make_clause_type_filter("") is None


def test_make_clause_type_filter_matches_clause_type() -> None:
    query_filter = make_clause_type_filter("Audit Rights")

    assert query_filter is not None
    condition = query_filter.must[0]
    assert condition.key == "clause_type"
    assert condition.match.value == "Audit Rights"


def test_search_clause_evidence_embeds_and_queries_qdrant() -> None:
    client = FakeClient()
    model = FakeModel()

    results = search_clause_evidence(
        client=client,
        model=model,
        query="  audit rights  ",
        clause_type="Audit Rights",
        limit=3,
        collection_name="contracts_clause_evidence",
    )

    assert model.calls == [
        {
            "query": "audit rights",
            "normalize_embeddings": True,
        }
    ]
    assert len(client.calls) == 1
    assert client.calls[0]["collection_name"] == "contracts_clause_evidence"
    assert client.calls[0]["query"] == [0.1, 0.2, 0.3]
    assert client.calls[0]["limit"] == 3
    assert client.calls[0]["with_payload"] is True
    assert client.calls[0]["query_filter"] is not None

    assert len(results) == 1
    assert results[0].score == 0.91
    assert results[0].clause_type == "Audit Rights"
    assert results[0].source_pdf == "Example.pdf"
    assert results[0].text == "Audit evidence"


def test_search_clause_evidence_rejects_empty_query() -> None:
    with pytest.raises(ValueError, match="query must not be empty"):
        search_clause_evidence(
            client=FakeClient(),
            model=FakeModel(),
            query=" ",
        )


def test_search_clause_evidence_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="limit must be at least 1"):
        search_clause_evidence(
            client=FakeClient(),
            model=FakeModel(),
            query="assignment",
            limit=0,
        )


def test_search_clause_evidence_reranks_candidates_and_truncates_top_k() -> None:
    client = FakeClient(
        points=[
            SimpleNamespace(score=0.95, payload={"text": "vector first"}),
            SimpleNamespace(score=0.90, payload={"text": "reranker first"}),
            SimpleNamespace(score=0.85, payload={"text": "third"}),
        ]
    )
    reranker = FakeReranker([0.1, 0.9, 0.2])
    diagnostics = SearchDiagnostics()

    results = search_clause_evidence(
        client=client,
        model=FakeModel(),
        query="license rights",
        clause_type="License Grant",
        limit=2,
        candidate_limit=3,
        reranker=reranker,  # type: ignore[arg-type]
        rerank=True,
        diagnostics=diagnostics,
    )

    assert client.calls[0]["limit"] == 3
    assert client.calls[0]["query_filter"] is not None
    assert reranker.calls == [
        [
            ("license rights", "vector first"),
            ("license rights", "reranker first"),
            ("license rights", "third"),
        ]
    ]
    assert [result.text for result in results] == ["reranker first", "third"]
    assert results[0].score == 0.9
    assert results[0].vector_score == 0.90
    assert results[0].reranker_score == 0.9
    assert diagnostics.reranking_latency_ms >= 0.0


def test_search_clause_evidence_optional_reranking_keeps_vector_order() -> None:
    client = FakeClient(
        points=[
            SimpleNamespace(score=0.8, payload={"text": "first"}),
            SimpleNamespace(score=0.7, payload={"text": "second"}),
        ]
    )

    results = search_clause_evidence(
        client=client,
        model=FakeModel(),
        query="assignment",
        limit=1,
        rerank=False,
    )

    assert client.calls[0]["limit"] == 1
    assert [result.text for result in results] == ["first"]
    assert results[0].vector_score == 0.8
    assert results[0].reranker_score is None


def test_search_clause_evidence_requires_reranker_when_enabled() -> None:
    with pytest.raises(ValueError, match="reranker is required"):
        search_clause_evidence(
            client=FakeClient(),
            model=FakeModel(),
            query="assignment",
            rerank=True,
        )


def test_hybrid_search_fuses_and_deduplicates_documents() -> None:
    client = FakeClient(
        points=[
            SimpleNamespace(
                score=0.9,
                payload={"id": "dense-1", "document_id": "doc-a", "text": "assignment"},
            ),
            SimpleNamespace(
                score=0.8,
                payload={"id": "dense-2", "document_id": "doc-a", "text": "consent"},
            ),
        ]
    )
    lexical_index = LexicalIndex(
        [
            {
                "id": "lexical-1",
                "document_id": "doc-b",
                "clause_type": "Anti-Assignment",
                "text": "assignment requires written consent",
            }
        ]
    )

    results = search_clause_evidence(
        client=client,
        model=FakeModel(),
        query="assignment written consent",
        clause_type="Anti-Assignment",
        limit=3,
        lexical_index=lexical_index,
    )

    assert {result.document_id for result in results} == {"doc-a", "doc-b"}
    assert all(result.fused_score is not None for result in results)
