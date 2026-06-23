from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.api import SearchEngine, create_app
from app.chat import (
    ANSWER_SYSTEM_PROMPT,
    QUERY_REWRITE_SYSTEM_PROMPT,
    ChatEngine,
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
        assert normalize_embeddings is True
        return FakeVector()


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def collection_exists(self, *, collection_name: str) -> bool:
        assert collection_name == "contracts_clause_evidence"
        return True

    def query_points(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        query_filter = kwargs.get("query_filter")
        clause_type = None
        if query_filter is not None:
            clause_type = query_filter.must[0].match.value  # type: ignore[attr-defined]

        if clause_type == "Termination For Convenience":
            payload = {
                "clause_type": "Termination For Convenience",
                "source_pdf": "Example.pdf",
                "source_txt": "Example.txt",
                "document_id": "Example",
                "answer": "Yes",
                "text": "Either party may terminate for convenience upon notice.",
            }
        else:
            payload = {
                "clause_type": "Audit Rights",
                "source_pdf": "Example.pdf",
                "source_txt": "Example.txt",
                "document_id": "Example",
                "answer": "Yes",
                "text": "Customer may audit records during business hours.",
            }

        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    score=0.92,
                    payload=payload,
                )
            ]
        )


class FakeChatClient:
    def __init__(self) -> None:
        self.model = "fake-model"
        self.calls: list[dict[str, object]] = []

    def complete(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if system_prompt == QUERY_REWRITE_SYSTEM_PROMPT:
            return "termination for convenience after notice"
        if system_prompt == ANSWER_SYSTEM_PROMPT:
            return "Yes. The contract allows termination for convenience with notice. [1]"
        raise AssertionError(f"Unexpected prompt: {system_prompt}")


class FakeReranker:
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [0.75 for _ in pairs]


def build_test_client() -> tuple[TestClient, FakeClient]:
    app = create_app(warmup_enabled=False)
    fake_client = FakeClient()
    fake_llm = FakeChatClient()
    app.state.search_engine_override = SearchEngine(
        client=fake_client,
        model=FakeModel(),
    )
    app.state.chat_engine_override = ChatEngine(
        search_engine=SearchEngine(client=fake_client, model=FakeModel()),
        llm=fake_llm,
        model_name="fake-model",
    )
    return TestClient(app), fake_client


def test_root_describes_available_api_routes() -> None:
    client, _ = build_test_client()

    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "ClauseLens API"
    assert body["docs"] == "/docs"
    assert body["search"]["path"] == "/search"


def test_health_reports_collection_status() -> None:
    client, _ = build_test_client()

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["ready"] is True
    assert body["collection_ready"] is True


def test_lifespan_warms_models_before_reporting_ready(monkeypatch) -> None:
    import app.api as api_module

    fake_model = FakeModel()
    fake_client = FakeClient()
    fake_reranker = FakeReranker()
    fake_llm = FakeChatClient()

    monkeypatch.setattr(api_module, "create_qdrant_client", lambda **_: fake_client)
    monkeypatch.setattr(api_module, "load_embedding_model", lambda _: fake_model)
    monkeypatch.setattr(api_module, "load_reranker_model", lambda _: fake_reranker)
    monkeypatch.setattr(api_module, "create_chat_client", lambda: fake_llm)

    app = api_module.create_app(warmup_enabled=True)
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert fake_model.calls[0]["query"] == "contract clause retrieval warmup"


def test_clause_types_lists_supported_filters() -> None:
    client, _ = build_test_client()

    response = client.get("/clause-types")

    assert response.status_code == 200
    assert "Audit Rights" in response.json()["clause_types"]


def test_search_returns_cited_clause_results() -> None:
    client, fake_client = build_test_client()

    response = client.post(
        "/search",
        json={
            "query": "  audit rights  ",
            "clause_type": "Audit Rights",
            "limit": 3,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "audit rights"
    assert body["clause_type"] == "Audit Rights"
    assert body["result_count"] == 1
    assert body["results"][0]["score"] == 0.92
    assert body["results"][0]["vector_score"] == 0.92
    assert body["results"][0]["reranker_score"] is None
    assert body["results"][0]["source_pdf"] == "Example.pdf"
    assert body["results"][0]["source_txt"] == "Example.txt"
    assert body["results"][0]["document_id"] == "Example"
    assert body["results"][0]["answer"] == "Yes"
    assert body["results"][0]["text"].startswith("Customer may audit")
    assert fake_client.calls[0]["limit"] == 3
    assert fake_client.calls[0]["query_filter"] is not None


def test_search_uses_candidate_pool_when_reranking_is_enabled() -> None:
    app = create_app(reranking_enabled=True)
    fake_client = FakeClient()
    app.state.search_engine_override = SearchEngine(
        client=fake_client,
        model=FakeModel(),
        reranker=FakeReranker(),  # type: ignore[arg-type]
        reranking_enabled=True,
    )
    client = TestClient(app)

    response = client.post("/search", json={"query": "audit rights", "limit": 3})

    assert response.status_code == 200
    assert fake_client.calls[0]["limit"] == 3


def test_search_rejects_blank_query() -> None:
    client, _ = build_test_client()

    response = client.post("/search", json={"query": " ", "limit": 5})

    assert response.status_code == 422


def test_search_rejects_invalid_limit() -> None:
    client, _ = build_test_client()

    response = client.post("/search", json={"query": "audit rights", "limit": 0})

    assert response.status_code == 422


def test_search_accepts_blank_clause_type_as_no_filter() -> None:
    client, fake_client = build_test_client()

    response = client.post(
        "/search",
        json={"query": "audit rights", "clause_type": " ", "limit": 1},
    )

    assert response.status_code == 200
    assert response.json()["clause_type"] is None
    assert fake_client.calls[0]["query_filter"] is None


def test_chat_returns_grounded_answer_with_citations() -> None:
    client, fake_client = build_test_client()

    response = client.post(
        "/chat",
        json={
            "messages": [
                {"role": "user", "content": "What is termination for convenience?"},
                {"role": "assistant", "content": "It is a termination right."},
                {"role": "user", "content": "Can a party walk away after notice?"},
            ],
            "clause_type": "Termination For Convenience",
            "limit": 3,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "Can a party walk away after notice?"
    assert body["standalone_query"] == "Can a party walk away after notice?"
    assert body["result_count"] == 1
    assert body["answer"].startswith("Yes.")
    assert body["resolved_clause_type"] == "Termination For Convenience"
    assert body["abstained"] is False
    assert body["results"][0]["clause_type"] == "Termination For Convenience"
    assert fake_client.calls[0]["query_filter"] is not None
