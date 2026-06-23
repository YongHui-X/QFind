import json
from types import SimpleNamespace

from app.api import SearchEngine
from app.chat import (
    ANSWER_SYSTEM_PROMPT,
    QUERY_REWRITE_SYSTEM_PROMPT,
    ChatEngine,
    ChatMessage,
    ChatRequest,
    answer_chat_turn,
    build_contextualized_query,
    choose_reranking,
    contextualize_follow_up,
    generate_grounded_answer,
    infer_clause_type,
    infer_conversation_clause_type,
    needs_query_rewrite,
    rewrite_standalone_query,
    select_answer_results,
    select_relevant_evidence,
    short_source_label,
    stream_chat_turn,
    trim_messages,
)


class FakeVector:
    def tolist(self) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeModel:
    def encode(self, query: str, *, normalize_embeddings: bool) -> FakeVector:
        assert query
        assert normalize_embeddings is True
        return FakeVector()


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def query_points(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    score=0.76,
                    payload={
                        "clause_type": "Termination For Convenience",
                        "source_pdf": "Example.pdf",
                        "source_txt": "Example.txt",
                        "document_id": "Example",
                        "answer": "Yes",
                        "text": "Either party may terminate for convenience upon notice.",
                    },
                )
            ]
        )


class FakeLLM:
    def __init__(self) -> None:
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
            return "Yes. The clause allows termination for convenience with notice. [1]"
        raise AssertionError(f"Unexpected prompt: {system_prompt}")

    def stream_complete(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }
        )
        if system_prompt == ANSWER_SYSTEM_PROMPT:
            yield "Yes. "
            yield "The clause allows termination for convenience with notice. [1]"
            return
        raise AssertionError(f"Unexpected prompt: {system_prompt}")


def test_trim_messages_keeps_recent_window() -> None:
    messages = [
        ChatMessage(role="user", content=f"message {index}")
        for index in range(1, 11)
    ]

    trimmed = trim_messages(messages, max_messages=4)

    assert [message.content for message in trimmed] == [
        "message 7",
        "message 8",
        "message 9",
        "message 10",
    ]


def test_rewrite_standalone_query_uses_context() -> None:
    llm = FakeLLM()
    messages = [
        ChatMessage(role="user", content="What is termination for convenience?"),
        ChatMessage(role="assistant", content="It is a right to end the contract."),
        ChatMessage(role="user", content="Can a party walk away after notice?"),
    ]

    query = rewrite_standalone_query(
        llm=llm,
        messages=messages,
        clause_type="Termination For Convenience",
    )

    assert query == "Can a party walk away after notice?"
    assert llm.calls == []


def test_contextualize_follow_up_adds_explicit_clause_type_when_needed() -> None:
    query = contextualize_follow_up(
        [
            ChatMessage(role="user", content="What rights are provided?"),
            ChatMessage(role="assistant", content="Several rights are retrieved."),
            ChatMessage(role="user", content="Are they exclusive?"),
        ],
        clause_type="License Grant",
    )

    assert query == "Are they exclusive? License Grant"


def test_new_supported_question_does_not_include_prior_unsupported_topic() -> None:
    messages = [
        ChatMessage(role="user", content="Does the contract require arbitration?"),
        ChatMessage(role="assistant", content="That topic is unsupported."),
        ChatMessage(role="user", content="Does the license permit sublicensing?"),
    ]

    clause_type = infer_conversation_clause_type(messages)
    query = build_contextualized_query(messages, clause_type=clause_type or "")

    assert clause_type == "License Grant"
    assert query == "Does the license permit sublicensing?"


def test_referential_license_follow_ups_preserve_original_topic() -> None:
    messages = [
        ChatMessage(role="user", content="Does the license permit sublicensing?"),
        ChatMessage(role="assistant", content="Yes, the license permits sublicensing."),
        ChatMessage(role="user", content="Is it also transferable?"),
        ChatMessage(
            role="assistant",
            content="The license is generally nontransferable.",
        ),
        ChatMessage(role="user", content="How long does it remain effective?"),
    ]

    clause_type = infer_conversation_clause_type(messages)
    query = build_contextualized_query(messages, clause_type=clause_type or "")

    assert clause_type == "License Grant"
    assert query == "How long does it remain effective? License Grant"


def test_transferable_follow_up_stays_with_license_topic() -> None:
    messages = [
        ChatMessage(role="user", content="Does the license permit sublicensing?"),
        ChatMessage(role="assistant", content="Yes, the license permits sublicensing."),
        ChatMessage(role="user", content="Is it also transferable?"),
    ]

    assert infer_conversation_clause_type(messages) == "License Grant"


def test_first_turn_does_not_need_query_rewrite() -> None:
    assert needs_query_rewrite(
        [ChatMessage(role="user", content="What audit rights are provided?")]
    ) is False
    assert needs_query_rewrite(
        [
            ChatMessage(role="user", content="What audit rights are provided?"),
            ChatMessage(role="assistant", content="The evidence describes an audit."),
            ChatMessage(role="user", content="How much notice is required?"),
        ]
    ) is True


def test_choose_reranking_is_adaptive_for_ip_paraphrase() -> None:
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
    assert choose_reranking(
        mode="auto",
        query="Does the restriction cover transfers by operation of law?",
        resolved_clause_type="Anti-Assignment",
    ) == (True, "adaptive clause-detail question")


def test_infer_clause_type_handles_legal_paraphrases() -> None:
    assert (
        infer_clause_type("Does the agreement grant a right to use intellectual property?")
        == "License Grant"
    )
    assert (
        infer_clause_type("Can the agreement be transferred to another party?")
        == "Anti-Assignment"
    )
    assert (
        infer_clause_type(
            "What is the specific provision in the contract that defines the "
            "rights granted for intellectual property use?"
        )
        == "License Grant"
    )
    assert infer_clause_type("What law governs the agreement?") is None
    assert (
        infer_clause_type("Are lost profits and anticipated savings recoverable?")
        == "Cap On Liability"
    )


def test_answer_chat_turn_returns_grounded_results() -> None:
    fake_client = FakeClient()
    fake_llm = FakeLLM()
    engine = ChatEngine(
        search_engine=SearchEngine(client=fake_client, model=FakeModel()),
        llm=fake_llm,
        model_name="fake-model",
    )
    request = ChatRequest(
        messages=[
            ChatMessage(role="user", content="What is termination for convenience?"),
            ChatMessage(role="assistant", content="It is a right to end the contract."),
            ChatMessage(role="user", content="Can a party walk away after notice?"),
        ],
        clause_type="Termination For Convenience",
        limit=3,
    )

    result = answer_chat_turn(engine=engine, request=request)

    assert result.question == "Can a party walk away after notice?"
    assert result.standalone_query == "Can a party walk away after notice?"
    assert result.resolved_clause_type == "Termination For Convenience"
    assert result.abstained is False
    assert result.result_count == 1
    assert result.answer.startswith("Yes.")
    assert result.results[0]["clause_type"] == "Termination For Convenience"
    assert result.timings.total_latency_ms >= 0.0
    assert result.timings.rewrite_latency_ms >= 0.0
    assert result.timings.retrieval_latency_ms >= 0.0
    assert result.timings.answer_latency_ms >= 0.0
    assert fake_client.calls[0]["query_filter"] is not None
    assert fake_llm.calls[0]["system_prompt"] == ANSWER_SYSTEM_PROMPT
    assert fake_llm.calls[0]["max_tokens"] == 160


def test_answer_chat_turn_infers_filter_for_supported_topic() -> None:
    fake_client = FakeClient()
    fake_llm = FakeLLM()
    engine = ChatEngine(
        search_engine=SearchEngine(client=fake_client, model=FakeModel()),
        llm=fake_llm,
        model_name="fake-model",
    )
    result = answer_chat_turn(
        engine=engine,
        request=ChatRequest(
            messages=[
                ChatMessage(
                    role="user",
                    content="Does the agreement grant a right to use intellectual property?",
                )
            ],
            limit=3,
            rerank_mode="off",
        ),
    )

    assert result.resolved_clause_type == "License Grant"
    assert fake_client.calls[0]["query_filter"].must[0].match.value == "License Grant"
    assert result.timings.rewrite_latency_ms == 0.0
    assert len(fake_llm.calls) == 1
    assert fake_llm.calls[0]["system_prompt"] == ANSWER_SYSTEM_PROMPT


def test_answer_chat_turn_abstains_for_unsupported_topic() -> None:
    fake_client = FakeClient()
    fake_llm = FakeLLM()
    engine = ChatEngine(
        search_engine=SearchEngine(client=fake_client, model=FakeModel()),
        llm=fake_llm,
        model_name="fake-model",
    )
    result = answer_chat_turn(
        engine=engine,
        request=ChatRequest(
            messages=[
                ChatMessage(
                    role="user",
                    content="What law governs the agreement?",
                )
            ]
        ),
    )

    assert result.abstained is True
    assert result.resolved_clause_type is None
    assert result.result_count == 0
    assert fake_client.calls == []


def test_generate_grounded_answer_returns_fallback_without_evidence() -> None:
    fake_llm = FakeLLM()

    answer = generate_grounded_answer(
        llm=fake_llm,
        question="Can a party walk away after notice?",
        standalone_query="termination for convenience after notice",
        results=[],
        conversation=[
            ChatMessage(role="user", content="What is termination for convenience?"),
            ChatMessage(role="user", content="Can a party walk away after notice?"),
        ],
    )

    assert "enough supporting clause evidence" in answer
    assert fake_llm.calls == []


def test_generate_grounded_answer_marks_multiple_contract_sources() -> None:
    fake_llm = FakeLLM()
    results = [
        SimpleNamespace(
            clause_type="Audit Rights",
            source_pdf="ContractA.pdf",
            source_txt="ContractA.txt",
            document_id="ContractA",
            answer="Yes",
            text="Notice is required.",
        ),
        SimpleNamespace(
            clause_type="Audit Rights",
            source_pdf="ContractB.pdf",
            source_txt="ContractB.txt",
            document_id="ContractB",
            answer="Yes",
            text="Thirty days' notice is required.",
        ),
    ]

    answer = generate_grounded_answer(
        llm=fake_llm,
        question="What does the contract say about audit access?",
        standalone_query="audit rights notice period",
        results=results,
        conversation=[
            ChatMessage(role="user", content="What does the contract say about audit access?"),
        ],
    )

    assert answer.startswith("Yes.")
    prompt = fake_llm.calls[0]["messages"][0]["content"]
    assert "multiple contract sources" in prompt
    assert "Do not collapse them into one contract rule" in prompt
    assert "Never convert missing language into support" in prompt
    assert "never assume a defined relationship" in prompt
    assert "source: ContractA" in prompt
    assert "source: ContractB" in prompt
    assert "do not add a concluding summary" in prompt
    assert "Omit weaker evidence" in prompt


def test_select_relevant_evidence_prefers_query_matching_segments() -> None:
    text = (
        ("General administrative language without the requested concept. " * 20)
        + "The customer may inspect and audit books and records upon notice. "
        + ("Unrelated boilerplate concerning notices and counterparts. " * 20)
    )

    selected = select_relevant_evidence(
        text,
        "What audit rights allow inspection of records?",
        max_chars=300,
    )

    assert "inspect and audit books and records" in selected
    assert len(selected) <= 300


def test_short_source_label_removes_dataset_filename_noise() -> None:
    ceres = SimpleNamespace(
        document_id="CERES,INC_01_25_2012-EX-10.20-Collaboration Agreement",
        source_pdf=None,
    )
    biocept = SimpleNamespace(
        document_id="BIOCEPTINC_08_19_2013-EX-10-COLLABORATION AGREEMENT",
        source_pdf=None,
    )

    assert short_source_label(ceres) == "CERES"
    assert short_source_label(biocept) == "BIOCEPT"


def test_specific_provision_prompt_uses_only_two_strongest_results() -> None:
    results = [
        SimpleNamespace(text="first"),
        SimpleNamespace(text="second"),
        SimpleNamespace(text="third"),
    ]

    selected = select_answer_results(
        results,
        query="What is the specific provision defining intellectual property use?",
    )

    assert [result.text for result in selected] == ["first", "second"]


def test_create_openai_chat_client_ignores_blank_base_url(monkeypatch) -> None:
    from app.chat import create_openai_chat_client

    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "   ")

    client = create_openai_chat_client()

    assert client.model == "gpt-4.1-mini"
    assert client.api_key == "test-key"
    assert client.base_url is None
    assert client.service_tier is None


def test_create_ollama_chat_client_uses_local_defaults(monkeypatch) -> None:
    from app.chat import create_ollama_chat_client

    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    client = create_ollama_chat_client()

    assert client.model == "llama3.2:latest"
    assert client.api_key == "ollama"
    assert client.base_url == "http://localhost:11434/v1"
    assert client.reasoning_effort == "none"


def test_create_chat_client_uses_openai_even_with_legacy_provider(monkeypatch) -> None:
    from app.chat import create_chat_client

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:8b")

    client = create_chat_client()

    assert client.model == "gpt-4.1-mini"
    assert client.api_key == "test-key"
    assert client.base_url is None


def test_stream_chat_turn_emits_final_payload() -> None:
    fake_client = FakeClient()
    fake_llm = FakeLLM()
    engine = ChatEngine(
        search_engine=SearchEngine(client=fake_client, model=FakeModel()),
        llm=fake_llm,
        model_name="fake-model",
    )
    request = ChatRequest(
        messages=[
            ChatMessage(role="user", content="What is termination for convenience?"),
            ChatMessage(role="assistant", content="It is a right to end the contract."),
            ChatMessage(role="user", content="Can a party walk away after notice?"),
        ],
        clause_type="Termination For Convenience",
        limit=3,
    )

    events = [json.loads(line) for line in stream_chat_turn(engine=engine, request=request)]

    assert events[0] == {"event": "status", "stage": "contextualizing"}
    assert any(event["event"] == "token" for event in events)
    assert events[-1]["event"] == "final"
    assert events[-1]["data"]["answer"].startswith("Yes.")
    assert events[-1]["data"]["timings"]["total_latency_ms"] >= 0.0
    assert events[-1]["data"]["timings"]["first_token_latency_ms"] >= 0.0
    assert (
        events[-1]["data"]["timings"]["generation_first_token_latency_ms"] >= 0.0
    )
    assert events[-1]["data"]["generation"]["prompt_chars"] > 0
    assert events[-1]["data"]["generation"]["evidence_chars"] > 0
    assert events[-1]["data"]["generation"]["estimated_input_tokens"] > 0


def test_stream_chat_turn_abstains_without_retrieval() -> None:
    fake_client = FakeClient()
    fake_llm = FakeLLM()
    engine = ChatEngine(
        search_engine=SearchEngine(client=fake_client, model=FakeModel()),
        llm=fake_llm,
        model_name="fake-model",
    )

    events = [
        json.loads(line)
        for line in stream_chat_turn(
            engine=engine,
            request=ChatRequest(
                messages=[
                    ChatMessage(
                        role="user",
                        content="Does this agreement renew automatically?",
                    )
                ]
            ),
        )
    ]

    assert events[-1]["data"]["abstained"] is True
    assert events[-1]["data"]["results"] == []
    assert events[-1]["data"]["timings"]["total_latency_ms"] >= 0.0
    assert fake_client.calls == []
