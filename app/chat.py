"""Chat orchestration for ClauseLens.

This module turns the retrieval layer into a grounded chatbot:
1. keep a short message window from the current chat session,
2. rewrite the latest follow-up into a standalone retrieval query,
3. retrieve clause evidence from Qdrant,
4. generate a grounded answer from the retrieved evidence.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from app.rag import (
    COLLECTION,
    RERANK_CANDIDATE_LIMIT,
    ClauseSearchResult,
    SearchDiagnostics,
    search_clause_evidence,
    serialize_search_result,
)

MessageRole = Literal["system", "user", "assistant"]
RerankMode = Literal["auto", "off", "always"]

MAX_CONTEXT_MESSAGES = 8
MAX_EVIDENCE_CHARS = 1000
MAX_ANSWER_TOKENS = 160
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_OLLAMA_MODEL = "llama3.2:latest"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"

# Kept as a compatibility export for older test utilities. Follow-up
# contextualization no longer calls an LLM.
QUERY_REWRITE_SYSTEM_PROMPT = (
    "Legacy query-rewrite prompt; deterministic contextualization is active."
)

ANSWER_SYSTEM_PROMPT = (
    "You are a contract clause retrieval assistant. Answer only from the "
    "provided evidence. If the evidence is not enough to answer, say that the "
    "indexed clauses do not provide enough support. Do not invent facts. "
    "Cite evidence with bracketed numbers like [1] and [2]. Follow the word "
    "budget in the user prompt: one direct answer sentence, then at most one "
    "sentence per source. "
    "Include only the most material distinction and finish the final sentence. Refer to "
    "'the retrieved agreement' or 'the retrieved agreements' when describing "
    "scope. Do not merge clauses from different contracts into one synthetic "
    "contract rule. If sources differ, are silent, or contain exceptions, lead "
    "with a qualified comparison such as 'The retrieved agreements differ' "
    "instead of an unconditional yes or no. Silence in one source is not "
    "evidence for or against a proposition. "
    "Distinguish sublicensing, assignment, and transfer rights; do not infer one "
    "right solely from language addressing another. Do not infer that an "
    "Affiliate is a subsidiary, that a subsidiary is wholly owned, or that an "
    "entity fits a defined term unless the provided evidence says so. Every "
    "source-specific claim must be directly supported by its cited text. "
    "End after the last source-specific statement. Never add a final summary "
    "sentence that restates the opening conclusion."
)

UNSUPPORTED_TOPIC_ANSWER = (
    "The current ClauseLens index only covers assignment restrictions, liability "
    "caps, license grants, audit rights, and termination for convenience. I could "
    "not match this question to one of those supported clause types."
)

CLAUSE_TYPE_TERMS: dict[str, tuple[str, ...]] = {
    "Anti-Assignment": (
        "anti-assignment",
        "assign",
        "assignment",
        "transfer the agreement",
        "transfer this agreement",
        "transferred to another party",
    ),
    "Cap On Liability": (
        "cap on liability",
        "liability cap",
        "limit liability",
        "limitation of liability",
        "limited liability",
        "damages",
        "consequential loss",
        "consequential damages",
        "indirect damages",
        "lost profits",
        "categories of loss",
        "category of loss",
        "excluded losses",
        "losses excluded",
        "anticipated savings",
        "prospective profits",
        "special damages",
        "punitive damages",
        "responsibility for losses",
    ),
    "License Grant": (
        "license",
        "licence",
        "licensed materials",
        "usage rights",
        "right to use",
        "rights to use",
        "use intellectual property",
        "use the intellectual property",
    ),
    "Audit Rights": (
        "audit",
        "inspect records",
        "inspect books",
        "review records",
        "review compliance records",
        "books and records",
    ),
    "Termination For Convenience": (
        "terminate",
        "termination",
        "for convenience",
        "without cause",
        "end the agreement",
        "ending the agreement",
        "cancel the agreement",
        "walk away",
    ),
}


class ChatMessage(BaseModel):
    """One message in a chat conversation."""

    role: MessageRole
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def strip_content(cls, value: str) -> str:
        clean_value = value.strip()
        if not clean_value:
            raise ValueError("content must not be empty")
        return clean_value


class ChatRequest(BaseModel):
    """Request body for grounded chat."""

    # The frontend sends the full recent turn window on every request.
    # That keeps the API stateless while still allowing follow-up questions.
    messages: list[ChatMessage] = Field(min_length=1)
    clause_type: str | None = Field(
        default=None,
        description="Optional CUAD clause-type filter",
    )
    limit: int = Field(default=5, ge=1, le=20)
    rerank_mode: RerankMode = "auto"

    @field_validator("clause_type")
    @classmethod
    def blank_clause_type_becomes_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        clean_value = value.strip()
        return clean_value or None

    @field_validator("messages")
    @classmethod
    def messages_must_end_with_user(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        # The last turn must be the user's latest question; otherwise there is
        # nothing to rewrite, search, or answer.
        if not value:
            raise ValueError("messages must not be empty")
        if value[-1].role != "user":
            raise ValueError("last message must be from the user")
        return value


class ChatTurnTimings(BaseModel):
    """Latency breakdown for one chat turn."""

    total_latency_ms: float = 0.0
    rewrite_latency_ms: float = 0.0
    contextualization_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    embedding_latency_ms: float = 0.0
    vector_search_latency_ms: float = 0.0
    lexical_search_latency_ms: float = 0.0
    reranker_loading_latency_ms: float = 0.0
    reranking_latency_ms: float = 0.0
    answer_latency_ms: float = 0.0
    first_token_latency_ms: float = 0.0
    generation_first_token_latency_ms: float = 0.0


class GenerationDiagnostics(BaseModel):
    """Prompt-size diagnostics used to investigate generation TTFT."""

    prompt_chars: int = 0
    evidence_chars: int = 0
    estimated_input_tokens: int = 0
    output_chars: int = 0
    estimated_output_tokens: int = 0
    model: str | None = None
    requested_service_tier: str | None = None
    response_service_tier: str | None = None
    request_id: str | None = None


class ChatResult(BaseModel):
    """Assistant response plus the retrieval evidence behind it."""

    turn_id: str
    question: str
    standalone_query: str
    clause_type: str | None
    resolved_clause_type: str | None
    abstained: bool = False
    reranking_applied: bool = False
    rerank_reason: str
    limit: int
    result_count: int
    answer: str
    results: list[dict[str, object]]
    timings: ChatTurnTimings
    generation: GenerationDiagnostics = Field(
        default_factory=GenerationDiagnostics
    )


class ChatCompletionClient(Protocol):
    """Minimal interface required from an LLM client."""

    def complete(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str: ...

    def stream_complete(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Iterable[str]: ...


class RetrievalEngine(Protocol):
    """Minimal retrieval surface used by the chat orchestration layer."""

    client: Any
    model: Any
    reranker: Any
    reranking_enabled: bool


def _clean_env_value(value: str | None) -> str | None:
    """Treat blank environment values like missing values."""

    if value is None:
        return None
    clean_value = value.strip()
    return clean_value or None


@dataclass
class OpenAIChatCompletionClient:
    """OpenAI chat-completions wrapper with a tiny surface for tests."""

    model: str
    api_key: str | None = None
    base_url: str | None = None
    reasoning_effort: str | None = None
    service_tier: str | None = None
    timeout_seconds: float = 30.0
    max_retries: int = 2
    _client_instance: Any = field(default=None, init=False, repr=False)
    _last_response_metadata: dict[str, str | None] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def _client(self) -> Any:
        if self._client_instance is not None:
            return self._client_instance

        from openai import OpenAI

        kwargs: dict[str, str] = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client_instance = OpenAI(
            **kwargs,
            timeout=self.timeout_seconds,
            max_retries=self.max_retries,
        )
        return self._client_instance

    def _capture_metadata(
        self,
        response: Any,
        *,
        service_tier: Any = None,
        response_model: Any = None,
    ) -> None:
        self._last_response_metadata = {
            "model": str(response_model or self.model),
            "requested_service_tier": self.service_tier or "standard",
            "response_service_tier": (
                str(service_tier) if service_tier is not None else None
            ),
            "request_id": (
                str(getattr(response, "_request_id", "") or "") or None
            ),
        }

    def response_metadata(self) -> dict[str, str | None]:
        return dict(self._last_response_metadata)

    def complete(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        client = self._client()
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                *messages,
            ],
        }
        if not self.model.startswith("gpt-5"):
            request["temperature"] = temperature
        if max_tokens is not None:
            token_parameter = (
                "max_completion_tokens"
                if self.model.startswith("gpt-5")
                else "max_tokens"
            )
            request[token_parameter] = max_tokens
        if self.reasoning_effort is not None:
            request["reasoning_effort"] = self.reasoning_effort
        if self.service_tier is not None:
            request["service_tier"] = self.service_tier
        response = client.chat.completions.create(
            **request,
        )
        self._capture_metadata(
            response,
            service_tier=getattr(response, "service_tier", None),
            response_model=getattr(response, "model", None),
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI response did not include assistant text")
        return content.strip()

    def stream_complete(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        client = self._client()
        request: dict[str, Any] = {
            "model": self.model,
            "stream": True,
            "messages": [
                {"role": "system", "content": system_prompt},
                *messages,
            ],
        }
        if not self.model.startswith("gpt-5"):
            request["temperature"] = temperature
        if max_tokens is not None:
            token_parameter = (
                "max_completion_tokens"
                if self.model.startswith("gpt-5")
                else "max_tokens"
            )
            request[token_parameter] = max_tokens
        if self.reasoning_effort is not None:
            request["reasoning_effort"] = self.reasoning_effort
        if self.service_tier is not None:
            request["service_tier"] = self.service_tier
        response = client.chat.completions.create(**request)
        response_service_tier = None
        response_model = None
        for chunk in response:
            chunk_model = getattr(chunk, "model", None)
            if chunk_model is not None:
                response_model = chunk_model
            chunk_tier = getattr(chunk, "service_tier", None)
            if chunk_tier is not None:
                response_service_tier = chunk_tier
            content = chunk.choices[0].delta.content
            if content:
                yield content
        self._capture_metadata(
            response,
            service_tier=response_service_tier,
            response_model=response_model,
        )

@dataclass
class ChatEngine:
    """Dependencies needed to answer one grounded chat turn."""

    search_engine: RetrievalEngine
    llm: ChatCompletionClient
    model_name: str
    max_context_messages: int = MAX_CONTEXT_MESSAGES
    rerank_candidate_limit: int = RERANK_CANDIDATE_LIMIT


@dataclass
class ChatTurnContext:
    """Resolved chat state before answer generation."""

    turn_id: str
    question: str
    standalone_query: str
    requested_clause_type: str | None
    resolved_clause_type: str | None
    trimmed_messages: list[ChatMessage]
    results: list[ClauseSearchResult]
    timings: ChatTurnTimings
    reranking_applied: bool
    rerank_reason: str
    abstained: bool = False


def create_openai_chat_client(
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    reasoning_effort: str | None = None,
    service_tier: str | None = None,
) -> OpenAIChatCompletionClient:
    """Create the default OpenAI-backed chat client."""

    effective_model = (
        model or _clean_env_value(os.getenv("OPENAI_MODEL")) or DEFAULT_OPENAI_MODEL
    )
    configured_reasoning = reasoning_effort or _clean_env_value(
        os.getenv("OPENAI_REASONING_EFFORT")
    )
    if configured_reasoning is None and effective_model.startswith("gpt-5"):
        configured_reasoning = "none"
    configured_tier = service_tier or _clean_env_value(
        os.getenv("OPENAI_SERVICE_TIER")
    )
    if configured_tier in {None, "standard"}:
        configured_tier = None
    return OpenAIChatCompletionClient(
        model=effective_model,
        api_key=api_key or _clean_env_value(os.getenv("OPENAI_API_KEY")),
        base_url=base_url or _clean_env_value(os.getenv("OPENAI_BASE_URL")),
        reasoning_effort=configured_reasoning,
        service_tier=configured_tier,
        timeout_seconds=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30")),
        max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "2")),
    )


def create_ollama_chat_client(
    *,
    model: str | None = None,
    base_url: str | None = None,
) -> OpenAIChatCompletionClient:
    """Create an Ollama client through its OpenAI-compatible API."""

    return OpenAIChatCompletionClient(
        model=model or _clean_env_value(os.getenv("OLLAMA_MODEL")) or DEFAULT_OLLAMA_MODEL,
        api_key="ollama",
        base_url=(
            base_url
            or _clean_env_value(os.getenv("OLLAMA_BASE_URL"))
            or DEFAULT_OLLAMA_BASE_URL
        ),
        reasoning_effort="none",
    )


def create_chat_client() -> OpenAIChatCompletionClient:
    """Create the hosted OpenAI chat backend used by ClauseLens."""

    return create_openai_chat_client()


def trim_messages(
    messages: list[ChatMessage],
    *,
    max_messages: int = MAX_CONTEXT_MESSAGES,
) -> list[ChatMessage]:
    """Keep the most recent message window for context."""

    # We intentionally cap context so the prompt stays short and predictable.
    # For a first chatbot version, recent-turn memory is enough to resolve
    # follow-ups like "what about that clause?" without building long-term state.
    if max_messages < 1:
        raise ValueError("max_messages must be at least 1")
    if len(messages) <= max_messages:
        return messages
    return messages[-max_messages:]


def conversation_transcript(messages: list[ChatMessage]) -> str:
    """Render chat messages into a compact plain-text transcript."""

    # The rewrite and answer prompts both consume a simple transcript instead
    # of raw JSON so the model sees the conversation in plain language.
    lines: list[str] = []
    for message in messages:
        label = "User" if message.role == "user" else "Assistant"
        lines.append(f"{label}: {message.content}")
    return "\n".join(lines)


def normalize_standalone_query(raw_query: str, fallback: str) -> str:
    """Clean up the query rewrite output."""

    # LLMs sometimes add prefixes or surrounding quotes even when instructed
    # not to. Strip those so retrieval gets a clean semantic query string.
    clean_query = raw_query.strip().strip('"').strip("'")
    if not clean_query:
        return fallback
    for prefix in ("Query:", "Search query:", "Standalone query:"):
        if clean_query.lower().startswith(prefix.lower()):
            clean_query = clean_query[len(prefix) :].strip()
            break
    return clean_query or fallback


def infer_clause_type(query: str) -> str | None:
    """Infer a supported starter clause type from legal concept phrases."""

    normalized_query = " ".join(query.lower().replace("-", " ").split())
    matches: list[tuple[int, str]] = []
    for clause_type, terms in CLAUSE_TYPE_TERMS.items():
        score = sum(
            len(term.split())
            for term in terms
            if term.replace("-", " ") in normalized_query
        )
        if score:
            matches.append((score, clause_type))

    # Legal paraphrases often reverse word order, for example
    # "intellectual property use" instead of "use intellectual property".
    if "intellectual property" in normalized_query and any(
        term in normalized_query
        for term in ("use", "right", "rights", "grant", "granted", "license", "licence")
    ):
        matches.append((4, "License Grant"))

    if not matches:
        return None
    matches.sort(key=lambda item: item[0], reverse=True)
    return matches[0][1]


def needs_query_rewrite(messages: list[ChatMessage]) -> bool:
    """Return whether the latest question needs conversation context."""

    return sum(message.role == "user" for message in messages) > 1


def is_referential_follow_up(query: str) -> bool:
    """Return whether a question depends on a previously established subject."""

    normalized = " ".join(query.lower().strip(" \"'“”‘’").split())
    words = normalized.split()
    follow_up_phrases = (
        "what about",
        "how about",
        "how long",
        "how much",
        "how often",
        "is it",
        "does it",
        "can it",
        "are they",
        "do they",
        "can they",
        "is that",
        "does that",
        "also",
    )
    has_referential_shape = (
        len(words) <= 8
        or any(normalized.startswith(phrase) for phrase in follow_up_phrases)
    )
    has_reference_token = any(
        token in words
        for token in ("it", "its", "they", "them", "that", "this", "also")
    )
    return (
        has_referential_shape
        and has_reference_token
        or normalized.startswith(("how long", "how much", "how often", "what about"))
    )


def infer_conversation_clause_type(
    messages: list[ChatMessage],
    *,
    requested_clause_type: str | None = None,
) -> str | None:
    """Resolve the latest topic while preserving supported follow-up context."""

    if requested_clause_type:
        return requested_clause_type

    latest = messages[-1].content
    latest_type = infer_clause_type(latest)
    if latest_type is not None and not is_referential_follow_up(latest):
        return latest_type

    for message in reversed(messages[:-1]):
        prior_type = infer_clause_type(message.content)
        if prior_type is not None:
            return prior_type
    return latest_type


def contextualize_follow_up(
    messages: list[ChatMessage],
    *,
    clause_type: str | None = None,
) -> str:
    """Build a standalone retrieval query without an additional LLM call."""

    latest = messages[-1].content
    if clause_type is None:
        return latest
    return build_contextualized_query(messages, clause_type=clause_type)


def build_contextualized_query(
    messages: list[ChatMessage],
    *,
    clause_type: str,
) -> str:
    """Build a focused query without carrying unrelated prior topics forward."""

    latest = messages[-1].content
    latest_type = infer_clause_type(latest)
    if latest_type == clause_type and not is_referential_follow_up(latest):
        return latest

    parts = [latest]
    if clause_type.lower() not in " ".join(parts).lower():
        parts.append(clause_type)
    return " ".join(dict.fromkeys(parts))


def rewrite_standalone_query(
    *,
    llm: ChatCompletionClient,
    messages: list[ChatMessage],
    clause_type: str | None = None,
) -> str:
    """Compatibility wrapper for deterministic follow-up contextualization."""

    del llm
    return contextualize_follow_up(messages, clause_type=clause_type)


def choose_reranking(
    *,
    mode: RerankMode,
    query: str,
    resolved_clause_type: str,
) -> tuple[bool, str]:
    """Choose reranking only where the measured starter evaluation benefits."""

    if mode == "off":
        return False, "disabled by request"
    if mode == "always":
        return True, "enabled by request"

    normalized = " ".join(query.lower().replace("-", " ").split())
    has_ip_phrase = "intellectual property" in normalized
    has_usage_language = any(
        term in normalized
        for term in ("use", "usage", "right", "rights", "grant", "granted", "provision")
    )
    has_explicit_license = "license" in normalized or "licence" in normalized
    ip_paraphrase = (
        resolved_clause_type == "License Grant"
        and has_ip_phrase
        and has_usage_language
        and not has_explicit_license
    )
    detail_terms = (
        "affiliate",
        "anniversary",
        "cost",
        "consequence",
        "consequences",
        "days",
        "duration",
        "effective",
        "exception",
        "exceptions",
        "how much",
        "how often",
        "operation of law",
        "percent",
        "perpetual",
        "prior notice",
        "subsidiary",
        "territory",
        "threshold",
        "void",
        "voidable",
        "what happens",
        "wholly owned",
        "written notice",
    )
    nuanced_detail = any(term in normalized for term in detail_terms)
    if ip_paraphrase:
        return True, "adaptive intellectual-property paraphrase"
    if nuanced_detail:
        return True, "adaptive clause-detail question"
    return False, "adaptive vector search"


def _load_reranker(search_engine: RetrievalEngine) -> tuple[Any, float]:
    reranker = getattr(search_engine, "reranker", None)
    if reranker is not None:
        return reranker, 0.0

    loader = getattr(search_engine, "get_reranker", None)
    if loader is None:
        raise ValueError("reranker is not available")
    started = time.perf_counter()
    reranker = loader()
    return reranker, (time.perf_counter() - started) * 1000


def _prepare_chat_turn(
    *,
    engine: ChatEngine,
    request: ChatRequest,
) -> ChatTurnContext:
    """Resolve query rewrite, routing, and retrieval before answering."""

    started = time.perf_counter()
    timings = ChatTurnTimings()
    turn_id = uuid4().hex

    # Keep the request window bounded before doing any expensive work.
    trimmed_messages = trim_messages(
        request.messages,
        max_messages=engine.max_context_messages,
    )
    latest_user_message = trimmed_messages[-1].content

    resolved_clause_type = infer_conversation_clause_type(
        trimmed_messages,
        requested_clause_type=request.clause_type,
    )

    if needs_query_rewrite(trimmed_messages) and resolved_clause_type is not None:
        contextualization_started = time.perf_counter()
        standalone_query = build_contextualized_query(
            trimmed_messages,
            clause_type=resolved_clause_type,
        )
        elapsed = (time.perf_counter() - contextualization_started) * 1000
        timings.contextualization_latency_ms = elapsed
        timings.rewrite_latency_ms = elapsed
    else:
        standalone_query = latest_user_message

    if resolved_clause_type is None:
        timings.total_latency_ms = (time.perf_counter() - started) * 1000
        return ChatTurnContext(
            turn_id=turn_id,
            question=latest_user_message,
            standalone_query=standalone_query,
            requested_clause_type=request.clause_type,
            resolved_clause_type=None,
            trimmed_messages=trimmed_messages,
            results=[],
            timings=timings,
            reranking_applied=False,
            rerank_reason="unsupported topic",
            abstained=True,
        )

    reranking_applied, rerank_reason = choose_reranking(
        mode=request.rerank_mode,
        query=standalone_query,
        resolved_clause_type=resolved_clause_type,
    )
    reranker = None
    if reranking_applied:
        reranker, timings.reranker_loading_latency_ms = _load_reranker(
            engine.search_engine
        )

    diagnostics = SearchDiagnostics()
    results = search_clause_evidence(
        client=engine.search_engine.client,
        model=engine.search_engine.model,
        query=standalone_query,
        clause_type=resolved_clause_type,
        limit=request.limit,
        collection_name=COLLECTION,
        reranker=reranker,
        rerank=reranking_applied,
        candidate_limit=engine.rerank_candidate_limit,
        lexical_index=getattr(engine.search_engine, "lexical_index", None),
        adaptive_rerank=request.rerank_mode == "auto",
        diagnostics=diagnostics,
    )
    timings.retrieval_latency_ms = (
        diagnostics.embedding_latency_ms
        + diagnostics.vector_search_latency_ms
        + diagnostics.lexical_search_latency_ms
    )
    timings.reranking_latency_ms = diagnostics.reranking_latency_ms
    timings.embedding_latency_ms = diagnostics.embedding_latency_ms
    timings.vector_search_latency_ms = diagnostics.vector_search_latency_ms
    timings.lexical_search_latency_ms = diagnostics.lexical_search_latency_ms
    reranking_applied = diagnostics.reranking_applied
    if request.rerank_mode == "auto":
        rerank_reason = diagnostics.rerank_reason

    return ChatTurnContext(
        turn_id=turn_id,
        question=latest_user_message,
        standalone_query=standalone_query,
        requested_clause_type=request.clause_type,
        resolved_clause_type=resolved_clause_type,
        trimmed_messages=trimmed_messages,
        results=results,
        timings=timings,
        reranking_applied=reranking_applied,
        rerank_reason=rerank_reason,
    )


def resolve_clause_type(
    *,
    requested_clause_type: str | None,
    standalone_query: str,
) -> str | None:
    """Prefer an explicit filter, otherwise infer a supported starter topic."""

    return requested_clause_type or infer_clause_type(standalone_query)


def _query_terms(query: str) -> set[str]:
    """Return useful lexical terms for lightweight evidence extraction."""

    stopwords = {
        "about",
        "agreement",
        "contract",
        "does",
        "from",
        "have",
        "into",
        "provision",
        "retrieved",
        "right",
        "rights",
        "that",
        "the",
        "this",
        "what",
        "when",
        "which",
        "with",
    }
    return {
        term
        for term in re.findall(r"[a-z0-9]+", query.lower())
        if len(term) >= 4 and term not in stopwords
    }


def select_relevant_evidence(
    text: str,
    query: str,
    *,
    max_chars: int = MAX_EVIDENCE_CHARS,
) -> str:
    """Select query-relevant clause segments to reduce answer-prompt prefill."""

    clean_text = " ".join(text.split())
    if len(clean_text) <= max_chars:
        return clean_text

    segments = [
        segment.strip()
        for segment in re.split(r"(?<=[.;:])\s+", clean_text)
        if segment.strip()
    ]
    terms = _query_terms(query)
    ranked = sorted(
        enumerate(segments),
        key=lambda item: (
            -sum(term in item[1].lower() for term in terms),
            item[0],
        ),
    )
    selected_indexes: list[int] = []
    selected_chars = 0
    for index, segment in ranked:
        if selected_indexes and selected_chars + len(segment) + 1 > max_chars:
            continue
        selected_indexes.append(index)
        selected_chars += len(segment) + 1
        if selected_chars >= max_chars * 0.75:
            break

    if not selected_indexes:
        return clean_text[:max_chars].rstrip()
    selected = " ".join(segments[index] for index in sorted(selected_indexes))
    return selected[:max_chars].rstrip()


def short_source_label(result: ClauseSearchResult) -> str:
    """Return a compact human-readable source label for answer generation."""

    raw_source = result.document_id or result.source_pdf or "Unknown"
    stem = raw_source.rsplit(".", 1)[0]
    prefix = re.split(r"[,_]", stem, maxsplit=1)[0].strip()
    compact = re.sub(
        r"(INC|CORP|CORPORATION|LLC|LTD|LIMITED)$",
        "",
        prefix,
        flags=re.IGNORECASE,
    ).strip()
    return compact or prefix or "Unknown"


def select_answer_results(
    results: list[ClauseSearchResult],
    *,
    query: str,
) -> list[ClauseSearchResult]:
    """Choose evidence sent to the answer model without changing UI results."""

    limit = 3 if is_comparative_or_multi_source_query(query) else 2
    selected: list[ClauseSearchResult] = []
    seen_documents: set[str] = set()
    for result in results:
        document_key = (
            getattr(result, "document_id", None)
            or getattr(result, "record_id", None)
            or result.text
        )
        if document_key in seen_documents:
            continue
        seen_documents.add(document_key)
        selected.append(result)
        if len(selected) >= limit:
            break
    return selected


def asks_for_specific_provision(query: str) -> bool:
    """Return whether the user wants a clause or provision identified."""

    normalized = " ".join(query.lower().split())
    return any(
        phrase in normalized
        for phrase in (
            "specific provision",
            "specific clause",
            "which provision",
            "which clause",
        )
    )


def is_comparative_or_multi_source_query(query: str) -> bool:
    """Return whether the answer should preserve more than one evidence block."""

    normalized = " ".join(query.lower().split())
    comparative_terms = (
        "compare",
        "comparison",
        "difference",
        "different",
        "differ",
        "how do",
        "how does",
        "versus",
        "vs",
        "between",
        "across",
        "multiple",
        "both",
        "all agreements",
        "retrieved agreements",
    )
    multi_source_terms = (
        "how often",
        "how much",
        "what happens",
        "operation of law",
        "wholly owned subsidiary",
        "affiliate",
        "anniversary",
        "prior notice",
    )
    return any(term in normalized for term in comparative_terms + multi_source_terms)


def format_retrieval_context(
    results: list[ClauseSearchResult],
    *,
    query: str = "",
) -> str:
    """Turn retrieved clauses into a readable evidence block."""

    # Number each clause so the answer model can cite specific items using the
    # same bracketed labels shown to the user.
    blocks: list[str] = []
    for index, result in enumerate(results, start=1):
        evidence_text = select_relevant_evidence(result.text, query)
        blocks.append(
            "\n".join(
                [
                    f"[{index}] clause_type: {result.clause_type or 'Unknown'}",
                    f"source: {short_source_label(result)}",
                    f"answer_label: {result.answer or 'Unknown'}",
                    f"text: {evidence_text}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_answer_prompt(
    *,
    question: str,
    standalone_query: str,
    results: list[ClauseSearchResult],
    conversation: list[ChatMessage],
) -> tuple[str, GenerationDiagnostics]:
    """Build the compact grounded-answer prompt and its size diagnostics."""

    answer_results = select_answer_results(results, query=standalone_query)
    context_messages = trim_messages(conversation, max_messages=3)
    evidence_context = format_retrieval_context(
        answer_results,
        query=standalone_query,
    )
    safety_guidance = answer_safety_guidance(
        standalone_query,
        source_count=len({short_source_label(result) for result in answer_results}),
    )
    word_budget = 65 if asks_for_specific_provision(standalone_query) else 55
    prompt_parts = [
        "Conversation context:",
        conversation_transcript(context_messages) or "(no prior context)",
        "",
        f"User question: {question}",
        f"Standalone retrieval query: {standalone_query}",
        summarize_retrieval_scope(answer_results),
        "",
        "Retrieved evidence:",
        evidence_context,
        "",
        safety_guidance,
        f"Use at most {word_budget} words and only the evidence above. Write one direct "
        "answer sentence followed by at most one concise sentence per source. "
        "When sources differ or one is silent, qualify the opening conclusion "
        "and describe only what each cited source establishes. Never convert "
        "missing language into support, and never assume a defined relationship "
        "such as Affiliate, subsidiary, successor, transfer, or sublicense. "
        "Include only the most material distinction, cite supporting clauses, "
        "finish the final sentence, and do not add a concluding summary. Omit "
        "weaker evidence rather than leaving a sentence incomplete. If the user "
        "asks for a specific provision but no section or article identifier is "
        "present, say that the retrieved evidence does not include one.",
    ]
    prompt = "\n".join(prompt_parts)
    total_chars = len(ANSWER_SYSTEM_PROMPT) + len(prompt)
    return prompt, GenerationDiagnostics(
        prompt_chars=total_chars,
        evidence_chars=len(evidence_context),
        estimated_input_tokens=max(1, round(total_chars / 4)),
    )


def answer_safety_guidance(query: str, *, source_count: int) -> str:
    """Return deterministic grounding instructions for high-risk comparisons."""

    normalized = " ".join(query.lower().replace("-", " ").split())
    instructions: list[str] = []
    comparative_terms = (
        "compare",
        "anniversary",
        "between",
        "difference",
        "different",
        "differ",
        "how often",
        "multiple",
        "operation of law",
        "right available to both",
        "what happens",
        "wholly owned subsidiary",
    )
    if source_count > 1 and any(term in normalized for term in comparative_terms):
        instructions.append(
            "Required opening: begin with 'The retrieved agreements differ.'"
        )
    if "operation of law" in normalized:
        instructions.append(
            "Do not give an unconditional yes or no; state explicit coverage, "
            "exceptions, and silence separately."
        )
    if "wholly owned subsidiary" in normalized:
        instructions.append(
            "Do not equate Affiliate with wholly owned subsidiary. Unless the "
            "evidence explicitly says wholly owned subsidiary, state that the "
            "relationship is not established."
        )
    if "what happens" in normalized and "assign" in normalized:
        instructions.append(
            "Do not state that every unauthorized assignment is void. Attribute "
            "void or voidable consequences only to sources that expressly say so."
        )
    if not instructions:
        return "Answer safety: apply the general grounding rules."
    return "Answer safety: " + " ".join(instructions)


def summarize_retrieval_scope(results: list[ClauseSearchResult]) -> str:
    """Describe whether the retrieved evidence comes from one or many contracts."""

    sources = [
        short_source_label(result) for result in results
    ]
    unique_sources = list(dict.fromkeys(sources))
    if not unique_sources:
        return "Evidence scope: no sources returned."
    if len(unique_sources) == 1:
        return f"Evidence scope: one contract source ({unique_sources[0]})."
    source_list = "; ".join(unique_sources)
    return (
        "Evidence scope: multiple contract sources. "
        f"Do not collapse them into one contract rule. Sources: {source_list}."
    )


def generate_grounded_answer(
    *,
    llm: ChatCompletionClient,
    question: str,
    standalone_query: str,
    results: list[ClauseSearchResult],
    conversation: list[ChatMessage],
    diagnostics: GenerationDiagnostics | None = None,
) -> str:
    """Generate a grounded answer from retrieved clause evidence."""

    # If retrieval returns nothing useful, do not ask the LLM to improvise.
    # Return a direct fallback instead of fabricating an answer.
    if not results:
        return (
            "I could not find enough supporting clause evidence in the indexed "
            "contracts to answer that from the current dataset."
        )

    # The answer prompt includes both the recent chat context and the retrieved
    # evidence. The model should use the evidence as the source of truth and
    # cite it with [1], [2], etc.
    prompt, prompt_diagnostics = build_answer_prompt(
        question=question,
        standalone_query=standalone_query,
        results=results,
        conversation=conversation,
    )
    if diagnostics is not None:
        diagnostics.prompt_chars = prompt_diagnostics.prompt_chars
        diagnostics.evidence_chars = prompt_diagnostics.evidence_chars
        diagnostics.estimated_input_tokens = (
            prompt_diagnostics.estimated_input_tokens
        )
    answer = llm.complete(
        system_prompt=ANSWER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=MAX_ANSWER_TOKENS,
    )
    _populate_generation_output_diagnostics(diagnostics, llm, answer)
    return answer


def stream_grounded_answer(
    *,
    llm: ChatCompletionClient,
    question: str,
    standalone_query: str,
    results: list[ClauseSearchResult],
    conversation: list[ChatMessage],
    diagnostics: GenerationDiagnostics | None = None,
) -> Iterable[str]:
    """Stream a grounded answer from retrieved clause evidence."""

    if not results:
        yield (
            "I could not find enough supporting clause evidence in the indexed "
            "contracts to answer that from the current dataset."
        )
        return

    prompt, prompt_diagnostics = build_answer_prompt(
        question=question,
        standalone_query=standalone_query,
        results=results,
        conversation=conversation,
    )
    if diagnostics is not None:
        diagnostics.prompt_chars = prompt_diagnostics.prompt_chars
        diagnostics.evidence_chars = prompt_diagnostics.evidence_chars
        diagnostics.estimated_input_tokens = (
            prompt_diagnostics.estimated_input_tokens
        )
    chunks: list[str] = []
    for chunk in llm.stream_complete(
        system_prompt=ANSWER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=MAX_ANSWER_TOKENS,
    ):
        chunks.append(chunk)
        yield chunk
    _populate_generation_output_diagnostics(diagnostics, llm, "".join(chunks))


def _populate_generation_output_diagnostics(
    diagnostics: GenerationDiagnostics | None,
    llm: ChatCompletionClient,
    answer: str,
) -> None:
    if diagnostics is None:
        return
    diagnostics.output_chars = len(answer)
    diagnostics.estimated_output_tokens = max(1, round(len(answer) / 4))
    metadata_getter = getattr(llm, "response_metadata", None)
    metadata = metadata_getter() if callable(metadata_getter) else {}
    diagnostics.model = metadata.get("model")
    diagnostics.requested_service_tier = metadata.get("requested_service_tier")
    diagnostics.response_service_tier = metadata.get("response_service_tier")
    diagnostics.request_id = metadata.get("request_id")


def answer_chat_turn(
    *,
    engine: ChatEngine,
    request: ChatRequest,
) -> ChatResult:
    """Run one grounded chat turn end to end."""

    started = time.perf_counter()
    context = _prepare_chat_turn(engine=engine, request=request)
    if context.abstained:
        context.timings.total_latency_ms = (time.perf_counter() - started) * 1000
        return ChatResult(
            turn_id=context.turn_id,
            question=context.question,
            standalone_query=context.standalone_query,
            clause_type=context.requested_clause_type,
            resolved_clause_type=None,
            abstained=True,
            reranking_applied=False,
            rerank_reason=context.rerank_reason,
            limit=request.limit,
            result_count=0,
            answer=UNSUPPORTED_TOPIC_ANSWER,
            results=[],
            timings=context.timings,
        )

    # Only after retrieval succeeds do we ask the model to draft the final
    # grounded response from the evidence it just saw.
    answer_started = time.perf_counter()
    generation = GenerationDiagnostics()
    answer = generate_grounded_answer(
        llm=engine.llm,
        question=context.question,
        standalone_query=context.standalone_query,
        results=context.results,
        conversation=context.trimmed_messages,
        diagnostics=generation,
    )
    context.timings.answer_latency_ms = (time.perf_counter() - answer_started) * 1000
    context.timings.total_latency_ms = (time.perf_counter() - started) * 1000
    return ChatResult(
        turn_id=context.turn_id,
        question=context.question,
        standalone_query=context.standalone_query,
        clause_type=context.requested_clause_type,
        resolved_clause_type=context.resolved_clause_type,
        reranking_applied=context.reranking_applied,
        rerank_reason=context.rerank_reason,
        limit=request.limit,
        result_count=len(context.results),
        answer=answer,
        results=[serialize_search_result(result) for result in context.results],
        timings=context.timings,
        generation=generation,
    )


def stream_chat_turn(
    *,
    engine: ChatEngine,
    request: ChatRequest,
) -> Iterable[str]:
    """Stream the answer text and finish with a final structured payload."""

    started = time.perf_counter()
    if needs_query_rewrite(request.messages):
        yield json.dumps({"event": "status", "stage": "contextualizing"}) + "\n"
    else:
        yield json.dumps({"event": "status", "stage": "routing"}) + "\n"
    context = _prepare_chat_turn(engine=engine, request=request)
    if context.abstained:
        context.timings.total_latency_ms = (time.perf_counter() - started) * 1000
        yield json.dumps(
            {"event": "token", "delta": UNSUPPORTED_TOPIC_ANSWER},
            ensure_ascii=False,
        ) + "\n"
        final_result = ChatResult(
            turn_id=context.turn_id,
            question=context.question,
            standalone_query=context.standalone_query,
            clause_type=context.requested_clause_type,
            resolved_clause_type=None,
            abstained=True,
            reranking_applied=False,
            rerank_reason=context.rerank_reason,
            limit=request.limit,
            result_count=0,
            answer=UNSUPPORTED_TOPIC_ANSWER,
            results=[],
            timings=context.timings,
        )
        yield json.dumps(
            {"event": "final", "data": final_result.model_dump()},
            ensure_ascii=False,
        ) + "\n"
        return

    yield json.dumps({"event": "status", "stage": "retrieved"}) + "\n"
    if context.reranking_applied:
        yield json.dumps({"event": "status", "stage": "reranked"}) + "\n"
    yield json.dumps({"event": "status", "stage": "generating"}) + "\n"
    answer_parts: list[str] = []
    answer_started = time.perf_counter()
    generation = GenerationDiagnostics()
    for chunk in stream_grounded_answer(
        llm=engine.llm,
        question=context.question,
        standalone_query=context.standalone_query,
        results=context.results,
        conversation=context.trimmed_messages,
        diagnostics=generation,
    ):
        if not answer_parts:
            first_token_now = time.perf_counter()
            context.timings.first_token_latency_ms = (first_token_now - started) * 1000
            context.timings.generation_first_token_latency_ms = (
                first_token_now - answer_started
            ) * 1000
        answer_parts.append(chunk)
        yield json.dumps({"event": "token", "delta": chunk}, ensure_ascii=False) + "\n"
    context.timings.answer_latency_ms = (time.perf_counter() - answer_started) * 1000
    context.timings.total_latency_ms = (time.perf_counter() - started) * 1000

    final_result = ChatResult(
        turn_id=context.turn_id,
        question=context.question,
        standalone_query=context.standalone_query,
        clause_type=context.requested_clause_type,
        resolved_clause_type=context.resolved_clause_type,
        reranking_applied=context.reranking_applied,
        rerank_reason=context.rerank_reason,
        limit=request.limit,
        result_count=len(context.results),
        answer="".join(answer_parts).strip(),
        results=[serialize_search_result(result) for result in context.results],
        timings=context.timings,
        generation=generation,
    )
    yield json.dumps(
        {"event": "final", "data": final_result.model_dump()},
        ensure_ascii=False,
    ) + "\n"
