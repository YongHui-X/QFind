"""Shared RAG/Qdrant helpers for ClauseLens.

RAG means Retrieval-Augmented Generation. In this project, the retrieval part is:
1. turn contract/clause text into embeddings,
2. store those embeddings in Qdrant,
3. search Qdrant later to find relevant contract evidence.

This module intentionally does not create a global Qdrant client or load the
embedding model at import time. Those operations can be slow or fail if Qdrant
is not running, so we expose functions that callers can run explicitly.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sentence_transformers import CrossEncoder, SentenceTransformer

# Name of the Qdrant collection where contract clause vectors are stored.
COLLECTION = "contracts_clause_evidence"

# SentenceTransformer model used to convert text into 384-number vectors.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Cross-encoder used to rescore the vector-retrieval candidate pool.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Small candidate pool chosen for the latency-sensitive chat path. Retrieval
# evaluation must remain the gate for changing this value.
RERANK_CANDIDATE_LIMIT = int(os.getenv("RERANK_CANDIDATE_LIMIT", "3"))
HYBRID_CANDIDATE_LIMIT = int(os.getenv("HYBRID_CANDIDATE_LIMIT", "6"))
RRF_K = int(os.getenv("RRF_K", "60"))
DEFAULT_EVIDENCE_PATH = Path("data/processed/starter_clause_evidence.jsonl")

# Embedded/local Qdrant path. This works without Docker or a Qdrant server.
QDRANT_PATH = Path("data/qdrant_local")

# Server URL used when Qdrant is running separately, usually via Docker.
QDRANT_URL = "http://localhost:6333"

# __all__ is optional. It documents which names this module expects other code
# to import when someone writes "from app.rag import ...".
__all__ = [
    "ClauseSearchResult",
    "COLLECTION",
    "EMBEDDING_MODEL",
    "RERANKER_MODEL",
    "RERANK_CANDIDATE_LIMIT",
    "HYBRID_CANDIDATE_LIMIT",
    "DEFAULT_EVIDENCE_PATH",
    "QDRANT_PATH",
    "QDRANT_URL",
    "Distance",
    "FieldCondition",
    "Filter",
    "MatchValue",
    "PointStruct",
    "QdrantClient",
    "CrossEncoder",
    "SentenceTransformer",
    "VectorParams",
    "create_qdrant_client",
    "create_configured_qdrant_client",
    "embedding_content_hash",
    "ensure_collection",
    "load_jsonl_records",
    "make_clause_type_filter",
    "load_embedding_model",
    "load_reranker_model",
    "search_clause_evidence",
    "serialize_search_result",
    "stable_point_id",
    "SearchDiagnostics",
    "LexicalIndex",
    "load_lexical_index",
]


@dataclass(frozen=True)
class ClauseSearchResult:
    """One retrieved clause evidence result from Qdrant."""

    score: float
    payload: dict[str, Any]
    vector_score: float | None = None
    reranker_score: float | None = None
    lexical_score: float | None = None
    fused_score: float | None = None
    dense_rank: int | None = None
    lexical_rank: int | None = None

    @property
    def clause_type(self) -> str | None:
        value = self.payload.get("clause_type")
        return str(value) if value is not None else None

    @property
    def source_pdf(self) -> str | None:
        value = self.payload.get("source_pdf")
        return str(value) if value is not None else None

    @property
    def text(self) -> str:
        return str(self.payload.get("text", ""))

    @property
    def source_txt(self) -> str | None:
        value = self.payload.get("source_txt")
        return str(value) if value is not None else None

    @property
    def document_id(self) -> str | None:
        value = self.payload.get("document_id")
        return str(value) if value is not None else None

    @property
    def answer(self) -> str | None:
        value = self.payload.get("answer")
        return str(value) if value is not None else None

    @property
    def record_id(self) -> str | None:
        value = self.payload.get("id")
        return str(value) if value is not None else None


@dataclass
class SearchDiagnostics:
    """Optional timing details populated during retrieval."""

    embedding_latency_ms: float = 0.0
    vector_search_latency_ms: float = 0.0
    lexical_search_latency_ms: float = 0.0
    reranking_latency_ms: float = 0.0
    candidate_count: int = 0
    confidence: float = 0.0
    reranking_applied: bool = False
    rerank_reason: str = "not requested"


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def lexical_tokens(text: str) -> list[str]:
    """Tokenize contract text for the small in-memory BM25 index."""

    return TOKEN_PATTERN.findall(text.lower())


class LexicalIndex:
    """Small BM25 index over prepared clause evidence records."""

    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = [dict(record) for record in records]
        self.tokens = [lexical_tokens(str(record.get("text", ""))) for record in records]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = (
            sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        )
        document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            document_frequency.update(set(tokens))
        document_count = max(1, len(self.records))
        self.idf = {
            term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(
        self,
        query: str,
        *,
        clause_type: str | None,
        limit: int,
    ) -> list[tuple[float, dict[str, object]]]:
        query_terms = lexical_tokens(query)
        if not query_terms or not self.records:
            return []
        query_counts = Counter(query_terms)
        scores: list[tuple[float, dict[str, object]]] = []
        k1 = 1.5
        b = 0.75
        average_length = self.average_length or 1.0
        for record, tokens, length in zip(
            self.records, self.tokens, self.lengths, strict=True
        ):
            if clause_type and record.get("clause_type") != clause_type:
                continue
            frequencies = Counter(tokens)
            score = 0.0
            for term, query_frequency in query_counts.items():
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + k1 * (
                    1 - b + b * length / average_length
                )
                score += (
                    self.idf.get(term, 0.0)
                    * frequency
                    * (k1 + 1)
                    / denominator
                    * query_frequency
                )
            if score > 0:
                scores.append((score, record))
        scores.sort(key=lambda item: item[0], reverse=True)
        return scores[:limit]


def load_lexical_index(
    path: str | Path = DEFAULT_EVIDENCE_PATH,
) -> LexicalIndex:
    """Load the prepared evidence file into a reusable BM25 index."""

    return LexicalIndex(load_jsonl_records(Path(path)))


def stable_point_id(raw_id: str) -> int:
    """Turn a text record ID into a stable integer ID for Qdrant."""

    digest = hashlib.sha256(raw_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def embedding_content_hash(text: str) -> str:
    """Return a stable fingerprint for text that is sent to the embedder."""

    normalized_text = text.strip().replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def load_jsonl_records(path: Path) -> list[dict[str, object]]:
    """Load prepared ClauseLens evidence records from JSONL."""

    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            if not record.get("id") or not record.get("text"):
                raise ValueError(f"Record on line {line_number} must include id and text")
            records.append(record)

    return records


def create_qdrant_client(
    *,
    url: str | None = None,
    path: str | Path | None = None,
) -> QdrantClient:
    """Create a Qdrant client.

    The "*" in the function signature makes url and path keyword-only
    arguments. That means callers must write:

        create_qdrant_client(path=QDRANT_PATH)

    instead of passing values positionally. This makes calls clearer because
    url and path mean very different modes.

    Return type: "-> QdrantClient" means this function returns a QdrantClient
    object from the qdrant-client library.
    """

    if path is not None:
        # Embedded mode: Qdrant stores files in a local folder and no server
        # needs to be running on localhost:6333.
        return QdrantClient(path=str(path))

    # Server mode: connect to a running Qdrant service.
    # os.getenv("QDRANT_URL", QDRANT_URL) means:
    # use the environment variable QDRANT_URL if it exists, otherwise use the
    # default constant "http://localhost:6333".
    return QdrantClient(url=url or os.getenv("QDRANT_URL", QDRANT_URL))


def create_configured_qdrant_client(
    *,
    mode: str | None = None,
    url: str | None = None,
    path: str | Path | None = None,
) -> QdrantClient:
    """Create Qdrant using the shared server-or-embedded application setting."""

    effective_mode = (mode or os.getenv("QDRANT_MODE", "server")).strip().lower()
    if effective_mode == "server":
        return create_qdrant_client(url=url)
    if effective_mode == "embedded":
        return create_qdrant_client(path=path or QDRANT_PATH)
    raise ValueError("QDRANT_MODE must be 'server' or 'embedded'")


def ensure_collection(
    client: QdrantClient,
    *,
    collection_name: str = COLLECTION,
    vector_size: int = 384,
    distance: Distance = Distance.COSINE,
) -> None:
    """Create the Qdrant collection if it does not already exist.

    client: QdrantClient means the caller passes an already-created client.
    collection_name: str = COLLECTION means the default collection name is the
    COLLECTION constant unless the caller overrides it.
    vector_size: int = 384 matches BAAI/bge-small-en-v1.5, which outputs
    384-dimensional embeddings.
    distance controls how Qdrant compares vectors. COSINE is common for
    normalized text embeddings.
    """

    if client.collection_exists(collection_name=collection_name):
        # Return early if the collection is already there. This makes the
        # function safe to call more than once.
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=distance),
    )


def load_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """Load the sentence-transformers embedding model.

    This is a separate function because loading model weights can take time.
    Keeping it explicit helps notebooks/scripts control when the slow work
    happens.
    """

    return SentenceTransformer(model_name)


def load_reranker_model(model_name: str = RERANKER_MODEL) -> CrossEncoder:
    """Load the cross-encoder used to rerank vector-search candidates."""

    return CrossEncoder(model_name)


def make_clause_type_filter(clause_type: str | None) -> Filter | None:
    """Build a Qdrant payload filter for a CUAD clause type."""

    if not clause_type:
        return None

    return Filter(
        must=[
            FieldCondition(
                key="clause_type",
                match=MatchValue(value=clause_type),
            )
        ]
    )


def search_clause_evidence(
    *,
    client: QdrantClient,
    model: SentenceTransformer,
    query: str,
    clause_type: str | None = None,
    limit: int = 5,
    collection_name: str = COLLECTION,
    reranker: CrossEncoder | None = None,
    rerank: bool = False,
    candidate_limit: int = RERANK_CANDIDATE_LIMIT,
    lexical_index: LexicalIndex | None = None,
    hybrid_candidate_limit: int = HYBRID_CANDIDATE_LIMIT,
    adaptive_rerank: bool = False,
    diagnostics: SearchDiagnostics | None = None,
) -> list[ClauseSearchResult]:
    """Search indexed clause evidence using a natural-language query.

    The caller provides the Qdrant client and embedding model so this function
    stays easy to reuse from scripts, APIs, notebooks, or a future UI.
    """

    clean_query = query.strip()
    if not clean_query:
        raise ValueError("query must not be empty")

    if limit < 1:
        raise ValueError("limit must be at least 1")

    if candidate_limit < 1:
        raise ValueError("candidate_limit must be at least 1")

    if rerank and reranker is None:
        raise ValueError("reranker is required when rerank is enabled")

    embedding_started = time.perf_counter()
    query_vector = model.encode(
        clean_query,
        normalize_embeddings=True,
    )
    if diagnostics is not None:
        diagnostics.embedding_latency_ms = (
            time.perf_counter() - embedding_started
        ) * 1000
    if hasattr(query_vector, "tolist"):
        query_vector = query_vector.tolist()

    qdrant_limit = max(
        limit,
        candidate_limit if rerank else limit,
        hybrid_candidate_limit if lexical_index is not None else limit,
    )
    vector_search_started = time.perf_counter()
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=make_clause_type_filter(clause_type),
        limit=qdrant_limit,
        with_payload=True,
    )
    if diagnostics is not None:
        diagnostics.vector_search_latency_ms = (
            time.perf_counter() - vector_search_started
        ) * 1000

    dense_candidates = [
        ClauseSearchResult(
            score=float(point.score),
            payload=dict(point.payload or {}),
            vector_score=float(point.score),
            dense_rank=index,
        )
        for index, point in enumerate(results.points, start=1)
    ]

    candidates = dense_candidates
    if lexical_index is not None:
        lexical_started = time.perf_counter()
        lexical_results = lexical_index.search(
            clean_query,
            clause_type=clause_type,
            limit=hybrid_candidate_limit,
        )
        if diagnostics is not None:
            diagnostics.lexical_search_latency_ms = (
                time.perf_counter() - lexical_started
            ) * 1000
        candidates = reciprocal_rank_fusion(dense_candidates, lexical_results)

    candidates = deduplicate_by_document(candidates)
    if diagnostics is not None:
        diagnostics.candidate_count = len(candidates)
        diagnostics.confidence = retrieval_confidence(candidates)

    should_rerank = rerank
    rerank_reason = "enabled by request" if rerank else "not requested"
    if rerank and adaptive_rerank:
        should_rerank, rerank_reason = should_adaptively_rerank(candidates)
    if diagnostics is not None:
        diagnostics.reranking_applied = should_rerank
        diagnostics.rerank_reason = rerank_reason

    if not should_rerank:
        return candidates[:limit]

    rerank_candidates = candidates[:candidate_limit]
    remaining_candidates = candidates[candidate_limit:]
    pairs = [(clean_query, candidate.text) for candidate in rerank_candidates]
    started = time.perf_counter()
    scores = reranker.predict(pairs) if pairs else []
    if diagnostics is not None:
        diagnostics.reranking_latency_ms = (time.perf_counter() - started) * 1000

    reranked = [
        ClauseSearchResult(
            score=float(reranker_score),
            payload=candidate.payload,
            vector_score=candidate.vector_score,
            reranker_score=float(reranker_score),
            lexical_score=candidate.lexical_score,
            fused_score=candidate.fused_score,
            dense_rank=candidate.dense_rank,
            lexical_rank=candidate.lexical_rank,
        )
        for candidate, reranker_score in zip(
            rerank_candidates, scores, strict=True
        )
    ]
    reranked.sort(
        key=lambda result: (
            result.reranker_score
            if result.reranker_score is not None
            else float("-inf")
        ),
        reverse=True,
    )
    return (reranked + remaining_candidates)[:limit]


def result_key(result: ClauseSearchResult) -> str:
    """Return the stable record key used to merge dense and lexical rankings."""

    return result.record_id or hashlib.sha256(result.text.encode("utf-8")).hexdigest()


def reciprocal_rank_fusion(
    dense_results: list[ClauseSearchResult],
    lexical_results: list[tuple[float, dict[str, object]]],
    *,
    rrf_k: int = RRF_K,
) -> list[ClauseSearchResult]:
    """Fuse dense and BM25 rankings without requiring score calibration."""

    by_key: dict[str, ClauseSearchResult] = {
        result_key(result): result for result in dense_results
    }
    fused_scores: defaultdict[str, float] = defaultdict(float)
    for rank, result in enumerate(dense_results, start=1):
        fused_scores[result_key(result)] += 1.0 / (rrf_k + rank)
    for rank, (lexical_score, payload) in enumerate(lexical_results, start=1):
        lexical_result = ClauseSearchResult(
            score=lexical_score,
            payload=payload,
            lexical_score=lexical_score,
            lexical_rank=rank,
        )
        key = result_key(lexical_result)
        fused_scores[key] += 1.0 / (rrf_k + rank)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = lexical_result
        else:
            by_key[key] = ClauseSearchResult(
                score=existing.score,
                payload=existing.payload,
                vector_score=existing.vector_score,
                lexical_score=lexical_score,
                dense_rank=existing.dense_rank,
                lexical_rank=rank,
            )
    fused = [
        ClauseSearchResult(
            score=fused_scores[key],
            payload=result.payload,
            vector_score=result.vector_score,
            lexical_score=result.lexical_score,
            fused_score=fused_scores[key],
            dense_rank=result.dense_rank,
            lexical_rank=result.lexical_rank,
        )
        for key, result in by_key.items()
    ]
    fused.sort(key=lambda result: result.fused_score or 0.0, reverse=True)
    return fused


def deduplicate_by_document(
    results: list[ClauseSearchResult],
) -> list[ClauseSearchResult]:
    """Keep the highest-ranked passage from each contract document."""

    selected: list[ClauseSearchResult] = []
    seen: set[str] = set()
    for result in results:
        document_key = result.document_id or result.record_id or result.text
        if document_key in seen:
            continue
        seen.add(document_key)
        selected.append(result)
    return selected


def retrieval_confidence(results: list[ClauseSearchResult]) -> float:
    """Estimate ranking confidence from agreement and fused-score separation."""

    if not results:
        return 0.0
    top = results[0]
    agreement = 1.0 if top.dense_rank == 1 and top.lexical_rank == 1 else 0.0
    if len(results) == 1:
        margin = 1.0
    else:
        first = top.fused_score or top.vector_score or top.score
        second = (
            results[1].fused_score
            or results[1].vector_score
            or results[1].score
        )
        margin = max(0.0, (first - second) / max(abs(first), 1e-9))
    return min(1.0, 0.7 * agreement + 0.3 * margin)


def should_adaptively_rerank(
    results: list[ClauseSearchResult],
) -> tuple[bool, str]:
    """Rerank only when hybrid rankings do not strongly agree."""

    if len(results) < 2:
        return False, "high-confidence single candidate"
    top = results[0]
    if top.dense_rank == 1 and top.lexical_rank == 1:
        return False, "dense and lexical top ranks agree"
    confidence = retrieval_confidence(results)
    if confidence >= 0.68:
        return False, "dense and lexical rankings agree"
    return True, "low hybrid ranking confidence"


def serialize_search_result(result: ClauseSearchResult) -> dict[str, object]:
    """Convert a Qdrant result into the public API/UI result shape."""

    return {
        "score": result.score,
        "vector_score": result.vector_score,
        "reranker_score": result.reranker_score,
        "lexical_score": result.lexical_score,
        "fused_score": result.fused_score,
        "dense_rank": result.dense_rank,
        "lexical_rank": result.lexical_rank,
        "clause_type": result.clause_type,
        "source_pdf": result.source_pdf,
        "source_txt": result.source_txt,
        "document_id": result.document_id,
        "answer": result.answer,
        "text": result.text,
    }
