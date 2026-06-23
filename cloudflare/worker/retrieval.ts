import {
  EMBEDDING_MODEL,
  HYBRID_CANDIDATE_LIMIT,
  MAX_RESULTS,
  RERANK_CANDIDATE_LIMIT,
  RERANKER_MODEL,
  RRF_K,
} from "./constants";
import type {
  Env,
  EvidenceRecord,
  LexicalArtifact,
  SearchDiagnostics,
  SearchResult,
  StaticIndex,
} from "./types";

let cachedIndex: Promise<StaticIndex> | null = null;

function assetRequest(request: Request, path: string): Request {
  const url = new URL(request.url);
  url.pathname = path;
  url.search = "";
  return new Request(url, { method: "GET" });
}

export function normalizeVector(vector: number[]): number[] {
  const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0)) || 1;
  return vector.map((value) => value / norm);
}

export async function loadStaticIndex(request: Request, env: Env): Promise<StaticIndex> {
  if (!cachedIndex) {
    cachedIndex = (async () => {
      const [manifestResponse, recordsResponse, lexicalResponse, vectorsResponse] =
        await Promise.all([
          env.ASSETS.fetch(assetRequest(request, "/generated/manifest.json")),
          env.ASSETS.fetch(assetRequest(request, "/generated/records.json")),
          env.ASSETS.fetch(assetRequest(request, "/generated/lexical.json")),
          env.ASSETS.fetch(assetRequest(request, "/generated/vectors.f32")),
        ]);
      if (
        !manifestResponse.ok ||
        !recordsResponse.ok ||
        !lexicalResponse.ok ||
        !vectorsResponse.ok
      ) {
        throw new Error("Static retrieval index is missing from the deployment");
      }
      const manifest = await manifestResponse.json<StaticIndex["manifest"]>();
      const records = await recordsResponse.json<EvidenceRecord[]>();
      const lexical = await lexicalResponse.json<LexicalArtifact>();
      const vectorBuffer = await vectorsResponse.arrayBuffer();
      const vectors = new Float32Array(vectorBuffer);
      if (
        records.length !== manifest.record_count ||
        vectors.length !== manifest.record_count * manifest.dimensions
      ) {
        throw new Error("Static retrieval index failed its dimension check");
      }
      return { manifest, records, lexical, vectors };
    })();
  }
  return cachedIndex;
}

function tokenize(text: string): string[] {
  return text.toLowerCase().match(/[a-z0-9]+/g) ?? [];
}

export function denseSearch(
  index: StaticIndex,
  queryVector: number[],
  clauseType: string,
  limit = HYBRID_CANDIDATE_LIMIT,
): SearchResult[] {
  const results: SearchResult[] = [];
  const dimensions = index.manifest.dimensions;
  for (let recordIndex = 0; recordIndex < index.records.length; recordIndex += 1) {
    const record = index.records[recordIndex];
    if (record.clause_type !== clauseType) continue;
    let score = 0;
    const offset = recordIndex * dimensions;
    for (let dimension = 0; dimension < dimensions; dimension += 1) {
      score += queryVector[dimension] * index.vectors[offset + dimension];
    }
    results.push(toSearchResult(record, score, { vector_score: score }));
  }
  results.sort((a, b) => b.score - a.score);
  return results.slice(0, limit).map((result, indexValue) => ({
    ...result,
    dense_rank: indexValue + 1,
  }));
}

export function lexicalSearch(
  records: EvidenceRecord[],
  lexical: LexicalArtifact,
  query: string,
  clauseType: string,
  limit = HYBRID_CANDIDATE_LIMIT,
): SearchResult[] {
  const queryCounts = new Map<string, number>();
  for (const term of tokenize(query)) {
    queryCounts.set(term, (queryCounts.get(term) ?? 0) + 1);
  }
  const results: SearchResult[] = [];
  const k1 = 1.5;
  const b = 0.75;
  records.forEach((record, recordIndex) => {
    if (record.clause_type !== clauseType) return;
    const frequencies = lexical.term_frequencies[recordIndex] ?? {};
    const length = lexical.lengths[recordIndex] ?? 0;
    let score = 0;
    for (const [term, queryFrequency] of queryCounts) {
      const frequency = frequencies[term] ?? 0;
      if (!frequency) continue;
      const denominator =
        frequency + k1 * (1 - b + (b * length) / (lexical.average_length || 1));
      score +=
        (lexical.idf[term] ?? 0) *
        frequency *
        ((k1 + 1) / denominator) *
        queryFrequency;
    }
    if (score > 0) results.push(toSearchResult(record, score, { lexical_score: score }));
  });
  results.sort((a, b) => b.score - a.score);
  return results.slice(0, limit).map((result, indexValue) => ({
    ...result,
    lexical_rank: indexValue + 1,
  }));
}

export function reciprocalRankFusion(
  dense: SearchResult[],
  lexical: SearchResult[],
): SearchResult[] {
  const byId = new Map<string, SearchResult>();
  const scores = new Map<string, number>();
  dense.forEach((result, indexValue) => {
    byId.set(result.id, result);
    scores.set(result.id, (scores.get(result.id) ?? 0) + 1 / (RRF_K + indexValue + 1));
  });
  lexical.forEach((result, indexValue) => {
    scores.set(result.id, (scores.get(result.id) ?? 0) + 1 / (RRF_K + indexValue + 1));
    const existing = byId.get(result.id);
    byId.set(
      result.id,
      existing
        ? { ...existing, lexical_score: result.lexical_score, lexical_rank: indexValue + 1 }
        : result,
    );
  });
  return [...byId.values()]
    .map((result) => ({
      ...result,
      score: scores.get(result.id) ?? 0,
      fused_score: scores.get(result.id) ?? 0,
    }))
    .sort((a, b) => (b.fused_score ?? 0) - (a.fused_score ?? 0));
}

export function deduplicateByDocument(results: SearchResult[]): SearchResult[] {
  const seen = new Set<string>();
  return results.filter((result) => {
    const key = result.document_id || result.id;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export function shouldRetrievalRerank(results: SearchResult[]): [boolean, string] {
  if (results.length < 2) return [false, "high-confidence single candidate"];
  const top = results[0];
  if (top.dense_rank === 1 && top.lexical_rank === 1) {
    return [false, "dense and lexical top ranks agree"];
  }
  const first = top.fused_score ?? top.vector_score ?? top.score;
  const second =
    results[1].fused_score ?? results[1].vector_score ?? results[1].score;
  const margin = Math.max(0, (first - second) / Math.max(Math.abs(first), 1e-9));
  const confidence = Math.min(
    1,
    0.7 * (top.dense_rank === 1 && top.lexical_rank === 1 ? 1 : 0) + 0.3 * margin,
  );
  return confidence >= 0.68
    ? [false, "dense and lexical rankings agree"]
    : [true, "low hybrid ranking confidence"];
}

function parseRerankerResponse(value: unknown): Array<{ id: number; score: number }> {
  const response =
    typeof value === "object" && value !== null && "response" in value
      ? (value as { response: unknown }).response
      : value;
  if (!Array.isArray(response)) throw new Error("Workers AI reranker returned invalid data");
  return response.map((item, fallbackIndex) => {
    const row = item as Record<string, unknown>;
    return {
      id: Number(row.id ?? row.index ?? fallbackIndex),
      score: Number(row.score ?? 0),
    };
  });
}

export async function searchEvidence(
  request: Request,
  env: Env,
  query: string,
  clauseType: string,
  limit: number,
  queryWantsRerank: boolean,
): Promise<{ results: SearchResult[]; diagnostics: SearchDiagnostics }> {
  const diagnostics: SearchDiagnostics = {
    reranking_applied: false,
    rerank_reason: "not requested",
    embedding_latency_ms: 0,
    vector_search_latency_ms: 0,
    lexical_search_latency_ms: 0,
    reranking_latency_ms: 0,
  };
  const index = await loadStaticIndex(request, env);
  const embeddingStarted = performance.now();
  const embeddingResponse = (await env.AI.run(EMBEDDING_MODEL, {
    text: [query],
    pooling: "mean",
  })) as { data?: number[][] };
  const queryVector = normalizeVector(embeddingResponse.data?.[0] ?? []);
  diagnostics.embedding_latency_ms = performance.now() - embeddingStarted;
  if (queryVector.length !== index.manifest.dimensions) {
    throw new Error("Workers AI embedding dimension does not match the static index");
  }

  const denseStarted = performance.now();
  const dense = denseSearch(index, queryVector, clauseType);
  diagnostics.vector_search_latency_ms = performance.now() - denseStarted;
  const lexicalStarted = performance.now();
  const lexical = lexicalSearch(index.records, index.lexical, query, clauseType);
  diagnostics.lexical_search_latency_ms = performance.now() - lexicalStarted;
  const candidates = deduplicateByDocument(reciprocalRankFusion(dense, lexical));

  const [retrievalWantsRerank, retrievalReason] = shouldRetrievalRerank(candidates);
  const applyRerank = queryWantsRerank && retrievalWantsRerank;
  diagnostics.rerank_reason = queryWantsRerank
    ? retrievalReason
    : "adaptive vector search";
  if (!applyRerank) return { results: candidates.slice(0, limit), diagnostics };

  const rerankCandidates = candidates.slice(0, RERANK_CANDIDATE_LIMIT);
  const rerankStarted = performance.now();
  const rerankResponse = await env.AI.run(RERANKER_MODEL, {
    query,
    contexts: rerankCandidates.map((candidate) => ({ text: candidate.text })),
    top_k: rerankCandidates.length,
  });
  const rankings = parseRerankerResponse(rerankResponse);
  diagnostics.reranking_latency_ms = performance.now() - rerankStarted;
  diagnostics.reranking_applied = true;
  const reranked: SearchResult[] = [];
  for (const { id, score } of rankings) {
    const candidate = rerankCandidates[id];
    if (candidate) {
      reranked.push({ ...candidate, score, reranker_score: score });
    }
  }
  reranked.sort((a, b) => (b.reranker_score ?? 0) - (a.reranker_score ?? 0));
  return {
    results: [...reranked, ...candidates.slice(RERANK_CANDIDATE_LIMIT)].slice(
      0,
      Math.min(limit, MAX_RESULTS),
    ),
    diagnostics,
  };
}

function toSearchResult(
  record: EvidenceRecord,
  score: number,
  overrides: Partial<SearchResult> = {},
): SearchResult {
  return {
    ...record,
    score,
    vector_score: null,
    reranker_score: null,
    lexical_score: null,
    fused_score: null,
    dense_rank: null,
    lexical_rank: null,
    ...overrides,
  };
}
