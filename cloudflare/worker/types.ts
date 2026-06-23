export type MessageRole = "user" | "assistant";

export interface ChatMessage {
  role: MessageRole;
  content: string;
}

export interface ChatRequest {
  messages: ChatMessage[];
  clause_type?: string | null;
  limit?: number;
  turnstile_token: string;
}

export interface EvidenceRecord {
  id: string;
  document_id: string;
  source_pdf: string;
  source_txt: string;
  clause_type: string;
  answer: string;
  text: string;
}

export interface IndexManifest {
  version: number;
  record_count: number;
  dimensions: number;
  embedding_model: string;
  pooling: "mean";
  normalized: true;
  source_sha256: string;
  generated_at: string;
}

export interface StaticIndex {
  manifest: IndexManifest;
  records: EvidenceRecord[];
  vectors: Float32Array;
  lexical: LexicalArtifact;
}

export interface LexicalArtifact {
  average_length: number;
  lengths: number[];
  idf: Record<string, number>;
  term_frequencies: Array<Record<string, number>>;
}

export interface SearchResult extends EvidenceRecord {
  score: number;
  vector_score: number | null;
  reranker_score: number | null;
  lexical_score: number | null;
  fused_score: number | null;
  dense_rank: number | null;
  lexical_rank: number | null;
}

export interface SearchDiagnostics {
  reranking_applied: boolean;
  rerank_reason: string;
  embedding_latency_ms: number;
  vector_search_latency_ms: number;
  lexical_search_latency_ms: number;
  reranking_latency_ms: number;
}

export interface Env {
  ASSETS: Fetcher;
  AI: {
    run(model: string, input: unknown): Promise<unknown>;
  };
  RATE_LIMITER: DurableObjectNamespace;
  OPENAI_API_KEY: string;
  TURNSTILE_SECRET_KEY: string;
  IP_HASH_SECRET: string;
  BENCHMARK_TOKEN?: string;
  OPENAI_MODEL?: string;
  OPENAI_SERVICE_TIER?: string;
  ALLOWED_ORIGIN?: string;
}
