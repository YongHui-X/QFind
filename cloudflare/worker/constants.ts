export const CLAUSE_TYPES = [
  "Anti-Assignment",
  "Cap On Liability",
  "License Grant",
  "Audit Rights",
  "Termination For Convenience",
] as const;

export const MAX_CONTEXT_MESSAGES = 8;
export const MAX_MESSAGE_CHARS = 1_000;
export const MAX_CONTEXT_CHARS = 6_000;
export const MAX_REQUEST_BYTES = 16 * 1024;
export const MAX_RESULTS = 5;
export const HYBRID_CANDIDATE_LIMIT = 6;
export const RERANK_CANDIDATE_LIMIT = 3;
export const RRF_K = 60;
export const MAX_EVIDENCE_CHARS = 1_000;
export const MAX_ANSWER_TOKENS = 160;
export const UPSTREAM_TIMEOUT_MS = 45_000;
export const EMBEDDING_MODEL = "@cf/baai/bge-small-en-v1.5";
export const RERANKER_MODEL = "@cf/baai/bge-reranker-base";

export const UNSUPPORTED_TOPIC_ANSWER =
  "The current ClauseLens index only covers assignment restrictions, liability caps, license grants, audit rights, and termination for convenience. I could not match this question to one of those supported clause types.";

export const ANSWER_SYSTEM_PROMPT =
  "You are a contract clause retrieval assistant. Answer only from the provided evidence. " +
  "If the evidence is insufficient, say so. Cite evidence with bracketed numbers like [1] and [2]. " +
  "Do not merge different contracts into one rule. Qualify mixed, silent, or exceptional evidence. " +
  "Do not infer relationships between Affiliate, subsidiary, wholly owned subsidiary, assignment, transfer, and sublicensing. " +
  "Every source-specific claim must be supported by its cited text. Finish the final sentence and do not add a repeated summary.";
