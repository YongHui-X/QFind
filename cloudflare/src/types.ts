export interface Evidence {
  id: string;
  document_id: string;
  source_pdf: string;
  source_txt: string;
  clause_type: string;
  answer: string;
  text: string;
  score: number;
  vector_score: number | null;
  reranker_score: number | null;
  lexical_score: number | null;
  fused_score: number | null;
}

export interface ChatResponse {
  turn_id: string;
  answer: string;
  question: string;
  standalone_query: string;
  resolved_clause_type: string | null;
  abstained: boolean;
  reranking_applied: boolean;
  rerank_reason: string;
  results: Evidence[];
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  response?: ChatResponse;
}

export interface SavedChat {
  id: string;
  title: string;
  messages: Message[];
  clauseType: string;
  limit: number;
  updatedAt: string;
}
