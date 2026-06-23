import {
  ANSWER_SYSTEM_PROMPT,
  MAX_ANSWER_TOKENS,
  MAX_EVIDENCE_CHARS,
  UNSUPPORTED_TOPIC_ANSWER,
  UPSTREAM_TIMEOUT_MS,
} from "./constants";
import {
  chooseQueryReranking,
  contextualizeQuery,
  resolveConversationClauseType,
} from "./routing";
import { searchEvidence } from "./retrieval";
import type { ChatMessage, ChatRequest, Env, SearchResult } from "./types";

const encoder = new TextEncoder();

function shortSourceLabel(result: SearchResult): string {
  const raw = result.document_id || result.source_pdf || "Unknown";
  const stem = raw.replace(/\.[^.]+$/, "");
  const prefix = stem.split(/[,_]/, 1)[0]?.trim() || "Unknown";
  return prefix.replace(/(INC|CORP|CORPORATION|LLC|LTD|LIMITED)$/i, "").trim() || prefix;
}

function selectRelevantEvidence(text: string, query: string): string {
  const clean = text.replace(/\s+/g, " ").trim();
  if (clean.length <= MAX_EVIDENCE_CHARS) return clean;
  const terms = new Set(
    (query.toLowerCase().match(/[a-z0-9]+/g) ?? []).filter((term) => term.length >= 4),
  );
  const segments = clean.split(/(?<=[.;:])\s+/);
  const ranked = segments
    .map((segment, index) => ({
      segment,
      index,
      score: [...terms].filter((term) => segment.toLowerCase().includes(term)).length,
    }))
    .sort((a, b) => b.score - a.score || a.index - b.index);
  const selected: typeof ranked = [];
  let length = 0;
  for (const item of ranked) {
    if (selected.length && length + item.segment.length + 1 > MAX_EVIDENCE_CHARS) continue;
    selected.push(item);
    length += item.segment.length + 1;
    if (length >= MAX_EVIDENCE_CHARS * 0.75) break;
  }
  return selected
    .sort((a, b) => a.index - b.index)
    .map((item) => item.segment)
    .join(" ")
    .slice(0, MAX_EVIDENCE_CHARS);
}

function isMultiSourceQuery(query: string): boolean {
  const clean = query.toLowerCase();
  return [
    "compare",
    "difference",
    "different",
    "versus",
    "between",
    "across",
    "multiple",
    "both",
    "how often",
    "how much",
    "what happens",
    "operation of law",
    "wholly owned subsidiary",
    "affiliate",
    "anniversary",
    "prior notice",
  ].some((term) => clean.includes(term));
}

function answerPrompt(
  question: string,
  standaloneQuery: string,
  results: SearchResult[],
  messages: ChatMessage[],
): string {
  const evidenceLimit = isMultiSourceQuery(standaloneQuery) ? 3 : 2;
  const answerResults = results.slice(0, evidenceLimit);
  const evidence = answerResults
    .map(
      (result, index) =>
        `[${index + 1}] clause_type: ${result.clause_type}\n` +
        `source: ${shortSourceLabel(result)}\n` +
        `answer_label: ${result.answer || "Unknown"}\n` +
        `text: ${selectRelevantEvidence(result.text, standaloneQuery)}`,
    )
    .join("\n\n");
  const conversation = messages
    .slice(-3)
    .map((message) => `${message.role === "user" ? "User" : "Assistant"}: ${message.content}`)
    .join("\n");
  const specific = /specific (provision|clause)|which (provision|clause)/i.test(standaloneQuery);
  const wordBudget = specific ? 65 : 55;
  return (
    `Conversation context:\n${conversation}\n\n` +
    `User question: ${question}\nStandalone retrieval query: ${standaloneQuery}\n\n` +
    `Retrieved evidence:\n${evidence}\n\n` +
    `Use at most ${wordBudget} words and only the evidence above. Write one direct answer ` +
    "sentence followed by at most one concise sentence per source. Keep sources separate. " +
    "Treat silence as silence, not support. Include only material distinctions and finish the final sentence. " +
    (specific
      ? "If no section or article identifier is present, say the retrieved evidence does not include one."
      : "")
  );
}

async function openAiStream(
  env: Env,
  prompt: string,
  signal: AbortSignal,
): Promise<Response> {
  if (!env.OPENAI_API_KEY) throw new Error("OpenAI is not configured");
  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    signal,
    headers: {
      Authorization: `Bearer ${env.OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: env.OPENAI_MODEL || "gpt-4.1-mini-2025-04-14",
      service_tier: env.OPENAI_SERVICE_TIER || "standard",
      stream: true,
      temperature: 0,
      max_tokens: MAX_ANSWER_TOKENS,
      messages: [
        { role: "system", content: ANSWER_SYSTEM_PROMPT },
        { role: "user", content: prompt },
      ],
    }),
  });
  if (!response.ok || !response.body) {
    const detail = await response.text();
    throw new Error(`OpenAI request failed (${response.status}): ${detail.slice(0, 300)}`);
  }
  return response;
}

async function* readOpenAiTokens(response: Response): AsyncGenerator<string> {
  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const rawLine of lines) {
      const line = rawLine.trim();
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (!data || data === "[DONE]") continue;
      const chunk = JSON.parse(data) as {
        choices?: Array<{ delta?: { content?: string | null } }>;
      };
      const token = chunk.choices?.[0]?.delta?.content;
      if (token) yield token;
    }
  }
}

function event(value: unknown): Uint8Array {
  return encoder.encode(`${JSON.stringify(value)}\n`);
}

export function streamChat(request: Request, env: Env, body: ChatRequest): Response {
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const started = performance.now();
      try {
        const messages = body.messages.slice(-8);
        const question = messages.at(-1)!.content;
        const clauseType = resolveConversationClauseType(messages, body.clause_type);
        const standaloneQuery = clauseType
          ? contextualizeQuery(messages, clauseType)
          : question;
        controller.enqueue(event({ event: "status", stage: "routing" }));
        if (!clauseType) {
          controller.enqueue(event({ event: "token", delta: UNSUPPORTED_TOPIC_ANSWER }));
          controller.enqueue(
            event({
              event: "final",
              data: {
                turn_id: crypto.randomUUID(),
                question,
                standalone_query: standaloneQuery,
                clause_type: body.clause_type ?? null,
                resolved_clause_type: null,
                abstained: true,
                reranking_applied: false,
                rerank_reason: "unsupported topic",
                limit: body.limit ?? 5,
                result_count: 0,
                answer: UNSUPPORTED_TOPIC_ANSWER,
                results: [],
                timings: { total_latency_ms: performance.now() - started },
              },
            }),
          );
          return;
        }
        const queryRerank = chooseQueryReranking(standaloneQuery, clauseType);
        const retrievalStarted = performance.now();
        const { results, diagnostics } = await searchEvidence(
          request,
          env,
          standaloneQuery,
          clauseType,
          body.limit ?? 5,
          queryRerank,
        );
        controller.enqueue(event({ event: "status", stage: "retrieved" }));
        if (diagnostics.reranking_applied) {
          controller.enqueue(event({ event: "status", stage: "reranked" }));
        }
        controller.enqueue(event({ event: "status", stage: "generating" }));
        const prompt = answerPrompt(question, standaloneQuery, results, messages);
        const abortController = new AbortController();
        const timeout = setTimeout(() => abortController.abort(), UPSTREAM_TIMEOUT_MS);
        const answerParts: string[] = [];
        let firstTokenLatency = 0;
        try {
          const response = await openAiStream(env, prompt, abortController.signal);
          for await (const token of readOpenAiTokens(response)) {
            if (!answerParts.length) firstTokenLatency = performance.now() - started;
            answerParts.push(token);
            controller.enqueue(event({ event: "token", delta: token }));
          }
        } finally {
          clearTimeout(timeout);
        }
        const answer = answerParts.join("").trim();
        const retrievalLatency =
          diagnostics.embedding_latency_ms +
          diagnostics.vector_search_latency_ms +
          diagnostics.lexical_search_latency_ms;
        controller.enqueue(
          event({
            event: "final",
            data: {
              turn_id: crypto.randomUUID(),
              question,
              standalone_query: standaloneQuery,
              clause_type: body.clause_type ?? null,
              resolved_clause_type: clauseType,
              abstained: false,
              reranking_applied: diagnostics.reranking_applied,
              rerank_reason: diagnostics.rerank_reason,
              limit: body.limit ?? 5,
              result_count: results.length,
              answer,
              results,
              timings: {
                total_latency_ms: performance.now() - started,
                first_token_latency_ms: firstTokenLatency,
                retrieval_latency_ms: retrievalLatency,
                embedding_latency_ms: diagnostics.embedding_latency_ms,
                vector_search_latency_ms: diagnostics.vector_search_latency_ms,
                lexical_search_latency_ms: diagnostics.lexical_search_latency_ms,
                reranking_latency_ms: diagnostics.reranking_latency_ms,
              },
            },
          }),
        );
      } catch (error) {
        controller.enqueue(
          event({
            event: "error",
            detail:
              error instanceof DOMException && error.name === "AbortError"
                ? "The answer provider timed out"
                : error instanceof Error
                  ? error.message
                  : "Chat generation failed",
          }),
        );
      } finally {
        controller.close();
      }
    },
  });
  return new Response(stream, {
    headers: {
      "Content-Type": "application/x-ndjson; charset=utf-8",
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff",
    },
  });
}
