import {
  CLAUSE_TYPES,
  MAX_CONTEXT_CHARS,
  MAX_CONTEXT_MESSAGES,
  MAX_MESSAGE_CHARS,
  MAX_REQUEST_BYTES,
  MAX_RESULTS,
} from "./constants";
import type { ChatRequest } from "./types";

export class RequestError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

export async function parseChatRequest(request: Request): Promise<ChatRequest> {
  const lengthHeader = Number(request.headers.get("content-length") ?? 0);
  if (lengthHeader > MAX_REQUEST_BYTES) {
    throw new RequestError(413, "Request body is too large");
  }
  const bodyText = await request.text();
  if (new TextEncoder().encode(bodyText).byteLength > MAX_REQUEST_BYTES) {
    throw new RequestError(413, "Request body is too large");
  }
  let raw: unknown;
  try {
    raw = JSON.parse(bodyText);
  } catch {
    throw new RequestError(400, "Request body must be valid JSON");
  }
  if (!raw || typeof raw !== "object") {
    throw new RequestError(400, "Request body must be an object");
  }
  const value = raw as Record<string, unknown>;
  if (!Array.isArray(value.messages) || value.messages.length < 1) {
    throw new RequestError(422, "At least one message is required");
  }
  if (value.messages.length > MAX_CONTEXT_MESSAGES) {
    throw new RequestError(422, `At most ${MAX_CONTEXT_MESSAGES} messages are allowed`);
  }
  const messages = value.messages.map((message) => {
    if (!message || typeof message !== "object") {
      throw new RequestError(422, "Each message must be an object");
    }
    const item = message as Record<string, unknown>;
    if (item.role !== "user" && item.role !== "assistant") {
      throw new RequestError(422, "Message role must be user or assistant");
    }
    if (typeof item.content !== "string" || !item.content.trim()) {
      throw new RequestError(422, "Message content must not be empty");
    }
    const content = item.content.trim();
    if (content.length > MAX_MESSAGE_CHARS) {
      throw new RequestError(
        422,
        `Each message must be at most ${MAX_MESSAGE_CHARS} characters`,
      );
    }
    return { role: item.role as "user" | "assistant", content };
  });
  if (messages.at(-1)?.role !== "user") {
    throw new RequestError(422, "The final message must be from the user");
  }
  if (messages.reduce((total, message) => total + message.content.length, 0) > MAX_CONTEXT_CHARS) {
    throw new RequestError(422, "Conversation context is too large");
  }
  const clauseType =
    value.clause_type === null || value.clause_type === undefined || value.clause_type === ""
      ? null
      : String(value.clause_type);
  if (clauseType && !CLAUSE_TYPES.includes(clauseType as (typeof CLAUSE_TYPES)[number])) {
    throw new RequestError(422, "Unsupported clause type");
  }
  const limit = value.limit === undefined ? MAX_RESULTS : Number(value.limit);
  if (!Number.isInteger(limit) || limit < 1 || limit > MAX_RESULTS) {
    throw new RequestError(422, `Result limit must be between 1 and ${MAX_RESULTS}`);
  }
  if (typeof value.turnstile_token !== "string" || !value.turnstile_token.trim()) {
    throw new RequestError(422, "Turnstile verification is required");
  }
  return {
    messages,
    clause_type: clauseType,
    limit,
    turnstile_token: value.turnstile_token.trim(),
  };
}
