import type { ChatResponse, Message } from "./types";

export type StreamEvent =
  | { event: "status"; stage: string }
  | { event: "token"; delta: string }
  | { event: "final"; data: ChatResponse }
  | { event: "error"; detail: string };

export async function streamChat(
  messages: Message[],
  clauseType: string,
  limit: number,
  turnstileToken: string,
  onEvent: (event: StreamEvent) => void,
): Promise<void> {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: messages.slice(-8).map(({ role, content }) => ({ role, content })),
      clause_type: clauseType || null,
      limit,
      turnstile_token: turnstileToken,
    }),
  });
  if (!response.ok || !response.body) {
    const retryAfter = response.headers.get("Retry-After");
    const body = (await response.json().catch(() => ({}))) as { detail?: string };
    const suffix = retryAfter ? ` Try again in ${retryAfter} seconds.` : "";
    throw new Error(`${body.detail || `Request failed (${response.status})`}${suffix}`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.trim()) continue;
      onEvent(JSON.parse(line) as StreamEvent);
    }
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer) as StreamEvent);
}
