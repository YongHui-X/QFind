import { describe, expect, it } from "vitest";
import { parseChatRequest, RequestError } from "./validation";

function request(body: unknown): Request {
  return new Request("https://example.com/api/chat/stream", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

describe("request validation", () => {
  it("accepts a bounded chat request", async () => {
    const parsed = await parseChatRequest(
      request({
        messages: [{ role: "user", content: "Does the contract restrict assignment?" }],
        limit: 5,
        turnstile_token: "token",
      }),
    );
    expect(parsed.limit).toBe(5);
  });

  it("rejects invalid limits", async () => {
    await expect(
      parseChatRequest(
        request({
          messages: [{ role: "user", content: "Question" }],
          limit: 20,
          turnstile_token: "token",
        }),
      ),
    ).rejects.toBeInstanceOf(RequestError);
  });

  it("requires the final turn to be from the user", async () => {
    await expect(
      parseChatRequest(
        request({
          messages: [{ role: "assistant", content: "Answer" }],
          turnstile_token: "token",
        }),
      ),
    ).rejects.toThrow("final message");
  });
});
