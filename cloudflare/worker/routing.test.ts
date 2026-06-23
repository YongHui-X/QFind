import { describe, expect, it } from "vitest";
import {
  contextualizeQuery,
  inferClauseType,
  resolveConversationClauseType,
} from "./routing";

describe("routing", () => {
  it("routes supported questions", () => {
    expect(inferClauseType("What damages are excluded?")).toBe("Cap On Liability");
    expect(inferClauseType("Can the customer inspect records?")).toBe("Audit Rights");
  });

  it("preserves a topic for referential follow-ups", () => {
    const messages = [
      { role: "user" as const, content: "Does the license permit sublicensing?" },
      { role: "assistant" as const, content: "The agreements differ." },
      { role: "user" as const, content: "How long does it remain effective?" },
    ];
    expect(resolveConversationClauseType(messages)).toBe("License Grant");
    expect(contextualizeQuery(messages, "License Grant")).toContain("License Grant");
  });

  it("abstains on unsupported topics", () => {
    expect(inferClauseType("Does either party provide indemnification?")).toBeNull();
  });
});
