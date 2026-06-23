import { describe, expect, it } from "vitest";
import {
  deduplicateByDocument,
  denseSearch,
  lexicalSearch,
  normalizeVector,
  reciprocalRankFusion,
  shouldRetrievalRerank,
} from "./retrieval";
import type { StaticIndex } from "./types";

const records = [
  {
    id: "a",
    document_id: "doc-a",
    source_pdf: "a.pdf",
    source_txt: "a.txt",
    clause_type: "Anti-Assignment",
    answer: "Yes",
    text: "Assignment requires written consent.",
  },
  {
    id: "b",
    document_id: "doc-b",
    source_pdf: "b.pdf",
    source_txt: "b.txt",
    clause_type: "Anti-Assignment",
    answer: "Yes",
    text: "A transfer is void without approval.",
  },
];

const index: StaticIndex = {
  manifest: {
    version: 1,
    record_count: 2,
    dimensions: 2,
    embedding_model: "test",
    pooling: "mean",
    normalized: true,
    source_sha256: "test",
    generated_at: "2026-01-01T00:00:00Z",
  },
  records,
  vectors: new Float32Array([1, 0, 0, 1]),
  lexical: {
    average_length: 5,
    lengths: [5, 6],
    idf: { assignment: 1, written: 1, consent: 1, transfer: 1, void: 1 },
    term_frequencies: [
      { assignment: 1, written: 1, consent: 1 },
      { transfer: 1, void: 1 },
    ],
  },
};

describe("retrieval", () => {
  it("normalizes vectors", () => {
    expect(normalizeVector([3, 4])).toEqual([0.6, 0.8]);
  });

  it("ranks dense and lexical evidence", () => {
    expect(denseSearch(index, [1, 0], "Anti-Assignment")[0].id).toBe("a");
    expect(
      lexicalSearch(index.records, index.lexical, "void transfer", "Anti-Assignment")[0]
        .id,
    ).toBe("b");
  });

  it("fuses ranks and deduplicates documents", () => {
    const dense = denseSearch(index, [1, 0], "Anti-Assignment");
    const lexical = lexicalSearch(
      index.records,
      index.lexical,
      "written consent",
      "Anti-Assignment",
    );
    const fused = reciprocalRankFusion(dense, lexical);
    expect(fused[0].id).toBe("a");
    expect(deduplicateByDocument([...fused, fused[0]])).toHaveLength(2);
  });

  it("skips reranking when dense and lexical agree", () => {
    const dense = denseSearch(index, [1, 0], "Anti-Assignment");
    const lexical = lexicalSearch(
      index.records,
      index.lexical,
      "written consent",
      "Anti-Assignment",
    );
    expect(shouldRetrievalRerank(reciprocalRankFusion(dense, lexical))[0]).toBe(false);
  });
});
