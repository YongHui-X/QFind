# ClauseLens Product Requirements

## Product Goal

ClauseLens is a citation-grounded contract-intelligence chatbot. It helps users
ask plain-English questions, retrieve relevant clause evidence from CUAD, and
receive concise answers that remain traceable to the source agreements.

ClauseLens is a portfolio and research prototype, not a legal-advice system.

## Target Users

- Legal operations and contract-review users exploring agreement language.
- Engineers evaluating practical RAG architecture, retrieval quality, and
  latency tradeoffs.
- Portfolio reviewers who need observable evidence, citations, and repeatable
  evaluation rather than unsupported model claims.

## Core Requirements

### Retrieval

- Support the five starter clause types: Anti-Assignment, Cap On Liability,
  License Grant, Audit Rights, and Termination For Convenience.
- Retrieve clause evidence using dense semantic embeddings and Qdrant.
- Apply clause-type metadata filters before reranking.
- Use adaptive cross-encoder reranking only where evaluation shows a benefit.
- Preserve source agreement metadata and original clause text.

### Answering

- Answer only from retrieved evidence.
- Cite evidence with bracketed indexes matching the evidence panel.
- Keep different agreements separate rather than synthesizing a fictional
  combined contract.
- Abstain when the question is outside the supported starter scope.
- Produce complete, concise answers without truncated sentences.
- Use short source labels in generated answers while retaining full source
  metadata in the evidence panel.

### Performance

- P50 total latency: under 2 seconds.
- P95 total latency: under 5 seconds.
- P99 total latency: under 10 seconds.
- P95 first visible answer text: under 3 seconds.
- Keep embedding within 20-100 ms, vector search within 10-100 ms, and
  reranking within 50-500 ms under normal warm operation.

### Quality

- Maintain 11/11 passing retrieval evaluation cases.
- Maintain MRR 1.000 on the starter evaluation.
- Require valid citations and clause-type-consistent evidence.
- Add answer-faithfulness and citation-correctness benchmarks before claiming
  production readiness.

## Current RAG Design

ClauseLens is a routed, metadata-filtered, dense retrieval RAG system with
adaptive cross-encoder reranking and grounded generation.

The request flow is:

```text
User question
  -> deterministic topic routing
  -> dense query embedding
  -> Qdrant vector search with clause-type metadata filter
  -> adaptive cross-encoder reranking for difficult paraphrases
  -> query-focused evidence compression
  -> citation-grounded OpenAI answer generation
```

It is not currently hybrid search because it does not combine BM25/sparse
retrieval with dense retrieval. It is not agentic RAG because no autonomous
tool-selection or iterative retrieval loop is used.

## Evaluation And Experiment Logging Requirement

Every major behavioral or performance change must be recorded in
`docs/experiments.md` during the same implementation task.

A major change includes:

- Retrieval, embedding, filtering, candidate-pool, or reranking changes.
- Prompt, model, context-selection, answer-length, or citation changes.
- Startup, caching, streaming, telemetry, or latency changes.
- Dataset, evaluation-case, supported-topic, or abstention changes.

Each experiment entry must include:

1. Goal or problem being addressed.
2. Implementation changes.
3. Test or benchmark method.
4. Before/after measurements when available.
5. Quality regressions, failures, or inconclusive results.
6. Decision and follow-up action.

Results must not be described as P50, P95, or P99 unless the sample size and
workload are stated. Failed experiments must remain documented so they are not
repeated without new evidence.

## Documentation Ownership

- `README.md`: setup, commands, architecture overview, and demo instructions.
- `docs/prd.md`: product requirements and durable development rules.
- `docs/notes.md`: stable technical explanations and operating terminology.
- `docs/experiments.md`: chronological measured experiments and decisions.

## Current Priorities

1. Collect at least 100 representative warm chat requests.
2. Report client and backend P50, P95, and P99 by request category.
3. Add answer faithfulness and citation-correctness evaluation.
4. Add full-contract chunks and eventual PDF page citations.
5. Expand clause coverage only with corresponding labeled regression cases.
