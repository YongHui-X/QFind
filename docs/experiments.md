# ClauseLens Experiments

This document records measured retrieval, answer-quality, and latency
experiments. Stable architecture and terminology live in `docs/notes.md`.

## 12. Permanent Cloudflare Deployment Port

Goal: make the portfolio UI immediately available without a local Docker
process, always-on VM, Streamlit hibernation screen, or Qdrant Cloud lifecycle.

Implementation:

- Added a React/Vite frontend and Cloudflare Worker streaming API.
- Generated a static artifact from the exact 463 normalized vectors used by the
  validated local Qdrant collection.
- Ported clause routing, deterministic follow-up contextualization, BM25,
  dense search, RRF fusion, contract deduplication, adaptive top-three
  reranking, evidence compression, grounding rules, and citation payloads.
- Kept the Python/Streamlit/Qdrant stack as the research and benchmark
  reference.
- Added Turnstile plus SQLite Durable Object limits: 3 requests/minute/IP,
  10/day/IP, 100/day globally, 1 concurrent request/IP, and 5 globally.
- Added artifact checksum tests and TypeScript retrieval, routing, and request
  validation tests.

Migration rule:

- The cloud deployment cannot be promoted on structural tests alone. The
  Workers AI embedding and reranker path must pass the existing 11 retrieval
  cases and 120-request answer benchmark against a preview deployment.
- The 120-request local baseline remains P50 1.69 s, P95 2.43 s, P99 5.05 s,
  100% deterministic checks, 100% citation validity, and zero critical
  failures.

## 1. Initial Cross-Encoder Reranking

Goal: improve result ordering when vector search retrieves the right clause but
ranks it below less relevant evidence.

Method:

- Retrieve 20 vector candidates.
- Score each question-clause pair with
  `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Return the best requested top-k results.

Initial 10-question comparison:

| Metric | Vector baseline | Always rerank |
| --- | ---: | ---: |
| Top-1 accuracy | 90% | 90% |
| Top-5 accuracy | 100% | 100% |
| MRR | 0.925 | 0.950 |
| nDCG | 0.945 | 0.970 |
| Keyword hit rate | 100% | 100% |
| Average retrieval latency | 70.5 ms | 1430.1 ms |
| Average reranking latency | 0 ms | 1370.8 ms |

Result:

- The intellectual-property paraphrase improved from License Grant rank 4 to
  rank 1.
- Always-on reranking added about 1.37 seconds on the local CPU.
- Aggregate top-1 accuracy remained unchanged, so reranking every query was not
  an acceptable latency tradeoff.

## 2. Adaptive Reranking

Goal: retain the difficult paraphrase improvement without paying reranking cost
on ordinary questions.

Expanded 11-question evaluation:

| Mode | Passed | Top-1 | MRR | Average retrieval latency |
| --- | ---: | ---: | ---: | ---: |
| Vector only | 9/11 | 81.8% | 0.886 | 57.1 ms |
| Always rerank | 10/11 | 90.9% | 0.955 | 1375.8 ms |
| Adaptive | 11/11 | 100% | 1.000 | 305.3 ms |

Result:

- Adaptive mode reranked difficult intellectual-property paraphrases.
- Literal license questions retained vector ordering.
- Retrieval reached 11/11 passes and MRR 1.000.

## 3. Cold-Start GPT Observation

Question:

```text
Does the agreement grant a right to use intellectual property?
```

Observed timing:

| Stage | Time |
| --- | ---: |
| Total turn | 17.63 s |
| First token | 15.87 s |
| Context | 0.00 s |
| Retrieval | 1.62 s |
| Embedding | 0.41 s |
| Vector search | 0.01 s |
| Reranker cold load | 9.73 s |
| Reranking | 1.21 s |
| Answer generation | 6.28 s |

Answer assessment:

- Quality was approximately 8/10.
- The response was grounded, separated CERES and BIOCEPT, and used valid
  citations.
- It described exclusivity, sublicensing, scope, and technology exclusions.
- It was too detailed and ended mid-sentence after exhausting the 220-token
  budget.
- “Both retrieved agreements” would have been more precise than “both
  agreements.”

Primary findings:

- The 9.73-second cold reranker load was the largest avoidable delay.
- The answer prompt and token ceiling produced unnecessary detail and
  truncation.

## 4. Latency Optimization

Changes:

- Warm embedding and reranker models during API startup.
- Reduce the reranker candidate pool from 20 to 8.
- Keep adaptive rather than always-on reranking.
- Replace LLM follow-up rewriting with deterministic contextualization.
- Send the top three evidence results, rather than all five, to OpenAI.
- Reduce the answer ceiling from 220 to 120 tokens.
- Require direct answers in two to four complete sentences.
- Reuse the OpenAI SDK client and a persistent Streamlit HTTP connection.

Warm benchmark configuration:

```text
Answer model: gpt-4.1-mini
OpenAI service tier: standard
Cases: 11
Reranking mode: adaptive
Reranker candidates: 8
Generation evidence items: 3
Answer ceiling: 120 tokens
```

Results:

| Metric | Optimized result |
| --- | ---: |
| Wall-clock mean | 1.800 s |
| Wall-clock P50 | 1.814 s |
| Wall-clock P95 | 2.100 s |
| First-token mean | 0.906 s |
| First-token P95 | 1.203 s |
| Generation first-token mean | 0.812 s |
| Retrieval mean | 0.094 s |
| Answer mean | 1.705 s |
| Reranker loading during requests | 0 ms |

Retrieval quality after reducing candidates to eight:

| Metric | Result |
| --- | ---: |
| Passed | 11/11 |
| MRR | 1.000 |
| Top-1 hit rate | 100% |
| Keyword hit rate | 100% |
| Average adaptive retrieval latency | 277.1 ms |
| Average reranking latency | 227.6 ms |

Conclusion:

- The warm benchmark met P95 first useful text below 3 seconds.
- It also completed full answers below 3 seconds at P95.
- Retrieval quality was unchanged.

## 5. Latest Live Query

Question:

```text
Does the agreement grant a right to use intellectual property?
```

Observed timing:

| Stage | Time | Target assessment |
| --- | ---: | --- |
| Total turn | 4.17 s | Slightly above the typical 1-4 s band, but within the P95 objective of 5 s. |
| First token | 3.48 s | Above the preferred P95 first-visible objective of 3 s. |
| Context | 0.00 s | Expected for a first-turn question. |
| Retrieval | 0.59 s | Slightly above the desired component ranges due to reranking. |
| Embedding | 0.08 s | Within the 20-100 ms target. |
| Vector search | 0.02 s | Within the 10-100 ms target. |
| Reranker load | 0.00 s | Startup warmup worked. |
| Reranking | 0.49 s | At the upper edge of the 50-500 ms target. |
| Answer generation | 3.58 s | Within the 0.5-5 s target and the main latency contributor. |

Answer assessment:

- Quality was approximately 9/10.
- The answer was concise, complete, grounded, and correctly distinguished the
  CERES and BIOCEPT agreements.
- Citations `[1][2][3]` matched the retrieved evidence.
- The final sentence repeated the opening conclusion and could be removed.

Interpretation:

- This request was slower than the 11-case warm benchmark because hosted model
  latency varies between calls.
- Retrieval is no longer the dominant bottleneck; OpenAI answer generation is.
- One 4.17-second request does not invalidate the benchmark or determine P95.
  Stable percentile claims require at least 100 representative requests.

## Current Performance Objectives

| Stage | Typical target |
| --- | ---: |
| Embedding query | 20-100 ms |
| Vector search | 10-100 ms |
| Reranking | 50-500 ms |
| LLM generation | 0.5-5 s |
| Total | 1-4 s |

Service objectives:

```text
P50 total: under 2 seconds
P95 total: under 5 seconds
P99 total: under 10 seconds
P95 first visible text: under 3 seconds
Retrieval evaluation: 11/11
Citation validity: 100%
```

Next experiment:

- Run at least 100 representative warm requests.
- Report client-side and backend P50, P95, and P99.
- Separate vector-only, adaptively reranked, follow-up, and abstention cases.
- Consider OpenAI Priority processing only if standard processing misses the
  P95 objectives over that larger sample.

## 6. Prompt Compression And TTFT Instrumentation

Goal: test whether long retrieved clauses were causing excessive model prefill
before the first visible token.

Changes:

- Select query-matching clause segments before generation.
- Cap each of the three generation evidence items at 1,000 characters.
- Record total prompt characters, evidence characters, and approximate input
  tokens with each response.
- Display prompt-size diagnostics in the Streamlit evidence details.
- Strengthen the answer prompt so it ends after the last source-specific
  statement instead of repeating the opening conclusion.

Updated 11-case warm benchmark:

| Metric | Result |
| --- | ---: |
| Wall-clock mean | 1.939 s |
| Wall-clock P50 | 1.817 s |
| Wall-clock P95 | 2.595 s |
| First-token mean | 0.961 s |
| First-token P95 | 1.580 s |
| Generation first-token mean | 0.845 s |
| Retrieval mean | 0.116 s |
| Answer mean | 1.823 s |
| Mean prompt size | 3,053 characters |
| Mean estimated input | 763 tokens |

Target intellectual-property query:

```text
Prompt size: 3,095 characters
Evidence size: 1,746 characters
Estimated input: 774 tokens
Generation TTFT: 2.497 seconds
Total: 3.754 seconds
```

Conclusion:

- The generation prompt is already small enough that prompt prefill is unlikely
  to be the principal cause of occasional multi-second TTFT.
- Compression did not improve the small-sample benchmark relative to the prior
  run; hosted model latency variance was larger than the expected prefill gain.
- Prompt diagnostics should remain because they let future 100-request tests
  measure the relationship between prompt size and TTFT directly.
- Streaming remains correctly wired end to end. The remaining long pauses are
  model TTFT variation rather than full-response buffering.

Answer-format verification:

- A general “do not repeat” instruction removed the repeated summary but caused
  the model to spend too many tokens on the first source and truncate the
  BIOCEPT sentence.
- The final prompt contract uses at most 70 words normally and 90 words for
  specific-provision questions that may require an identifier disclaimer.
- The target query then returned a complete 63-word answer with citations
  `[1][2][3]`, no repeated conclusion, and no truncation.

## 7. Answer Completion And Source-Label Refinement

Goal: prevent specific-provision answers from truncating while keeping normal
answers concise and readable.

Problem observed:

- A specific-provision answer ended at `without sublic`, despite a 70-word
  instruction, because the 120-token hard ceiling was too small.
- Dataset-style source names such as
  `CERES,INC_01_25_2012-EX-10.20` consumed output space and reduced readability.
- A lower-ranked third agreement was not necessary to answer the
  specific-provision question.

Changes:

- Raise the hard completion ceiling from 120 to 160 tokens.
- Keep normal answers at a 70-word target.
- Allow a 90-word target for specific-provision questions that may require a
  section-identifier disclaimer.
- Convert dataset filenames to short generation labels such as `CERES`,
  `BIOCEPT`, and `ENERGOUS`.
- Send only the two strongest evidence items for specific-provision questions.
- Keep all retrieved evidence visible in the UI.
- Require the model to omit weaker evidence rather than leave a sentence
  incomplete.
- If no section or article metadata exists, state that explicitly instead of
  inventing an identifier.

Verification:

- Normal intellectual-property answer: 69 words, complete, citations
  `[1][2][3]`.
- Specific-provision answer: complete, used the strongest two evidence items,
  and disclosed that the retrieved evidence lacked section/article identifiers.
- Tests increased to 53 passing cases.
- Ruff and Python compilation checks passed.

Decision:

- Retain the 160-token safety ceiling with query-specific word budgets.
- Treat answer completeness as more important than forcing every answer under a
  single fixed word count.

## 8. Follow-Up Topic Persistence

Goal: preserve the active supported clause type across short referential
follow-ups without carrying unrelated unsupported questions into new topics.

Problem observed:

- A new license question inherited a preceding arbitration question in its
  standalone retrieval query.
- A later question, `How long does it remain effective?`, abstained because
  only the immediately preceding user message was considered.
- `Is it also transferable?` could be misrouted to Anti-Assignment even when
  `it` referred to a license.

Changes:

- Resolve explicit new supported questions from the latest message alone.
- Treat short pronoun-based questions as contextual follow-ups.
- Recover their topic from recent conversation messages.
- Anchor retrieval to the most recent explicit user question for that topic,
  skipping unrelated unsupported turns.
- Instruct answer generation not to infer sublicensing rights from assignment
  or transfer language.

Verification:

- Arbitration followed by a license question produces only the license
  question as the standalone query.
- Transferability and duration follow-ups remain under `License Grant`.
- The duration query is anchored to the original sublicensing question rather
  than the immediately preceding ambiguous question.

## 9. Persistent Multi-Chat History

Goal: make reset behavior start a genuinely new conversation and allow prior
completed chats to be reviewed after reruns or application restarts.

Changes:

- Add local SQLite persistence at `data/processed/chat_history.db`.
- Save completed user and assistant turns automatically, including answer
  evidence and diagnostics.
- Add sidebar controls to create, reopen, and delete chats.
- Restore each chat's clause filter, result limit, and reranking mode.
- Use the first user question as a deterministic chat title.
- Keep empty new chats in session only so repeated clicks do not create blank
  history records.

Verification:

- Database setup is idempotent.
- Message metadata and retrieval controls survive save and reload.
- Updated chats move to the top of recent history.
- Deleting one chat leaves other conversations unchanged.

## 10. Grounding And Answer-Accuracy Hardening

Goal: prevent broad legal conclusions when retrieved agreements differ, remain
silent, or use defined terms whose definitions are not present in evidence.

Problems observed:

- Unauthorized-assignment consequences were generalized as "generally void or
  voidable" even though one retrieved source did not state a consequence.
- An operation-of-law question received an unconditional yes despite mixed and
  silent evidence.
- `Affiliate` was treated as including a wholly owned subsidiary without a
  retrieved definition.
- Explicit follow-up questions included the previous full question in the
  retrieval query, adding irrelevant terms.

Changes:

- Use the latest explicit question alone when it identifies the clause topic.
- Resolve referential follow-ups using the latest question plus a short clause
  label rather than copying a prior question.
- Adaptively rerank nuanced detail questions involving consequences,
  exceptions, operation of law, defined relationships, duration, territory,
  thresholds, and notice.
- Require qualified comparisons for mixed evidence and prohibit treating
  silence as support.
- Prohibit unstated inferences between Affiliate, subsidiary, transfer,
  assignment, and sublicensing concepts.
- Add a 12-case generated-answer benchmark across all five supported clause
  types plus abstention.
- Add deterministic critical regressions and an optional OpenAI judge with a
  90% pass-rate gate.

Verification:

- Focused contextualization, reranking decisions, grounding instructions,
  answer-case loading, deterministic scoring, judge parsing, and quality gates
  are covered by unit tests.
- The existing retrieval evaluation could not be rerun while documenting this
  change because the running API held the embedded Qdrant storage lock. Stop
  the API before running `evaluation/eval.py` or `evaluation/answer_eval.py`.

## 11. Reliable Latency And Output Benchmarking

Goal: make latency measurements reproducible and select a model and processing
tier only when both performance and grounding targets pass.

Changes:

- Add a Docker Compose Qdrant service and default the application to server
  mode, avoiding embedded-storage locking and OneDrive I/O variance.
- Retain `QDRANT_MODE=embedded` as a development fallback.
- Reduce normal answers to 55 words, specific-provision answers to 65 words,
  and the hard completion ceiling to 128 tokens.
- Add OpenAI timeout and retry configuration.
- Capture model, requested and returned service tier, request ID, and estimated
  output size in response diagnostics and telemetry.
- Make the reranker candidate pool configurable.
- Add liability routing phrases for lost profits, anticipated savings,
  prospective profits, special damages, and punitive damages.
- Extend the latency benchmark with server mode, model/tier selection,
  candidate-pool selection, randomized repeats, and response metadata.
- Add `evaluation/performance_benchmark.py` for model/tier screening and a
  120-turn acceptance workload with latency and deterministic-quality gates.

Acceptance gates:

```text
P50 total < 2 seconds
P95 total < 5 seconds
P99 total < 10 seconds
P95 first token < 3 seconds
Vector P95 total < 3 seconds
Reranked and follow-up P95 total < 5 seconds
Abstention P95 total < 250 ms
Citation validity = 100%
Answer-mode consistency >= 95%
Critical deterministic failures = 0
```

Status:

- Docker Qdrant 1.18.0 is running and contains all 463 starter evidence
  records.
- A 12-case Standard-tier smoke run with pinned
  `gpt-4.1-mini-2025-04-14` produced:

  | Metric | Result |
  | --- | ---: |
  | P50 total | 1.82 s |
  | P95 total | 2.58 s |
  | P99 total | 2.65 s |
  | P95 first token | 1.86 s |
  | Citation validity | 100% |
  | Critical deterministic failures | 0 |

- The initial consistency gate exposed three benchmark-label mismatches. The
  cases were corrected so transferability and excluded-damages questions are
  `supported`, while absent license duration is `insufficient`.
- Reranker candidate comparison:

  | Candidates | Passed | MRR | nDCG | Avg rerank |
  | ---: | ---: | ---: | ---: | ---: |
  | 5 | 11/11 | 1.000 | 0.971 | 50.7 ms |
  | 6 | 11/11 | 1.000 | 0.977 | 55.3 ms |
  | 8 | 11/11 | 1.000 | 0.982 | 75.1 ms |

- Candidate pool 5 also passed all three critical answer-grounding cases, so it
  is now the default.
- The four-configuration matrix was run with 36 requests per configuration:

  | Configuration | P50 total | P95 total | P99 total | P95 first token | Deterministic | Citations |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | GPT-4.1 mini Standard | 1.73 s | 2.61 s | 2.76 s | 1.44 s | 100% | 100% |
  | GPT-4.1 mini Priority | 1.42 s | 2.73 s | 2.96 s | 2.26 s | 100% | 100% |
  | GPT-5.4 mini Standard | 1.38 s | 1.91 s | 2.09 s | 1.50 s | 86.1% | 100% |
  | GPT-5.4 mini Priority | 1.08 s | 1.48 s | 1.82 s | 1.21 s | 83.3% | 100% |

- Priority was returned for 91.7% of requested Priority calls, below the 95%
  reliability gate.
- GPT-4.1 mini Standard remains the quality-first default.
- GPT-5.4 mini Standard is the latency-first candidate because it reaches
  P95 total below two seconds without Priority pricing.
- GPT-5.4 failures were partly strict lexical-evaluator mismatches, but some
  responses also ended incompletely. It will not become the default until
  normalized evaluation and repeated completion checks pass.
- No tested configuration achieved P95 first token below 500 ms.
- Full details are recorded in
  `docs/performance_results_2026-06-22.md`.
