# ClauseLens Technical Notes

This document contains stable architecture notes, terminology, operating
expectations, test prompts, and the engineering backlog. Measured results and
performance changes belong in `docs/experiments.md`. Setup and run commands
belong in the root `README.md`.

## Current Architecture

ClauseLens is a contract-intelligence RAG application:

1. `app/cuad.py` prepares labeled CUAD clause evidence.
2. `scripts/prepare_cuad_subset.py` writes the starter JSONL dataset.
3. `scripts/index_qdrant.py` embeds changed evidence and stores it in Qdrant.
4. `app/rag.py` performs vector retrieval and optional cross-encoder reranking.
5. `app/chat.py` routes supported topics, contextualizes follow-ups, retrieves
   evidence, and asks OpenAI for a grounded answer with citations.
6. FastAPI exposes search and chat endpoints; Streamlit provides the demo UI
   and local telemetry.

Current model stack:

```text
Embeddings: BAAI/bge-small-en-v1.5
Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
Vector database: Qdrant
Answer model: gpt-4.1-mini-2025-04-14
```

The main application uses OpenAI directly. Ollama helpers remain available for
local experiments but are not the production application path.

## Dataset Snapshot

Local CUAD inputs:

```text
data/cuad/CUAD_v1/master_clauses.csv
data/cuad/CUAD_v1/CUAD_v1.json
data/cuad/CUAD_v1/full_contract_txt/Part_I
data/cuad/CUAD_v1/full_contract_txt/Part_II
```

Observed source data:

- 200 local text contracts: 100 in each TXT partition.
- 510 labeled CSV rows and 83 columns.
- 194 labeled rows match local TXT files by filename stem.
- Empty evidence lists mean that the clause type was not identified and should
  not be embedded.

The starter index deliberately covers five well-represented clause types:

```text
Anti-Assignment
Cap On Liability
License Grant
Audit Rights
Termination For Convenience
```

CUAD CSV filenames use `.pdf`, while the supplied full-text files use `.txt`.
The preparation layer matches them using the filename stem and retains the PDF
name for source citations.

## Retrieval And Reranking

Vector search quickly finds semantically similar clauses. The cross-encoder
reranker then evaluates the question and each candidate clause together and
reorders them using a more precise relevance score.

The reranker score is a relative ranking signal, not a percentage or an
eight-out-of-ten score. A score such as `8.052` means that clause ranked highly
relative to other candidates for that query.

“Reranker candidates” is the number of vector-search results sent to the
cross-encoder:

```text
Question
  -> vector search returns 5 candidates
  -> reranker compares the question with all 5
  -> best 5 are returned to the UI
  -> top 3 are included in the answer-generation prompt
```

ClauseLens uses adaptive reranking. Difficult intellectual-property
paraphrases are reranked, while ordinary literal queries stay on faster vector
ordering. See `docs/experiments.md` for measured quality and latency.

## Latency Statistics

The Streamlit timing line reports:

| Statistic | Meaning |
| --- | --- |
| Turn | Total backend time from receiving the request through the completed answer. |
| First token | Time until the first visible answer text is generated. |
| Context | Time spent converting a follow-up into a standalone retrieval query. First-turn questions are normally zero. |
| Retrieval | Total evidence lookup time, including embedding, vector search, and reranking. |
| Embed | Time to convert the question into an embedding vector. |
| Vector | Time Qdrant spends finding semantically similar clauses. |
| Reranker load | Time required to load the cross-encoder. This should be zero after startup warmup. |
| Rerank | Time spent scoring candidate question-clause pairs. |
| Answer | Time spent streaming the generated answer from OpenAI. |

The response details also show generation prompt characters, evidence
characters, and an approximate input-token count. These fields help test
whether slow generation TTFT correlates with prompt size.

Target operating ranges:

| Stage | Typical target |
| --- | ---: |
| Embedding query | 20-100 ms |
| Vector search | 10-100 ms |
| Reranking | 50-500 ms |
| LLM generation | 0.5-5 s |
| Total | 1-4 s |

Service objectives:

- P50 total latency: under 2 seconds.
- P95 total latency: under 5 seconds.
- P99 total latency: under 10 seconds.
- P95 first visible text: under 3 seconds.
- Citation validity: 100%.
- Retrieval evaluation: 11/11 passing.

P50 is the median: half of requests finish at or below that value. P95 means
95% finish at or below that value. P99 means 99% finish at or below that value.
A single request can be compared with a target, but it cannot establish a
percentile; percentile reporting requires a representative collection of
requests.

## Routing And Abstention

The chatbot maps plain-English questions to the five starter clause types before
retrieval. Explicit user filters remain authoritative.

Unsupported topics such as governing law, automatic renewal, and breach
remedies must abstain without returning unrelated evidence. The raw `/search`
endpoint remains available for unbiased retrieval experiments.

## Manual Test Prompts

Core demo sequence:

```text
Does the agreement grant a right to use intellectual property?
What is the specific provision that defines the rights granted for intellectual property use?

What does the contract say about ending the agreement early?
How much notice is required?

What does the contract say about governing law?
```

Expected behavior:

- Intellectual-property questions resolve to `License Grant`, cite retrieved
  evidence, and use adaptive reranking for paraphrased wording.
- The termination follow-up is contextualized while keeping the legal topic.
- Governing-law questions abstain and return no evidence.
- First-turn questions report zero contextualization latency.

Additional neutral prompts:

```text
What does the contract say about transferring the agreement?
What does the contract say about limits on damages?
What does the contract say about audit access?
Who is allowed to use the licensed materials?
Can either party end the agreement after giving notice?
```

Acceptance checks:

- The answer is supported by retrieved text.
- Citations refer to returned evidence.
- Sources from different agreements are not merged into a single rule.
- The resolved clause type matches the question.
- Unsupported topics abstain.
- Follow-up questions preserve recent context.
- Telemetry separates contextualization, retrieval, reranking, and answer
  latency.

## Engineering Decisions

- Stable Qdrant point IDs are derived from record IDs so repeated indexing
  updates existing records instead of creating duplicates.
- SHA-256 content hashes and embedding-model metadata prevent unnecessary
  re-embedding.
- Embeddings are normalized for cosine similarity.
- The clause-type filter is applied during Qdrant retrieval, before reranking.
- Follow-ups are contextualized deterministically without an extra LLM call.
- Offline labeled evaluation is the source of retrieval accuracy. Live user
  telemetry is operational data, not ground-truth accuracy.

## Current Backlog

1. Add answer faithfulness and citation-correctness benchmarks.
2. Collect at least 100 representative chat turns for stable P50/P95/P99
   reporting.
3. Add full-contract chunks with character spans and eventual PDF page
   citations.
4. Expand beyond five clause types only after adding labeled regression cases.
5. Capture final Streamlit and API screenshots for the README.
