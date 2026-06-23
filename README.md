# ClauseLens

> A citation-grounded RAG chatbot that helps legal, procurement, and compliance
> teams find contract risks, rights, and obligations faster.

ClauseLens answers plain-English questions using evidence retrieved from
real-world contracts in the CUAD dataset. Every generated answer includes
source-backed citations so users can verify the result against the retrieved
contract language.

This is a portfolio project and not a legal advice tool.

## Verified Results

The final configuration was evaluated on 463 clause records from 30 CUAD
contracts.

### Retrieval

| Metric | Result |
| --- | ---: |
| Recall@5 | **100%** |
| Context precision | **98.2%** |
| MRR | **1.000** |
| nDCG | **0.998** |
| P95 retrieval latency | **68.2 ms** |
| P95 reranking latency | **124.6 ms** |
| Evaluation cases passed | **11/11** |

### End-to-End Chat

Measured over 120 sequential requests using GPT-4.1 mini Standard:

| Metric | Result |
| --- | ---: |
| Deterministic answer accuracy | **100%** |
| Citation validity | **100%** |
| Answer-mode consistency | **100%** |
| Critical failures | **0** |
| P50 response latency | **1.69 s** |
| P95 response latency | **2.43 s** |
| P95 first-token latency | **1.22 s** |

The system meets the practical P95 target of 2.5 seconds. It does not yet meet
the experimental targets of P95 below 2.0 seconds or first-token latency below
700 ms. The remaining tail latency is primarily hosted-model response time.

Detailed methodology and claim boundaries are documented in
[the performance report](docs/performance_results_2026-06-22.md).

### Token Usage and Estimated Cost

ClauseLens keeps generation costs low by running dense retrieval, BM25 search,
reciprocal-rank fusion, and cross-encoder reranking locally. The hosted model is
used only for the final grounded answer, with compact evidence prompts and a
160-token completion ceiling.

| Cost indicator | Result |
| --- | ---: |
| Average estimated output per benchmark request | **93 tokens** |
| Representative estimated input per generated answer | **~1,100 tokens** |
| Estimated GPT-4.1 mini cost per generated answer | **~$0.00060** |
| Estimated cost per 1,000 generated answers | **~$0.60** |
| Estimated model cost for the 120-request benchmark | **~$0.07** |

The estimate uses GPT-4.1 mini Standard pricing of $0.40 per million input
tokens and $1.60 per million output tokens, as listed in the
[OpenAI model documentation](https://developers.openai.com/api/docs/models/gpt-4.1-mini).
Input size is based on representative recorded telemetry, while the output
average comes from the final 120-request benchmark. These are estimated
generation costs rather than an exported billing total and exclude hosting or
infrastructure charges.

## Business Value

- Helps legal, procurement, and compliance teams locate important contract
  language without manually scanning entire agreements.
- Presents cited evidence for review instead of returning unsupported model
  conclusions.
- Surfaces assignment restrictions, audit rights, liability limits, license
  grants, and termination rights through natural-language questions.
- Preserves contract-level distinctions when agreements contain different
  terms, exceptions, or missing information.
- Saves past conversations automatically so users can return to earlier
  contract reviews.

## Core Capabilities

- Citation-grounded answers with expandable source evidence.
- Hybrid dense and lexical retrieval.
- Adaptive cross-encoder reranking.
- Deterministic follow-up contextualization.
- Supported-topic routing and safe abstention.
- Persistent chat history with open, delete, and new-chat controls.
- Per-query latency telemetry and user feedback.
- FastAPI backend and Streamlit chat interface.
- Incremental indexing that skips unchanged contract clauses.
- Repeatable retrieval, answer-quality, and performance evaluations.

## How It Works

```text
CUAD contracts
      |
      v
Clause extraction and preparation
      |
      v
Sentence Transformer embeddings
      |
      v
Qdrant dense search + BM25 lexical search
      |
      v
Reciprocal-rank fusion
      |
      v
Contract deduplication
      |
      v
Adaptive top-3 cross-encoder reranking
      |
      v
Grounded answer generation with citations
```

The current retrieval configuration is:

```text
BGE dense retrieval: 6 candidates
BM25 lexical retrieval: 6 candidates
Fusion: reciprocal-rank fusion, k=60
Deduplication: one leading passage per contract
Reranking: adaptive, top 3 candidates
Returned evidence: up to 5 passages
```

## Technology Stack

| Layer | Technology |
| --- | --- |
| Dataset | CUAD |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Vector database | Qdrant |
| Lexical retrieval | In-memory BM25 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Answer model | OpenAI GPT-4.1 mini |
| API | FastAPI |
| Local interface | Streamlit |
| Public deployment | Cloudflare Workers and React |
| Persistence | SQLite and JSONL telemetry |
| Testing | pytest |

## Supported Contract Topics

ClauseLens currently supports five clause categories:

1. Assignment restrictions
2. Liability caps
3. License grants
4. Audit rights
5. Termination for convenience

Questions outside these categories are rejected safely instead of being
answered using unrelated evidence.

## Dataset Snapshot

The current evaluated subset contains:

| Item | Count |
| --- | ---: |
| Contracts | 30 |
| Clause evidence records | 463 |
| Supported clause categories | 5 |

| Clause category | Records |
| --- | ---: |
| Anti-Assignment | 71 |
| Audit Rights | 165 |
| Cap On Liability | 90 |
| License Grant | 116 |
| Termination For Convenience | 21 |

CUAD contains approximately 500 commercial contracts and 41 clause
categories. ClauseLens deliberately starts with a smaller evaluated scope
rather than claiming broad coverage without sufficient testing.

## Quick Start

### 1. Create the environment file

```powershell
Copy-Item .env.example .env
```

Configure the answer model:

```text
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-mini-2025-04-14
OPENAI_SERVICE_TIER=standard
MODEL_WARMUP_ENABLED=true
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

The commands below use the existing project environment:

```powershell
.\.conda-clauselens\python.exe
```

### 3. Start Qdrant

```powershell
docker compose up -d qdrant
```

Server mode is recommended because the API and evaluation tools can share one
Qdrant service. Embedded mode is available for single-process development but
cannot be opened by multiple processes simultaneously.

### 4. Prepare and index CUAD

```powershell
.\.conda-clauselens\python.exe scripts\prepare_cuad_subset.py
.\.conda-clauselens\python.exe scripts\index_qdrant.py
```

The indexer stores a SHA-256 content hash and embedding model name with every
record. Later runs:

- Skip unchanged records.
- Re-embed changed clauses.
- Update metadata without regenerating vectors.
- Embed duplicate clause text only once per indexing run.

Use `--recreate` only when intentionally rebuilding the collection.

### 5. Run the API

```powershell
.\.conda-clauselens\python.exe -m uvicorn app.api:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

Check `/health` and wait for `"ready": true`. Startup warmup loads the
embedding model and reranker before the application begins accepting normal
traffic.

### 6. Run the Streamlit interface

```powershell
.\.conda-clauselens\python.exe -m streamlit run app\streamlit_app.py
```

Keep the API running while using Streamlit. The interface sends requests to:

```text
http://127.0.0.1:8000
```

## Application Surfaces

### Streamlit Chat

- Collapsible chat-history sidebar.
- Automatic conversation persistence.
- New, reopen, and delete chat controls.
- Clause-type and result-limit controls.
- Adaptive, disabled, or forced reranking.
- Streamed answers and stage indicators.
- Expandable evidence and latency details.
- Helpful and not-helpful feedback.

### FastAPI

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Readiness and model configuration |
| `GET /clause-types` | Supported clause categories |
| `POST /search` | Clause evidence retrieval |
| `POST /chat` | Grounded non-streaming answer |
| `POST /chat/stream` | Grounded streaming answer |

Example search:

```powershell
curl -X POST http://localhost:8000/search `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"Does the contract restrict assignment?\",\"clause_type\":\"Anti-Assignment\",\"limit\":5}"
```

### Command-Line Search

```powershell
.\.conda-clauselens\python.exe scripts\search_qdrant.py "Does the contract restrict assignment?"
```

## Evaluation

### Retrieval Quality

```powershell
.\.conda-clauselens\python.exe evaluation\eval.py `
  --qdrant-mode server `
  --top-k 5 `
  --rerank-mode auto `
  --candidate-limit 3 `
  --output data\processed\eval_hybrid_adaptive.json
```

The evaluator uses the production clause router and scores passage relevance,
not only clause-category matches.

### Answer Grounding

Run deterministic route, abstention, citation, required-concept, and
overclaim checks:

```powershell
.\.conda-clauselens\python.exe evaluation\answer_eval.py
```

Add an optional model judge for claim support, attribution, uncertainty, and
directness:

```powershell
.\.conda-clauselens\python.exe evaluation\answer_eval.py --judge
```

The model judge incurs API usage and is intentionally excluded from normal
unit tests.

### End-to-End Performance

```powershell
.\.conda-clauselens\python.exe evaluation\performance_benchmark.py `
  --model gpt-4.1-mini-2025-04-14 `
  --repeats 10 `
  --candidate-limit 3 `
  --output data\processed\performance_hybrid_final_120.json
```

This runs 120 sequential answer-quality and latency requests.

### Model and Service-Tier Comparison

```powershell
.\.conda-clauselens\python.exe evaluation\performance_benchmark.py `
  --matrix `
  --repeats 3 `
  --output data\processed\performance_matrix.json
```

This benchmark incurs hosted-model usage and does not automatically change the
live model.

## Telemetry and Persistence

Completed Streamlit queries are recorded in:

```text
data/processed/query_metrics.jsonl
```

Recorded data includes stage latency, model metadata, evidence IDs, citation
checks, routing consistency, and explicit user feedback.

Chat history is stored in:

```text
data/processed/chat_history.db
```

Live telemetry checks are operational signals, not labeled accuracy metrics.
Accuracy claims come from the repeatable offline evaluations.

## Public Deployment

The deployable application is under
[cloudflare/](cloudflare/README.md). It uses a React interface, Cloudflare
Worker streaming API, and a committed static retrieval index generated from
the same 463 vectors validated by the Python benchmark.

Deployment protections include:

- Turnstile verification.
- Per-IP minute and daily limits.
- Global daily AI budget.
- Concurrency leases.
- Strict request-size limits.
- Server-only API keys.

A free `*.workers.dev` address is sufficient. A custom domain is optional.

## Project Structure

```text
app/
  api.py                   FastAPI service
  chat.py                  chat routing and grounded generation
  chat_history.py          SQLite conversation persistence
  cuad.py                  CUAD preparation helpers
  rag.py                   hybrid retrieval and reranking
  streamlit_app.py         local chat interface
  telemetry.py             query metrics and feedback

evaluation/
  answer_eval.py           generated-answer evaluation
  answer_tests.jsonl       answer-quality cases
  chat_benchmark.py        stage-level latency benchmark
  eval.py                  passage-level retrieval evaluation
  performance_benchmark.py end-to-end acceptance workload
  tests.jsonl              retrieval cases

scripts/
  prepare_cuad_subset.py   prepares clause evidence
  index_qdrant.py          incrementally indexes Qdrant
  search_qdrant.py         terminal retrieval utility

tests/                     unit and integration tests
docs/                      requirements, experiments, and results
```

## Testing

```powershell
.\.conda-clauselens\python.exe -m pytest
.\.conda-clauselens\python.exe -m ruff check .
```

The current suite contains 72 passing tests covering:

- CUAD parsing and record preparation.
- Incremental indexing.
- Dense, lexical, fused, and reranked retrieval.
- Contract-level deduplication.
- Clause routing and follow-up contextualization.
- Citation and grounding behavior.
- API endpoints and validation.
- Chat persistence and telemetry.
- Retrieval and performance evaluation logic.

## Current Limitations

- Only five clause categories are supported.
- The benchmark uses a curated, sequential workload rather than concurrent
  production traffic.
- The 11-case retrieval suite is useful but still small.
- The system assists contract review and does not replace legal judgment.
- Hosted-model latency prevents the stricter 2.0-second P95 and 700 ms
  first-token targets from being met consistently.

## Documentation

- [Product requirements](docs/prd.md)
- [Architecture and operating notes](docs/notes.md)
- [Experiment history](docs/experiments.md)
- [Latest performance report](docs/performance_results_2026-06-22.md)
- [Evaluation questions](docs/questions.md)

## Resume Summary

- Built a hybrid contract RAG pipeline using Qdrant dense retrieval, BM25,
  reciprocal-rank fusion, and adaptive cross-encoder reranking over 463 clause
  passages from 30 CUAD contracts across five clause types.
- Achieved 100% Recall@5 and 98.2% context precision on an 11-case retrieval
  evaluation, plus 100% deterministic answer accuracy and citation validity
  with 2.43-second P95 latency across 120 sequential GPT-4.1 mini requests.
- Kept estimated generation cost near $0.00060 per answer, or approximately
  $0.60 per 1,000 answers, by using local retrieval and reranking with compact
  grounded prompts.

## Next Steps

- Expand coverage to 10 to 15 thoroughly evaluated clause categories.
- Add full-contract chunking with character-level source spans.
- Increase the passage-level and adversarial evaluation sets.
- Add concurrent-user load testing.
- Evaluate a deterministic fast path for frequent contract questions.
