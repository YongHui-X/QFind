# ClauseLens

ClauseLens is a contract intelligence RAG chatbot. It helps a user ask plain-English questions about legal agreements, retrieve the most relevant clause evidence, and generate grounded answers with source information attached so the result can be checked against the original contract.

Project documentation:

- [`docs/prd.md`](docs/prd.md): product requirements and development rules.
- [`docs/notes.md`](docs/notes.md): architecture, terminology, and operating notes.
- [`docs/experiments.md`](docs/experiments.md): retrieval, quality, and latency experiments.
- [`docs/performance_results_2026-06-22.md`](docs/performance_results_2026-06-22.md): latest model, tier, and reranking benchmark.

The project is built around the CUAD contract dataset. It prepares labeled contract clauses, embeds them with Sentence Transformers, stores them in Qdrant, and exposes the search layer through a command-line tool, FastAPI service, Streamlit demo, and retrieval evaluation script.

This repository is not a legal advice tool. It is a portfolio project showing how retrieval-augmented generation foundations can be built responsibly for contract review.

## Why This Exists

Contract review is slow because useful information is often buried inside long agreements. ClauseLens focuses on the first step: finding the right evidence quickly.

Instead of asking a model to immediately generate an answer, the system first retrieves the clauses that support an answer. This makes the workflow easier to inspect, easier to evaluate, and safer for a domain where citation and traceability matter.

## Benefits

- Finds contract evidence using natural-language questions, not only exact keyword search.
- Supports filtering by clause type, such as audit rights, assignment restrictions, liability caps, license grants, and termination rights.
- Returns source metadata with each result, including source PDF name, TXT path, document ID, answer label, score, and evidence text.
- Includes a repeatable evaluation script so retrieval quality can be measured instead of judged only by manual testing.
- Includes live per-query telemetry and a chat latency benchmark so you can
  isolate rewrite, embedding, vector search, reranking, and answer cost.
- Provides both a FastAPI backend and a Streamlit demo UI for easier review.

## Current Demo Surfaces

- CLI search for quick local testing.
- FastAPI service with `/health`, `/clause-types`, `/search`, and `/chat`.
- Streamlit chat UI with clause-type filter, top-k control, retrieved evidence panel, and evaluation summary panel.
- Cloudflare Worker + React public deployment that does not require an
  always-on computer, Docker process, custom domain, or Streamlit wake screen.
- Persistent local chat history with new-chat, reopen, and delete controls.
- JSONL-based evaluation cases for repeatable retrieval checks.

## Permanent Public Deployment

The deployable application is under [`cloudflare/`](cloudflare/README.md). It
serves the React UI and streaming API from one Cloudflare Worker and uses a
committed static retrieval index generated from the same 463 vectors validated
by the Python benchmark.

The public deployment preserves the final hybrid retrieval configuration:

```text
BGE dense retrieval (6 candidates)
  + BM25 retrieval (6 candidates)
  -> reciprocal-rank fusion, k=60
  -> deduplicate by contract
  -> adaptive top-3 BGE reranking
  -> up to 5 cited evidence results
```

Abuse controls include Turnstile, per-IP minute and daily limits, a global
daily AI budget, concurrency leases, strict request sizes, and server-only API
keys. A free `*.workers.dev` address is sufficient; a custom domain is
optional.

## Run Commands

Start the local Qdrant server:

```powershell
docker compose up -d qdrant
```

Use the project conda environment if you are working in this existing workspace:

```powershell
.\.conda-clauselens\python.exe scripts\prepare_cuad_subset.py
.\.conda-clauselens\python.exe scripts\index_qdrant.py
```

The application defaults to `QDRANT_MODE=server` for stable access and
benchmarking. Set `QDRANT_MODE=embedded` to use `data/qdrant_local` for simple
single-process development; embedded storage cannot be opened concurrently by
the API and evaluation commands.

Indexing stores a SHA-256 content hash and embedding model name in each Qdrant
payload. Subsequent runs skip unchanged records, update metadata without
re-embedding, and embed duplicate clause text only once per run. Use
`--recreate` only when you intentionally want to rebuild the collection.

### Incremental Embedding With Content Hashes

Each clause is assigned a SHA-256 hash based on its text. Before embedding, the
indexer compares this hash and the embedding model name with the values already
stored in Qdrant. New or changed clauses are embedded, unchanged clauses are
skipped, and metadata-only changes are updated without rebuilding the vector.
Identical clause text is embedded once per indexing run and reused.

Benefits:

- Faster indexing after the initial run.
- Lower CPU, GPU, and memory usage.
- No unnecessary embedding of unchanged clauses.
- Automatic re-embedding when clause text or the embedding model changes.
- Safe metadata updates without recomputing vectors.

Create `.env` from the example configuration when setting up a new checkout:

```powershell
Copy-Item .env.example .env
```

The default chat configuration is:

```text
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-mini-2025-04-14
OPENAI_SERVICE_TIER=standard
MODEL_WARMUP_ENABLED=true
```

The main application uses OpenAI directly. Ollama client settings remain
available for local development utilities:

```text
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=llama3.2:latest
```

Run the API:

```powershell
.\.conda-clauselens\python.exe -m uvicorn app.api:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Run the Streamlit demo:

```powershell
.\.conda-clauselens\python.exe -m streamlit run app\streamlit_app.py
```

Keep the API running while using Streamlit. The UI sends chat requests to `http://127.0.0.1:8000` so Qdrant is opened by only one backend process.

Run retrieval evaluation:

```powershell
.\.conda-clauselens\python.exe evaluation\eval.py --top-k 5 --no-rerank
.\.conda-clauselens\python.exe evaluation\eval.py --top-k 5 --rerank --candidate-limit 20
.\.conda-clauselens\python.exe evaluation\eval.py --top-k 5 --rerank-mode auto
```

Save evaluation output for the Streamlit sidebar:

```powershell
.\.conda-clauselens\python.exe evaluation\eval.py --top-k 5 --rerank-mode auto --output data\processed\eval_results.json
```

Run the chat latency benchmark:

```powershell
.\.conda-clauselens\python.exe evaluation\chat_benchmark.py --qdrant-mode server --repeats 3 --output data\processed\chat_latency_benchmark.json
```

Run the 120-turn latency and deterministic-quality acceptance workload:

```powershell
.\.conda-clauselens\python.exe evaluation\performance_benchmark.py --repeats 10 --enforce-gates
```

Screen the pinned model snapshots on Standard and Priority processing:

```powershell
.\.conda-clauselens\python.exe evaluation\performance_benchmark.py --matrix --repeats 3 --output data\processed\performance_matrix.json
```

The matrix intentionally incurs API usage. It does not change the live model
automatically; select a winner only after reviewing both latency and
deterministic quality results.

Run generated-answer grounding evaluation:

```powershell
.\.conda-clauselens\python.exe evaluation\answer_eval.py
.\.conda-clauselens\python.exe evaluation\answer_eval.py --judge
```

The first command runs deterministic route, abstention, citation, required
concept, and forbidden-claim checks. `--judge` additionally uses
`ANSWER_EVAL_MODEL` (default `gpt-4.1-mini`) to judge claim support, source
attribution, uncertainty handling, and directness. The judge is intentionally
not part of normal pytest runs because it requires an API key and incurs model
cost. Stop the local API before running evaluations against embedded Qdrant,
because the local storage directory permits only one client process.

Benchmark output includes:

- `wall_clock_latency_ms`: end-to-end time from request start to final response.
- `first_token_latency_ms`: time until the first streamed token appears.
- `contextualization_latency_ms`, `embedding_latency_ms`,
  `vector_search_latency_ms`,
  `reranker_loading_latency_ms`, `reranking_latency_ms`, and
  `answer_latency_ms`: the stage breakdown.

Use the comparison mode to tell whether the slowdown is mostly reranking,
hosted answer generation, or local retrieval.

Chat uses adaptive reranking by default. It applies the cross-encoder to the
measured difficult intellectual-property paraphrase while keeping ordinary
queries on faster vector search. It also reranks nuanced detail questions about
exceptions, operation of law, defined relationships, duration, territory,
thresholds, and notice periods. The Streamlit control can force reranking off
or always on. The raw `/search` endpoint uses vector ordering by default; set
`RERANKING_ENABLED=true` to enable reranking there.

First-turn chat questions go directly to retrieval. Follow-up questions are
contextualized deterministically from the previous user question, avoiding a
second LLM request.

The API warms the embedding model and reranker before reporting ready. Check
`/health` and wait for `"ready": true` before opening the demo.

Each completed Streamlit query appends local telemetry to
`data/processed/query_metrics.jsonl`. The sidebar updates query count, latency,
deterministic citation/evidence checks, and explicit helpful/not-helpful
feedback. These live checks are not labeled accuracy. Retrieval accuracy still
comes from the repeatable questions in `evaluation/tests.jsonl`.

Completed conversations are also saved automatically to
`data/processed/chat_history.db`. Use the collapsible Streamlit sidebar to start
a new chat, reopen a previous conversation, or delete saved history. Reopening
a chat restores its messages and retrieval controls.

The chat layer also resolves plain-English questions to one of the five starter
clause types before retrieval. This improves paraphrase handling and prevents
unsupported topics, such as governing law or automatic renewal, from being
answered with unrelated evidence.

Run tests:

```powershell
.\.conda-clauselens\python.exe -m pytest
.\.conda-clauselens\python.exe -m ruff check .
```

For a fresh environment, install dependencies first:

```powershell
python -m pip install -r requirements.txt
```

## Screenshots

Add screenshots here after capturing the local demo.

### Streamlit Chat Demo

### API Docs

### Evaluation Summary

## Evaluation Insights

### Latest Performance Matrix

The latest benchmark used Docker Qdrant, five reranker candidates, and 36
sequential requests per model/tier configuration:

| Configuration | P95 total | P99 total | P95 first token | Deterministic pass | Citation validity |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-4.1 mini Standard | 2.61 s | 2.76 s | 1.44 s | 100% | 100% |
| GPT-4.1 mini Priority | 2.73 s | 2.96 s | 2.26 s | 100% | 100% |
| GPT-5.4 mini Standard | 1.91 s | 2.09 s | 1.50 s | 86.1% | 100% |
| GPT-5.4 mini Priority | 1.48 s | 1.82 s | 1.21 s | 83.3% | 100% |

GPT-4.1 mini Standard is the current quality-first default. GPT-5.4 mini
Standard is the latency-first candidate, but it is not promoted until
normalized deterministic evaluation and answer-completion checks pass.
Priority was returned for only 91.7% of requested Priority calls and did not
reach the proposed 500 ms P95 first-token target.

These are portfolio benchmark results from a curated sequential workload, not
production SLA claims. See
[`docs/performance_results_2026-06-22.md`](docs/performance_results_2026-06-22.md)
for methodology and claim boundaries.

### Retrieval Quality

The retrieval evaluation contains plain-English contract review questions
across the starter clause types, including a regression case for the
intellectual-property usage wording.

Baseline vector-search results:

```text
Passed: 9/11
MRR: 0.886
nDCG: 0.921
Top result was the right clause type: 81.8%
Right clause appeared somewhere in the top 5: 100%
Expected evidence words were found: 100%
```

Cross-encoder reranking results over 20 candidates:

```text
Passed: 10/11
MRR: 0.955
nDCG: 0.964
Top result was the right clause type: 90.9%
Right clause appeared somewhere in the top 5: 100%
Average retrieval latency: 1375.8 ms on the local CPU
```

Adaptive reranking results:

```text
Passed: 11/11
MRR: 1.000
nDCG: 0.985
Top result was the right clause type: 100%
Average retrieval latency: 305.3 ms across all cases
```

The current Docker-Qdrant configuration uses five reranker candidates and
retains 11/11 passes, MRR 1.000, and 100% top-1 clause-type accuracy. In the
candidate comparison it averaged 89.6 ms retrieval and 50.7 ms reranking.

## Dataset Snapshot

The current starter subset contains:

```text
Documents: 30
Clause evidence records: 463
```

Clause evidence counts:

```text
Anti-Assignment: 71
Audit Rights: 165
Cap On Liability: 90
License Grant: 116
Termination For Convenience: 21
```

### Why The Starter Scope Is Limited

ClauseLens intentionally starts with five clause types and 30 contracts so the
full ingestion, retrieval, reranking, citation, and evaluation pipeline can be
tested before expanding coverage. The selected contracts contain the most
evidence for the supported clause types.

CUAD contains roughly 500 contracts and 41 clause categories. The next planned
step is to expand to 10-15 well-tested clause types across more contracts,
rather than claim broad coverage without sufficient evaluation.

Expected CUAD files:

```text
data/cuad/CUAD_v1/master_clauses.csv
data/cuad/CUAD_v1/CUAD_v1.json
data/cuad/CUAD_v1/full_contract_txt/Part_I
data/cuad/CUAD_v1/full_contract_txt/Part_II
```

Raw CUAD data and local Qdrant storage are ignored by Git because they are large local artifacts.

## Architecture

```text
CUAD CSV + TXT contracts
        |
        v
scripts/prepare_cuad_subset.py
        |
        v
data/processed/starter_clause_evidence.jsonl
        |
        v
scripts/index_qdrant.py
        |
        v
SentenceTransformer embeddings -> Qdrant
        |
        v
app/rag.py shared retrieval helpers
        |
        +--> scripts/search_qdrant.py
        +--> app/api.py
        +--> app/streamlit_app.py
        +--> evaluation/eval.py
```

## Project Structure

```text
app/
  api.py                  FastAPI retrieval service
  streamlit_app.py        local demo UI
  rag.py                  shared Qdrant and embedding helpers
  cuad.py                 CUAD data preparation helpers

scripts/
  prepare_cuad_subset.py  creates starter JSONL evidence records
  index_qdrant.py         embeds and indexes records into Qdrant
  search_qdrant.py        searches indexed evidence from the terminal

evaluation/
  cases.py                loads retrieval evaluation cases
  eval.py                 runs retrieval metrics against Qdrant
  tests.jsonl             retrieval test cases

tests/                    unit tests for data prep, retrieval, eval, and API
docs/                     setup notes, dataset notes, developer notes, plan
```

## API Example

Start the API:

```powershell
.\.conda-clauselens\python.exe -m uvicorn app.api:app --reload
```

Search:

```powershell
curl -X POST http://localhost:8000/search `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"Does the contract restrict assignment?\",\"clause_type\":\"Anti-Assignment\",\"limit\":5}"
```

Example response shape:

```json
{
  "query": "Does the contract restrict assignment?",
  "clause_type": "Anti-Assignment",
  "limit": 5,
  "result_count": 1,
  "results": [
    {
      "score": 0.87,
      "clause_type": "Anti-Assignment",
      "source_pdf": "Example.pdf",
      "source_txt": "data/cuad/CUAD_v1/full_contract_txt/Part_I/Example.txt",
      "document_id": "Example",
      "answer": "Yes",
      "text": "This Agreement may not be assigned without consent..."
    }
  ]
}
```

## Tests And Quality Checks

Current tests cover:

- CUAD filename matching and evidence parsing.
- starter-record selection.
- retrieval query validation and Qdrant call shape.
- retrieval evaluation scoring and export.
- FastAPI health, clause-type, search, and validation endpoints.

Verification commands:

```powershell
.\.conda-clauselens\python.exe -m pytest
.\.conda-clauselens\python.exe -m ruff check .
.\.conda-clauselens\python.exe -m py_compile app\api.py app\cuad.py app\rag.py app\streamlit_app.py scripts\prepare_cuad_subset.py scripts\index_qdrant.py scripts\search_qdrant.py evaluation\cases.py evaluation\eval.py
```

## Resume Summary

Built ClauseLens, a contract intelligence RAG chatbot over CUAD using Sentence
Transformers, Qdrant, and OpenAI for grounded answer generation. Implemented
metadata-filtered semantic search, source-grounded clause evidence, FastAPI and
Streamlit chat surfaces, and retrieval evaluation with clear quality insights.

## Current Status

Implemented:

- CUAD evidence extraction and starter JSONL generation.
- embedded-local and server Qdrant indexing.
- reusable retrieval helpers.
- CLI search.
- FastAPI search and chat services.
- Streamlit chat UI with recent-turn context.
- grounded LLM answer generation with citations.
- retrieval evaluation CLI and JSONL test cases.
- unit tests for core behavior and API endpoints.

Next:

- add screenshots to this README.
- add full-contract chunking with character spans.
- add citation correctness and answer faithfulness evaluation.
