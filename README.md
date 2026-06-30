# QFind

**A RAG chatbot for asking contract questions and checking the source evidence.**

QFind helps legal, procurement, and compliance reviewers ask plain-English
questions about contracts and quickly see the clauses that support the answer.
Instead of making users manually scan long agreements, this RAG chatbot retrieves
relevant contract language, writes a concise answer, and shows the cited evidence
used to produce it.

The evaluated version of this RAG chatbot was tested on 463 clause evidence
records from 30 CUAD contracts. On the 11-case retrieval suite it reached
100% Recall@5, 98.2% average context precision, 1.000 MRR, and 0.998 nDCG.
On the final 120-request answer benchmark, it passed deterministic
route/citation/concept checks for repeated runs of 12 curated answer cases.

**Live production demo:** [qfind-736872970476.asia-southeast1.run.app](https://qfind-736872970476.asia-southeast1.run.app/)  

This is the live QFind RAG chatbot deployed on Google Cloud Run. Note that there is a 20 seconds warm up time when first loaded due to the nature of Google Cloud Run having to do a cold start, a trade off for hosting free on their platform.

This is a portfolio research prototype. It is not a legal advice tool.

## Contents

- [Verified Results](#verified-results)
- [Why It Matters](#why-it-matters)
- [Core Capabilities](#core-capabilities)
- [Demo Protections](#demo-protections)
- [How It Works](#how-it-works)
- [Technology Stack](#technology-stack)
- [Supported Contract Topics](#supported-contract-topics)
- [Quick Start](#quick-start)
- [Application Surfaces](#application-surfaces)
- [Evaluation](#evaluation)
- [CI/CD](#cicd)
- [Deployment](#public-deployment)
- [Project Structure](#project-structure)

## At a Glance

| Question | Answer |
| --- | --- |
| What is it? | A RAG chatbot for contract review. |
| Who is it for? | Legal, procurement, and compliance teams reviewing commercial agreements. |
| What does it help with? | Finding contract risks, rights, and obligations faster. |
| How does it build trust? | Every answer is tied to retrieved contract evidence and numbered citations. |
| What can it answer today? | Questions about assignment, liability caps, license grants, audit rights, and termination for convenience. |
| What happens outside that scope? | The chatbot refuses unsupported topics instead of guessing. |
| How is it delivered? | A browser chat interface backed by FastAPI, Qdrant, and OpenAI. |

## Product Demo

### Video Walkthrough

<img src="docs/Media/shortdemomp4.gif" alt="QFind RAG chatbot demo preview" width="900">

For a more detailed demo, please refer to the
[full MP4 walkthrough](docs/Media/QFind%20demo.mp4).

### Cited Evidence

Answers include numbered citations and expandable retrieved passages, so a
reviewer can verify claims against the source contract language.

<img src="docs/Media/Evidence.png" alt="QFind cited evidence panel" width="900">

### Conflicting Sublicensing Rights

When retrieved contracts differ, QFind qualifies the answer instead of
overstating a single global rule. Follow-up questions remain anchored to the
conversation context.

<img src="docs/Media/sublicensing.png" alt="QFind sublicensing answer with follow-up context" width="900">

### Unsupported Topic Guardrail

Questions outside the evaluated clause categories are rejected safely rather
than answered from unrelated evidence.

<img src="docs/Media/unsupported%20topics.png" alt="QFind unsupported topic response" width="900">

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

These answer-quality metrics are deterministic route, citation, abstention,
required-concept, and forbidden-overclaim checks over 12 curated scenarios. They
are not a broad semantic or legal-correctness score.

The system meets the practical P95 target of 2.5 seconds. It does not yet meet
the experimental targets of P95 below 2.0 seconds or first-token latency below
700 ms. The remaining tail latency is primarily hosted-model response time.

Detailed methodology and claim boundaries are documented in
[the performance report](docs/performance_results_2026-06-22.md).

### Token Usage and Estimated Cost

QFind keeps generation costs low by running dense retrieval, BM25 search,
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

## Why It Matters

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
- React chat interface served by FastAPI, plus an optional Streamlit interface.
- Browser-session API protection with signed `HttpOnly` cookies, origin checks,
  and per-session/IP rate limits for public demo traffic.
- Incremental indexing that skips unchanged contract clauses.
- Repeatable retrieval, answer-quality, and performance evaluations.

## Demo Protections

QFind includes several practical safeguards for a public portfolio demo.
They are designed to reduce casual abuse and make answers easier to verify,
not to replace a full enterprise security review.

| Protection | What it does | Why it matters |
| --- | --- | --- |
| Signed browser session | The React app first calls `GET /api/session`, which sets a signed `HttpOnly` cookie. Protected API routes require that cookie. | Blocks simple unauthenticated calls directly against `/search`, `/chat`, and `/chat/stream`. |
| Same-origin check | Write-style API calls must come from the same browser origin, or from the configured `ALLOWED_ORIGIN`. | Reduces cross-site request abuse against the public demo. |
| Rate limiting | Each browser session and client IP is limited by a sliding window. Defaults are 5 requests per minute and 50 per day. | Prevents one visitor from quickly burning API/model quota. |
| `Retry-After` response | When rate limited, the API returns `429` with a wait time. | Gives the frontend and users a clear reason instead of failing silently. |
| Narrow topic routing | The app only answers the five evaluated clause categories. Unsupported topics are rejected safely. | Avoids pretending the index covers every legal issue. |
| Evidence-only answering | The answer model is instructed to answer only from retrieved evidence and cite sources like `[1]`. | Keeps responses tied to visible contract text. |
| Prompt-injection guard | Retrieved contract text is treated as untrusted data, not as instructions to the model. | Prevents contract text from overriding the assistant's rules. |
| Multi-contract overclaim guard | If retrieved contracts differ, are silent, or contain exceptions, the answer must qualify the result instead of giving a global yes/no. | Prevents answers like "all liability limits exclude punitive damages" when only some retrieved clauses say that. |
| Compact evidence prompts | Only the strongest, query-relevant evidence is sent to the answer model. | Reduces cost, latency, and chances of irrelevant text influencing the answer. |

The in-memory rate limiter is intentionally lightweight for demo-scale Cloud
Run instances. For production multi-region or high-traffic use, this should be
replaced with shared rate limiting such as Redis, Cloud Armor, API Gateway, or
another centralized control.

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
Returned evidence: top 3 passages in the React chat UI
```

## Technology Stack

| Layer | Technology |
| --- | --- |
| Dataset | [CUAD](https://huggingface.co/datasets/theatticusproject/cuad) |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Vector database | Qdrant |
| Lexical retrieval | In-memory BM25 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Answer model | OpenAI GPT-4.1 mini |
| API and web service | FastAPI |
| Primary interface | React 19, TypeScript, and Vite served by FastAPI |
| Optional local interface | Streamlit |
| Public deployment | Cloud Run, Qdrant Cloud, and the built React app |
| Persistence | SQLite and JSONL telemetry |
| Testing | pytest |

## Supported Contract Topics

QFind currently supports five clause categories:

1. Assignment restrictions
2. Liability caps
3. License grants
4. Audit rights
5. Termination for convenience

Questions outside these categories are rejected safely instead of being
answered using unrelated evidence.

## Dataset Snapshot

Original dataset source: [theatticusproject/cuad on Hugging Face](https://huggingface.co/datasets/theatticusproject/cuad).

The current evaluated subset contains:

| Item | QFind | Full CUAD | Coverage |
| --- | ---: | ---: | ---: |
| Contracts | 30 | 510 | 5.9% |
| Supported clause categories | 5 | 41 | 12.2% |
| Clause evidence records | 463 | Not directly comparable | - |

An indexed clause evidence record is one searchable clause passage with its
source contract, category, citation metadata, and vector embedding. A contract
can contribute multiple records, so the 463 records represent searchable
passages from 30 contracts, not 463 of CUAD's 510 contracts.

| Clause category | Records |
| --- | ---: |
| Anti-Assignment | 71 |
| Audit Rights | 165 |
| Cap On Liability | 90 |
| License Grant | 116 |
| Termination For Convenience | 21 |

CUAD contains 510 commercial contracts and 41 clause categories. QFind
deliberately starts with a smaller evaluated scope rather than claiming broad
coverage without sufficient testing.

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
SESSION_SIGNING_SECRET=replace-with-a-random-local-secret
```

### 2. Install dependencies

Use Python 3.11 or 3.12 for the full development environment. Python 3.14 can
install and run the main application, but skips the optional Ragas dependency
because one of its transitive packages does not publish Python 3.14 wheels on
Windows.

```powershell
python -m pip install -r requirements.txt
```

The commands below use the existing project environment:

```powershell
python
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
python scripts\prepare_cuad_subset.py
python scripts\index_qdrant.py
```

The indexer stores a SHA-256 content hash and embedding model name with every
record. Later runs:

- Skip unchanged records.
- Re-embed changed clauses.
- Update metadata without regenerating vectors.
- Embed duplicate clause text only once per indexing run.

Use `--recreate` only when intentionally rebuilding the collection.

### 5. Build the React UI

The FastAPI service serves the built React app from `frontend/dist` at `/`.

```powershell
Set-Location frontend
npm install
npm run build
Set-Location ..
```

### 6. Start the full local app

Server mode uses the Docker Qdrant service from step 3:

```powershell
$env:QDRANT_MODE="server"
$env:MODEL_WARMUP_ENABLED="false"
$env:SESSION_SIGNING_SECRET="local-dev-session-secret"
python -m uvicorn app.api:app --reload
```

Open the React chat UI:

```text
http://127.0.0.1:8000/
```

API documentation and readiness:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/health
```

For quick local UI checks against the existing embedded index, use embedded
mode instead of Docker Qdrant:

```powershell
$env:QDRANT_MODE="embedded"
$env:MODEL_WARMUP_ENABLED="false"
$env:SESSION_SIGNING_SECRET="local-dev-session-secret"
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Keep `MODEL_WARMUP_ENABLED=false` for faster local startup. In production,
Cloud Run can also run with warmup disabled to avoid startup failures on cold
instances.

### 7. Optional Streamlit interface

```powershell
python -m streamlit run app\streamlit_app.py
```

Keep the API running while using Streamlit. The interface sends requests to:

```text
http://127.0.0.1:8000
```

## Startup Commands

Use these from the repository root after dependencies are installed.

Build the React UI:

```powershell
Set-Location frontend
npm run build
Set-Location ..
```

Start the full local React + FastAPI app with the existing embedded index:

```powershell
$env:QDRANT_MODE="embedded"
$env:MODEL_WARMUP_ENABLED="false"
$env:SESSION_SIGNING_SECRET="local-dev-session-secret"
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Start the full local app against Docker Qdrant:

```powershell
docker compose up -d qdrant
$env:QDRANT_MODE="server"
$env:MODEL_WARMUP_ENABLED="false"
$env:SESSION_SIGNING_SECRET="local-dev-session-secret"
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Open the app at:

```text
http://127.0.0.1:8000/
```

## Application Surfaces

### React Chat

- Served from the FastAPI root path after `npm run build`.
- Left-rail new chat, search history, and collapsible chat history controls.
- Browser session bootstrap through `GET /api/session`.
- Sends chat requests with credentials so the signed session cookie is included.
- Shows up to three retrieved evidence passages with the grounded answer.

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
| `GET /` | React chat UI when `frontend/dist` exists |
| `GET /api` | API metadata |
| `GET /api/session` | Signed browser session cookie |
| `GET /health` | Readiness and model configuration |
| `GET /clause-types` | Supported clause categories |
| `POST /search` | Clause evidence retrieval |
| `POST /chat` | Grounded non-streaming answer |
| `POST /chat/stream` | Grounded streaming answer |

`/clause-types`, `/search`, `/chat`, and `/chat/stream` require the browser
session cookie and same-origin request headers. Direct unauthenticated API calls
return `401`.

Example protected search from PowerShell:

```powershell
$origin = "http://127.0.0.1:8000"
Invoke-WebRequest "$origin/api/session" -SessionVariable clauseSession | Out-Null
Invoke-WebRequest "$origin/search" `
  -Method POST `
  -WebSession $clauseSession `
  -Headers @{ Origin = $origin } `
  -ContentType "application/json" `
  -Body '{"query":"Does the contract restrict assignment?","clause_type":"Anti-Assignment","limit":3}'
```

### Command-Line Search

```powershell
python scripts\search_qdrant.py "Does the contract restrict assignment?"
```

## Evaluation

### Retrieval Quality

```powershell
python evaluation\eval.py `
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
python evaluation\answer_eval.py
```

CI uses the offline deterministic mode so pull requests do not spend OpenAI
credits:

```powershell
python evaluation\answer_eval.py --offline
```

Add an optional model judge for claim support, attribution, uncertainty, and
directness:

```powershell
python evaluation\answer_eval.py --judge
```

The model judge incurs API usage and is intentionally excluded from normal
unit tests.

### Ragas Semantic Evaluation

[Ragas](https://docs.ragas.io/en/stable/) is available as a complementary
hosted-judge benchmark for release validation. It reuses the 12 curated answer
scenarios, collects the generated answer and retrieved contexts, and scores
faithfulness, answer relevancy, context precision, and context recall using the
[Ragas evaluation workflow](https://docs.ragas.io/en/stable/getstarted/evals/):

Run this evaluation from Python 3.11 or 3.12. On Windows with Python 3.14, pip
may try to build `scikit-network` from source and fail without Microsoft C++
Build Tools.

```powershell
$env:RAGAS_JUDGE_MODEL="gpt-4.1-mini-2025-04-14"
python evaluation\ragas_eval.py `
  --qdrant-mode server `
  --qdrant-url http://localhost:6333 `
  --rerank-mode auto `
  --output data\processed\ragas_eval_results.json `
  --enforce-gates
```

Initial release-quality gates are mean faithfulness >= 0.90, mean answer
relevancy >= 0.80, mean context precision >= 0.80, mean context recall >= 0.80,
and no critical case below 0.75 faithfulness. Ragas is not a PR gate because it
uses hosted judge calls and can vary with external model behavior.

### End-to-End Performance

```powershell
python evaluation\performance_benchmark.py `
  --model gpt-4.1-mini-2025-04-14 `
  --repeats 10 `
  --candidate-limit 3 `
  --output data\processed\performance_hybrid_final_120.json
```

This runs 120 sequential answer-quality and latency requests.

### Model and Service-Tier Comparison

```powershell
python evaluation\performance_benchmark.py `
  --matrix `
  --repeats 3 `
  --output data\processed\performance_matrix.json
```

This benchmark incurs hosted-model usage and does not automatically change the
live model.

## CI/CD

QFind uses GitHub Actions with manual deployment gates:

- `CI` runs on pull requests, pushes to `main`, and manual dispatch. It installs
  Python dependencies, runs `pytest` and `ruff`, checks and builds the React
  frontend, builds the Cloud Run Docker image, runs warning-only dependency
  audits, and fails only on critical container vulnerabilities.
- `RAG Quality` runs manually and weekly. It starts Qdrant, prepares the CUAD
  subset, indexes the collection, runs retrieval evaluation, runs offline
  deterministic answer checks, optionally runs Ragas release-quality gates when
  `OPENAI_API_KEY` is available, and uploads JSON reports as artifacts. It is
  intentionally not a pull-request gate.
- `Deploy to Cloud Run` is manual only. It first checks that `CI` passed for the
  selected commit, then builds and pushes the image to Artifact Registry,
  deploys Cloud Run with cost controls, and smoke-tests `/health` plus the root
  URL.

Required GitHub repository variables:

```text
GCP_PROJECT_ID
GCP_REGION=asia-southeast1
CLOUD_RUN_SERVICE=qfind
QDRANT_CLOUD_URL
OPENAI_MODEL=gpt-4.1-mini-2025-04-14
```

Required GitHub repository secrets:

```text
GCP_WORKLOAD_IDENTITY_PROVIDER
GCP_SERVICE_ACCOUNT
```

Runtime secrets remain in Google Secret Manager:

```text
OPENAI_API_KEY
QDRANT_API_KEY
SESSION_SIGNING_SECRET
```

Hosted-model judging and full performance benchmarks are not part of default PR
CI because they spend API credits and measure external hosted-model latency
variance. Run them manually when validating release claims.

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

The primary deployment path is Cloud Run with Qdrant Cloud:

```text
Cloud Run FastAPI service
  -> Qdrant Cloud live vector database
  -> BM25 lexical index built from Qdrant payloads
  -> OpenAI grounded answer generation
  -> React chat UI served from the same origin
```

See [docs/cloud_run.md](docs/cloud_run.md) for the exact setup commands.

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

frontend/
  src/                     React chat interface
  dist/                    built assets served by FastAPI

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
Dockerfile                 Cloud Run container entrypoint
```

## Testing

```powershell
python -m pytest
python -m ruff check .
```

The current suite covers:

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

## Future Plans

- Expand coverage to 10 to 15 thoroughly evaluated clause categories.
- Add full-contract chunking with character-level source spans.
- Increase the passage-level and adversarial evaluation sets.
- Add concurrent-user load testing.
- Evaluate a deterministic fast path for frequent contract questions.
