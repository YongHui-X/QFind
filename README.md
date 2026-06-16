# ClauseLens

ClauseLens is a contract intelligence search prototype. It helps a user ask plain-English questions about legal agreements and retrieve the most relevant clause evidence, with source information attached so the answer can be checked against the original contract.

The project is built around the CUAD contract dataset. It prepares labeled contract clauses, embeds them with Sentence Transformers, stores them in Qdrant, and exposes the search layer through a command-line tool, FastAPI service, Streamlit demo, and retrieval evaluation script.

This repository is not a legal advice tool. It is a portfolio project showing how retrieval-augmented generation foundations can be built responsibly before adding an LLM answer layer.

## Why This Exists

Contract review is slow because useful information is often buried inside long agreements. ClauseLens focuses on the first step: finding the right evidence quickly.

Instead of asking a model to immediately generate an answer, the system first retrieves the clauses that support an answer. This makes the workflow easier to inspect, easier to evaluate, and safer for a domain where citation and traceability matter.

## Benefits

- Finds contract evidence using natural-language questions, not only exact keyword search.
- Supports filtering by clause type, such as audit rights, assignment restrictions, liability caps, license grants, and termination rights.
- Returns source metadata with each result, including source PDF name, TXT path, document ID, answer label, score, and evidence text.
- Includes a repeatable evaluation script so retrieval quality can be measured instead of judged only by manual testing.
- Provides both a FastAPI backend and a Streamlit demo UI for easier review.

## Current Demo Surfaces

- CLI search for quick local testing.
- FastAPI service with `/health`, `/clause-types`, and `/search`.
- Streamlit UI with query input, clause-type filter, top-k control, evidence cards, and evaluation summary panel.
- JSONL-based evaluation cases for repeatable retrieval checks.

## Run Commands

Use the project conda environment if you are working in this existing workspace:

```powershell
.\.conda-clauselens\python.exe scripts\prepare_cuad_subset.py
.\.conda-clauselens\python.exe scripts\index_qdrant.py --qdrant-path data/qdrant_local --recreate
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

Run retrieval evaluation:

```powershell
.\.conda-clauselens\python.exe evaluation\eval.py --top-k 5
```

Save evaluation output for the Streamlit sidebar:

```powershell
.\.conda-clauselens\python.exe evaluation\eval.py --top-k 5 --output data\processed\eval_results.json
```

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

### Streamlit Search Demo

### API Docs

### Evaluation Summary

## Evaluation Insights

The latest local retrieval evaluation used 10 plain-English contract review questions across the starter clause types.

Results:

```text
Passed: 9/10
Top result was the right clause type: 90%
Right clause appeared somewhere in the top 5: 100%
Expected evidence words were found: 100%
```

In plain English, ClauseLens usually puts the right clause at the top of the results. Even when it missed the top spot, it still found the right evidence within the first five results.

The one weaker case was a question about intellectual property usage rights. The system found a relevant license-grant clause, but ranked it fourth instead of first. That is useful feedback: the next quality improvement should focus on better ranking for questions where the wording changes from "license" to related concepts like "right to use intellectual property."

Overall, the evaluation suggests the current prototype is strong enough for a portfolio demo. It also shows a clear next step: add reranking or more diverse evaluation examples before claiming production-level retrieval quality.

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

Built ClauseLens, a contract intelligence retrieval prototype over CUAD using Sentence Transformers and Qdrant. Implemented metadata-filtered semantic search, source-grounded clause evidence, FastAPI and Streamlit demo surfaces, and retrieval evaluation with clear quality insights.

## Current Status

Implemented:

- CUAD evidence extraction and starter JSONL generation.
- embedded-local and server Qdrant indexing.
- reusable retrieval helpers.
- CLI search.
- FastAPI search service.
- Streamlit demo UI.
- retrieval evaluation CLI and JSONL test cases.
- unit tests for core behavior and API endpoints.

Next:

- add screenshots to this README.
- add full-contract chunking with character spans.
- add reranking for harder semantic matches.
- add grounded LLM answer generation with citations.
- add citation correctness and answer faithfulness evaluation.
