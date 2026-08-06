# QFind Retrieval Evaluation Notes

Date: 2026-08-06

## Goal

Improve and correctly frame QFind retrieval quality on the strict 25-case gold
evidence benchmark.

The important distinction is:

- Raw retrieval quality: whether the retriever can find the gold evidence before
  product post-processing.
- Production evidence selection: what the user sees after diversity filters such
  as document-level deduplication.

The work below improves the evaluation methodology and adds a product-facing
middle ground. It should not be described as simply "retrieval recall improved
from 0.660 to 0.940" because the retriever itself was not changed in that first
measurement step.

## What Was Implemented

### 1. Document Deduplication Was Made Configurable

Before the change, retrieval always kept only one passage per contract document.
That behavior was implemented as document-level deduplication after dense and
BM25 hybrid candidates were fused.

This was useful for chat UX because it encouraged evidence diversity across
contracts, but it was harmful for strict gold recall evaluation because some
questions have multiple gold evidence records from the same contract.

Implemented:

- `search_clause_evidence(..., deduplicate_documents=True)`
- `evaluation/eval.py --deduplicate-documents`
- `evaluation/eval.py --no-deduplicate-documents`

Default behavior remains unchanged for production:

```text
deduplicate_documents=True
```

Strict raw retrieval measurement uses:

```text
--no-deduplicate-documents
```

### 2. Per-Document Passage Cap Was Added

Binary deduplication was then generalized into a configurable cap:

```text
max_passages_per_document = 1     current production behavior
max_passages_per_document = 2     middle-ground product ablation
max_passages_per_document = None  raw retrieval, no document cap
```

Implemented:

- `limit_passages_per_document`
- `search_clause_evidence(..., max_passages_per_document=1)`
- `evaluation/eval.py --max-passages-per-document 2`

This gives a product-facing compromise: preserve some cross-contract diversity,
but avoid dropping every additional passage from the same contract.

### 3. Evaluation Diagnostics Were Improved

Each evaluation row now includes:

```text
expected_record_ids
retrieved_record_ids
gold_record_ids_found
missing_gold_record_ids
candidate_count
max_passages_per_document
rerank_reason
```

This makes failures inspectable. Instead of only seeing a low recall number, we
can identify exactly which gold evidence IDs were missed and whether they were
dropped by post-processing.

### 4. Candidate-Limit Sweep Was Added

Added:

```text
evaluation/candidate_sweep.py
tests/test_candidate_sweep.py
```

The sweep runs multiple `candidate_limit` values through the existing evaluator
and reports:

```text
recall_at_k
hit_rate_at_k
context_precision
mrr
ndcg
keyword_hit_rate
average_candidate_count
average_retrieval_latency_ms
average_reranking_latency_ms
```

This tests whether recall is capped because the reranker sees too few
candidates.

## Benchmark Results

Gold set:

```text
evaluation/tests_recall_gold.jsonl
25 questions
33 manually labeled gold evidence IDs
5 clause types
```

Shared settings:

```text
top_k = 5
rerank_mode = auto
candidate_limit = 3 unless swept
qdrant_mode = server
require_gold_record_ids = true
```

### Three Retrieval Modes

| Mode | Recall@5 | Top 5 Evidence Hit Rate | Context Precision |
| --- | ---: | ---: | ---: |
| Production-style, max 1 passage per document | 0.660 | 80.0% | 0.272 |
| Middle ground, max 2 passages per document | 0.860 | 96.0% | 0.277 |
| Raw retrieval, no document cap | 0.940 | 96.0% | 0.248 |

Interpretation:

- No document cap is the raw retrieval ceiling, not production behavior.
- Max 2 passages per document is the strongest product-facing improvement so
  far because it recovers hit rate to 96.0% while preserving some diversity.
- Raw retrieval has higher recall but lower context precision because more
  same-contract passages can enter the final top 5.

### What The Max-2 Cap Fixed

The max-2 setting turned these baseline misses into hits:

```text
Case 12: TomOnline Skype intellectual property license grant
Case 14: Cerence Nuance license under SpinCo patents
Case 15: HealthGate termination software license grant, partially fixed
Case 18: HealthGate backup and disaster recovery audit right
```

This confirms that top-1 document deduplication was sometimes evicting the gold
passage in favor of a different same-contract passage.

Remaining gap from max-2 to raw retrieval:

```text
Case 4
Case 15
Case 19
Case 20
```

These cases still miss one gold record under max-2 but recover under no document
cap.

## Candidate-Limit Sweep

Command:

```powershell
.\.conda-clauselens\python.exe evaluation\candidate_sweep.py `
  --tests evaluation\tests_recall_gold.jsonl `
  --qdrant-mode server `
  --top-k 5 `
  --rerank-mode auto `
  --candidate-limits 3 5 10 20 50 `
  --max-passages-per-document 2 `
  --require-gold-record-ids `
  --output data\processed\candidate_limit_sweep_max2.json
```

Results with max 2 passages per document:

| Candidate Limit | Recall@5 | Hit Rate@5 | Context Precision | Avg Rerank Latency |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 0.860 | 96.0% | 0.277 | 18.2 ms |
| 5 | 0.860 | 96.0% | 0.277 | 19.5 ms |
| 10 | 0.860 | 96.0% | 0.277 | 72.1 ms |
| 20 | 0.860 | 96.0% | 0.277 | 151.7 ms |
| 50 | 0.860 | 96.0% | 0.277 | 242.8 ms |

Results with document deduplication disabled:

| Candidate Limit | Recall@5 | Hit Rate@5 | Context Precision |
| ---: | ---: | ---: | ---: |
| 3 | 0.940 | 96.0% | 0.248 |
| 5 | 0.940 | 96.0% | 0.248 |
| 10 | 0.940 | 96.0% | 0.248 |
| 20 | 0.940 | 96.0% | 0.248 |
| 50 | 0.940 | 96.0% | 0.248 |

Conclusion:

Widening the reranker candidate limit is not the current bottleneck under the
current hybrid retrieval setup. Recall and context precision stay flat while
reranking latency increases.

## Verification

Tests:

```powershell
.\.conda-clauselens\python.exe -m pytest
```

Result:

```text
104 passed
```

Lint:

```powershell
.\.conda-clauselens\python.exe -m ruff check app\rag.py evaluation\eval.py evaluation\candidate_sweep.py tests\test_rag.py tests\test_evaluation.py tests\test_candidate_sweep.py
```

Result:

```text
All checks passed
```

Generated benchmark outputs:

```text
data/processed/eval_true_recall_k5.json
data/processed/eval_true_recall_k5_max2_per_doc.json
data/processed/eval_true_recall_k5_no_dedup.json
data/processed/candidate_limit_sweep_max2.json
data/processed/candidate_limit_sweep_no_dedup.json
```

## Resume Strength Assessment

This is strong enough for a resume if framed correctly.

Strong framing:

```text
Built a contract RAG evaluation harness with gold evidence IDs, separated raw
retrieval recall from production evidence-diversity filtering, and identified a
document-level deduplication confound. Introduced a configurable per-document
passage cap that improved production-style gold Recall@5 from 0.660 to 0.860
and Top 5 evidence hit rate from 80.0% to 96.0%, while preserving a raw
retrieval ceiling of 0.940 Recall@5.
```

Shorter resume bullet:

```text
Improved QFind contract RAG evaluation and retrieval post-processing by adding
gold evidence ID diagnostics, candidate-depth ablations, and a configurable
per-document passage cap, raising production-style Recall@5 from 0.660 to 0.860
and Top 5 evidence hit rate from 80.0% to 96.0%.
```

Interview-safe wording:

```text
The first discovery was not a model improvement. It was a measurement confound:
document deduplication was mixed into recall evaluation. I separated raw
retrieval recall from product-layer evidence selection, then tested a product
middle ground with up to two passages per document. That improved the
production-style metric while keeping the raw retrieval ceiling visible.
```

Avoid saying:

```text
Improved retrieval recall from 0.660 to 0.940.
```

That is misleading because 0.940 is the raw no-dedup retrieval ceiling, not the
current production path.

## Next Best Step

Candidate depth is not the bottleneck. The next serious retrieval-improvement
step should be an indexing or chunking ablation:

1. Compare current clause-level chunks against larger or overlap-aware chunks.
2. Test whether missed gold records are absent from the candidate pool or ranked
   below same-contract distractors.
3. Consider contextual retrieval after chunking diagnostics, not before.

