# ClauseLens Performance Results — 2026-06-22

## Test Configuration

The performance matrix ran 36 requests per configuration using:

```text
Qdrant: local Docker server, version 1.18.0
Indexed evidence records: 463
Reranker candidate pool: 5
Workload: 12 answer-quality cases repeated 3 times in randomized order
Traffic profile: single-user sequential
```

The results are stored in:

```text
data/processed/performance_matrix.json
```

## Model And Service-Tier Matrix

| Configuration | P50 total | P95 total | P99 total | P95 first token | Deterministic pass | Citation validity | Answer-mode consistency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-4.1 mini Standard | 1.73 s | 2.61 s | 2.76 s | 1.44 s | 100% | 100% | 100% |
| GPT-4.1 mini Priority | 1.42 s | 2.73 s | 2.96 s | 2.26 s | 100% | 100% | 100% |
| GPT-5.4 mini Standard | 1.38 s | 1.91 s | 2.09 s | 1.50 s | 86.1% | 100% | 100% |
| GPT-5.4 mini Priority | 1.08 s | 1.48 s | 1.82 s | 1.21 s | 83.3% | 100% | 97.2% |

Priority processing was returned for 91.7% of requests in both Priority
configurations, below the benchmark's 95% reliability gate.

## Interpretation

### Quality-first configuration

`gpt-4.1-mini-2025-04-14` on Standard processing was the most reliable tested
configuration:

- Every deterministic answer check passed.
- Citation validity and answer-mode consistency were both 100%.
- P95 total latency was 2.61 seconds.
- P95 first-token latency was 1.44 seconds.

### Latency-first configuration

`gpt-5.4-mini-2026-03-17` on Standard processing was the best non-premium
latency result:

- P95 total latency was 1.91 seconds.
- P99 total latency was 2.09 seconds.
- Citation validity and answer-mode consistency were 100%.
- The raw deterministic pass rate was 86.1%.

Most GPT-5.4 deterministic failures involved strict lexical checks such as
`3%` versus `three percent` and `party` versus `parties`. However, some answers
also reached the output ceiling or ended incompletely, so GPT-5.4 should not
replace the quality-first default until the normalized evaluator and repeated
answer-completion checks pass.

### Priority processing

Priority reduced median latency, especially for GPT-5.4 mini, but did not reach
the proposed 500 ms P95 first-token target. It also failed the 95% returned-tier
reliability gate and would add premium processing cost.

## Retrieval Candidate Evaluation

| Candidates | Passed | MRR | nDCG | Average retrieval | Average reranking |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 11/11 | 1.000 | 0.971 | 89.6 ms | 50.7 ms |
| 6 | 11/11 | 1.000 | 0.977 | 92.9 ms | 55.3 ms |
| 8 | 11/11 | 1.000 | 0.982 | 111.1 ms | 75.1 ms |

The original dense-only experiment selected five candidates. After hybrid
fusion was added, a three-candidate cross-encoder pool preserved 11/11
passage-level retrieval passes while reducing reranking latency, so three is
the final default.

## Decision

Keep `gpt-4.1-mini-2025-04-14` Standard as the quality-first production/demo
default.

Treat GPT-5.4 mini Standard as the next candidate because it meets the
two-second P95 total-latency goal without Priority pricing. Promote it only
after:

1. Normalizing equivalent evaluator terms such as `3%` and `three percent`.
2. Preventing incomplete answers at the output ceiling.
3. Passing the full deterministic suite over at least 100 repeated requests.

The proposed 500 ms P95 first-token target was not achieved by any tested
hosted configuration. Reaching it likely requires a deterministic response
path for common questions rather than another model or service-tier change.

## Latency-Tuning Rerun

After the prompt and streaming changes, the same 36-request matrix was rerun and
saved to:

```text
data/processed/performance_matrix_after_latency_tuning.json
```

| Configuration | Baseline P95 total | Rerun P95 total | Baseline P95 first token | Rerun P95 first token | Baseline deterministic pass | Rerun deterministic pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-4.1 mini Standard | 2.61 s | 2.32 s | 1.44 s | 1.19 s | 100% | 91.7% |
| GPT-4.1 mini Priority | 2.73 s | 2.26 s | 2.26 s | 1.52 s | 100% | 88.9% |
| GPT-5.4 mini Standard | 1.91 s | 2.15 s | 1.50 s | 1.55 s | 86.1% | 72.2% |
| GPT-5.4 mini Priority | 1.48 s | 1.72 s | 1.21 s | 1.21 s | 83.3% | 80.6% |

The rerun improved first-token latency for the gpt-4.1-mini configurations, but
it also reduced deterministic pass rates. The main failures were:

- `liability-excluded-damages`, where the answer no longer repeated the
  required concept `damages`.
- `assignment-wholly-owned-subsidiary`, where the priority configuration
  overgeneralized an affiliate clause.
- GPT-5.4 retained its answer-mode brittleness on `license-duration` and
  related cases, so it still does not meet the quality bar for promotion.

## Claim Boundaries

These results are suitable for a portfolio benchmark, not a production SLA:

- Each model/tier configuration used 36 sequential requests.
- The workload is curated and limited to five supported clause categories.
- No concurrent-user load was tested.
- Model-judge faithfulness was not included in this matrix.

## Hybrid Retrieval And Accuracy Recovery

The final implementation restored the evidence budget to 1,000 characters,
added BM25 plus dense reciprocal-rank fusion, deduplicated results by contract,
and skipped cross-encoder reranking when dense and lexical top ranks agreed.

Passage-level retrieval results are stored in:

```text
data/processed/eval_hybrid_adaptive.json
```

| Metric | Final result | Target |
| --- | ---: | ---: |
| Cases passed | 11/11 | 11/11 |
| Recall@5 | 1.000 | > 0.80 |
| Context precision | 0.982 | > 0.80 |
| MRR | 1.000 | > 0.80 |
| nDCG | 0.998 | — |
| Average retrieval latency | 46.0 ms | < 200 ms |
| P95 retrieval latency | 58.2 ms | < 200 ms |
| Average reranking latency | 17.0 ms | < 200 ms |
| P95 reranking latency | 102.2 ms | < 200 ms |

The evaluator now resolves clause types through the production router and
scores passage relevance rather than filtering with the expected label.

The final 120-request GPT-4.1 mini Standard benchmark is stored in:

```text
data/processed/performance_hybrid_final_120.json
```

| Run | P50 total | P95 total | P95 first token | Deterministic pass |
| --- | ---: | ---: | ---: | ---: |
| Original baseline | 1.73 s | 2.61 s | 1.44 s | 100% |
| Prompt latency tuning | 1.78 s | 2.32 s | 1.19 s | 91.7% |
| Hybrid final, 120 requests | 1.69 s | 2.43 s | 1.22 s | 100% |

The final configuration recovered 100% deterministic checks, citation
validity, and answer-mode consistency with no critical failures over 120
requests. End-to-end retrieval P95 was 68.2 ms and reranking P95 was 124.6 ms.
It did not
meet the strict hosted-model goals of P95 total below 2.0 seconds or P95 first
token below 700 ms. It does meet the earlier P95-at-most-2.5-second objective.
Retrieval is within target; remaining tail latency is
primarily answer-model/provider time and requires a faster generation path or
deterministic answers for common questions.
