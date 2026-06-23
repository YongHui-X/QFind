# ClauseLens Chat Evaluation — 2026-06-21

## Scope

This report evaluates the five chats that were saved in
`data/processed/chat_history.db` on June 21, 2026.

```text
Saved chats: 5
Total turns: 11
Generated answers: 8
Abstentions: 3
```

The latency calculations use the response metadata stored with those saved
chats. Older telemetry from previous application versions was excluded.

This is a small manual sample. Its percentile values describe this sample only
and are not stable service-level measurements.

## Performance

Generated answers only:

| Metric | Result | Project target | Assessment |
| --- | ---: | ---: | --- |
| P50 total latency | 2.57 s | Under 2 s | Miss |
| P95 total latency | 5.24 s | Under 5 s | Slight miss |
| P95 first visible text | 4.16 s | Under 3 s | Miss |
| Citation validity | 8/8 | 100% | Pass |

Vector-only generated turns:

| Metric | Result |
| --- | ---: |
| Turns | 4 |
| P50 total latency | 1.67 s |
| P95 total latency | 2.25 s |
| P95 first visible text | 1.15 s |

Adaptively reranked generated turns:

| Metric | Result |
| --- | ---: |
| Turns | 4 |
| P50 total latency | 3.70 s |
| P95 total latency | 5.48 s |
| P95 first visible text | 4.35 s |

Interpretation:

- Ordinary vector-search questions performed well.
- Reranked questions were the main source of slow turns.
- Hosted answer generation remained a significant part of latency.
- One reranked query had an unusual 1.42-second vector-search time; ordinary
  vector searches were approximately 16–20 ms.
- The sample is too small to establish production P50 or P95 performance.

## Routing

Expected routing accuracy was 9/11, or 81.8%.

Correct behavior:

- Confidentiality correctly abstained as an unsupported topic.
- License follow-ups preserved `License Grant`.
- `Is it also transferable?` used the focused standalone query
  `Is it also transferable? License Grant`.
- `How long does it remain effective?` used
  `How long does it remain effective? License Grant`.

Routing failure:

```text
Are lost profits and anticipated savings recoverable?
```

This question abstained twice. It should resolve to `Cap On Liability`.
The router recognizes terms such as damages and consequential loss but does not
currently recognize lost profits or anticipated savings.

## Answer Quality

Manual assessment of the eight generated answers:

```text
Strongly grounded: 4
Partially grounded: 1
Material overclaim: 3
```

### Material overclaims

1. Unauthorized assignment

   The answer concluded that an assignment without consent was void. Two cited
   agreements supported that consequence, but another only prohibited the
   assignment and did not state that it was void.

2. Transfers by operation of law

   The answer opened with a universal conclusion even though the retrieved
   agreements contained different rules and exceptions.

3. Wholly owned subsidiary

   The answer treated permission to assign to an `Affiliate` as permission to
   assign to a wholly owned subsidiary without retrieving the agreement's
   definition of `Affiliate`.

### Partial grounding

The sublicensing answer implied that the retrieved agreements generally
permitted sublicensing under conditions, while one cited source instead
prohibited general third-party access except where expressly allowed.

### Stronger answers

The answers concerning punitive and consequential damages, license
exclusivity, license transferability, and license duration were generally
source-grounded and appropriately cited.

## Final Assessment

Overall prototype rating: **7/10**

Strengths:

- Valid citations and source traceability.
- Fast ordinary vector retrieval.
- Correct unsupported-topic abstention for confidentiality.
- Improved preservation of license follow-up context.
- Source-specific evidence remains visible for inspection.

Weaknesses:

- Routing vocabulary does not cover all supported legal paraphrases.
- Generated answers can still overgeneralize across agreements.
- Defined legal relationships can be inferred without retrieved definitions.
- Reranked turns sometimes miss latency targets.

## Recommended Actions

1. Add `lost profits`, `anticipated savings`, and related phrases to
   `Cap On Liability` routing.
2. Require mixed-source answers to begin with a qualified comparison.
3. Reject or revise generated claims that rely on unstated definitions such as
   `Affiliate`, subsidiary, successor, transfer, or sublicense.
4. Investigate the isolated 1.42-second local vector-search delay.
5. Run the answer-quality benchmark after stopping the API:

   ```powershell
   .\.conda-clauselens\python.exe evaluation\answer_eval.py
   .\.conda-clauselens\python.exe evaluation\answer_eval.py --judge
   ```

6. Collect at least 100 representative turns before making stable P50, P95, or
   P99 claims.

## Data Sources

```text
data/processed/chat_history.db
data/processed/query_metrics.jsonl
```
