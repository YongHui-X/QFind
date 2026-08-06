# Repository Agent Guidance

## RAG Evaluation Metric Wording

Use precise metric names when updating QFind documentation, reports, or agent
notes. The repository has both deterministic custom retrieval metrics and
optional hosted Ragas metrics, and their names must not be mixed.

Answer Relevancy means whether the answer directly addresses the user question.
In QFind this is covered by deterministic answer checks, with optional Ragas
answer relevancy for hosted semantic evaluation.

Faithfulness means whether generated claims are supported by retrieved evidence.
In QFind this is covered by Ragas faithfulness plus deterministic citation and
overclaim checks.

Contextual Relevancy means whether retrieved contexts are relevant to the
question. QFind does not currently report this as a separate hosted metric.
The custom retrieval relevance checks and custom context precision partially
reflect it.

Contextual Recall means how much required evidence was retrieved. QFind reports
strict gold evidence ID Recall@5 on the 25-question gold set, and can optionally
report Ragas context recall in hosted semantic evaluation.

Contextual Precision means whether relevant contexts are ranked ahead of less
relevant contexts. Ragas context precision is the hosted judge metric for this.
QFind also reports MRR and nDCG for deterministic ranking quality.

Guardrails:

- Do not call the 11-query curated retrieval metric strict Recall@5.
- Do not claim raw no-dedup `0.940 Recall@5` as production behavior. It is a
  raw retrieval ceiling before product-layer evidence diversity filters.
- Distinguish QFind's custom `context_precision` from Ragas context precision.
  Qualify the custom metric as `custom context precision` or `context precision
  (custom Precision@K)`.
