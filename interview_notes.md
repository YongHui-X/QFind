# QFind Interview Notes

## How The Metrics Were Verified

I verified the reported metrics using two separate evaluation runs.

### Retrieval Metrics

The **100% Recall@5** and **98.2% context precision** came from the retrieval evaluation.

I ran the retrieval evaluator on 11 curated contract questions over 463 clause passages from 30 CUAD contracts. For each question, the system retrieved the top 5 passages. Recall@5 checked whether the correct supporting passage appeared anywhere in those top 5 results.

Recall@5 was 100% because every test question had the required evidence in the top 5 retrieved passages.

Context precision measured how much of the retrieved context was actually relevant. Most queries returned 5 out of 5 relevant passages. One query returned 4 out of 5 relevant passages, so the average became 98.2% instead of 100%.

### Latency Metric

The **2.43s P95 latency** came from the end to end chat benchmark, not just retrieval.

I ran 120 sequential requests using GPT-4.1 mini. The benchmark measured full wall clock response time, including retrieval, reranking, prompt construction, and answer generation. The 95th percentile response time was 2.433 seconds, which I rounded to 2.43 seconds.

### Important Clarification

These numbers were not all from the exact same test.

Retrieval quality was measured using an 11 case retrieval benchmark. The 2.43s P95 latency was measured separately using a 120 request end to end answer benchmark. This avoided mixing retrieval accuracy with hosted model response latency.

Interview phrasing:

> I verified retrieval quality with an 11 case retrieval benchmark, and verified latency separately with a 120 request end to end benchmark. This avoided mixing retrieval accuracy with hosted model response latency.

## How To Rerun The Metrics

Run these commands in PowerShell from the project root.

```powershell
cd "C:\Users\User\OneDrive - National University of Singapore\Latest personal projects\QFind"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Start Qdrant:

```powershell
docker compose up -d qdrant
```

Run the retrieval benchmark:

```powershell
python evaluation\eval.py `
  --qdrant-mode server `
  --top-k 5 `
  --rerank-mode auto `
  --candidate-limit 3 `
  --output data\processed\eval_hybrid_adaptive.json
```

Confirm Recall@5 and context precision:

```powershell
$rows = Get-Content data\processed\eval_hybrid_adaptive.json -Raw | ConvertFrom-Json
"Cases: $($rows.Count)"
"Recall@5: $([math]::Round((($rows | Measure-Object recall_at_k -Average).Average) * 100, 1))%"
"Context precision: $([math]::Round((($rows | Measure-Object context_precision -Average).Average) * 100, 1))%"
```

Run the 120 request latency benchmark:

```powershell
python evaluation\performance_benchmark.py `
  --model gpt-4.1-mini-2025-04-14 `
  --repeats 10 `
  --candidate-limit 3 `
  --output data\processed\performance_hybrid_final_120.json
```

Confirm P95 response latency:

```powershell
$result = Get-Content data\processed\performance_hybrid_final_120.json -Raw | ConvertFrom-Json
$p95 = $result.configurations[0].summary.overall.p95_total_ms
"P95 response latency: $([math]::Round($p95 / 1000, 2))s"
```

Note: the latency benchmark calls GPT-4.1 mini, so it needs the OpenAI API key configured and will use API credits.
