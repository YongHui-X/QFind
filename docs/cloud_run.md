# Cloud Run deployment

This deployment runs the real FastAPI + Qdrant RAG backend:

```text
Cloud Run FastAPI service
  -> Qdrant Cloud vector database
  -> OpenAI answer generation
```

Use this path when the public demo should show live Qdrant usage.

## 1. Create Qdrant Cloud

Create a Qdrant Cloud cluster and copy:

```text
QDRANT_CLOUD_URL
QDRANT_API_KEY
```

The free tier is enough for the current starter index.

## 2. Index the evidence into Qdrant Cloud

Run this locally from the repo root:

```powershell
$env:QDRANT_CLOUD_URL="https://YOUR-CLUSTER-url"
$env:QDRANT_API_KEY="YOUR-QDRANT-API-KEY"

python scripts\index_qdrant.py --recreate
```

Expected output includes:

```text
Collection count: 463
```

## 3. Create Google Secret Manager secrets

```powershell
gcloud secrets create OPENAI_API_KEY --replication-policy="automatic"
gcloud secrets versions add OPENAI_API_KEY --data-file=-

gcloud secrets create QDRANT_API_KEY --replication-policy="automatic"
gcloud secrets versions add QDRANT_API_KEY --data-file=-

gcloud secrets create SESSION_SIGNING_SECRET --replication-policy="automatic"
gcloud secrets versions add SESSION_SIGNING_SECRET --data-file=-
```

Paste each secret value when prompted, then press `Ctrl+Z` and Enter on
Windows PowerShell.

Use a high-entropy random value for `SESSION_SIGNING_SECRET`; it signs the
browser-only session cookie issued by `/api/session`.

## 4. Deploy to Cloud Run

The preferred production-style path is the manual GitHub Actions workflow
`Deploy to Cloud Run`. It requires the `CI` workflow to have passed for the
selected commit, pushes the Docker image to Artifact Registry, deploys Cloud
Run, and checks `/health` plus the root URL.

Configure these GitHub repository variables:

```text
GCP_PROJECT_ID
GCP_REGION=asia-southeast1
CLOUD_RUN_SERVICE=qfind
QDRANT_CLOUD_URL
OPENAI_MODEL=gpt-4.1-mini-2025-04-14
```

Configure these GitHub repository secrets for Workload Identity Federation:

```text
GCP_WORKLOAD_IDENTITY_PROVIDER
GCP_SERVICE_ACCOUNT
```

The workflow assumes an Artifact Registry Docker repository named the same as
`CLOUD_RUN_SERVICE`, for example `qfind`.

For local/manual deployment, use `min instances = 0` to keep cost near zero for
occasional demo traffic.

```powershell
gcloud run deploy qfind `
  --source . `
  --region asia-southeast1 `
  --allow-unauthenticated `
  --memory 2Gi `
  --cpu 1 `
  --min-instances 0 `
  --max-instances 2 `
  --set-env-vars QDRANT_MODE=server,MODEL_WARMUP_ENABLED=false,RERANKING_ENABLED=false,OPENAI_MODEL=gpt-4.1-mini-2025-04-14,QDRANT_CLOUD_URL=https://YOUR-CLUSTER-url `
  --set-secrets OPENAI_API_KEY=OPENAI_API_KEY:latest,QDRANT_API_KEY=QDRANT_API_KEY:latest,SESSION_SIGNING_SECRET=SESSION_SIGNING_SECRET:latest
```

## 5. Test the deployed API

Open:

```text
https://YOUR-CLOUD-RUN-URL/health
https://YOUR-CLOUD-RUN-URL/docs
```

`/health` should show:

```json
{
  "status": "ok",
  "collection_ready": true,
  "lexical_ready": true,
  "lexical_source": "qdrant_payloads",
  "lexical_record_count": 463
}
```

Then open the Cloud Run root URL in a browser and send a chat message from the
React UI. Direct unauthenticated POSTs to `/search`, `/chat`, and
`/chat/stream` should return `401` unless the browser has first loaded
`/api/session` and sends the signed `HttpOnly` cookie from the same origin.

## Cost guardrails

Keep these settings unless you intentionally want a warm always-on service:

```text
min instances: 0
max instances: 2
CPU: 1
memory: 2Gi
GPU: none
```

Do not set `min instances = 1` for the portfolio demo; that creates an
always-on monthly cost.
