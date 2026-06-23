# ClauseLens Cloudflare deployment

This directory contains the permanent public demo:

- React/Vite frontend served as Worker static assets.
- Cloudflare Worker streaming API.
- Workers AI query embeddings and adaptive reranking.
- Static dense-vector and BM25 retrieval artifacts.
- OpenAI grounded answer generation.
- Turnstile and Durable Object abuse controls.

The existing Python, FastAPI, Streamlit, and Qdrant implementation remains the
research and evaluation reference.

## Local checks

```powershell
cd cloudflare
npm install
npm run check
npm test
npm run build
```

The Worker itself requires Cloudflare bindings, so use Wrangler for integrated
local testing:

```powershell
Copy-Item .dev.vars.example .dev.vars
npm run dev
```

Cloudflare's documented Turnstile test key is present in the example file.
Replace `OPENAI_API_KEY` and `IP_HASH_SECRET` before exercising chat.

## Regenerate the retrieval artifacts

Start the local Qdrant server containing the validated collection, then run:

```powershell
.\.conda-clauselens\python.exe scripts\build_cloudflare_index.py
```

This writes the manifest, evidence records, lexical index, and normalized
vectors under `cloudflare/public/generated/`. Commit all four files together.

## Cloudflare setup

1. Authenticate Wrangler with `npx wrangler login`.
2. Create a Turnstile widget for the deployed hostname.
3. Configure secrets:

   ```powershell
   npx wrangler secret put OPENAI_API_KEY
   npx wrangler secret put TURNSTILE_SECRET_KEY
   npx wrangler secret put IP_HASH_SECRET
   npx wrangler secret put BENCHMARK_TOKEN
   ```

4. Build and deploy:

   ```powershell
   $env:VITE_TURNSTILE_SITE_KEY="your-site-key"
   npm run deploy
   ```

5. Set `ALLOWED_ORIGIN` in `wrangler.toml` to the exact assigned
   `https://*.workers.dev` origin and deploy again.

The public URL does not require a custom domain.

## Enforced limits

- Turnstile on every chat request.
- Three requests per rolling minute per hashed IP.
- Ten AI requests per hashed IP per UTC day.
- 100 total AI requests per UTC day.
- One concurrent request per IP and five globally.
- 16 KB request body, eight messages, 1,000 characters per message, and 160
  answer tokens.
- No raw IP address is persisted.

`BENCHMARK_TOKEN` is for the non-browser preview benchmark only. Requests with
the matching `X-ClauseLens-Benchmark` header bypass public visitor counters so
the 120-request regression workload can run. Keep it separate from all browser
configuration and rotate it if exposed.
