import { streamChat } from "./chat";
import { RateLimiter } from "./rateLimiter";
import {
  hashIp,
  isBenchmarkRequest,
  securityHeaders,
  validateRequestOrigin,
  verifyTurnstile,
} from "./security";
import type { Env } from "./types";
import { parseChatRequest, RequestError } from "./validation";

export { RateLimiter };

function jsonResponse(body: unknown, status = 200, extra?: HeadersInit): Response {
  const headers = securityHeaders();
  headers.set("Content-Type", "application/json; charset=utf-8");
  headers.set("Cache-Control", "no-store");
  if (extra) new Headers(extra).forEach((value, key) => headers.set(key, value));
  return new Response(JSON.stringify(body), { status, headers });
}

async function reserve(
  env: Env,
  ipHash: string,
  reservationId: string,
): Promise<{ allowed: boolean; reason?: string; retryAfter?: number }> {
  const id = env.RATE_LIMITER.idFromName("global");
  const stub = env.RATE_LIMITER.get(id);
  const response = await stub.fetch("https://rate-limiter.internal/reserve", {
    method: "POST",
    body: JSON.stringify({
      action: "reserve",
      ipHash,
      reservationId,
      now: Math.floor(Date.now() / 1000),
    }),
  });
  return response.json();
}

async function release(env: Env, ipHash: string, reservationId: string): Promise<void> {
  const id = env.RATE_LIMITER.idFromName("global");
  await env.RATE_LIMITER.get(id).fetch("https://rate-limiter.internal/release", {
    method: "POST",
    body: JSON.stringify({
      action: "release",
      ipHash,
      reservationId,
      now: Math.floor(Date.now() / 1000),
    }),
  });
}

export default {
  async fetch(request: Request, env: Env, context: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === "/api/health" && request.method === "GET") {
      return jsonResponse({
        status: "ok",
        deployment: "cloudflare",
        model: env.OPENAI_MODEL || "gpt-4.1-mini-2025-04-14",
        retrieval: "static-hybrid-bge-bm25",
      });
    }
    if (url.pathname === "/api/chat/stream") {
      if (request.method !== "POST") {
        return jsonResponse({ detail: "Method not allowed" }, 405, { Allow: "POST" });
      }
      let ipHash = "";
      let reservationId = "";
      try {
        validateRequestOrigin(request, env);
        const ip = request.headers.get("CF-Connecting-IP");
        if (!ip) throw new RequestError(400, "Client IP metadata is unavailable");
        if (!env.IP_HASH_SECRET) throw new RequestError(503, "Rate limiting is not configured");
        const body = await parseChatRequest(request);
        const benchmarkRequest = await isBenchmarkRequest(request, env);
        if (benchmarkRequest) {
          return streamChat(request, env, body);
        }
        await verifyTurnstile(body.turnstile_token, ip, env);
        ipHash = await hashIp(ip, env.IP_HASH_SECRET);
        reservationId = crypto.randomUUID();
        const limit = await reserve(env, ipHash, reservationId);
        if (!limit.allowed) {
          return jsonResponse(
            { detail: `Request limit reached: ${limit.reason ?? "rate_limit"}` },
            429,
            { "Retry-After": String(Math.max(1, limit.retryAfter ?? 60)) },
          );
        }
        const response = streamChat(request, env, body);
        const [clientBody, monitorBody] = response.body!.tee();
        context.waitUntil(
          (async () => {
            try {
              const reader = monitorBody.getReader();
              while (!(await reader.read()).done) {
                // Drain the monitoring branch so the lease covers the full stream.
              }
            } finally {
              await release(env, ipHash, reservationId);
            }
          })(),
        );
        return new Response(clientBody, response);
      } catch (error) {
        if (reservationId && ipHash) context.waitUntil(release(env, ipHash, reservationId));
        if (error instanceof RequestError) {
          return jsonResponse({ detail: error.message }, error.status);
        }
        return jsonResponse(
          { detail: error instanceof Error ? error.message : "Internal error" },
          500,
        );
      }
    }
    const assetResponse = await env.ASSETS.fetch(request);
    const headers = new Headers(assetResponse.headers);
    securityHeaders().forEach((value, key) => headers.set(key, value));
    return new Response(assetResponse.body, {
      status: assetResponse.status,
      statusText: assetResponse.statusText,
      headers,
    });
  },
} satisfies ExportedHandler<Env>;
