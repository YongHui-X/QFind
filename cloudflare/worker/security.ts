import type { Env } from "./types";
import { RequestError } from "./validation";

export function securityHeaders(): Headers {
  return new Headers({
    "Content-Security-Policy":
      "default-src 'self'; script-src 'self' https://challenges.cloudflare.com; " +
      "frame-src https://challenges.cloudflare.com; connect-src 'self'; img-src 'self' data:; " +
      "style-src 'self' 'unsafe-inline'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
  });
}

export function validateRequestOrigin(request: Request, env: Env): void {
  const url = new URL(request.url);
  if (url.protocol !== "https:" && url.hostname !== "localhost" && url.hostname !== "127.0.0.1") {
    throw new RequestError(400, "HTTPS is required");
  }
  const expected = env.ALLOWED_ORIGIN?.trim() || url.origin;
  const origin = request.headers.get("origin");
  if (!origin || origin !== expected) {
    throw new RequestError(403, "Cross-origin requests are not allowed");
  }
}

export async function hashIp(ip: string, secret: string): Promise<string> {
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const signature = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(ip));
  return [...new Uint8Array(signature)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

export async function verifyTurnstile(
  token: string,
  ip: string,
  env: Env,
): Promise<void> {
  if (!env.TURNSTILE_SECRET_KEY) {
    throw new RequestError(503, "Turnstile is not configured");
  }
  const form = new FormData();
  form.set("secret", env.TURNSTILE_SECRET_KEY);
  form.set("response", token);
  form.set("remoteip", ip);
  const response = await fetch(
    "https://challenges.cloudflare.com/turnstile/v0/siteverify",
    { method: "POST", body: form },
  );
  const result = (await response.json()) as { success?: boolean; action?: string };
  if (!result.success || (result.action && result.action !== "chat")) {
    throw new RequestError(403, "Turnstile verification failed");
  }
}

export async function isBenchmarkRequest(request: Request, env: Env): Promise<boolean> {
  const supplied = request.headers.get("X-ClauseLens-Benchmark");
  if (!supplied || !env.BENCHMARK_TOKEN) return false;
  const [suppliedHash, expectedHash] = await Promise.all([
    crypto.subtle.digest("SHA-256", new TextEncoder().encode(supplied)),
    crypto.subtle.digest("SHA-256", new TextEncoder().encode(env.BENCHMARK_TOKEN)),
  ]);
  const left = new Uint8Array(suppliedHash);
  const right = new Uint8Array(expectedHash);
  let difference = 0;
  for (let index = 0; index < left.length; index += 1) {
    difference |= left[index] ^ right[index];
  }
  return difference === 0;
}
