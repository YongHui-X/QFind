interface ReservationRequest {
  action: "reserve" | "release";
  ipHash: string;
  reservationId: string;
  now: number;
}

interface LimitResponse {
  allowed: boolean;
  reason?: string;
  retryAfter?: number;
}

const MINUTE_LIMIT = 3;
const IP_DAILY_LIMIT = 10;
const GLOBAL_DAILY_LIMIT = 100;
const GLOBAL_CONCURRENCY_LIMIT = 5;
const LEASE_SECONDS = 60;

export class RateLimiter implements DurableObject {
  private sql: SqlStorage;

  constructor(state: DurableObjectState) {
    this.sql = state.storage.sql;
    this.sql.exec(`
      CREATE TABLE IF NOT EXISTS requests (
        ip_hash TEXT NOT NULL,
        created_at INTEGER NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_requests_ip_time
        ON requests(ip_hash, created_at);
      CREATE INDEX IF NOT EXISTS idx_requests_time
        ON requests(created_at);
      CREATE TABLE IF NOT EXISTS leases (
        reservation_id TEXT PRIMARY KEY,
        ip_hash TEXT NOT NULL,
        expires_at INTEGER NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_leases_ip
        ON leases(ip_hash);
      CREATE INDEX IF NOT EXISTS idx_leases_expiry
        ON leases(expires_at);
    `);
  }

  async fetch(request: Request): Promise<Response> {
    const body = (await request.json()) as ReservationRequest;
    if (body.action === "release") {
      this.sql.exec(
        "DELETE FROM leases WHERE reservation_id = ? AND ip_hash = ?",
        body.reservationId,
        body.ipHash,
      );
      return Response.json({ allowed: true });
    }
    return Response.json(this.reserve(body));
  }

  private reserve(body: ReservationRequest): LimitResponse {
    const now = Math.floor(body.now);
    const minuteAgo = now - 60;
    const dayStart = Math.floor(now / 86_400) * 86_400;
    const retentionCutoff = dayStart - 86_400;
    this.sql.exec("DELETE FROM leases WHERE expires_at <= ?", now);
    this.sql.exec("DELETE FROM requests WHERE created_at < ?", retentionCutoff);

    const minuteCount = this.count(
      "SELECT COUNT(*) AS count FROM requests WHERE ip_hash = ? AND created_at > ?",
      body.ipHash,
      minuteAgo,
    );
    if (minuteCount >= MINUTE_LIMIT) {
      return { allowed: false, reason: "minute_limit", retryAfter: 60 };
    }
    const ipDailyCount = this.count(
      "SELECT COUNT(*) AS count FROM requests WHERE ip_hash = ? AND created_at >= ?",
      body.ipHash,
      dayStart,
    );
    if (ipDailyCount >= IP_DAILY_LIMIT) {
      return {
        allowed: false,
        reason: "ip_daily_limit",
        retryAfter: dayStart + 86_400 - now,
      };
    }
    const globalDailyCount = this.count(
      "SELECT COUNT(*) AS count FROM requests WHERE created_at >= ?",
      dayStart,
    );
    if (globalDailyCount >= GLOBAL_DAILY_LIMIT) {
      return {
        allowed: false,
        reason: "global_daily_limit",
        retryAfter: dayStart + 86_400 - now,
      };
    }
    const ipLeases = this.count(
      "SELECT COUNT(*) AS count FROM leases WHERE ip_hash = ?",
      body.ipHash,
    );
    if (ipLeases >= 1) {
      return { allowed: false, reason: "ip_concurrency", retryAfter: LEASE_SECONDS };
    }
    const globalLeases = this.count("SELECT COUNT(*) AS count FROM leases");
    if (globalLeases >= GLOBAL_CONCURRENCY_LIMIT) {
      return { allowed: false, reason: "global_concurrency", retryAfter: 15 };
    }
    this.sql.exec(
      "INSERT INTO leases (reservation_id, ip_hash, expires_at) VALUES (?, ?, ?)",
      body.reservationId,
      body.ipHash,
      now + LEASE_SECONDS,
    );
    this.sql.exec(
      "INSERT INTO requests (ip_hash, created_at) VALUES (?, ?)",
      body.ipHash,
      now,
    );
    return { allowed: true };
  }

  private count(query: string, ...bindings: Array<string | number>): number {
    const row = [...this.sql.exec<{ count: number }>(query, ...bindings)][0];
    return Number(row?.count ?? 0);
  }
}
