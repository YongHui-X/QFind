"""Run ClauseLens answer regressions against a deployed Cloudflare Worker."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.answer_cases import (  # noqa: E402
    DEFAULT_ANSWER_TEST_FILE,
    load_answer_tests,
)
from evaluation.answer_eval import evaluate_deterministically  # noqa: E402


def request_case(
    client: httpx.Client,
    *,
    endpoint: str,
    benchmark_token: str,
    messages: list[dict[str, str]],
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    final: dict[str, Any] | None = None
    with client.stream(
        "POST",
        f"{endpoint.rstrip('/')}/api/chat/stream",
        headers={
            "Origin": endpoint.rstrip("/"),
            "X-ClauseLens-Benchmark": benchmark_token,
        },
        json={
            "messages": messages,
            "clause_type": None,
            "limit": 5,
            "turnstile_token": "benchmark-bypass",
        },
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            event = json.loads(line)
            if event.get("event") == "error":
                raise RuntimeError(str(event.get("detail", "Cloud benchmark failed")))
            if event.get("event") == "final":
                final = dict(event["data"])
    if final is None:
        raise RuntimeError("Cloud Worker did not return a final event")
    return final, (time.perf_counter() - started) * 1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument(
        "--benchmark-token",
        default=os.getenv("CLAUSELENS_BENCHMARK_TOKEN"),
    )
    parser.add_argument("--tests", type=Path, default=DEFAULT_ANSWER_TEST_FILE)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.benchmark_token:
        raise SystemExit("Set --benchmark-token or CLAUSELENS_BENCHMARK_TOKEN")

    rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=60.0) as client:
        for _ in range(args.repeats):
            for case in load_answer_tests(args.tests):
                response, latency_ms = request_case(
                    client,
                    endpoint=args.endpoint,
                    benchmark_token=args.benchmark_token,
                    messages=[
                        {"role": message.role, "content": message.content}
                        for message in case.messages
                    ],
                )
                checks = evaluate_deterministically(case, response)
                rows.append(
                    {
                        "case_id": case.case_id,
                        "latency_ms": round(latency_ms, 3),
                        "passed": checks.passed,
                        "failures": checks.failures,
                        "citation_valid": checks.citation_valid,
                        "response": response,
                    }
                )

    failed = [row for row in rows if not row["passed"]]
    print(f"Requests: {len(rows)}")
    print(f"Passed: {len(rows) - len(failed)}/{len(rows)}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
