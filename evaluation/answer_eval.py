"""Offline generated-answer evaluation with deterministic and optional LLM checks."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chat import ChatMessage, ChatRequest, answer_chat_turn  # noqa: E402
from app.rag import QDRANT_PATH, QDRANT_URL  # noqa: E402
from app.telemetry import citation_indexes  # noqa: E402
from evaluation.answer_cases import (  # noqa: E402
    DEFAULT_ANSWER_TEST_FILE,
    AnswerTestCase,
    load_answer_tests,
)
from evaluation.chat_benchmark import build_engine  # noqa: E402

DEFAULT_OUTPUT_PATH = Path("data/processed/answer_eval_results.json")
DEFAULT_JUDGE_MODEL = "gpt-4.1-mini"
VARIATION_MARKERS = (
    "differ",
    "depends on",
    "varies",
    "some",
    "while",
    "by agreement",
    "not all",
)
INSUFFICIENT_MARKERS = (
    "not enough",
    "does not establish",
    "does not specify",
    "do not provide",
    "not stated",
    "no explicit",
    "without stating",
    "cannot determine",
    "insufficient",
)


class OfflineAnswerClient:
    """Deterministic answer client for CI checks that must not call OpenAI."""

    model = "offline-deterministic"

    def __init__(self, case: AnswerTestCase) -> None:
        self.case = case

    def complete(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        concepts = ", ".join(self.case.required_concepts)
        concept_text = f" about {concepts}" if concepts else ""
        if self.case.answer_mode == "varies":
            return (
                "The retrieved agreements differ. "
                f"The evidence varies by agreement{concept_text} [1]."
            )
        if self.case.answer_mode == "insufficient":
            return (
                "The retrieved evidence does not specify enough information"
                f"{concept_text} [1]."
            )
        return f"The retrieved evidence supports the answer{concept_text} [1]."

    def stream_complete(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ):
        yield self.complete(
            system_prompt=system_prompt,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def response_metadata(self) -> dict[str, str | None]:
        return {"model": self.model}


def normalize_legal_concepts(text: str) -> str:
    """Normalize harmless lexical variants without relaxing semantic checks."""

    normalized = " ".join(text.lower().split())
    replacements = {
        "three percent": "3 percent",
        "3%": "3 percent",
        "parties": "party",
        "losses": "damage",
        "loss": "damage",
        "damages": "damage",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return normalized


class JudgeDecision(BaseModel):
    """Validated structured output from the answer-quality judge."""

    passed: bool
    claim_support: int = Field(ge=1, le=5)
    source_attribution: int = Field(ge=1, le=5)
    uncertainty_handling: int = Field(ge=1, le=5)
    directness: int = Field(ge=1, le=5)
    rationale: str


@dataclass(frozen=True)
class DeterministicChecks:
    route_valid: bool
    abstention_valid: bool
    citation_valid: bool
    required_concepts_valid: bool
    forbidden_claims_valid: bool
    answer_mode_valid: bool
    failures: list[str]

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class AnswerEvalResult:
    case_id: str
    critical: bool
    question: str
    expected_clause_type: str | None
    resolved_clause_type: str | None
    answer_mode: str
    answer: str
    abstained: bool
    results: list[dict[str, object]]
    deterministic: DeterministicChecks
    judge: JudgeDecision | None = None

    @property
    def passed(self) -> bool:
        return self.deterministic.passed and (
            self.judge is None or self.judge.passed
        )


def evaluate_deterministically(
    case: AnswerTestCase,
    response: dict[str, Any],
) -> DeterministicChecks:
    """Apply reproducible route, citation, concept, and overclaim checks."""

    answer = str(response.get("answer", ""))
    normalized = normalize_legal_concepts(answer)
    abstained = bool(response.get("abstained", False))
    results = list(response.get("results", []))
    citations = citation_indexes(answer)
    resolved = response.get("resolved_clause_type")
    failures: list[str] = []

    route_valid = resolved == case.expected_clause_type
    if not route_valid:
        failures.append(
            f"route expected {case.expected_clause_type!r}, got {resolved!r}"
        )

    abstention_valid = (
        abstained and not results
        if case.answer_mode == "abstain"
        else not abstained
    )
    if not abstention_valid:
        failures.append(f"abstention behavior did not match {case.answer_mode}")

    citation_valid = True
    if case.citation_required:
        citation_valid = bool(results) and bool(citations) and all(
            1 <= index <= len(results) for index in citations
        )
    elif case.answer_mode == "abstain":
        citation_valid = not citations and not results
    if not citation_valid:
        failures.append("citations were missing or outside the returned evidence")

    missing_concepts = [
        concept
        for concept in case.required_concepts
        if normalize_legal_concepts(concept) not in normalized
    ]
    required_concepts_valid = not missing_concepts
    if missing_concepts:
        failures.append(f"missing required concepts: {missing_concepts}")

    matched_forbidden = [
        pattern
        for pattern in case.forbidden_patterns
        if re.search(pattern, normalized, flags=re.IGNORECASE)
    ]
    forbidden_claims_valid = not matched_forbidden
    if matched_forbidden:
        failures.append(f"matched forbidden claims: {matched_forbidden}")

    answer_mode_valid = True
    if case.answer_mode == "varies":
        answer_mode_valid = any(marker in normalized for marker in VARIATION_MARKERS)
    elif case.answer_mode == "insufficient":
        answer_mode_valid = any(
            marker in normalized for marker in INSUFFICIENT_MARKERS
        )
    elif case.answer_mode == "abstain":
        answer_mode_valid = abstained
    if not answer_mode_valid:
        failures.append(f"answer did not express expected mode {case.answer_mode}")

    return DeterministicChecks(
        route_valid=route_valid,
        abstention_valid=abstention_valid,
        citation_valid=citation_valid,
        required_concepts_valid=required_concepts_valid,
        forbidden_claims_valid=forbidden_claims_valid,
        answer_mode_valid=answer_mode_valid,
        failures=failures,
    )


def judge_answer(
    *,
    case: AnswerTestCase,
    response: dict[str, Any],
    model: str,
    client: OpenAI | None = None,
) -> JudgeDecision:
    """Judge answer faithfulness against only the returned evidence."""

    effective_client = client or OpenAI()
    payload = {
        "question": case.messages[-1].content,
        "expected_answer_mode": case.answer_mode,
        "answer": response.get("answer"),
        "evidence": [
            {
                "index": index,
                "source": result.get("document_id"),
                "text": result.get("text"),
            }
            for index, result in enumerate(response.get("results", []), start=1)
            if isinstance(result, dict)
        ],
    }
    completion = effective_client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Evaluate a contract-answer strictly against the supplied "
                    "evidence. Pass only if every material claim is supported by "
                    "its cited source, differences and silence are handled "
                    "correctly, no defined relationships are assumed, citations "
                    "identify supporting evidence, and the answer directly "
                    "addresses the question. Return JSON with passed, integer "
                    "scores 1-5 for claim_support, source_attribution, "
                    "uncertainty_handling, directness, and a short rationale."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    content = completion.choices[0].message.content
    if not content:
        raise ValueError("answer judge returned no content")
    return JudgeDecision.model_validate_json(content)


def run_case(
    *,
    case: AnswerTestCase,
    engine: Any,
    judge: bool,
    judge_model: str,
    offline: bool,
) -> AnswerEvalResult:
    """Generate and evaluate one answer-quality case."""

    original_llm = engine.llm
    if offline:
        engine.llm = OfflineAnswerClient(case)
    try:
        response = answer_chat_turn(
            engine=engine,
            request=ChatRequest(
                messages=[
                    ChatMessage(role=message.role, content=message.content)
                    for message in case.messages
                ],
                limit=5,
                rerank_mode="auto",
            ),
        ).model_dump()
    finally:
        engine.llm = original_llm
    deterministic = evaluate_deterministically(case, response)
    judge_decision = (
        judge_answer(case=case, response=response, model=judge_model)
        if judge
        else None
    )
    return AnswerEvalResult(
        case_id=case.case_id,
        critical=case.critical,
        question=case.messages[-1].content,
        expected_clause_type=case.expected_clause_type,
        resolved_clause_type=response.get("resolved_clause_type"),
        answer_mode=case.answer_mode,
        answer=str(response.get("answer", "")),
        abstained=bool(response.get("abstained", False)),
        results=list(response.get("results", [])),
        deterministic=deterministic,
        judge=judge_decision,
    )


def quality_gate(
    results: list[AnswerEvalResult],
    *,
    judge_required: bool,
    minimum_judge_pass_rate: float = 0.90,
) -> tuple[bool, list[str]]:
    """Apply critical, citation, and model-judge acceptance thresholds."""

    failures: list[str] = []
    critical_failures = [
        result.case_id
        for result in results
        if result.critical and not result.deterministic.passed
    ]
    if critical_failures:
        failures.append(f"critical deterministic failures: {critical_failures}")
    invalid_citations = [
        result.case_id
        for result in results
        if not result.deterministic.citation_valid
    ]
    if invalid_citations:
        failures.append(f"citation failures: {invalid_citations}")
    if judge_required:
        judged = [result for result in results if result.judge is not None]
        pass_rate = (
            sum(result.judge.passed for result in judged if result.judge) / len(judged)
            if judged
            else 0.0
        )
        if pass_rate < minimum_judge_pass_rate:
            failures.append(
                f"judge pass rate {pass_rate:.1%} below {minimum_judge_pass_rate:.1%}"
            )
    return not failures, failures


def serialize_result(result: AnswerEvalResult) -> dict[str, Any]:
    """Convert a result into JSON-compatible detail."""

    row = asdict(result)
    row["passed"] = result.passed
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated QFind answers against grounded cases."
    )
    parser.add_argument("--tests", type=Path, default=DEFAULT_ANSWER_TEST_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument(
        "--qdrant-mode",
        choices=["server", "embedded"],
        default="embedded",
    )
    parser.add_argument("--qdrant-url", default=QDRANT_URL)
    parser.add_argument("--qdrant-path", type=Path, default=QDRANT_PATH)
    parser.add_argument(
        "--rerank-mode",
        choices=["off", "auto", "always"],
        default="auto",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use deterministic local answer text instead of hosted generation.",
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("ANSWER_EVAL_MODEL", DEFAULT_JUDGE_MODEL),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_answer_tests(args.tests)
    engine = build_engine(
        rerank_mode=args.rerank_mode,
        qdrant_mode=args.qdrant_mode,
        qdrant_url=args.qdrant_url,
        qdrant_path=args.qdrant_path,
    )
    results = [
        run_case(
            case=case,
            engine=engine,
            judge=args.judge,
            judge_model=args.judge_model,
            offline=args.offline,
        )
        for case in cases
    ]
    passed, gate_failures = quality_gate(results, judge_required=args.judge)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "passed": passed,
                "judge_enabled": args.judge,
                "gate_failures": gate_failures,
                "results": [serialize_result(result) for result in results],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} {result.case_id}: {result.answer}")
        for failure in result.deterministic.failures:
            print(f"  - {failure}")
        if result.judge:
            print(f"  - judge: {result.judge.passed} ({result.judge.rationale})")
    print(f"Wrote answer evaluation to {args.output}")
    if not passed:
        raise SystemExit("; ".join(gate_failures))


if __name__ == "__main__":
    main()
