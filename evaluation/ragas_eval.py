"""Manual Ragas evaluation for QFind generated answers."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chat import ChatMessage, ChatRequest, answer_chat_turn  # noqa: E402
from app.rag import QDRANT_PATH, QDRANT_URL  # noqa: E402
from evaluation.chat_benchmark import build_engine  # noqa: E402
from evaluation.ragas_cases import (  # noqa: E402
    DEFAULT_RAGAS_CASE_FILE,
    RagasCase,
    load_ragas_cases,
)

DEFAULT_OUTPUT_PATH = Path("data/processed/ragas_eval_results.json")
DEFAULT_JUDGE_MODEL = "gpt-4.1-mini-2025-04-14"
METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)
DEFAULT_THRESHOLDS = {
    "faithfulness": 0.90,
    "answer_relevancy": 0.80,
    "context_precision": 0.80,
    "context_recall": 0.80,
}
DEFAULT_CRITICAL_FAITHFULNESS_MIN = 0.75


@dataclass(frozen=True)
class RagasCollectedCase:
    case_id: str
    critical: bool
    user_input: str
    response: str
    reference: str
    retrieved_contexts: list[str]
    expected_clause_type: str | None
    resolved_clause_type: str | None
    abstained: bool


def collect_case(*, case: RagasCase, engine: Any, rerank_mode: str) -> RagasCollectedCase:
    """Run one QFind case and collect Ragas-compatible inputs."""

    response = answer_chat_turn(
        engine=engine,
        request=ChatRequest(
            messages=[
                ChatMessage(role=message.role, content=message.content)
                for message in case.messages
            ],
            limit=5,
            rerank_mode=rerank_mode,
        ),
    ).model_dump()
    return RagasCollectedCase(
        case_id=case.case_id,
        critical=case.critical,
        user_input=case.messages[-1].content,
        response=str(response.get("answer", "")),
        reference=case.reference,
        retrieved_contexts=[
            str(result.get("text", ""))
            for result in response.get("results", [])
            if isinstance(result, dict) and result.get("text")
        ],
        expected_clause_type=case.expected_clause_type,
        resolved_clause_type=response.get("resolved_clause_type"),
        abstained=bool(response.get("abstained", False)),
    )


def ragas_dataset_rows(rows: list[RagasCollectedCase]) -> list[dict[str, Any]]:
    """Convert collected cases to the field names expected by Ragas."""

    return [
        {
            "user_input": row.user_input,
            "response": row.response,
            "reference": row.reference,
            "retrieved_contexts": row.retrieved_contexts,
        }
        for row in rows
    ]


def _metric_value(score: dict[str, Any], name: str) -> float | None:
    aliases = {
        "answer_relevancy": ("answer_relevancy", "response_relevancy"),
        "context_precision": (
            "context_precision",
            "llm_context_precision_with_reference",
        ),
        "context_recall": ("context_recall", "llm_context_recall"),
    }
    for key in aliases.get(name, (name,)):
        value = score.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def summarize_results(
    *,
    rows: list[RagasCollectedCase],
    scores: list[dict[str, Any]],
    judge_model: str,
    thresholds: dict[str, float] | None = None,
    critical_faithfulness_min: float = DEFAULT_CRITICAL_FAITHFULNESS_MIN,
) -> dict[str, Any]:
    """Build the saved report and apply release-quality gates."""

    effective_thresholds = thresholds or DEFAULT_THRESHOLDS
    per_case: list[dict[str, Any]] = []
    for row, score in zip(rows, scores, strict=True):
        metrics = {
            name: _metric_value(score, name)
            for name in METRIC_NAMES
            if _metric_value(score, name) is not None
        }
        per_case.append(
            {
                **asdict(row),
                "scores": metrics,
            }
        )

    aggregates = {}
    for name in METRIC_NAMES:
        values = [
            case["scores"][name]
            for case in per_case
            if case["scores"].get(name) is not None
        ]
        aggregates[name] = round(sum(values) / len(values), 4) if values else None

    failures: list[str] = []
    for name, threshold in effective_thresholds.items():
        value = aggregates.get(name)
        if value is None:
            failures.append(f"{name} was not scored")
        elif value < threshold:
            failures.append(f"{name} mean {value:.3f} below {threshold:.3f}")

    critical_failures = [
        case["case_id"]
        for case in per_case
        if case["critical"]
        and (case["scores"].get("faithfulness") is None
             or case["scores"]["faithfulness"] < critical_faithfulness_min)
    ]
    if critical_failures:
        failures.append(
            "critical faithfulness below "
            f"{critical_faithfulness_min:.3f}: {critical_failures}"
        )

    return {
        "passed": not failures,
        "timestamp": datetime.now(UTC).isoformat(),
        "judge_model": judge_model,
        "thresholds": {
            **effective_thresholds,
            "critical_faithfulness_min": critical_faithfulness_min,
        },
        "aggregate_means": aggregates,
        "failures": failures,
        "cases": per_case,
    }


def install_ragas_vertexai_import_shim() -> None:
    """Provide Ragas' unused legacy VertexAI import path when LangChain lacks it."""

    module_name = "langchain_community.chat_models.vertexai"
    try:
        if importlib.util.find_spec(module_name) is not None:
            return
    except (ImportError, ModuleNotFoundError, ValueError):
        pass

    module = types.ModuleType(module_name)

    class ChatVertexAI:  # pragma: no cover - only used if Ragas selects VertexAI.
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "ChatVertexAI is not installed. QFind Ragas CI uses OpenAI; "
                "install langchain-google-vertexai only for VertexAI judges."
            )

    module.ChatVertexAI = ChatVertexAI
    sys.modules.setdefault(module_name, module)


def run_ragas_scores(
    dataset_rows: list[dict[str, Any]],
    *,
    judge_model: str,
) -> list[dict[str, Any]]:
    """Score collected rows with Ragas.

    Ragas is imported lazily so deterministic tests and PR CI do not need
    hosted judge configuration.
    """

    try:
        install_ragas_vertexai_import_shim()
        from ragas import EvaluationDataset, evaluate
        from ragas.llms import llm_factory
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
        )
        try:
            from ragas.metrics import ResponseRelevancy
        except ImportError:
            from ragas.metrics import AnswerRelevancy as ResponseRelevancy
    except ImportError as exc:
        raise RuntimeError(
            "Ragas evaluation requires the optional ragas dependency. "
            "Install requirements.txt and configure OPENAI_API_KEY."
        ) from exc

    dataset = EvaluationDataset.from_list(dataset_rows)
    llm = llm_factory(judge_model)
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=llm,
        raise_exceptions=True,
    )
    if hasattr(result, "to_pandas"):
        return result.to_pandas().to_dict(orient="records")
    if hasattr(result, "scores"):
        return list(result.scores)
    raise TypeError("Unsupported Ragas result object; cannot extract per-case scores")


def run_evaluation(
    *,
    cases: list[RagasCase],
    engine: Any,
    judge_model: str,
    rerank_mode: str,
    scorer: Callable[[list[dict[str, Any]], str], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Collect QFind answers, score them, and summarize gates."""

    collected = [
        collect_case(case=case, engine=engine, rerank_mode=rerank_mode)
        for case in cases
    ]
    dataset_rows = ragas_dataset_rows(collected)
    score_fn = scorer or (
        lambda rows, model: run_ragas_scores(rows, judge_model=model)
    )
    scores = score_fn(dataset_rows, judge_model)
    return summarize_results(
        rows=collected,
        scores=scores,
        judge_model=judge_model,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run judge-based Ragas evaluation for QFind."
    )
    parser.add_argument("--tests", type=Path, default=DEFAULT_RAGAS_CASE_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
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
        "--judge-model",
        default=os.getenv("RAGAS_JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
    )
    parser.add_argument(
        "--enforce-gates",
        action="store_true",
        help="Exit unsuccessfully when Ragas release-quality gates fail.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_ragas_cases(args.tests)
    engine = build_engine(
        rerank_mode=args.rerank_mode,
        qdrant_mode=args.qdrant_mode,
        qdrant_url=args.qdrant_url,
        qdrant_path=args.qdrant_path,
    )
    report = run_evaluation(
        cases=cases,
        engine=engine,
        judge_model=args.judge_model,
        rerank_mode=args.rerank_mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote Ragas evaluation to {args.output}")
    print(json.dumps(report["aggregate_means"], indent=2))
    if args.enforce_gates and not report["passed"]:
        raise SystemExit("; ".join(report["failures"]))


if __name__ == "__main__":
    main()
