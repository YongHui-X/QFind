from types import SimpleNamespace

from evaluation.answer_cases import AnswerTestCase, load_answer_tests
from evaluation.answer_eval import (
    AnswerEvalResult,
    DeterministicChecks,
    JudgeDecision,
    evaluate_deterministically,
    judge_answer,
    normalize_legal_concepts,
    quality_gate,
)


def test_normalize_legal_concepts_handles_equivalent_forms() -> None:
    assert normalize_legal_concepts("Three percent losses by the parties") == (
        "3 percent damage by the party"
    )


def make_case(**overrides):
    data = {
        "case_id": "assignment-operation-of-law",
        "messages": [
            {
                "role": "user",
                "content": "Does this cover transfers by operation of law?",
            }
        ],
        "expected_clause_type": "Anti-Assignment",
        "answer_mode": "varies",
        "required_concepts": ["operation of law"],
        "forbidden_patterns": [r"^yes[,\s]"],
        "citation_required": True,
        "critical": True,
    }
    data.update(overrides)
    return AnswerTestCase(**data)


def make_response(answer: str):
    return {
        "answer": answer,
        "resolved_clause_type": "Anti-Assignment",
        "abstained": False,
        "results": [{"text": "Transfer by operation of law requires consent."}],
    }


def test_load_answer_tests_covers_supported_scope() -> None:
    cases = load_answer_tests()

    assert len(cases) >= 10
    assert {
        case.expected_clause_type for case in cases if case.expected_clause_type
    } == {
        "Anti-Assignment",
        "Cap On Liability",
        "License Grant",
        "Audit Rights",
        "Termination For Convenience",
    }
    assert sum(case.critical for case in cases) == 3


def test_deterministic_checks_accept_qualified_supported_answer() -> None:
    checks = evaluate_deterministically(
        make_case(),
        make_response(
            "The retrieved agreements differ on operation of law: one expressly "
            "requires consent, while another is silent. [1]"
        ),
    )

    assert checks.passed is True
    assert checks.citation_valid is True


def test_deterministic_checks_reject_unqualified_overclaim() -> None:
    checks = evaluate_deterministically(
        make_case(),
        make_response(
            "Yes, the assignment restriction applies to transfers by operation "
            "of law. [1]"
        ),
    )

    assert checks.passed is False
    assert checks.forbidden_claims_valid is False
    assert checks.answer_mode_valid is False


def test_deterministic_checks_validate_abstention() -> None:
    case = make_case(
        case_id="unsupported",
        expected_clause_type=None,
        answer_mode="abstain",
        required_concepts=[],
        forbidden_patterns=[],
        citation_required=False,
        critical=False,
    )

    checks = evaluate_deterministically(
        case,
        {
            "answer": "The topic is outside the supported index.",
            "resolved_clause_type": None,
            "abstained": True,
            "results": [],
        },
    )

    assert checks.passed is True


def test_deterministic_checks_accept_explicit_missing_duration() -> None:
    case = make_case(
        case_id="license-duration",
        expected_clause_type="License Grant",
        answer_mode="insufficient",
        required_concepts=["license"],
        forbidden_patterns=[],
        critical=False,
    )

    checks = evaluate_deterministically(
        case,
        {
            "answer": (
                "The retrieved agreements do not provide explicit information "
                "on how long the license remains effective. [1]"
            ),
            "resolved_clause_type": "License Grant",
            "abstained": False,
            "results": [{"text": "A license is granted."}],
        },
    )

    assert checks.passed is True


def test_judge_answer_parses_structured_json() -> None:
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=(
                        '{"passed":true,"claim_support":5,'
                        '"source_attribution":5,"uncertainty_handling":5,'
                        '"directness":4,"rationale":"Grounded."}'
                    )
                )
            )
        ]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: completion)
        )
    )

    decision = judge_answer(
        case=make_case(),
        response=make_response(
            "The retrieved agreements differ on operation of law. [1]"
        ),
        model="judge-model",
        client=client,
    )

    assert decision.passed is True
    assert decision.claim_support == 5


def test_quality_gate_requires_critical_checks_and_ninety_percent_judge() -> None:
    passing_checks = DeterministicChecks(
        route_valid=True,
        abstention_valid=True,
        citation_valid=True,
        required_concepts_valid=True,
        forbidden_claims_valid=True,
        answer_mode_valid=True,
        failures=[],
    )
    results = [
        AnswerEvalResult(
            case_id=f"case-{index}",
            critical=index == 0,
            question="Question",
            expected_clause_type="Audit Rights",
            resolved_clause_type="Audit Rights",
            answer_mode="supported",
            answer="Supported. [1]",
            abstained=False,
            results=[{"text": "Supported."}],
            deterministic=passing_checks,
            judge=JudgeDecision(
                passed=index < 9,
                claim_support=5,
                source_attribution=5,
                uncertainty_handling=5,
                directness=5,
                rationale="Result",
            ),
        )
        for index in range(10)
    ]

    passed, failures = quality_gate(results, judge_required=True)

    assert passed is True
    assert failures == []
