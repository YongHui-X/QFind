"""Load generated-answer quality cases for ClauseLens."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

DEFAULT_ANSWER_TEST_FILE = Path(__file__).parent / "answer_tests.jsonl"
AnswerMode = Literal["supported", "varies", "insufficient", "abstain"]


class AnswerMessage(BaseModel):
    """One conversation message supplied to the answer benchmark."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class AnswerTestCase(BaseModel):
    """Expected routing and answer behavior for one generated response."""

    case_id: str = Field(min_length=1)
    messages: list[AnswerMessage] = Field(min_length=1)
    expected_clause_type: str | None
    answer_mode: AnswerMode
    required_concepts: list[str] = Field(default_factory=list)
    forbidden_patterns: list[str] = Field(default_factory=list)
    citation_required: bool = True
    critical: bool = False


def load_answer_tests(
    path: str | Path = DEFAULT_ANSWER_TEST_FILE,
) -> list[AnswerTestCase]:
    """Load answer-quality cases from JSONL."""

    test_path = Path(path)
    cases: list[AnswerTestCase] = []
    with test_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc
            cases.append(AnswerTestCase(**data))
    return cases
