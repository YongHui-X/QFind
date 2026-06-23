"""Load ClauseLens retrieval evaluation cases."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_TEST_FILE = Path(__file__).parent / "tests.jsonl"


class RetrievalTestCase(BaseModel):
    """A retrieval test case for ClauseLens."""

    question: str = Field(description="Natural-language contract review query")
    expected_clause_type: str = Field(description="Clause type expected in top-k results")
    keywords: list[str] = Field(
        description="Keywords or phrases expected in the retrieved evidence"
    )
    expected_record_ids: list[str] = Field(default_factory=list)
    category: str = Field(description="Evaluation category")


def load_tests(path: str | Path = DEFAULT_TEST_FILE) -> list[RetrievalTestCase]:
    """Load retrieval test cases from a JSONL file."""

    test_path = Path(path)
    tests: list[RetrievalTestCase] = []

    with test_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            tests.append(RetrievalTestCase(**data))

    return tests
