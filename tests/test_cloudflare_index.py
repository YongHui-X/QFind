from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATED = PROJECT_ROOT / "cloudflare" / "public" / "generated"
SOURCE = PROJECT_ROOT / "data" / "processed" / "starter_clause_evidence.jsonl"


def test_cloudflare_index_matches_source_and_dimensions() -> None:
    manifest = json.loads((GENERATED / "manifest.json").read_text(encoding="utf-8"))
    records = json.loads((GENERATED / "records.json").read_text(encoding="utf-8"))
    vector_bytes = (GENERATED / "vectors.f32").read_bytes()

    assert manifest["source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert manifest["record_count"] == len(records) == 463
    assert manifest["dimensions"] == 384
    assert len(vector_bytes) == len(records) * manifest["dimensions"] * 4
    first = struct.unpack("<384f", vector_bytes[: 384 * 4])
    magnitude = sum(value * value for value in first) ** 0.5
    assert magnitude == pytest.approx(1.0, abs=1e-5)


def test_cloudflare_lexical_artifact_aligns_with_records() -> None:
    records = json.loads((GENERATED / "records.json").read_text(encoding="utf-8"))
    lexical = json.loads((GENERATED / "lexical.json").read_text(encoding="utf-8"))

    assert len(lexical["lengths"]) == len(records)
    assert len(lexical["term_frequencies"]) == len(records)
    assert lexical["average_length"] > 0
    assert lexical["idf"]["assignment"] > 0
