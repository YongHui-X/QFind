"""Build deterministic static retrieval artifacts for the Cloudflare deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag import (  # noqa: E402
    COLLECTION,
    EMBEDDING_MODEL,
    QDRANT_URL,
    stable_point_id,
)

DEFAULT_SOURCE = PROJECT_ROOT / "data" / "processed" / "starter_clause_evidence.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "cloudflare" / "public" / "generated"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / magnitude for value in vector]


def load_vectors(
    client: QdrantClient,
    records: list[dict[str, Any]],
    collection: str,
) -> list[list[float]]:
    point_ids = [stable_point_id(str(record["id"])) for record in records]
    by_id: dict[int, list[float]] = {}
    for start in range(0, len(point_ids), 128):
        points = client.retrieve(
            collection_name=collection,
            ids=point_ids[start : start + 128],
            with_payload=False,
            with_vectors=True,
        )
        for point in points:
            raw_vector = point.vector
            if isinstance(raw_vector, dict):
                raw_vector = next(iter(raw_vector.values()))
            if not isinstance(raw_vector, list):
                raise ValueError(f"Point {point.id} does not contain a dense vector")
            by_id[int(point.id)] = normalize([float(value) for value in raw_vector])
    missing = [point_id for point_id in point_ids if point_id not in by_id]
    if missing:
        raise ValueError(f"Qdrant is missing {len(missing)} evidence vectors")
    return [by_id[point_id] for point_id in point_ids]


def lexical_artifact(records: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = [
        TOKEN_PATTERN.findall(str(record.get("text", "")).lower()) for record in records
    ]
    lengths = [len(row) for row in tokens]
    document_frequency: Counter[str] = Counter()
    for row in tokens:
        document_frequency.update(set(row))
    document_count = max(1, len(records))
    idf = {
        term: math.log(
            1 + (document_count - frequency + 0.5) / (frequency + 0.5)
        )
        for term, frequency in sorted(document_frequency.items())
    }
    return {
        "average_length": sum(lengths) / len(lengths) if lengths else 0.0,
        "lengths": lengths,
        "idf": idf,
        "term_frequencies": [
            dict(sorted(Counter(row).items())) for row in tokens
        ],
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )


def build(
    *,
    source: Path,
    output: Path,
    qdrant_url: str,
    collection: str,
) -> None:
    records = load_records(source)
    client = QdrantClient(url=qdrant_url)
    vectors = load_vectors(client, records, collection)
    dimensions = len(vectors[0]) if vectors else 0
    if dimensions != 384:
        raise ValueError(f"Expected 384-dimensional vectors, received {dimensions}")

    output.mkdir(parents=True, exist_ok=True)
    public_records = [
        {
            "id": str(record.get("id", "")),
            "document_id": str(record.get("document_id", "")),
            "source_pdf": str(record.get("source_pdf", "")),
            "source_txt": str(record.get("source_txt", "")),
            "clause_type": str(record.get("clause_type", "")),
            "answer": str(record.get("answer", "")),
            "text": str(record.get("text", "")),
        }
        for record in records
    ]
    write_json(output / "records.json", public_records)
    write_json(output / "lexical.json", lexical_artifact(records))
    with (output / "vectors.f32").open("wb") as file:
        for vector in vectors:
            file.write(struct.pack(f"<{dimensions}f", *vector))
    write_json(
        output / "manifest.json",
        {
            "version": 1,
            "record_count": len(records),
            "dimensions": dimensions,
            "embedding_model": EMBEDDING_MODEL,
            "pooling": "mean",
            "normalized": True,
            "source_sha256": source_hash(source),
            "generated_at": datetime.now(UTC).isoformat(),
        },
    )
    print(
        f"Wrote {len(records)} records and {len(records) * dimensions} "
        f"vector dimensions to {output}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", QDRANT_URL))
    parser.add_argument("--collection", default=COLLECTION)
    return parser.parse_args()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    args = parse_args()
    build(
        source=args.source,
        output=args.output,
        qdrant_url=args.qdrant_url,
        collection=args.collection,
    )


if __name__ == "__main__":
    main()
