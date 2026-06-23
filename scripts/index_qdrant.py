"""Index prepared ClauseLens evidence records into Qdrant."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client.models import PointStruct  # noqa: E402

from app.rag import (  # noqa: E402
    COLLECTION,
    EMBEDDING_MODEL,
    QDRANT_URL,
    create_qdrant_client,
    embedding_content_hash,
    ensure_collection,
    load_embedding_model,
    load_jsonl_records,
    stable_point_id,
)

HASH_PAYLOAD_KEY = "_embedding_content_hash"
MODEL_PAYLOAD_KEY = "_embedding_model"


def batched(values: list[int], batch_size: int) -> Iterable[list[int]]:
    """Yield fixed-size batches without adding another dependency."""

    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def load_existing_points(
    client: Any,
    *,
    collection_name: str,
    point_ids: list[int],
    batch_size: int = 256,
) -> dict[int, dict[str, object]]:
    """Load hash metadata for points that may already be indexed."""

    existing: dict[int, dict[str, object]] = {}
    for point_id_batch in batched(point_ids, batch_size):
        points = client.retrieve(
            collection_name=collection_name,
            ids=point_id_batch,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            existing[int(point.id)] = dict(point.payload or {})
    return existing


def prepare_record(record: dict[str, object], model_name: str) -> dict[str, object]:
    """Attach the fingerprint needed for incremental indexing."""

    prepared = dict(record)
    prepared[HASH_PAYLOAD_KEY] = embedding_content_hash(str(record["text"]))
    prepared[MODEL_PAYLOAD_KEY] = model_name
    return prepared


def index_records(
    *,
    client: Any,
    model: Any,
    records: list[dict[str, object]],
    collection_name: str,
    model_name: str,
    batch_size: int,
) -> tuple[int, int, int]:
    """Embed only new or changed records and update unchanged metadata."""

    prepared_records = [prepare_record(record, model_name) for record in records]
    point_ids = [stable_point_id(str(record["id"])) for record in prepared_records]
    existing_points = load_existing_points(
        client,
        collection_name=collection_name,
        point_ids=point_ids,
    )

    records_to_embed: list[tuple[int, dict[str, object]]] = []
    metadata_updates = 0
    unchanged = 0

    for point_id, record in zip(point_ids, prepared_records, strict=True):
        existing_payload = existing_points.get(point_id)
        same_embedding = (
            existing_payload is not None
            and existing_payload.get(HASH_PAYLOAD_KEY) == record[HASH_PAYLOAD_KEY]
            and existing_payload.get(MODEL_PAYLOAD_KEY) == model_name
        )
        if not same_embedding:
            records_to_embed.append((point_id, record))
            continue

        if existing_payload != record:
            client.overwrite_payload(
                collection_name=collection_name,
                payload=record,
                points=[point_id],
            )
            metadata_updates += 1
        else:
            unchanged += 1

    # Identical text is embedded once even if it appears under multiple IDs.
    unique_text_by_hash: dict[str, str] = {}
    for _, record in records_to_embed:
        unique_text_by_hash.setdefault(
            str(record[HASH_PAYLOAD_KEY]),
            str(record["text"]),
        )

    vector_by_hash: dict[str, list[float]] = {}
    if unique_text_by_hash:
        hashes = list(unique_text_by_hash)
        embeddings = model.encode(
            [unique_text_by_hash[content_hash] for content_hash in hashes],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        vector_by_hash = {
            content_hash: embedding.tolist()
            for content_hash, embedding in zip(hashes, embeddings, strict=True)
        }

    points = [
        PointStruct(
            id=point_id,
            vector=vector_by_hash[str(record[HASH_PAYLOAD_KEY])],
            payload=record,
        )
        for point_id, record in records_to_embed
    ]
    if points:
        client.upsert(collection_name=collection_name, points=points)

    return len(points), unchanged, metadata_updates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index prepared ClauseLens evidence records into Qdrant."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/starter_clause_evidence.jsonl"),
    )
    parser.add_argument(
        "--url",
        default=QDRANT_URL,
        help="Qdrant server URL. Ignored when --qdrant-path is provided.",
    )
    parser.add_argument(
        "--qdrant-path",
        type=Path,
        help="Use embedded local Qdrant storage instead of a running Qdrant server.",
    )
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--model", default=EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the target collection before indexing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl_records(args.input)
    if not records:
        raise ValueError(f"No records found in {args.input}")

    model = load_embedding_model(args.model)
    client = create_qdrant_client(path=args.qdrant_path) if args.qdrant_path else (
        create_qdrant_client(url=args.url)
    )

    if args.recreate and client.collection_exists(collection_name=args.collection):
        client.delete_collection(collection_name=args.collection)

    ensure_collection(client, collection_name=args.collection)

    indexed, unchanged, metadata_updates = index_records(
        client=client,
        model=model,
        records=records,
        collection_name=args.collection,
        model_name=args.model,
        batch_size=args.batch_size,
    )
    count = client.count(collection_name=args.collection, exact=True).count
    print(f"Embedded and indexed: {indexed}")
    print(f"Skipped unchanged: {unchanged}")
    print(f"Metadata-only updates: {metadata_updates}")
    print(f"Collection count: {count}")


if __name__ == "__main__":
    main()
