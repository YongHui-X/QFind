from types import SimpleNamespace

from app.rag import stable_point_id
from scripts.index_qdrant import (
    HASH_PAYLOAD_KEY,
    MODEL_PAYLOAD_KEY,
    index_records,
    prepare_record,
)


class FakeEmbedding:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def tolist(self) -> list[float]:
        return self.values


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str], **_: object) -> list[FakeEmbedding]:
        self.calls.append(texts)
        return [FakeEmbedding([float(index), 0.5]) for index, _ in enumerate(texts)]


class FakeClient:
    def __init__(self, payloads: dict[int, dict[str, object]] | None = None) -> None:
        self.payloads = payloads or {}
        self.upserts: list[object] = []
        self.overwrites: list[dict[str, object]] = []

    def retrieve(self, *, ids: list[int], **_: object) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(id=point_id, payload=self.payloads[point_id])
            for point_id in ids
            if point_id in self.payloads
        ]

    def upsert(self, *, points: list[object], **_: object) -> None:
        self.upserts.extend(points)

    def overwrite_payload(self, **kwargs: object) -> None:
        self.overwrites.append(kwargs)


def test_index_records_skips_unchanged_embeddings() -> None:
    record = {"id": "record-1", "text": "Same clause", "source_pdf": "a.pdf"}
    prepared = prepare_record(record, "embedding-model")
    point_id = stable_point_id("record-1")
    client = FakeClient({point_id: prepared})
    model = FakeModel()

    result = index_records(
        client=client,
        model=model,
        records=[record],
        collection_name="clauses",
        model_name="embedding-model",
        batch_size=64,
    )

    assert result == (0, 1, 0)
    assert model.calls == []
    assert client.upserts == []


def test_index_records_reembeds_when_text_changes() -> None:
    record = {"id": "record-1", "text": "Updated clause"}
    point_id = stable_point_id("record-1")
    client = FakeClient(
        {
            point_id: {
                "id": "record-1",
                "text": "Old clause",
                HASH_PAYLOAD_KEY: "old-hash",
                MODEL_PAYLOAD_KEY: "embedding-model",
            }
        }
    )
    model = FakeModel()

    result = index_records(
        client=client,
        model=model,
        records=[record],
        collection_name="clauses",
        model_name="embedding-model",
        batch_size=64,
    )

    assert result == (1, 0, 0)
    assert model.calls == [["Updated clause"]]
    assert len(client.upserts) == 1


def test_index_records_embeds_duplicate_text_once() -> None:
    records = [
        {"id": "record-1", "text": "Duplicate clause"},
        {"id": "record-2", "text": "Duplicate clause"},
    ]
    client = FakeClient()
    model = FakeModel()

    result = index_records(
        client=client,
        model=model,
        records=records,
        collection_name="clauses",
        model_name="embedding-model",
        batch_size=64,
    )

    assert result == (2, 0, 0)
    assert model.calls == [["Duplicate clause"]]
    assert len(client.upserts) == 2
    assert client.upserts[0].vector == client.upserts[1].vector


def test_index_records_updates_metadata_without_embedding() -> None:
    old_record = {"id": "record-1", "text": "Same clause", "source_pdf": "old.pdf"}
    new_record = {"id": "record-1", "text": "Same clause", "source_pdf": "new.pdf"}
    point_id = stable_point_id("record-1")
    client = FakeClient({point_id: prepare_record(old_record, "embedding-model")})
    model = FakeModel()

    result = index_records(
        client=client,
        model=model,
        records=[new_record],
        collection_name="clauses",
        model_name="embedding-model",
        batch_size=64,
    )

    assert result == (0, 0, 1)
    assert model.calls == []
    assert client.overwrites[0]["payload"]["source_pdf"] == "new.pdf"
