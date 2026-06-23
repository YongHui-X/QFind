from pathlib import Path

from app.chat_history import (
    build_chat_title,
    delete_chat,
    initialize_chat_history,
    list_chats,
    load_chat,
    save_chat,
)


def sample_messages(question: str = "What audit rights exist?"):
    return [
        {"role": "assistant", "content": "Ask a contract question."},
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": "The customer may audit records. [1]",
            "response": {
                "turn_id": "turn-1",
                "results": [{"document_id": "doc-1"}],
                "timings": {"total_latency_ms": 1000.0},
            },
        },
    ]


def test_initialize_chat_history_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "history.db"

    initialize_chat_history(path)
    initialize_chat_history(path)

    assert list_chats(path) == []


def test_save_and_load_chat_preserves_messages_and_settings(tmp_path: Path) -> None:
    path = tmp_path / "history.db"
    messages = sample_messages()

    chat_id = save_chat(
        messages=messages,
        clause_type="Audit Rights",
        limit=7,
        rerank_mode="always",
        path=path,
    )
    saved = load_chat(chat_id, path=path)

    assert saved is not None
    assert saved.messages == messages
    assert saved.clause_type == "Audit Rights"
    assert saved.limit == 7
    assert saved.rerank_mode == "always"
    assert saved.title == "What audit rights exist?"


def test_updating_chat_moves_it_to_top_and_keeps_created_at(tmp_path: Path) -> None:
    path = tmp_path / "history.db"
    older_id = save_chat(
        messages=sample_messages("First question"),
        clause_type=None,
        limit=5,
        rerank_mode="auto",
        path=path,
    )
    newer_id = save_chat(
        messages=sample_messages("Second question"),
        clause_type=None,
        limit=5,
        rerank_mode="auto",
        path=path,
    )
    original = load_chat(older_id, path=path)

    save_chat(
        chat_id=older_id,
        messages=[*sample_messages("First question"), {"role": "user", "content": "More?"}],
        clause_type="Audit Rights",
        limit=3,
        rerank_mode="off",
        path=path,
    )
    updated = load_chat(older_id, path=path)

    assert original is not None and updated is not None
    assert updated.created_at == original.created_at
    assert updated.updated_at >= original.updated_at
    assert [chat.chat_id for chat in list_chats(path)] == [older_id, newer_id]


def test_delete_chat_does_not_affect_other_chats(tmp_path: Path) -> None:
    path = tmp_path / "history.db"
    first_id = save_chat(
        messages=sample_messages("First"),
        clause_type=None,
        limit=5,
        rerank_mode="auto",
        path=path,
    )
    second_id = save_chat(
        messages=sample_messages("Second"),
        clause_type=None,
        limit=5,
        rerank_mode="auto",
        path=path,
    )

    assert delete_chat(first_id, path=path) is True
    assert delete_chat(first_id, path=path) is False
    assert load_chat(first_id, path=path) is None
    assert load_chat(second_id, path=path) is not None


def test_build_chat_title_normalizes_and_truncates() -> None:
    messages = [
        {
            "role": "user",
            "content": "  Does   this agreement permit sublicensing under every circumstance?  ",
        }
    ]

    assert build_chat_title(messages, max_length=24) == "Does this agreement per…"
    assert build_chat_title([], max_length=24) == "New chat"
