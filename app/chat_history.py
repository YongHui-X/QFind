"""Local SQLite persistence for Streamlit chat conversations."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

CHAT_HISTORY_PATH = Path("data/processed/chat_history.db")
DEFAULT_CHAT_TITLE = "New chat"
MAX_TITLE_LENGTH = 52


@dataclass(frozen=True)
class ChatSummary:
    """Compact chat metadata used by the sidebar."""

    chat_id: str
    title: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class SavedChat:
    """A complete persisted conversation and its retrieval controls."""

    chat_id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[dict[str, Any]]
    clause_type: str | None
    limit: int
    rerank_mode: str


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def initialize_chat_history(path: Path = CHAT_HISTORY_PATH) -> None:
    """Create the local chat-history schema if it does not already exist."""

    with _connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chats (
                chat_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                messages_json TEXT NOT NULL,
                clause_type TEXT,
                result_limit INTEGER NOT NULL,
                rerank_mode TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chats_updated_at
            ON chats(updated_at DESC)
            """
        )


def build_chat_title(
    messages: list[dict[str, Any]],
    *,
    max_length: int = MAX_TITLE_LENGTH,
) -> str:
    """Build a deterministic sidebar title from the first user question."""

    first_question = next(
        (
            " ".join(str(message.get("content", "")).split())
            for message in messages
            if message.get("role") == "user" and str(message.get("content", "")).strip()
        ),
        "",
    )
    if not first_question:
        return DEFAULT_CHAT_TITLE
    if len(first_question) <= max_length:
        return first_question
    return first_question[: max(1, max_length - 1)].rstrip() + "…"


def save_chat(
    *,
    messages: list[dict[str, Any]],
    clause_type: str | None,
    limit: int,
    rerank_mode: str,
    chat_id: str | None = None,
    path: Path = CHAT_HISTORY_PATH,
) -> str:
    """Create or atomically update a completed conversation."""

    initialize_chat_history(path)
    effective_chat_id = chat_id or uuid4().hex
    timestamp = datetime.now(UTC).isoformat()
    title = build_chat_title(messages)
    messages_json = json.dumps(messages, ensure_ascii=False)

    with _connect(path) as connection:
        existing = connection.execute(
            "SELECT created_at FROM chats WHERE chat_id = ?",
            (effective_chat_id,),
        ).fetchone()
        created_at = str(existing["created_at"]) if existing else timestamp
        connection.execute(
            """
            INSERT INTO chats (
                chat_id,
                title,
                created_at,
                updated_at,
                messages_json,
                clause_type,
                result_limit,
                rerank_mode
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
                title = excluded.title,
                updated_at = excluded.updated_at,
                messages_json = excluded.messages_json,
                clause_type = excluded.clause_type,
                result_limit = excluded.result_limit,
                rerank_mode = excluded.rerank_mode
            """,
            (
                effective_chat_id,
                title,
                created_at,
                timestamp,
                messages_json,
                clause_type,
                limit,
                rerank_mode,
            ),
        )
    return effective_chat_id


def list_chats(path: Path = CHAT_HISTORY_PATH) -> list[ChatSummary]:
    """Return saved chats ordered by most recent activity."""

    initialize_chat_history(path)
    with _connect(path) as connection:
        rows = connection.execute(
            """
            SELECT chat_id, title, created_at, updated_at
            FROM chats
            ORDER BY updated_at DESC, created_at DESC
            """
        ).fetchall()
    return [
        ChatSummary(
            chat_id=str(row["chat_id"]),
            title=str(row["title"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )
        for row in rows
    ]


def load_chat(
    chat_id: str,
    *,
    path: Path = CHAT_HISTORY_PATH,
) -> SavedChat | None:
    """Load one saved conversation by ID."""

    initialize_chat_history(path)
    with _connect(path) as connection:
        row = connection.execute(
            "SELECT * FROM chats WHERE chat_id = ?",
            (chat_id,),
        ).fetchone()
    if row is None:
        return None
    messages = json.loads(str(row["messages_json"]))
    if not isinstance(messages, list):
        raise ValueError("saved chat messages must be a list")
    return SavedChat(
        chat_id=str(row["chat_id"]),
        title=str(row["title"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        messages=messages,
        clause_type=(
            str(row["clause_type"]) if row["clause_type"] is not None else None
        ),
        limit=int(row["result_limit"]),
        rerank_mode=str(row["rerank_mode"]),
    )


def delete_chat(chat_id: str, *, path: Path = CHAT_HISTORY_PATH) -> bool:
    """Delete one saved conversation and report whether it existed."""

    initialize_chat_history(path)
    with _connect(path) as connection:
        cursor = connection.execute(
            "DELETE FROM chats WHERE chat_id = ?",
            (chat_id,),
        )
    return cursor.rowcount > 0
