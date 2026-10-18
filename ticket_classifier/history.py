"""Optional prediction history stored in SQLite (standard library only)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ticket_classifier.config import HISTORY_DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    text TEXT NOT NULL,
    predicted_label TEXT NOT NULL,
    confidence REAL NOT NULL,
    needs_review INTEGER NOT NULL
)
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute(_SCHEMA)
    return connection


def save_prediction(text: str, result: dict[str, Any], db_path: Path = HISTORY_DB_PATH) -> int:
    connection = _connect(db_path)
    try:
        with connection:
            cursor = connection.execute(
                """
                INSERT INTO predictions (created_at, text, predicted_label, confidence, needs_review)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    text,
                    result["label"],
                    result["confidence"],
                    int(result["needs_review"]),
                ),
            )
        return int(cursor.lastrowid)
    finally:
        connection.close()


def recent_predictions(limit: int = 20, db_path: Path = HISTORY_DB_PATH) -> list[dict[str, Any]]:
    connection = _connect(db_path)
    try:
        rows = connection.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    finally:
        connection.close()
    return [
        {
            "id": row["id"],
            "created_at": row["created_at"],
            "text": row["text"],
            "predicted_label": row["predicted_label"],
            "confidence": row["confidence"],
            "needs_review": bool(row["needs_review"]),
        }
        for row in rows
    ]
