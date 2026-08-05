"""Camada de acesso a dados (SQLite, sem ORM para manter tudo simples)."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from . import config

_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    input_path TEXT NOT NULL UNIQUE,
    output_path TEXT,
    thumb_input_path TEXT,
    thumb_output_path TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    num_plates INTEGER NOT NULL DEFAULT 0,
    detections TEXT,
    error_message TEXT,
    width INTEGER,
    height INTEGER,
    source TEXT NOT NULL DEFAULT 'upload',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    processed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_photos_status ON photos(status);
"""


def get_conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(config.DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
    return conn


def init_db() -> None:
    conn = get_conn()
    conn.executescript(SCHEMA)
    conn.commit()


@contextmanager
def tx() -> Iterator[sqlite3.Connection]:
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    if d.get("detections"):
        try:
            d["detections"] = json.loads(d["detections"])
        except (TypeError, ValueError):
            d["detections"] = []
    else:
        d["detections"] = []
    return d


def insert_photo(filename: str, input_path: str, source: str) -> Optional[int]:
    with tx() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO photos (filename, input_path, source) VALUES (?, ?, ?)",
                (filename, input_path, source),
            )
        except sqlite3.IntegrityError:
            return None
        return cur.lastrowid


def get_photo(photo_id: int) -> Optional[dict[str, Any]]:
    row = get_conn().execute("SELECT * FROM photos WHERE id = ?", (photo_id,)).fetchone()
    return row_to_dict(row) if row else None


def list_pending_ids(statuses: tuple[str, ...] = ("pending",)) -> list[int]:
    placeholders = ",".join("?" for _ in statuses)
    rows = get_conn().execute(
        f"SELECT id FROM photos WHERE status IN ({placeholders}) ORDER BY id", statuses
    ).fetchall()
    return [r["id"] for r in rows]


def list_photos(
    status: Optional[str] = None, page: int = 1, page_size: int = 60
) -> tuple[list[dict[str, Any]], int]:
    conn = get_conn()
    where = ""
    params: list[Any] = []
    if status and status != "all":
        where = "WHERE status = ?"
        params.append(status)
    total = conn.execute(f"SELECT COUNT(*) FROM photos {where}", params).fetchone()[0]
    offset = max(0, (page - 1) * page_size)
    rows = conn.execute(
        f"SELECT * FROM photos {where} ORDER BY id DESC LIMIT ? OFFSET ?",
        [*params, page_size, offset],
    ).fetchall()
    return [row_to_dict(r) for r in rows], total


def stats() -> dict[str, int]:
    conn = get_conn()
    rows = conn.execute("SELECT status, COUNT(*) as c FROM photos GROUP BY status").fetchall()
    counts = {r["status"]: r["c"] for r in rows}
    total = sum(counts.values())
    return {
        "total": total,
        "pending": counts.get("pending", 0),
        "processing": counts.get("processing", 0),
        "success": counts.get("success", 0),
        "no_plate": counts.get("no_plate", 0),
        "error": counts.get("error", 0),
    }


def set_processing(photo_id: int) -> None:
    with tx() as conn:
        conn.execute("UPDATE photos SET status = 'processing' WHERE id = ?", (photo_id,))


def set_result(
    photo_id: int,
    status: str,
    output_path: Optional[str] = None,
    thumb_input_path: Optional[str] = None,
    thumb_output_path: Optional[str] = None,
    num_plates: int = 0,
    detections: Optional[list[dict[str, Any]]] = None,
    error_message: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
    with tx() as conn:
        conn.execute(
            """
            UPDATE photos SET
                status = ?, output_path = ?, thumb_input_path = ?, thumb_output_path = ?,
                num_plates = ?, detections = ?, error_message = ?,
                width = ?, height = ?, processed_at = datetime('now')
            WHERE id = ?
            """,
            (
                status,
                output_path,
                thumb_input_path,
                thumb_output_path,
                num_plates,
                json.dumps(detections or []),
                error_message,
                width,
                height,
                photo_id,
            ),
        )


def delete_photo(photo_id: int) -> Optional[dict[str, Any]]:
    photo = get_photo(photo_id)
    if not photo:
        return None
    with tx() as conn:
        conn.execute("DELETE FROM photos WHERE id = ?", (photo_id,))
    return photo


def reset_to_pending(photo_ids: list[int]) -> int:
    if not photo_ids:
        return 0
    with tx() as conn:
        placeholders = ",".join("?" for _ in photo_ids)
        cur = conn.execute(
            f"UPDATE photos SET status = 'pending' WHERE id IN ({placeholders})", photo_ids
        )
        return cur.rowcount


def clear_all() -> int:
    with tx() as conn:
        cur = conn.execute("DELETE FROM photos")
        return cur.rowcount
