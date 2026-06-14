"""
SQLite storage layer.

Two tables:
  raw_posts  — every scraped post, preserved for re-processing.
  events     — validated, deduplicated events (keyed on dedupe_key).

Both tables are append-only by design; updating a record means inserting
a new version, not mutating the old one.
"""
from __future__ import annotations
import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import config
from models.event import RawPost, ExtractedEvent

_SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_posts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT NOT NULL,
    username    TEXT NOT NULL,
    post_url    TEXT NOT NULL,
    taken_at    TEXT NOT NULL,
    caption     TEXT,
    ocr_text    TEXT,
    image_path  TEXT,
    stored_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    dedupe_key          TEXT UNIQUE,
    status              TEXT NOT NULL DEFAULT 'review',
    confidence          REAL,
    confidence_reason   TEXT,
    title               TEXT NOT NULL,
    date                TEXT,
    time                TEXT,
    location            TEXT,
    description         TEXT,
    source_url          TEXT NOT NULL,
    organizer           TEXT,
    event_type          TEXT,
    is_event            INTEGER,
    raw_ai_response     TEXT,
    validation_errors   TEXT,
    stored_at           TEXT NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def save_raw_posts(posts: list[RawPost]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn = _connect()
    conn.executemany(
        """INSERT INTO raw_posts
           (source, username, post_url, taken_at, caption, ocr_text, image_path, stored_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (p.source, p.username, p.post_url, p.taken_at,
             p.caption, p.ocr_text, p.image_path, now)
            for p in posts
        ],
    )
    conn.commit()
    conn.close()


def save_events(events: list[ExtractedEvent]) -> dict[str, int]:
    """
    Upsert events by dedupe_key.  Returns counts by status.
    """
    now = datetime.now(timezone.utc).isoformat()
    counts: dict[str, int] = {"published": 0, "review": 0, "rejected": 0, "duplicate": 0}
    conn = _connect()

    for ev in events:
        key = ev.dedupe_key or ""
        existing = conn.execute(
            "SELECT id FROM events WHERE dedupe_key = ?", (key,)
        ).fetchone()
        if existing:
            counts["duplicate"] += 1
            continue

        conn.execute(
            """INSERT OR IGNORE INTO events
               (dedupe_key, status, confidence, confidence_reason, title, date, time,
                location, description, source_url, organizer, event_type, is_event,
                raw_ai_response, validation_errors, stored_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                key,
                ev.status,
                ev.confidence,
                ev.confidence_reason,
                ev.title,
                ev.date,
                ev.time,
                ev.location,
                ev.description,
                ev.source_url,
                ev.organizer,
                ev.event_type,
                int(ev.is_event),
                ev.raw_ai_response,
                json.dumps(ev.validation_errors),
                now,
            ),
        )
        counts[ev.status] = counts.get(ev.status, 0) + 1

    conn.commit()
    conn.close()
    return counts


def fetch_events(status: Optional[str] = None, limit: int = 200) -> list[dict]:
    conn = _connect()
    if status:
        rows = conn.execute(
            "SELECT * FROM events WHERE status = ? ORDER BY date, stored_at DESC LIMIT ?",
            (status, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM events ORDER BY date, stored_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def search_events(
    q: str = "",
    status: Optional[str] = None,
    event_type: Optional[str] = None,
    limit: int = 200,
) -> list[dict]:
    conn = _connect()
    clauses: list[str] = []
    params: list = []

    if status:
        clauses.append("status = ?")
        params.append(status)
    if event_type:
        clauses.append("event_type = ?")
        params.append(event_type)
    if q:
        pattern = f"%{q}%"
        clauses.append(
            "(title LIKE ? OR organizer LIKE ? OR location LIKE ? OR description LIKE ?)"
        )
        params.extend([pattern, pattern, pattern, pattern])

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = conn.execute(
        f"SELECT * FROM events {where} ORDER BY date ASC, stored_at DESC LIMIT ?",
        params + [limit],
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_event(event_id: int) -> Optional[dict]:
    conn = _connect()
    row = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


_EDITABLE_FIELDS = {"title", "date", "time", "location", "description", "event_type", "organizer"}


def update_event(event_id: int, fields: dict) -> Optional[dict]:
    """Update whitelisted fields on an event. Returns the updated row or None."""
    safe = {k: v for k, v in fields.items() if k in _EDITABLE_FIELDS}
    if not safe:
        return get_event(event_id)
    set_clause = ", ".join(f"{k} = ?" for k in safe)
    conn = _connect()
    conn.execute(
        f"UPDATE events SET {set_clause} WHERE id = ?",
        list(safe.values()) + [event_id],
    )
    conn.commit()
    row = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def set_event_status(event_id: int, status: str) -> Optional[dict]:
    if status not in ("published", "review", "rejected"):
        raise ValueError(f"Invalid status: {status!r}")
    conn = _connect()
    conn.execute("UPDATE events SET status = ? WHERE id = ?", (status, event_id))
    conn.commit()
    row = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def bulk_set_status(event_ids: list[int], status: str) -> int:
    if status not in ("published", "review", "rejected"):
        raise ValueError(f"Invalid status: {status!r}")
    if not event_ids:
        return 0
    placeholders = ",".join("?" * len(event_ids))
    conn = _connect()
    cur = conn.execute(
        f"UPDATE events SET status = ? WHERE id IN ({placeholders})",
        [status] + event_ids,
    )
    count = cur.rowcount
    conn.commit()
    conn.close()
    return count


def status_counts() -> dict[str, int]:
    conn = _connect()
    rows = conn.execute(
        "SELECT status, COUNT(*) as n FROM events GROUP BY status"
    ).fetchall()
    conn.close()
    return {r["status"]: r["n"] for r in rows}
