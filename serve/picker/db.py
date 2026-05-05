"""SQLite-backed conversation store for the picker.

Schema is intentionally tiny: conversations + messages. No multi-user
(PLAN.md §13.3 — single-user assumed).
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id           TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    model        TEXT NOT NULL,
    rag_enabled  INTEGER NOT NULL DEFAULT 0,
    rag_filter   TEXT,                           -- JSON list of subfolder filters
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,               -- system | user | assistant
    content         TEXT NOT NULL,
    model           TEXT,
    sources         TEXT,                        -- JSON list of RAG source paths
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_msgs_conv ON messages(conversation_id, id);
"""


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ----- conversations
    def create_conversation(self, model: str, title: str = "New conversation",
                            rag_enabled: bool = False,
                            rag_filter: list[str] | None = None) -> dict:
        cid = uuid.uuid4().hex[:12]
        now = time.time()
        with self._conn() as c:
            c.execute(
                "INSERT INTO conversations(id, title, model, rag_enabled, rag_filter, created_at, updated_at)"
                " VALUES(?,?,?,?,?,?,?)",
                (cid, title, model, int(rag_enabled),
                 json.dumps(rag_filter or []), now, now),
            )
        return self.get_conversation(cid)  # type: ignore[return-value]

    def list_conversations(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC"
            ).fetchall()
        return [self._conv_row(r) for r in rows]

    def get_conversation(self, cid: str) -> dict | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM conversations WHERE id=?", (cid,)
            ).fetchone()
        return self._conv_row(row) if row else None

    def update_conversation(self, cid: str, **fields) -> dict | None:
        if not fields:
            return self.get_conversation(cid)
        cols, vals = [], []
        for k, v in fields.items():
            if k == "rag_filter":
                v = json.dumps(v or [])
                cols.append("rag_filter = ?")
            elif k == "rag_enabled":
                v = int(bool(v))
                cols.append("rag_enabled = ?")
            elif k in {"title", "model"}:
                cols.append(f"{k} = ?")
            else:
                continue
            vals.append(v)
        cols.append("updated_at = ?")
        vals.append(time.time())
        vals.append(cid)
        with self._conn() as c:
            c.execute(
                f"UPDATE conversations SET {', '.join(cols)} WHERE id = ?",
                vals,
            )
        return self.get_conversation(cid)

    def delete_conversation(self, cid: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM messages WHERE conversation_id = ?", (cid,))
            c.execute("DELETE FROM conversations WHERE id = ?", (cid,))

    # ----- messages
    def add_message(self, cid: str, role: str, content: str,
                    model: str | None = None,
                    sources: list[str] | None = None) -> dict:
        now = time.time()
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO messages(conversation_id, role, content, model, sources, created_at)"
                " VALUES(?,?,?,?,?,?)",
                (cid, role, content, model,
                 json.dumps(sources) if sources else None, now),
            )
            mid = cur.lastrowid
            c.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, cid))
        return {
            "id": mid, "conversation_id": cid, "role": role, "content": content,
            "model": model, "sources": sources or [], "created_at": now,
        }

    def list_messages(self, cid: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC",
                (cid,),
            ).fetchall()
        return [self._msg_row(r) for r in rows]

    @staticmethod
    def _conv_row(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["rag_enabled"] = bool(d["rag_enabled"])
        try:
            d["rag_filter"] = json.loads(d["rag_filter"] or "[]")
        except json.JSONDecodeError:
            d["rag_filter"] = []
        return d

    @staticmethod
    def _msg_row(row: sqlite3.Row) -> dict:
        d = dict(row)
        if d.get("sources"):
            try:
                d["sources"] = json.loads(d["sources"])
            except json.JSONDecodeError:
                d["sources"] = []
        else:
            d["sources"] = []
        return d
