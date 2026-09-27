"""
JARVIS long-term memory — SQLite.

Stores conversation turns, experiences, learned facts, and skill events.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional


class Memory:
    """Thread-safe SQLite memory store."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    meta TEXT,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS experiences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    goal TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    skill_name TEXT,
                    details TEXT,
                    success INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL UNIQUE,
                    value TEXT NOT NULL,
                    source TEXT,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS skill_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    skill_name TEXT NOT NULL,
                    event TEXT NOT NULL,
                    status TEXT,
                    details TEXT,
                    created_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_conv_created ON conversations(created_at);
                CREATE INDEX IF NOT EXISTS idx_exp_created ON experiences(created_at);
                CREATE INDEX IF NOT EXISTS idx_skill_events ON skill_events(skill_name);
                """
            )
            self._conn.commit()

    def add_message(self, role: str, content: str, meta: Optional[dict] = None) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO conversations (role, content, meta, created_at) VALUES (?,?,?,?)",
                (role, content, json.dumps(meta or {}), time.time()),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def recent_messages(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, meta, created_at FROM conversations "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out = []
        for r in reversed(rows):
            out.append({
                "role": r["role"],
                "content": r["content"],
                "meta": json.loads(r["meta"] or "{}"),
                "created_at": r["created_at"],
            })
        return out

    def chat_history_for_llm(self, limit: int = 12) -> list[dict[str, str]]:
        msgs = self.recent_messages(limit)
        return [
            {"role": m["role"] if m["role"] in ("user", "assistant", "system") else "assistant",
             "content": m["content"]}
            for m in msgs
            if m["role"] in ("user", "assistant")
        ]

    def save_experience(
        self,
        goal: str,
        outcome: str,
        success: bool,
        skill_name: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO experiences (goal, outcome, skill_name, details, success, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (
                    goal,
                    outcome,
                    skill_name,
                    json.dumps(details or {}, default=str),
                    1 if success else 0,
                    time.time(),
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def recent_experiences(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM experiences ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def set_fact(self, key: str, value: Any, source: str = "jarvis") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO facts (key, value, source, updated_at) VALUES (?,?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                "source=excluded.source, updated_at=excluded.updated_at",
                (key, json.dumps(value, default=str), source, time.time()),
            )
            self._conn.commit()

    def get_fact(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM facts WHERE key=?", (key,)
            ).fetchone()
        if not row:
            return default
        try:
            return json.loads(row["value"])
        except Exception:
            return row["value"]

    def log_skill_event(
        self,
        skill_name: str,
        event: str,
        status: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO skill_events (skill_name, event, status, details, created_at) "
                "VALUES (?,?,?,?,?)",
                (
                    skill_name,
                    event,
                    status,
                    json.dumps(details or {}, default=str),
                    time.time(),
                ),
            )
            self._conn.commit()

    def stats(self) -> dict[str, int]:
        with self._lock:
            conversations = self._conn.execute(
                "SELECT COUNT(*) AS c FROM conversations"
            ).fetchone()["c"]
            experiences = self._conn.execute(
                "SELECT COUNT(*) AS c FROM experiences"
            ).fetchone()["c"]
            successes = self._conn.execute(
                "SELECT COUNT(*) AS c FROM experiences WHERE success=1"
            ).fetchone()["c"]
            facts = self._conn.execute(
                "SELECT COUNT(*) AS c FROM facts"
            ).fetchone()["c"]
            skill_events = self._conn.execute(
                "SELECT COUNT(*) AS c FROM skill_events"
            ).fetchone()["c"]
        return {
            "conversations": conversations,
            "experiences": experiences,
            "successes": successes,
            "facts": facts,
            "skill_events": skill_events,
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
