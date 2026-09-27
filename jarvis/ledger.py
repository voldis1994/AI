"""
JARVIS task/action ledger — append-only audit trail of the learning cycle.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional


class Ledger:
    """Records REQUEST → PLAN → … → DONE phases for every task."""

    PHASES = (
        "REQUEST",
        "PLAN",
        "CHECK_CAPABILITIES",
        "RESEARCH",
        "LEARN",
        "BUILD_SKILL",
        "INSTALL_DEPS",
        "TEST",
        "OBSERVE",
        "DIAGNOSE",
        "REPAIR",
        "RETEST",
        "VERIFY",
        "SAVE_SKILL",
        "EXECUTE",
        "SAVE_EXPERIENCE",
        "DONE",
        "FAIL",
    )

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Faster durable writes on the hot path (many phase logs per attempt)
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        except Exception:
            pass
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    goal TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    result TEXT
                );

                CREATE TABLE IF NOT EXISTS actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    message TEXT,
                    data TEXT,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES tasks(id)
                );

                CREATE INDEX IF NOT EXISTS idx_actions_task ON actions(task_id);
                """
            )
            self._conn.commit()

    def start_task(self, goal: str) -> str:
        """Create task row only — callers log the detailed REQUEST phase once."""
        task_id = uuid.uuid4().hex[:12]
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO tasks (id, goal, status, created_at, updated_at, result) "
                "VALUES (?,?,?,?,?,?)",
                (task_id, goal, "REQUEST", now, now, None),
            )
            self._conn.commit()
        return task_id

    def log(
        self,
        task_id: str,
        phase: str,
        message: str = "",
        data: Optional[dict] = None,
    ) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO actions (task_id, phase, message, data, created_at) "
                "VALUES (?,?,?,?,?)",
                (task_id, phase, message, json.dumps(data or {}, default=str), now),
            )
            self._conn.execute(
                "UPDATE tasks SET status=?, updated_at=? WHERE id=?",
                (phase, now, task_id),
            )
            self._conn.commit()

    def finish(self, task_id: str, success: bool, result: Optional[dict] = None) -> None:
        phase = "DONE" if success else "FAIL"
        self.log(task_id, phase, "Task completed" if success else "Task failed", result)
        with self._lock:
            self._conn.execute(
                "UPDATE tasks SET status=?, result=?, updated_at=? WHERE id=?",
                (phase, json.dumps(result or {}, default=str), time.time(), task_id),
            )
            self._conn.commit()

    def get_task(self, task_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_actions(self, task_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM actions WHERE task_id=? ORDER BY id ASC",
                (task_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def recent_tasks(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict[str, int]:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) AS c FROM tasks").fetchone()["c"]
            done = self._conn.execute(
                "SELECT COUNT(*) AS c FROM tasks WHERE status='DONE'"
            ).fetchone()["c"]
            failed = self._conn.execute(
                "SELECT COUNT(*) AS c FROM tasks WHERE status='FAIL'"
            ).fetchone()["c"]
            actions = self._conn.execute(
                "SELECT COUNT(*) AS c FROM actions"
            ).fetchone()["c"]
        return {"tasks": total, "done": done, "failed": failed, "actions": actions}

    def close(self) -> None:
        with self._lock:
            self._conn.close()
