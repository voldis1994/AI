"""
JARVIS capability registry — tracks skills and their lifecycle status.

Statuses:
  CANDIDATE → TESTING → ACTIVE
  BROKEN → REPAIRING → TESTING → ACTIVE
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional


VALID_STATUSES = (
    "CANDIDATE",
    "TESTING",
    "ACTIVE",
    "BROKEN",
    "REPAIRING",
)


class CapabilityRegistry:
    """Maps capabilities → skills and enforces trust rules (only ACTIVE is trusted)."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS skills (
                    name TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    capabilities TEXT NOT NULL DEFAULT '[]',
                    dependencies TEXT NOT NULL DEFAULT '[]',
                    success_count INTEGER NOT NULL DEFAULT 0,
                    fail_count INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    activated_at REAL
                );

                CREATE TABLE IF NOT EXISTS capability_index (
                    capability TEXT NOT NULL,
                    skill_name TEXT NOT NULL,
                    PRIMARY KEY (capability, skill_name),
                    FOREIGN KEY(skill_name) REFERENCES skills(name)
                );

                CREATE INDEX IF NOT EXISTS idx_skills_status ON skills(status);
                """
            )
            self._conn.commit()

    def register_candidate(
        self,
        name: str,
        description: str,
        file_path: str,
        capabilities: list[str],
        dependencies: Optional[list[str]] = None,
        version: int = 1,
    ) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            existing = self._conn.execute(
                "SELECT name FROM skills WHERE name=?", (name,)
            ).fetchone()
            if existing:
                self._conn.execute(
                    "UPDATE skills SET description=?, file_path=?, status=?, version=?, "
                    "capabilities=?, dependencies=?, updated_at=?, last_error=NULL WHERE name=?",
                    (
                        description,
                        file_path,
                        "CANDIDATE",
                        version,
                        json.dumps(capabilities),
                        json.dumps(dependencies or []),
                        now,
                        name,
                    ),
                )
            else:
                self._conn.execute(
                    "INSERT INTO skills (name, description, file_path, status, version, "
                    "capabilities, dependencies, created_at, updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        name,
                        description,
                        file_path,
                        "CANDIDATE",
                        version,
                        json.dumps(capabilities),
                        json.dumps(dependencies or []),
                        now,
                        now,
                    ),
                )
            self._rebuild_capability_index(name, capabilities)
            self._conn.commit()
        return self.get_skill(name)  # type: ignore

    def set_status(self, name: str, status: str, error: Optional[str] = None) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}")
        now = time.time()
        with self._lock:
            activated = now if status == "ACTIVE" else None
            if activated:
                self._conn.execute(
                    "UPDATE skills SET status=?, updated_at=?, activated_at=?, last_error=? WHERE name=?",
                    (status, now, activated, error, name),
                )
            else:
                self._conn.execute(
                    "UPDATE skills SET status=?, updated_at=?, last_error=? WHERE name=?",
                    (status, now, error, name),
                )
            self._conn.commit()

    def mark_success(self, name: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE skills SET success_count=success_count+1, updated_at=? WHERE name=?",
                (time.time(), name),
            )
            self._conn.commit()

    def mark_failure(self, name: str, error: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE skills SET fail_count=fail_count+1, status=?, last_error=?, "
                "updated_at=? WHERE name=?",
                ("BROKEN", error[:2000], time.time(), name),
            )
            self._conn.commit()

    def get_skill(self, name: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM skills WHERE name=?", (name,)
            ).fetchone()
        return self._row_to_skill(row) if row else None

    def list_skills(self, status: Optional[str] = None) -> list[dict[str, Any]]:
        with self._lock:
            if status:
                rows = self._conn.execute(
                    "SELECT * FROM skills WHERE status=? ORDER BY name", (status,)
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM skills ORDER BY name"
                ).fetchall()
        return [self._row_to_skill(r) for r in rows]

    def active_capabilities(self) -> list[str]:
        """Human-readable list of what ACTIVE skills can do."""
        lines: list[str] = []
        for skill in self.list_skills("ACTIVE"):
            caps = ", ".join(skill["capabilities"]) or skill["description"]
            lines.append(f"{skill['name']} (v{skill['version']}): {caps}")
        return lines

    def find_by_capability(self, query: str, only_active: bool = True) -> list[dict[str, Any]]:
        q = query.lower()
        results = []
        skills = self.list_skills("ACTIVE") if only_active else self.list_skills()
        for skill in skills:
            hay = " ".join(
                [skill["name"], skill["description"]] + list(skill["capabilities"])
            ).lower()
            if q in hay or any(tok in hay for tok in q.split() if len(tok) > 3):
                results.append(skill)
        return results

    def match_skills(self, keywords: list[str], only_active: bool = True) -> list[dict[str, Any]]:
        skills = self.list_skills("ACTIVE") if only_active else self.list_skills()
        scored: list[tuple[int, dict]] = []
        for skill in skills:
            hay = " ".join(
                [skill["name"], skill["description"]] + list(skill["capabilities"])
            ).lower()
            score = sum(1 for kw in keywords if kw.lower() in hay)
            if score:
                scored.append((score, skill))
        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored]

    def next_version(self, name: str) -> int:
        skill = self.get_skill(name)
        return (skill["version"] + 1) if skill else 1

    def stats(self) -> dict[str, int]:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) AS c FROM skills").fetchone()["c"]
            by_status = {}
            for st in VALID_STATUSES:
                by_status[st] = self._conn.execute(
                    "SELECT COUNT(*) AS c FROM skills WHERE status=?", (st,)
                ).fetchone()["c"]
        return {"total": total, **by_status}

    def _rebuild_capability_index(self, name: str, capabilities: list[str]) -> None:
        self._conn.execute(
            "DELETE FROM capability_index WHERE skill_name=?", (name,)
        )
        for cap in capabilities:
            self._conn.execute(
                "INSERT OR IGNORE INTO capability_index (capability, skill_name) VALUES (?,?)",
                (cap.lower().strip(), name),
            )

    @staticmethod
    def _row_to_skill(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "name": row["name"],
            "description": row["description"],
            "file_path": row["file_path"],
            "status": row["status"],
            "version": row["version"],
            "capabilities": json.loads(row["capabilities"] or "[]"),
            "dependencies": json.loads(row["dependencies"] or "[]"),
            "success_count": row["success_count"],
            "fail_count": row["fail_count"],
            "last_error": row["last_error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "activated_at": row["activated_at"],
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
