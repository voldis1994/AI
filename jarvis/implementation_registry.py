"""
ImplementationRegistry — TOOL / IMPLEMENTATION lifecycle.

Stores versioned executable tools (skill .py files) and enforces trust:
only ACTIVE implementations may serve EXECUTE. Archive + pending-candidate
protect rollback/versioning.

Ontology: this registry owns TOOL/IMPLEMENTATION rows. Competence domains and
CAPABILITY classes live in CompetenceRegistry. KNOWLEDGE/EXPERIENCE attach via
Memory. USER REQUEST truth is TaskContract only.

Statuses:
  CANDIDATE → TESTING → ACTIVE
  BROKEN → REPAIRING → TESTING → ACTIVE
  ACTIVE (old) → ARCHIVED  (only after new version PASS)
"""

from __future__ import annotations

import json
import shutil
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
    "ARCHIVED",
)


class ImplementationRegistry:
    """Maps tools/implementations and enforces trust rules (only ACTIVE is trusted)."""

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
                    activated_at REAL,
                    pending_path TEXT,
                    pending_version INTEGER,
                    pending_meta TEXT
                );

                CREATE TABLE IF NOT EXISTS skill_archive (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    file_path TEXT NOT NULL,
                    description TEXT,
                    capabilities TEXT,
                    archived_at REAL NOT NULL,
                    reason TEXT
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
            # migrate older DBs missing pending columns
            cols = {
                r["name"]
                for r in self._conn.execute("PRAGMA table_info(skills)").fetchall()
            }
            for col, decl in (
                ("pending_path", "TEXT"),
                ("pending_version", "INTEGER"),
                ("pending_meta", "TEXT"),
            ):
                if col not in cols:
                    self._conn.execute(f"ALTER TABLE skills ADD COLUMN {col} {decl}")
            self._conn.commit()

    def register_candidate(
        self,
        name: str,
        description: str,
        file_path: str,
        capabilities: list[str],
        dependencies: Optional[list[str]] = None,
        version: int = 1,
        protect_active: bool = True,
    ) -> dict[str, Any]:
        """
        Register a candidate skill.

        If an ACTIVE skill already exists and protect_active=True, the ACTIVE
        row/file_path is NOT overwritten — candidate is stored in pending_*.
        """
        now = time.time()
        meta = {
            "description": description,
            "capabilities": capabilities,
            "dependencies": dependencies or [],
            "version": version,
            "file_path": file_path,
        }
        with self._lock:
            existing = self._conn.execute(
                "SELECT * FROM skills WHERE name=?", (name,)
            ).fetchone()

            if existing and existing["status"] == "ACTIVE" and protect_active:
                # Keep ACTIVE intact; only set pending fields
                self._conn.execute(
                    "UPDATE skills SET pending_path=?, pending_version=?, pending_meta=?, "
                    "updated_at=? WHERE name=?",
                    (file_path, version, json.dumps(meta), now, name),
                )
                self._conn.commit()
                skill = self.get_skill(name)
                assert skill is not None
                skill["pending"] = meta
                return skill

            if existing:
                # Non-ACTIVE: safe to move into CANDIDATE (file already written aside)
                self._conn.execute(
                    "UPDATE skills SET description=?, file_path=?, status=?, version=?, "
                    "capabilities=?, dependencies=?, updated_at=?, last_error=NULL, "
                    "pending_path=NULL, pending_version=NULL, pending_meta=NULL WHERE name=?",
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
            if status == "ACTIVE":
                self._conn.execute(
                    "UPDATE skills SET status=?, updated_at=?, activated_at=?, last_error=? WHERE name=?",
                    (status, now, now, error, name),
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
        """Mark skill BROKEN for trust, but do not delete/overwrite its file."""
        with self._lock:
            self._conn.execute(
                "UPDATE skills SET fail_count=fail_count+1, status=?, last_error=?, "
                "updated_at=? WHERE name=?",
                ("BROKEN", error[:2000], time.time(), name),
            )
            self._conn.commit()

    def clear_pending(self, name: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE skills SET pending_path=NULL, pending_version=NULL, "
                "pending_meta=NULL, updated_at=? WHERE name=?",
                (time.time(), name),
            )
            self._conn.commit()

    def set_pending(
        self,
        name: str,
        candidate_path: str,
        version: int,
        meta: dict[str, Any],
        keep_file_path: Optional[str] = None,
    ) -> None:
        """Attach a pending candidate without replacing the trusted file_path."""
        now = time.time()
        with self._lock:
            self._conn.execute(
                "UPDATE skills SET pending_path=?, pending_version=?, pending_meta=?, "
                "updated_at=? WHERE name=?",
                (candidate_path, version, json.dumps(meta), now, name),
            )
            if keep_file_path:
                self._conn.execute(
                    "UPDATE skills SET file_path=? WHERE name=?",
                    (keep_file_path, name),
                )
            self._conn.commit()

    def promote_candidate(
        self,
        name: str,
        candidate_path: str,
        version: int,
        description: str,
        capabilities: list[str],
        dependencies: Optional[list[str]] = None,
        archive_dir: Optional[str | Path] = None,
    ) -> dict[str, Any]:
        """
        After PASS: archive old ACTIVE file, install candidate as main, set ACTIVE.

        Old working file is never overwritten before this call.
        """
        now = time.time()
        candidate = Path(candidate_path)
        if not candidate.exists():
            raise FileNotFoundError(f"Candidate missing: {candidate}")

        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM skills WHERE name=?", (name,)
            ).fetchone()
            old_path = Path(row["file_path"]) if row else None
            old_version = int(row["version"]) if row else None
            old_status = row["status"] if row else None

            # Archive previous file if it exists and differs from candidate
            archived_to = None
            if old_path and old_path.exists() and old_path.resolve() != candidate.resolve():
                adir = Path(archive_dir) if archive_dir else old_path.parent / "archive"
                adir.mkdir(parents=True, exist_ok=True)
                archived_to = adir / f"{name}.v{old_version or 0}.py"
                shutil.copy2(old_path, archived_to)
                self._conn.execute(
                    "INSERT INTO skill_archive (name, version, file_path, description, "
                    "capabilities, archived_at, reason) VALUES (?,?,?,?,?,?,?)",
                    (
                        name,
                        old_version or 0,
                        str(archived_to),
                        row["description"] if row else description,
                        row["capabilities"] if row else json.dumps(capabilities),
                        now,
                        f"replaced_by_v{version}",
                    ),
                )

            # Install candidate as the canonical skill file
            final_path = candidate.parent / f"{name}.py"
            if candidate.resolve() != final_path.resolve():
                shutil.copy2(candidate, final_path)
                # keep candidate copy as versioned artifact
                versioned = candidate.parent / f"{name}.v{version}.py"
                if candidate.resolve() != versioned.resolve():
                    try:
                        shutil.copy2(candidate, versioned)
                    except Exception:
                        pass

            if row:
                self._conn.execute(
                    "UPDATE skills SET description=?, file_path=?, status=?, version=?, "
                    "capabilities=?, dependencies=?, updated_at=?, activated_at=?, "
                    "last_error=NULL, pending_path=NULL, pending_version=NULL, "
                    "pending_meta=NULL WHERE name=?",
                    (
                        description,
                        str(final_path),
                        "ACTIVE",
                        version,
                        json.dumps(capabilities),
                        json.dumps(dependencies or []),
                        now,
                        now,
                        name,
                    ),
                )
            else:
                self._conn.execute(
                    "INSERT INTO skills (name, description, file_path, status, version, "
                    "capabilities, dependencies, created_at, updated_at, activated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        name,
                        description,
                        str(final_path),
                        "ACTIVE",
                        version,
                        json.dumps(capabilities),
                        json.dumps(dependencies or []),
                        now,
                        now,
                        now,
                    ),
                )
            self._rebuild_capability_index(name, capabilities)
            self._conn.commit()

        skill = self.get_skill(name)
        assert skill is not None
        skill["archived_from"] = str(archived_to) if archived_to else None
        skill["previous_status"] = old_status
        return skill

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

    def list_archive(self, name: Optional[str] = None) -> list[dict[str, Any]]:
        with self._lock:
            if name:
                rows = self._conn.execute(
                    "SELECT * FROM skill_archive WHERE name=? ORDER BY archived_at DESC",
                    (name,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM skill_archive ORDER BY archived_at DESC"
                ).fetchall()
        return [dict(r) for r in rows]

    def active_capabilities(self) -> list[str]:
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
        if not skill:
            return 1
        pending_v = skill.get("pending_version") or 0
        return max(int(skill["version"]) + 1, int(pending_v) + 1)

    def stats(self) -> dict[str, int]:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) AS c FROM skills").fetchone()["c"]
            by_status = {}
            for st in VALID_STATUSES:
                by_status[st] = self._conn.execute(
                    "SELECT COUNT(*) AS c FROM skills WHERE status=?", (st,)
                ).fetchone()["c"]
            archived = self._conn.execute(
                "SELECT COUNT(*) AS c FROM skill_archive"
            ).fetchone()["c"]
        return {"total": total, "archived_versions": archived, **by_status}

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
        keys = row.keys()
        pending_meta = None
        if "pending_meta" in keys and row["pending_meta"]:
            try:
                pending_meta = json.loads(row["pending_meta"])
            except Exception:
                pending_meta = None
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
            "pending_path": row["pending_path"] if "pending_path" in keys else None,
            "pending_version": row["pending_version"] if "pending_version" in keys else None,
            "pending_meta": pending_meta,
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
