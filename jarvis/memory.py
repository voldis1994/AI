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

                CREATE TABLE IF NOT EXISTS learning_failures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    skill_name TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    version INTEGER,
                    phase TEXT,
                    observation TEXT NOT NULL,
                    code_fingerprint TEXT,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS learning_diagnoses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    skill_name TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    failure_id INTEGER,
                    diagnosis TEXT NOT NULL,
                    approach TEXT,
                    approach_fingerprint TEXT,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(failure_id) REFERENCES learning_failures(id)
                );

                CREATE TABLE IF NOT EXISTS learning_solutions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    skill_name TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    version INTEGER,
                    approach TEXT,
                    approach_fingerprint TEXT,
                    diagnosis_summary TEXT,
                    code_fingerprint TEXT,
                    details TEXT,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS failed_approaches (
                    skill_name TEXT NOT NULL,
                    approach_fingerprint TEXT NOT NULL,
                    approach TEXT NOT NULL,
                    fail_count INTEGER NOT NULL DEFAULT 1,
                    last_error TEXT,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (skill_name, approach_fingerprint)
                );

                CREATE INDEX IF NOT EXISTS idx_conv_created ON conversations(created_at);
                CREATE INDEX IF NOT EXISTS idx_exp_created ON experiences(created_at);
                CREATE INDEX IF NOT EXISTS idx_skill_events ON skill_events(skill_name);
                CREATE INDEX IF NOT EXISTS idx_learn_fail_skill ON learning_failures(skill_name);
                CREATE INDEX IF NOT EXISTS idx_learn_diag_skill ON learning_diagnoses(skill_name);
                CREATE INDEX IF NOT EXISTS idx_learn_sol_skill ON learning_solutions(skill_name);
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

    # ── Universal learning memory ───────────────────────────────────────

    def save_failure(
        self,
        skill_name: str,
        goal: str,
        observation: dict[str, Any],
        version: Optional[int] = None,
        phase: Optional[str] = None,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO learning_failures "
                "(skill_name, goal, version, phase, observation, code_fingerprint, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    skill_name,
                    goal,
                    version,
                    phase or observation.get("phase"),
                    json.dumps(observation, default=str),
                    observation.get("code_fingerprint"),
                    time.time(),
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def save_diagnosis(
        self,
        skill_name: str,
        goal: str,
        diagnosis: dict[str, Any],
        failure_id: Optional[int] = None,
    ) -> int:
        approach = str(diagnosis.get("approach") or diagnosis.get("fix_plan") or "")
        fp = str(diagnosis.get("approach_fingerprint") or "")
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO learning_diagnoses "
                "(skill_name, goal, failure_id, diagnosis, approach, approach_fingerprint, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    skill_name,
                    goal,
                    failure_id,
                    json.dumps(diagnosis, default=str),
                    approach,
                    fp,
                    time.time(),
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def save_solution(
        self,
        skill_name: str,
        goal: str,
        version: int,
        approach: str,
        approach_fingerprint: str,
        diagnosis_summary: str,
        code_fingerprint: str,
        details: Optional[dict] = None,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO learning_solutions "
                "(skill_name, goal, version, approach, approach_fingerprint, "
                "diagnosis_summary, code_fingerprint, details, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    skill_name,
                    goal,
                    version,
                    approach,
                    approach_fingerprint,
                    diagnosis_summary,
                    code_fingerprint,
                    json.dumps(details or {}, default=str),
                    time.time(),
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def record_failed_approach(
        self,
        skill_name: str,
        approach: str,
        approach_fingerprint: str,
        last_error: str = "",
    ) -> None:
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT fail_count FROM failed_approaches "
                "WHERE skill_name=? AND approach_fingerprint=?",
                (skill_name, approach_fingerprint),
            ).fetchone()
            if row:
                self._conn.execute(
                    "UPDATE failed_approaches SET fail_count=fail_count+1, "
                    "approach=?, last_error=?, updated_at=? "
                    "WHERE skill_name=? AND approach_fingerprint=?",
                    (approach, last_error[:2000], now, skill_name, approach_fingerprint),
                )
            else:
                self._conn.execute(
                    "INSERT INTO failed_approaches "
                    "(skill_name, approach_fingerprint, approach, fail_count, last_error, updated_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (skill_name, approach_fingerprint, approach, 1, last_error[:2000], now),
                )
            self._conn.commit()

    def get_failed_approaches(self, skill_name: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM failed_approaches WHERE skill_name=? "
                "ORDER BY fail_count DESC, updated_at DESC",
                (skill_name,),
            ).fetchall()
        return [dict(r) for r in rows]

    def recent_failures(self, skill_name: str, limit: int = 5) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM learning_failures WHERE skill_name=? "
                "ORDER BY id DESC LIMIT ?",
                (skill_name, limit),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["observation"] = json.loads(d["observation"])
            except Exception:
                pass
            out.append(d)
        return out

    def recent_diagnoses(self, skill_name: str, limit: int = 5) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM learning_diagnoses WHERE skill_name=? "
                "ORDER BY id DESC LIMIT ?",
                (skill_name, limit),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["diagnosis"] = json.loads(d["diagnosis"])
            except Exception:
                pass
            out.append(d)
        return out

    def recent_solutions(
        self, skill_name: Optional[str] = None, goal: Optional[str] = None, limit: int = 5
    ) -> list[dict[str, Any]]:
        with self._lock:
            if skill_name:
                rows = self._conn.execute(
                    "SELECT * FROM learning_solutions WHERE skill_name=? "
                    "ORDER BY id DESC LIMIT ?",
                    (skill_name, limit),
                ).fetchall()
            elif goal:
                rows = self._conn.execute(
                    "SELECT * FROM learning_solutions WHERE goal LIKE ? "
                    "ORDER BY id DESC LIMIT ?",
                    (f"%{goal[:80]}%", limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM learning_solutions ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

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
            failures = self._conn.execute(
                "SELECT COUNT(*) AS c FROM learning_failures"
            ).fetchone()["c"]
            diagnoses = self._conn.execute(
                "SELECT COUNT(*) AS c FROM learning_diagnoses"
            ).fetchone()["c"]
            solutions = self._conn.execute(
                "SELECT COUNT(*) AS c FROM learning_solutions"
            ).fetchone()["c"]
        knowledge_entries = self._conn.execute(
            "SELECT COUNT(*) AS c FROM facts WHERE key LIKE 'knowledge:%'"
        ).fetchone()["c"]
        return {
            "conversations": conversations,
            "experiences": experiences,
            "successes": successes,
            "facts": facts,
            "skill_events": skill_events,
            "learning_failures": failures,
            "learning_diagnoses": diagnoses,
            "learning_solutions": solutions,
            "knowledge_topics": knowledge_entries,
        }

    # ── Research knowledge persistence ──────────────────────────────────

    def save_research_knowledge(
        self,
        skill_name: str,
        research: dict[str, Any],
        *,
        goal: str = "",
        queries: Optional[list] = None,
    ) -> dict[str, Any]:
        """
        Persist research insights for a skill and return the updated history.

        Keeps a rolling history under knowledge:{skill_name} and the latest
        snapshot under research:{skill_name}.
        """
        entry = {
            "ts": time.time(),
            "goal": goal,
            "queries": list(queries or [])[:8],
            "approach": research.get("approach"),
            "libraries": list(research.get("libraries") or [])[:12],
            "key_apis": list(research.get("key_apis") or [])[:12],
            "pitfalls": list(research.get("pitfalls") or [])[:12],
            "repair_insight": research.get("repair_insight") or "",
            "test_idea": research.get("test_idea") or "",
            "sources": list(research.get("sources") or [])[:12],
            "result_count": len(research.get("results") or []),
        }
        key = f"knowledge:{skill_name}"
        prior = self.get_fact(key) or []
        if not isinstance(prior, list):
            prior = [prior]
        prior.append(entry)
        prior = prior[-20:]
        self.set_fact(key, prior, source="research")
        self.set_fact(f"research:{skill_name}", research, source="research")
        return {"entry": entry, "history": prior}

    def get_research_knowledge(self, skill_name: str, limit: int = 8) -> list[dict[str, Any]]:
        prior = self.get_fact(f"knowledge:{skill_name}") or []
        if not isinstance(prior, list):
            return [prior] if prior else []
        return list(prior)[-limit:]

    def get_latest_research(self, skill_name: str) -> Optional[dict[str, Any]]:
        val = self.get_fact(f"research:{skill_name}")
        return val if isinstance(val, dict) else None

    # ── Topic knowledge (learning requests — NOT tied to a skill) ───────

    def save_topic_knowledge(
        self,
        topic: str,
        research: dict[str, Any],
        *,
        goal: str = "",
        queries: Optional[list] = None,
        summary: str = "",
        verified: bool = False,
    ) -> dict[str, Any]:
        """
        Persist knowledge for a learning topic (independent of any skill).

        Keys: knowledge:topic:{slug} (history) and topic:{slug} (latest).
        """
        from jarvis.intent import IntentClassifier

        slug = IntentClassifier.topic_slug(topic or goal or "topic")
        entry = {
            "ts": time.time(),
            "topic": topic or slug,
            "goal": goal,
            "queries": list(queries or [])[:8],
            "summary": (summary or research.get("approach") or "")[:4000],
            "approach": research.get("approach"),
            "libraries": list(research.get("libraries") or [])[:12],
            "key_apis": list(research.get("key_apis") or [])[:20],
            "pitfalls": list(research.get("pitfalls") or [])[:12],
            "test_idea": research.get("test_idea") or "",
            "practical_result": research.get("practical_result"),
            "sources": list(research.get("sources") or [])[:12],
            "result_count": len(research.get("results") or []),
            "verified": bool(verified),
        }
        key = f"knowledge:topic:{slug}"
        prior = self.get_fact(key) or []
        if not isinstance(prior, list):
            prior = [prior]
        prior.append(entry)
        prior = prior[-20:]
        self.set_fact(key, prior, source="learning")
        self.set_fact(f"topic:{slug}", entry, source="learning")
        return {"topic": slug, "entry": entry, "history": prior}

    def get_topic_knowledge(self, topic: str, limit: int = 8) -> list[dict[str, Any]]:
        from jarvis.intent import IntentClassifier

        slug = IntentClassifier.topic_slug(topic)
        prior = self.get_fact(f"knowledge:topic:{slug}") or []
        if not isinstance(prior, list):
            return [prior] if prior else []
        return list(prior)[-limit:]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
