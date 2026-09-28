"""
Competence registry — hierarchical SKILL / CAPABILITY / TOOL index.

Executable TOOL code continues to live in CapabilityRegistry (skills table)
and on-disk .py files. This registry adds:
  - competence tree (auto-branching)
  - tool → competence/capability mapping
  - migration of legacy task-specific skills → tools under domains
  - VERIFIED knowledge/experience hooks (via Memory keys)

Does not replace TaskGoal, verifier, recovery, rollback, or model routing.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

from jarvis.capability_match import evaluate_capability
from jarvis.capability_registry import CapabilityRegistry
from jarvis.competence import (
    LAYER_CAPABILITY,
    LAYER_SKILL,
    LAYER_TOOL,
    CompetencePlan,
    CompetenceRef,
    classify_task_competences,
    format_competence_log,
    preferred_tool_name,
    slugify,
)
from jarvis.task_goal import TaskGoal


class CompetenceRegistry:
    """Hierarchical competence index layered on CapabilityRegistry."""

    def __init__(
        self,
        db_path: str | Path,
        skills: CapabilityRegistry,
    ) -> None:
        self.db_path = Path(db_path)
        self.skills = skills
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS competences (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    parent_id TEXT NOT NULL DEFAULT '',
                    kind TEXT NOT NULL,
                    meta TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tools (
                    id TEXT PRIMARY KEY,
                    competence_id TEXT NOT NULL,
                    capability TEXT NOT NULL DEFAULT '',
                    skill_name TEXT NOT NULL,
                    verified INTEGER NOT NULL DEFAULT 0,
                    meta TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_tools_competence
                    ON tools(competence_id);
                CREATE INDEX IF NOT EXISTS idx_tools_skill
                    ON tools(skill_name);
                CREATE INDEX IF NOT EXISTS idx_competences_parent
                    ON competences(parent_id);
                """
            )
            self._conn.commit()

    # ── Tree ops ────────────────────────────────────────────────────────

    def ensure_node(
        self,
        node_id: str,
        *,
        title: str = "",
        parent_id: str = "",
        kind: str = LAYER_SKILL,
        meta: Optional[dict[str, Any]] = None,
    ) -> CompetenceRef:
        """Create or return a competence node (auto-branch)."""
        # Keep dotted skill ids and capability ids with '/' intact
        nid = (node_id or "").strip() or "general"
        now = time.time()
        title = (title or nid).strip()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM competences WHERE id=?", (nid,)
            ).fetchone()
            if row:
                return self._row_to_node(row)
            if parent_id and parent_id != nid:
                # Parent may not exist yet — create outside this locked insert
                pass
        if parent_id and parent_id != nid and self.get_node(parent_id) is None:
            self.ensure_node(parent_id, title=parent_id, kind=LAYER_SKILL)
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM competences WHERE id=?", (nid,)
            ).fetchone()
            if row:
                return self._row_to_node(row)
            self._conn.execute(
                """
                INSERT INTO competences (id, title, parent_id, kind, meta, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?)
                """,
                (
                    nid,
                    title,
                    parent_id or "",
                    kind,
                    json.dumps(meta or {}),
                    now,
                    now,
                ),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM competences WHERE id=?", (nid,)
            ).fetchone()
        return self._row_to_node(row)

    def ensure_path(self, skill_ids: list[str], capability_ids: list[str]) -> list[str]:
        """Ensure skill branches + capability nodes; return created/existing ids."""
        out: list[str] = []
        for sid in skill_ids or []:
            parts = str(sid).split(".")
            parent = ""
            for i in range(len(parts)):
                cur = ".".join(parts[: i + 1])
                self.ensure_node(
                    cur,
                    title=cur,
                    parent_id=parent,
                    kind=LAYER_SKILL,
                )
                parent = cur
                if cur not in out:
                    out.append(cur)
        for cid in capability_ids or []:
            if "/" in cid:
                skill_part, cap_part = cid.split("/", 1)
                self.ensure_node(
                    skill_part, title=skill_part, parent_id="", kind=LAYER_SKILL
                )
                self.ensure_node(
                    cid,
                    title=cap_part,
                    parent_id=skill_part,
                    kind=LAYER_CAPABILITY,
                    meta={"capability": cap_part},
                )
            else:
                self.ensure_node(cid, title=cid, kind=LAYER_CAPABILITY)
            if cid not in out:
                out.append(cid)
        return out

    def get_node(self, node_id: str) -> Optional[CompetenceRef]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM competences WHERE id=?", (node_id,)
            ).fetchone()
        return self._row_to_node(row) if row else None

    def list_nodes(self, kind: Optional[str] = None) -> list[CompetenceRef]:
        with self._lock:
            if kind:
                rows = self._conn.execute(
                    "SELECT * FROM competences WHERE kind=? ORDER BY id", (kind,)
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM competences ORDER BY id"
                ).fetchall()
        return [self._row_to_node(r) for r in rows]

    # ── Tools ───────────────────────────────────────────────────────────

    def register_tool(
        self,
        skill_name: str,
        *,
        competence_id: str,
        capability: str = "",
        verified: bool = False,
        meta: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Map an executable skill row to a TOOL under a competence."""
        now = time.time()
        tid = skill_name
        self.ensure_node(
            competence_id,
            title=competence_id,
            kind=LAYER_SKILL if "/" not in competence_id else LAYER_CAPABILITY,
        )
        with self._lock:
            prev = self._conn.execute(
                "SELECT * FROM tools WHERE id=?", (tid,)
            ).fetchone()
            payload = json.dumps(meta or {})
            if prev:
                self._conn.execute(
                    """
                    UPDATE tools SET competence_id=?, capability=?, skill_name=?,
                        verified=?, meta=?, updated_at=?
                    WHERE id=?
                    """,
                    (
                        competence_id,
                        capability,
                        skill_name,
                        1 if verified else int(prev["verified"] or 0),
                        payload,
                        now,
                        tid,
                    ),
                )
            else:
                self._conn.execute(
                    """
                    INSERT INTO tools
                    (id, competence_id, capability, skill_name, verified, meta, created_at, updated_at)
                    VALUES (?,?,?,?,?,?,?,?)
                    """,
                    (
                        tid,
                        competence_id,
                        capability,
                        skill_name,
                        1 if verified else 0,
                        payload,
                        now,
                        now,
                    ),
                )
            self._conn.commit()
        return self.get_tool(tid) or {}

    def mark_tool_verified(self, skill_name: str, verified: bool = True) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE tools SET verified=?, updated_at=? WHERE skill_name=?",
                (1 if verified else 0, time.time(), skill_name),
            )
            self._conn.commit()

    def get_tool(self, tool_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tools WHERE id=?", (tool_id,)
            ).fetchone()
        if not row:
            return None
        return self._row_to_tool(row)

    def list_tools(
        self,
        *,
        competence_id: str = "",
        verified_only: bool = False,
    ) -> list[dict[str, Any]]:
        with self._lock:
            sql = "SELECT * FROM tools WHERE 1=1"
            args: list[Any] = []
            if competence_id:
                sql += " AND (competence_id=? OR competence_id LIKE ?)"
                args.extend([competence_id, competence_id + "%"])
            if verified_only:
                sql += " AND verified=1"
            sql += " ORDER BY updated_at DESC"
            rows = self._conn.execute(sql, args).fetchall()
        return [self._row_to_tool(r) for r in rows]

    def tools_as_skills(
        self,
        *,
        competence_ids: Optional[list[str]] = None,
        only_active: bool = True,
    ) -> list[dict[str, Any]]:
        """Resolve TOOL index → CapabilityRegistry skill records."""
        tools = self.list_tools()
        if competence_ids:
            allowed = set(competence_ids)
            tools = [
                t
                for t in tools
                if t.get("competence_id") in allowed
                or any(
                    str(t.get("competence_id") or "").startswith(c + ".")
                    or str(t.get("competence_id") or "").startswith(c + "/")
                    for c in allowed
                )
            ]
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for t in tools:
            name = str(t.get("skill_name") or "")
            if not name or name in seen:
                continue
            sk = self.skills.get_skill(name)
            if not sk:
                continue
            if only_active and sk.get("status") != "ACTIVE":
                continue
            sk = dict(sk)
            sk["competence_id"] = t.get("competence_id")
            sk["capability"] = t.get("capability")
            sk["tool_verified"] = bool(t.get("verified"))
            sk["layer"] = LAYER_TOOL
            out.append(sk)
            seen.add(name)
        return out

    # ── Planning ────────────────────────────────────────────────────────

    def plan_for_goal(self, task_goal: TaskGoal) -> CompetencePlan:
        """
        Classify TaskGoal → ensure branches → find reusable VERIFIED tools.

        Decision:
          REUSE_COMPOSE — one or more compatible ACTIVE tools
          EXTEND        — competence exists, no compatible tool (build under domain)
          OPEN_BRANCH   — need new skill/capability nodes + tool
          BUILD_TOOL    — fallback gap fill
        """
        clf = classify_task_competences(task_goal)
        skill_ids = list(clf["skill_ids"])
        capability_ids = list(clf["capability_ids"])
        created = self.ensure_path(skill_ids, capability_ids)

        # Candidates: tools under these competences + all ACTIVE (legacy) for match
        candidates = self.tools_as_skills(competence_ids=skill_ids, only_active=True)
        # Also consider unmigrated ACTIVE skills (will be indexed on migrate)
        known_names = {c.get("name") for c in candidates}
        for sk in self.skills.list_skills("ACTIVE"):
            if sk.get("name") not in known_names:
                candidates.append(dict(sk))

        matched: list[dict[str, Any]] = []
        reasons: list[str] = []
        for sk in candidates:
            report = evaluate_capability(sk, task_goal)
            if report.get("compatible"):
                row = dict(sk)
                row["_match"] = report
                matched.append(row)
                reasons.append(
                    f"{sk.get('name')}: {report.get('reason')} "
                    f"score={report.get('score')}"
                )

        matched.sort(key=lambda x: -float((x.get("_match") or {}).get("score") or 0))
        tool_name = preferred_tool_name(task_goal, clf)

        # Existing tool with same preferred name → prefer EXTEND/REUSE over new slug
        existing_preferred = self.skills.get_skill(tool_name)
        domain_tools = self.list_tools(competence_id=str(clf.get("primary_skill") or ""))

        if matched:
            decision = "REUSE_COMPOSE"
            needs_new = False
        elif existing_preferred and existing_preferred.get("status") in (
            "ACTIVE",
            "BROKEN",
            "REPAIRING",
        ):
            decision = "EXTEND"
            needs_new = True
            reasons.append(f"extend existing tool {tool_name}")
        elif domain_tools:
            decision = "EXTEND"
            needs_new = True
            reasons.append("competence exists; build/extend tool under domain")
        elif created:
            decision = "OPEN_BRANCH"
            needs_new = True
            reasons.append("new competence branch; build first tool")
        else:
            decision = "BUILD_TOOL"
            needs_new = True
            reasons.append("no reusable tool; build under classified competence")

        # Branches opened for this goal: subject sub-skills + capability nodes
        open_branches = [s for s in skill_ids if "subject." in s]
        if needs_new:
            for cid in capability_ids:
                if cid not in open_branches:
                    open_branches.append(cid)

        return CompetencePlan(
            task_goal=task_goal.to_dict(),
            skill_ids=skill_ids,
            capability_ids=capability_ids,
            tools=matched,
            open_branches=open_branches,
            needs_new_tool=needs_new,
            decision=decision,
            preferred_tool_name=tool_name,
            reasons=reasons,
        )

    def bind_built_tool(
        self,
        skill_name: str,
        task_goal: TaskGoal,
        *,
        verified: bool = False,
    ) -> dict[str, Any]:
        """After BUILD/PROMOTE — attach tool to classified competence (not a new skill domain)."""
        clf = classify_task_competences(task_goal)
        self.ensure_path(clf["skill_ids"], clf["capability_ids"])
        primary = clf["primary_skill"]
        cap = (clf["capabilities"] or ["act.unknown"])[0]
        return self.register_tool(
            skill_name,
            competence_id=primary,
            capability=cap,
            verified=verified,
            meta={
                "subject": task_goal.subject,
                "actions": list(task_goal.actions),
                "artifacts": list(task_goal.artifacts),
                "layer": LAYER_TOOL,
            },
        )

    # ── Migration ───────────────────────────────────────────────────────

    def migrate_legacy_skills(self) -> dict[str, Any]:
        """
        Map existing registry skills → TOOLS under competence domains.

        Keeps working .py files and skill rows; only adds competence index.
        Task-specific slugs become tools under generic io/net/learn branches
        derived from their capability tags / description — never deleted.
        """
        migrated = 0
        skipped = 0
        details: list[dict[str, Any]] = []
        for sk in self.skills.list_skills():
            name = str(sk.get("name") or "")
            if not name:
                skipped += 1
                continue
            if self.get_tool(name):
                skipped += 1
                continue
            # Build a synthetic TaskGoal from skill meta for classification
            caps = list(sk.get("capabilities") or [])
            desc = str(sk.get("description") or name)
            # Prefer capability tags to pick domain; fallback to description
            pseudo = TaskGoal.from_request(desc, goal=desc)
            clf = classify_task_competences(pseudo)
            # If skill already declares io-like capabilities, honor them
            primary = clf["primary_skill"]
            for c in caps:
                cl = str(c).lower()
                if any(x in cl for x in ("write", "file", "create", "path")):
                    primary = "io.files"
                    break
                if any(x in cl for x in ("http", "fetch", "url", "download")):
                    primary = "net.http"
                    break
                if any(x in cl for x in ("learn", "research", "knowledge")):
                    primary = "learn.topic"
                    break
            cap = (clf["capabilities"] or ["act.unknown"])[0]
            if caps:
                # Map first declared capability into dotted form if plain
                raw = str(caps[0]).strip().lower().replace(" ", ".")
                if raw:
                    cap = raw if "." in raw else f"cap.{slugify(raw, 24)}"
            self.ensure_path([primary], [f"{primary}/{cap}"])
            verified = sk.get("status") == "ACTIVE" and int(sk.get("success_count") or 0) > 0
            self.register_tool(
                name,
                competence_id=primary,
                capability=cap,
                verified=verified,
                meta={
                    "migrated": True,
                    "legacy_name": name,
                    "legacy_status": sk.get("status"),
                    "legacy_capabilities": caps,
                },
            )
            migrated += 1
            details.append(
                {
                    "skill": name,
                    "competence_id": primary,
                    "capability": cap,
                    "verified": verified,
                }
            )
        return {
            "migrated": migrated,
            "skipped": skipped,
            "details": details[:40],
            "nodes": len(self.list_nodes()),
            "tools": len(self.list_tools()),
        }

    # ── Knowledge / Experience attachment (Memory keys) ─────────────────

    @staticmethod
    def knowledge_key(competence_id: str) -> str:
        return f"knowledge:competence:{slugify(competence_id, 64)}"

    @staticmethod
    def experience_key(competence_id: str) -> str:
        return f"experience:competence:{slugify(competence_id, 64)}"

    def attach_knowledge(
        self,
        memory: Any,
        competence_id: str,
        entry: dict[str, Any],
        *,
        verified: bool = False,
    ) -> None:
        """Persist VERIFIED knowledge under a competence (KNOWLEDGE layer)."""
        if memory is None or not hasattr(memory, "set_fact"):
            return
        key = self.knowledge_key(competence_id)
        prior = memory.get_fact(key) or []
        if not isinstance(prior, list):
            prior = [prior] if prior else []
        row = dict(entry or {})
        row["ts"] = time.time()
        row["competence_id"] = competence_id
        row["layer"] = "knowledge"
        row["verified"] = bool(verified)
        prior.append(row)
        memory.set_fact(key, prior[-40:], source="competence")

    def attach_experience(
        self,
        memory: Any,
        competence_id: str,
        entry: dict[str, Any],
    ) -> None:
        """Persist experience/errors under a competence (EXPERIENCE layer)."""
        if memory is None or not hasattr(memory, "set_fact"):
            return
        key = self.experience_key(competence_id)
        prior = memory.get_fact(key) or []
        if not isinstance(prior, list):
            prior = [prior] if prior else []
        row = dict(entry or {})
        row["ts"] = time.time()
        row["competence_id"] = competence_id
        row["layer"] = "experience"
        prior.append(row)
        memory.set_fact(key, prior[-40:], source="competence")

    def stats(self) -> dict[str, Any]:
        nodes = self.list_nodes()
        tools = self.list_tools()
        return {
            "competences": len(nodes),
            "skills": sum(1 for n in nodes if n.kind == LAYER_SKILL),
            "capabilities": sum(1 for n in nodes if n.kind == LAYER_CAPABILITY),
            "tools": len(tools),
            "verified_tools": sum(1 for t in tools if t.get("verified")),
        }

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_node(row: sqlite3.Row) -> CompetenceRef:
        meta = {}
        try:
            meta = json.loads(row["meta"] or "{}")
        except Exception:
            meta = {}
        return CompetenceRef(
            id=row["id"],
            title=row["title"],
            parent_id=row["parent_id"] or "",
            kind=row["kind"],
            meta=meta if isinstance(meta, dict) else {},
        )

    @staticmethod
    def _row_to_tool(row: sqlite3.Row) -> dict[str, Any]:
        meta = {}
        try:
            meta = json.loads(row["meta"] or "{}")
        except Exception:
            meta = {}
        return {
            "id": row["id"],
            "competence_id": row["competence_id"],
            "capability": row["capability"],
            "skill_name": row["skill_name"],
            "verified": bool(row["verified"]),
            "meta": meta if isinstance(meta, dict) else {},
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "layer": LAYER_TOOL,
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# Re-export helpers for callers
__all__ = [
    "CompetenceRegistry",
    "CompetencePlan",
    "classify_task_competences",
    "preferred_tool_name",
    "format_competence_log",
]
