"""
Hierarchical competence model for JARVIS.

Layers (self-developing, not task-hardcoded):
  SKILL       — learned competence / knowledge domain (tree node)
  CAPABILITY  — what JARVIS can do (action class under a skill)
  TOOL        — executable code / API (backed by skill .py + registry)
  KNOWLEDGE   — verified knowledge attached to a skill domain
  EXPERIENCE  — verified experience and failures attached to a skill/tool

TaskGoal remains the immutable source of truth. Classification derives
competence paths from actions / artifact kinds / subject — never from
invented task-specific keywords.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from jarvis.task_goal import SUBJECT_AMBIGUOUS, SUBJECT_UNKNOWN, TaskGoal


# ── Layer labels ───────────────────────────────────────────────────────
LAYER_SKILL = "skill"
LAYER_CAPABILITY = "capability"
LAYER_TOOL = "tool"
LAYER_KNOWLEDGE = "knowledge"
LAYER_EXPERIENCE = "experience"

# Generic action → capability class (verbs only — no filenames/tasks).
_ACTION_CAPABILITY = {
    "create": "io.create",
    "write": "io.write",
    "make": "io.create",
    "build": "io.create",
    "run": "exec.run",
    "execute": "exec.run",
    "fetch": "net.fetch",
    "download": "net.fetch",
    "install": "env.install",
    "learn": "learn.topic",
    "research": "learn.research",
    "test": "exec.test",
    "verify": "exec.verify",
    "summarize": "text.summarize",
    "summary": "text.summarize",
    "izveido": "io.create",
    "uzraksti": "io.write",
    "palaid": "exec.run",
    "iemacies": "learn.topic",
    "iemācies": "learn.topic",
    "paradi": "text.show",
    "parādi": "text.show",
}

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, limit: int = 48) -> str:
    s = _SLUG_RE.sub("_", (text or "").lower()).strip("_")
    return (s or "node")[:limit]


def artifact_kind(value: str) -> str:
    """Classify an artifact path/url into a generic kind."""
    sv = str(value or "").strip()
    if not sv:
        return "unknown"
    low = sv.lower()
    if low.startswith(("http://", "https://")):
        return "url"
    p = Path(sv.replace("\\", "/"))
    if p.suffix:
        return "file"
    if "/" in sv or "\\" in sv:
        return "path"
    return "token"


def capability_for_action(action: str) -> str:
    a = (action or "").strip().lower()
    return _ACTION_CAPABILITY.get(a, f"act.{slugify(a, 24)}" if a else "act.unknown")


def skill_branch_for_artifact_kind(kind: str) -> str:
    """Top-level skill domain from artifact kind (generic)."""
    return {
        "file": "io.files",
        "path": "io.paths",
        "url": "net.http",
        "token": "io.generic",
        "unknown": "general",
    }.get(kind, "general")


@dataclass(frozen=True)
class CompetenceRef:
    """One node in the competence tree."""

    id: str
    title: str
    parent_id: str = ""
    kind: str = LAYER_SKILL  # skill | capability
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "parent_id": self.parent_id,
            "kind": self.kind,
            "meta": dict(self.meta or {}),
        }


@dataclass
class CompetencePlan:
    """
    Multi-competence plan for one TaskGoal.

    tools — reusable TOOL records (skill rows) already VERIFIED/ACTIVE
    open_branches — skill/capability ids to create
    needs_new_tool — gap under an existing competence (extend, don't new-skill-slug)
    """

    task_goal: dict[str, Any]
    skill_ids: list[str] = field(default_factory=list)
    capability_ids: list[str] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    open_branches: list[str] = field(default_factory=list)
    needs_new_tool: bool = False
    decision: str = "BUILD_TOOL"  # REUSE_COMPOSE | EXTEND | BUILD_TOOL | OPEN_BRANCH
    preferred_tool_name: str = ""
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_ids": list(self.skill_ids),
            "capability_ids": list(self.capability_ids),
            "tools": [
                {
                    "name": t.get("name"),
                    "competence_id": t.get("competence_id"),
                    "capability": t.get("capability"),
                    "status": t.get("status"),
                }
                for t in self.tools[:12]
            ],
            "open_branches": list(self.open_branches),
            "needs_new_tool": self.needs_new_tool,
            "decision": self.decision,
            "preferred_tool_name": self.preferred_tool_name,
            "reasons": list(self.reasons)[:12],
        }


def classify_task_competences(task_goal: TaskGoal) -> dict[str, Any]:
    """
    Classify a TaskGoal into skill domains + capability classes.

    Universal rules only:
      - actions → capability ids
      - artifact kinds → skill branches
      - subject tokens → optional subject sub-branch (surface form, not invented sense)
    """
    if not isinstance(task_goal, TaskGoal):
        raise TypeError("task_goal must be TaskGoal")

    actions = list(task_goal.actions or ()) or ["execute"]
    capabilities = []
    for a in actions:
        cap = capability_for_action(a)
        if cap not in capabilities:
            capabilities.append(cap)

    kinds = {artifact_kind(a) for a in (task_goal.artifacts or ())}
    # content_source URLs/paths/memory are origins — open matching skill domains
    for src in task_goal.content_source or ():
        if str(src).lower() == "memory":
            kinds.add("memory")
            continue
        kinds.add(artifact_kind(str(src)))
    if not kinds:
        kinds = {"unknown"}
    skill_roots = []
    for k in sorted(kinds):
        branch = (
            "learn.memory"
            if k == "memory"
            else skill_branch_for_artifact_kind(k)
        )
        if branch not in skill_roots:
            skill_roots.append(branch)

    # Capability prefixes may open additional skill domains (multi-competence)
    for cap in capabilities:
        if cap.startswith("net.") and "net.http" not in skill_roots:
            skill_roots.append("net.http")
        elif cap.startswith("learn.") and "learn.topic" not in skill_roots:
            skill_roots.append("learn.topic")
        elif cap.startswith("exec.") and "exec.runtime" not in skill_roots:
            skill_roots.append("exec.runtime")

    # Subject sub-branch — keep surface tokens; never invent meaning
    subject = (task_goal.subject or "").strip()
    subject_branch = ""
    if subject and subject not in (SUBJECT_UNKNOWN, SUBJECT_AMBIGUOUS, ""):
        # Use at most 3 content tokens for a stable sub-competence id
        toks = [
            t for t in re.findall(r"[A-Za-z0-9_]{3,}", subject.lower())
            if t not in {"about", "with", "from", "using"}
        ][:3]
        if len(toks) >= 2:
            subject_branch = "subject." + slugify("_".join(toks), 40)
        elif len(toks) == 1:
            # Single token is ambiguous — do not open a specialized branch
            subject_branch = ""

    skill_ids: list[str] = []
    for root in skill_roots:
        skill_ids.append(root)
        if subject_branch:
            skill_ids.append(f"{root}.{subject_branch}")

    # Capability nodes hang under the matching skill domain (multi-competence)
    primary = skill_roots[0] if skill_roots else "general"

    def _root_for_cap(cap: str) -> str:
        if cap.startswith("net."):
            return "net.http"
        if cap.startswith("learn."):
            return "learn.topic"
        if cap.startswith("exec."):
            return "exec.runtime"
        if cap.startswith("text."):
            return "text.language"
        return primary

    capability_ids = []
    for c in capabilities:
        cid = f"{_root_for_cap(c)}/{c}"
        if cid not in capability_ids:
            capability_ids.append(cid)
        root = _root_for_cap(c)
        if root not in skill_ids:
            skill_ids.append(root)
            if root not in skill_roots:
                skill_roots.append(root)

    return {
        "skill_ids": skill_ids,
        "capability_ids": capability_ids,
        "capabilities": capabilities,
        "artifact_kinds": sorted(kinds),
        "subject_branch": subject_branch,
        "primary_skill": primary,
        "actions": actions,
    }


def preferred_tool_name(task_goal: TaskGoal, classification: Optional[dict] = None) -> str:
    """
    Stable tool name from competence + capability (+ artifact class) —
    NOT a slug of the full request or a specific filename.

    Prevents every codegen from becoming a new task-specific skill, while
    still separating tool variants by artifact kind/extension class so that
    extending ``io.create`` for ``.md`` does not overwrite a ``.txt`` tool.
    """
    clf = classification or classify_task_competences(task_goal)
    primary = str(clf.get("primary_skill") or "general").replace(".", "_")
    caps = list(clf.get("capabilities") or ["act.unknown"])
    cap = str(caps[0]).replace(".", "_")
    # Artifact class only (extension / kind) — never the basename/stem
    art_class = ""
    for a in task_goal.artifacts or ():
        suf = Path(str(a)).suffix.lower().lstrip(".")
        if suf and suf.isalnum() and len(suf) <= 12:
            art_class = suf
            break
    if not art_class:
        kinds = list(clf.get("artifact_kinds") or [])
        if kinds and kinds[0] not in ("unknown", "token"):
            art_class = str(kinds[0])
    if art_class:
        return slugify(f"{primary}_{cap}_{art_class}", 40)
    return slugify(f"{primary}_{cap}", 40)


def format_competence_log(plan: CompetencePlan) -> str:
    tools = ",".join(t.get("name") or "?" for t in plan.tools[:4]) or "-"
    return (
        f"COMPETENCE DECISION: {plan.decision} "
        f"skills={plan.skill_ids[:4]} caps={plan.capability_ids[:4]} "
        f"tools=[{tools}] open={plan.open_branches[:4]} "
        f"new_tool={plan.needs_new_tool}"
    )
