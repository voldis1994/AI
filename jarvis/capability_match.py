"""
Universal TaskGoal ↔ skill capability compatibility.

Semantic / keyword overlap alone is never enough to REUSE or REPAIR a skill.
Selection is based on actions, artifacts, constraints, content requirements
and success criteria derived from the TaskGoal.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Optional

from jarvis.task_goal import TaskGoal

# Tokens that never prove a skill matches a specific TaskGoal.
_GENERIC = frozenset(
    {
        "create", "write", "make", "build", "file", "files", "skill", "skills",
        "data", "test", "tests", "run", "exec", "execute", "handle", "new",
        "with", "that", "this", "from", "into", "please", "using", "about",
        "learn", "study", "basics", "knowledge", "result", "output", "input",
        "path", "name", "text", "content", "value", "item", "items", "task",
        "goal", "args", "meta", "code", "coding", "program", "programming",
        "script", "scripts", "function", "module", "package", "library",
        "python", "java", "javascript", "typescript", "rust", "golang",
        "izveido", "uzraksti", "failu", "saturu", "izpildi", "uzdevums",
        "pamatus", "iemacies", "macities", "programma", "programmesanu",
        "true", "false", "none", "null", "string", "number", "object",
        "active", "candidate", "broken", "version", "jarvis",
    }
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{3,}")
_COMPAT_SCORE = 0.55


def _tokens(text: str) -> set[str]:
    out: set[str] = set()
    for m in _TOKEN_RE.finditer(str(text or "").lower()):
        t = m.group(0)
        if t in _GENERIC or t.isdigit():
            continue
        if len(t) < 4 and t not in {"zip", "csv", "pdf", "url", "api", "sql", "json", "xml", "html", "http"}:
            continue
        out.add(t)
    return out


def _path_tokens(paths: list[str] | tuple[str, ...]) -> set[str]:
    out: set[str] = set()
    for raw in paths or ():
        p = Path(str(raw).replace("\\", "/"))
        stem = p.stem.lower()
        if stem and stem not in _GENERIC and len(stem) >= 2:
            out.add(stem)
        suf = p.suffix.lower().lstrip(".")
        if suf:
            out.add(suf)
        for part in p.parts:
            part_l = part.lower().replace("-", "_")
            if part_l in (".", "..", "") or part_l in _GENERIC:
                continue
            if Path(part_l).suffix:
                continue
            if len(part_l) >= 3:
                out.add(part_l.replace(".", "_"))
        # split stem on _ -
        for piece in re.split(r"[_\-.]+", stem):
            if piece and piece not in _GENERIC and len(piece) >= 3:
                out.add(piece)
    return out


def _skill_text_blob(skill: dict[str, Any]) -> str:
    caps = skill.get("capabilities") or []
    if not isinstance(caps, (list, tuple)):
        caps = [str(caps)]
    return " ".join(
        [
            str(skill.get("name") or ""),
            str(skill.get("description") or ""),
            " ".join(str(c) for c in caps),
        ]
    )


def _read_skill_meta(skill: dict[str, Any]) -> dict[str, Any]:
    """Best-effort SKILL_META from disk (inputs/outputs/capabilities)."""
    path = skill.get("file_path") or skill.get("pending_path")
    if not path:
        return {}
    try:
        src = Path(str(path)).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}
    try:
        tree = ast.parse(src)
    except Exception:
        return {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "SKILL_META":
                    try:
                        return ast.literal_eval(node.value)
                    except Exception:
                        return {}
    # Fallback: scrape capability-ish strings from source (no execution)
    meta: dict[str, Any] = {}
    m = re.search(r"SKILL_META\s*=\s*(\{[\s\S]*?\n\})", src)
    if m:
        try:
            meta = ast.literal_eval(m.group(1))
        except Exception:
            pass
    return meta if isinstance(meta, dict) else {}


def _skill_domain_tokens(skill: dict[str, Any], meta: Optional[dict] = None) -> set[str]:
    meta = meta if meta is not None else _read_skill_meta(skill)
    blob = _skill_text_blob(skill)
    toks = _tokens(blob)
    # Name parts are strong specialization signals
    name = str(skill.get("name") or "")
    for part in re.split(r"[_\-\s]+", name.lower()):
        if part and part not in _GENERIC and len(part) >= 4:
            toks.add(part)
    if isinstance(meta, dict):
        for key in ("capabilities", "tags", "outputs", "produces", "artifacts"):
            val = meta.get(key)
            if isinstance(val, (list, tuple)):
                for item in val:
                    toks |= _tokens(str(item))
                    toks |= _path_tokens([str(item)])
            elif val:
                toks |= _tokens(str(val))
                toks |= _path_tokens([str(val)])
        for key in ("description", "summary", "goal"):
            if meta.get(key):
                toks |= _tokens(str(meta.get(key)))
    return toks


def _goal_domain_tokens(task_goal: TaskGoal) -> set[str]:
    toks = _tokens(task_goal.user_request)
    toks |= _tokens(task_goal.goal)
    toks |= _path_tokens(list(task_goal.artifacts))
    for c in task_goal.content_requirements:
        # short content cues still matter as domain markers when long enough
        toks |= _tokens(str(c))
    for a in task_goal.actions:
        if a not in _GENERIC and len(a) >= 4:
            toks.add(a.lower())
    for crit in task_goal.success_criteria:
        if ":" in crit:
            toks |= _path_tokens([crit.split(":", 1)[1]])
            toks |= _tokens(crit.split(":", 1)[1])
    return toks


def evaluate_capability(
    skill: dict[str, Any],
    task_goal: TaskGoal,
) -> dict[str, Any]:
    """
    Score whether ``skill`` can fulfill ``task_goal``.

    Returns:
      skill, score (0..1), compatible (bool), reason (str), signals (dict)
    """
    name = str((skill or {}).get("name") or "")
    if not skill or not task_goal:
        return {
            "skill": name,
            "score": 0.0,
            "compatible": False,
            "reason": "missing skill or TaskGoal",
            "signals": {},
        }

    meta = _read_skill_meta(skill)
    skill_toks = _skill_domain_tokens(skill, meta)
    goal_toks = _goal_domain_tokens(task_goal)
    goal_arts = _path_tokens(list(task_goal.artifacts))
    # Artifacts claimed by skill meta / description paths
    blob = _skill_text_blob(skill)
    skill_art_candidates: list[str] = list(
        re.findall(r"[\w./\\-]+\.[A-Za-z0-9]{1,8}", blob)
    )
    if isinstance(meta, dict):
        for key in ("artifacts", "outputs", "produces", "files"):
            val = meta.get(key)
            if isinstance(val, (list, tuple)):
                skill_art_candidates.extend(str(x) for x in val)
            elif isinstance(val, dict):
                skill_art_candidates.extend(str(x) for x in val.values())
            elif val:
                skill_art_candidates.append(str(val))
    skill_arts = _path_tokens(skill_art_candidates)

    # Action overlap
    actions = {a.lower() for a in (task_goal.actions or ()) if a}
    skill_action_hay = set(_tokens(blob)) | {
        p for p in re.split(r"[_\-\s]+", name.lower()) if p
    }
    if actions:
        action_hits = sum(1 for a in actions if a in skill_action_hay or a in blob.lower())
        action_score = action_hits / max(1, len(actions))
    else:
        action_score = 0.3

    # Artifact overlap — critical when TaskGoal demands concrete artifacts
    if goal_arts:
        if skill_arts:
            art_hits = len(goal_arts & skill_arts)
            art_score = art_hits / max(1, len(goal_arts))
        else:
            # Skill does not declare artifacts — require domain token overlap
            # with path stems in skill hay
            hay_toks = _tokens(blob) | skill_toks
            art_hits = len(goal_arts & hay_toks)
            art_score = art_hits / max(1, len(goal_arts))
    else:
        art_score = 0.4  # no artifact demand — neutral

    # Distinctive domain overlap (excludes language/generic tokens)
    if skill_toks and goal_toks:
        overlap = skill_toks & goal_toks
        denom = max(1, min(len(skill_toks), len(goal_toks)))
        domain_score = len(overlap) / denom
    elif not skill_toks and not goal_toks:
        domain_score = 0.2
    else:
        domain_score = 0.0

    # Content requirements: if skill description hard-codes unrelated content cue
    content_score = 0.5
    if task_goal.content_requirements:
        content_hits = 0
        for c in task_goal.content_requirements:
            cl = str(c).lower()
            if len(cl) >= 4 and cl in blob.lower():
                content_hits += 1
            else:
                # token overlap with content
                if _tokens(cl) & skill_toks:
                    content_hits += 1
        content_score = content_hits / max(1, len(task_goal.content_requirements))

    # Weighted score — artifacts + domain dominate; actions support
    score = (
        0.35 * domain_score
        + 0.35 * art_score
        + 0.20 * action_score
        + 0.10 * content_score
    )

    reasons: list[str] = []
    compatible = True

    # Veto: skill specialized for a different domain (distinctive tokens disjoint)
    if skill_toks and goal_toks and len(skill_toks & goal_toks) == 0:
        if goal_arts and art_score < 0.34:
            compatible = False
            reasons.append("domain tokens disjoint and artifacts uncovered")
        elif not goal_arts and domain_score < 0.34 and action_score < 0.5:
            compatible = False
            reasons.append("domain tokens disjoint; semantic/language overlap only")

    # Veto: TaskGoal asks for artifacts the skill never relates to
    if goal_arts and art_score < 0.34 and domain_score < 0.5:
        compatible = False
        reasons.append("TaskGoal artifacts not covered by skill")

    # Veto: name similarity alone / weak score
    if score < _COMPAT_SCORE:
        compatible = False
        if not reasons:
            reasons.append(f"score {score:.2f} below {_COMPAT_SCORE:.2f}")

    # Require at least one strong signal for REUSE
    strong = (
        art_score >= 0.5
        or domain_score >= 0.5
        or (action_score >= 0.5 and domain_score >= 0.34)
    )
    if compatible and not strong:
        compatible = False
        reasons.append("no strong action/artifact/domain signal")

    if compatible and not reasons:
        reasons.append("actions/artifacts/domain align with TaskGoal")

    return {
        "skill": name,
        "score": round(float(score), 3),
        "compatible": bool(compatible),
        "reason": "; ".join(reasons),
        "signals": {
            "domain_score": round(domain_score, 3),
            "artifact_score": round(art_score, 3),
            "action_score": round(action_score, 3),
            "content_score": round(content_score, 3),
            "skill_tokens": sorted(skill_toks)[:12],
            "goal_tokens": sorted(goal_toks)[:12],
            "goal_artifacts": list(task_goal.artifacts)[:8],
        },
    }


def verify_implies_capability_mismatch(
    verification: dict[str, Any],
    skill: dict[str, Any],
    task_goal: TaskGoal,
) -> bool:
    """
    If VERIFY failed because required artifacts/content were not produced,
    re-check capability fit before REPAIR. Incompatible → BUILD_NEW.
    """
    match = evaluate_capability(skill, task_goal)
    if not match["compatible"]:
        return True
    # Compatible skill with missing artifacts → execution/repair problem, not mismatch.
    # Only escalate when domain signals show the skill was never for this TaskGoal.
    signals = match.get("signals") or {}
    domain = float(signals.get("domain_score") or 0.0)
    art = float(signals.get("artifact_score") or 0.0)
    reason = str(verification.get("reason") or "").lower()
    missing_artifact = any(
        k in reason
        for k in ("missing", "not found", "does not exist", "no such file", "artifact")
    )
    checks = verification.get("checks") or []
    if isinstance(checks, list):
        for ch in checks:
            if not isinstance(ch, dict) or ch.get("ok") is not False:
                continue
            name = str(ch.get("name") or ch.get("check") or "").lower()
            detail = str(ch.get("detail") or ch.get("reason") or "").lower()
            if any(
                k in name or k in detail
                for k in ("file", "artifact", "missing", "exist", "path", "contains")
            ):
                missing_artifact = True
                break
    if missing_artifact and domain < 0.34 and art < 0.34:
        return True
    return False


def format_match_log(match: dict[str, Any]) -> str:
    return (
        f"CAPABILITY MATCH: skill={match.get('skill')!s} "
        f"score={match.get('score')} "
        f"compatible={match.get('compatible')} "
        f"reason={match.get('reason')}"
    )


def format_decision_log(decision: str, *, skill: str = "", detail: str = "") -> str:
    extra = f" skill={skill}" if skill else ""
    more = f" ({detail})" if detail else ""
    return f"CAPABILITY DECISION: {decision}{extra}{more}"
