"""
Universal TaskContract ↔ skill capability compatibility.

Semantic / keyword / file-extension overlap alone is never enough to REUSE
or REPAIR a skill. Selection is based on required outcomes, behaviors,
artifacts (identity), constraints, and acceptance criteria from the
immutable TaskContract.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Optional

from jarvis.task_contract import TaskContract

# Tokens that never prove a skill matches a specific TaskContract.
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
        # Bare extensions never prove identity with a TaskContract artifact
        "txt", "md", "py", "json", "csv", "pdf", "html", "xml", "yml", "yaml",
        "log", "bin", "dat", "tmp", "zip",
    }
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{3,}")
_COMPAT_SCORE = 0.55
# Common file extensions — weak signals only (never sole REUSE evidence)
_EXTENSIONS = frozenset(
    {
        "txt", "md", "py", "json", "csv", "pdf", "html", "xml", "yml", "yaml",
        "log", "bin", "dat", "tmp", "zip", "jpg", "png", "gif", "svg", "js",
        "ts", "tsx", "jsx", "rs", "go", "java", "c", "cpp", "h", "hpp",
    }
)


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


def _path_identity_tokens(paths: list[str] | tuple[str, ...] | None) -> set[str]:
    """Artifact identity tokens — stems/parts only (NOT bare extensions)."""
    out: set[str] = set()
    for raw in paths or ():
        p = Path(str(raw).replace("\\", "/"))
        stem = p.stem.lower()
        if stem and stem not in _GENERIC and len(stem) >= 2:
            out.add(stem)
        for part in p.parts:
            part_l = part.lower().replace("-", "_")
            if part_l in (".", "..", "") or part_l in _GENERIC:
                continue
            if Path(part_l).suffix:
                # directory-like parts only; skip file basenames with ext here
                continue
            if len(part_l) >= 3:
                out.add(part_l.replace(".", "_"))
        for piece in re.split(r"[_\-.]+", stem):
            if piece and piece not in _GENERIC and len(piece) >= 3:
                out.add(piece)
    return out


def _path_extension_tokens(paths: list[str] | tuple[str, ...] | None) -> set[str]:
    out: set[str] = set()
    for raw in paths or ():
        suf = Path(str(raw).replace("\\", "/")).suffix.lower().lstrip(".")
        if suf and suf in _EXTENSIONS:
            out.add(suf)
    return out


def _path_tokens(paths: list[str] | tuple[str, ...] | None) -> set[str]:
    """Backward-compatible path tokens (identity + weak extensions)."""
    return _path_identity_tokens(paths) | _path_extension_tokens(paths)


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
                    toks |= _path_identity_tokens([str(item)])
            elif val:
                toks |= _tokens(str(val))
                toks |= _path_identity_tokens([str(val)])
        for key in ("description", "summary", "goal"):
            if meta.get(key):
                toks |= _tokens(str(meta.get(key)))
    return toks


def _goal_domain_tokens(contract: TaskContract) -> set[str]:
    toks = _tokens(contract.original_request)
    toks |= _tokens(contract.goal)
    toks |= _path_identity_tokens(list(contract.artifacts))
    for c in contract.content_requirements:
        toks |= _tokens(str(c))
    for a in contract.actions:
        if a not in _GENERIC and len(a) >= 4:
            toks.add(a.lower())
    for crit in contract.acceptance_criteria:
        if ":" in crit:
            body = crit.split(":", 1)[1]
            toks |= _path_identity_tokens([body])
            toks |= _tokens(body)
    for o in contract.required_outcomes or ():
        toks |= _tokens(str(o))
    for b in contract.behaviors or ():
        if isinstance(b, dict):
            toks |= _tokens(str(b.get("kind") or ""))
            toks |= _tokens(str(b.get("target") or ""))
    return toks


def _behavior_score(skill: dict[str, Any], contract: TaskContract, blob: str) -> float:
    behs = [
        b for b in (contract.behaviors or ())
        if isinstance(b, dict) and b.get("kind") and b.get("target")
    ]
    if not behs:
        return 0.4  # neutral when contract has no behaviors
    hay = blob.lower() + " " + str(skill.get("name") or "").lower()
    hits = 0
    for b in behs:
        kind = str(b.get("kind") or "").lower()
        target = str(b.get("target") or "").lower()
        if target and target in hay:
            hits += 1
        elif kind and kind.replace("_", " ") in hay:
            hits += 0.5
        elif target and any(t in hay for t in _tokens(target)):
            hits += 0.5
    return min(1.0, hits / max(1, len(behs)))


def _outcome_score(skill: dict[str, Any], contract: TaskContract, blob: str) -> float:
    """How well skill signals cover required outcomes (not exclusive labels)."""
    outcomes = [str(o).lower() for o in (contract.required_outcomes or ()) if str(o).strip()]
    if not outcomes:
        return 0.4
    hay = blob.lower()
    # Map outcomes to capability cues a skill may declare
    cues = {
        "artifact": ("write", "file", "create", "artifact", "path"),
        "execution": ("run", "execute", "exec", "perform"),
        "test": ("test",),
        "capability": ("capability", "skill", "tool"),
        "side_effect": ("http", "fetch", "network", "download"),
        "learning": ("learn", "knowledge", "research"),
        "research": ("research", "search"),
        "conversation": ("converse", "chat", "reply", "answer"),
        "final_output": ("format", "reply", "response"),
    }
    hits = 0.0
    for o in outcomes:
        if o in ("conversation", "final_output", "learning", "research"):
            # These are not skill-lifecycle outcomes — neutral
            hits += 0.5
            continue
        words = cues.get(o, (o,))
        if any(w in hay for w in words):
            hits += 1.0
        elif contract.artifacts and o in ("artifact", "execution", "capability", "test"):
            # Artifact-bearing contracts: skill must look file-capable
            if any(w in hay for w in ("write", "file", "path", "create")):
                hits += 0.7
    return min(1.0, hits / max(1, len(outcomes)))


def evaluate_capability(
    skill: dict[str, Any],
    contract: TaskContract,
) -> dict[str, Any]:
    """
    Score whether ``skill`` can fulfill ``contract``.

    Returns:
      skill, score (0..1), compatible (bool), reason (str), signals (dict)
    """
    name = str((skill or {}).get("name") or "")
    if not skill or not contract:
        return {
            "skill": name,
            "score": 0.0,
            "compatible": False,
            "reason": "missing skill or TaskContract",
            "signals": {},
        }

    meta = _read_skill_meta(skill)
    skill_toks = _skill_domain_tokens(skill, meta)
    goal_toks = _goal_domain_tokens(contract)
    goal_id = _path_identity_tokens(list(contract.artifacts))
    goal_ext = _path_extension_tokens(list(contract.artifacts))
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
    skill_id = _path_identity_tokens(skill_art_candidates)
    skill_ext = _path_extension_tokens(skill_art_candidates)

    # Action overlap
    actions = {a.lower() for a in (contract.actions or ()) if a}
    skill_action_hay = set(_tokens(blob)) | {
        p for p in re.split(r"[_\-\s]+", name.lower()) if p
    }
    if actions:
        action_hits = sum(1 for a in actions if a in skill_action_hay or a in blob.lower())
        action_score = action_hits / max(1, len(actions))
    else:
        action_score = 0.3

    # Artifact identity overlap — extensions alone cannot authorize REUSE
    if goal_id:
        hay_toks = _tokens(blob) | skill_toks | skill_id
        id_hits = len(goal_id & (skill_id | hay_toks))
        id_score = id_hits / max(1, len(goal_id))
        ext_bonus = 0.0
        if goal_ext and skill_ext and (goal_ext & skill_ext):
            ext_bonus = 0.15  # weak supporting signal only
        art_score = min(1.0, id_score + ext_bonus)
        # Extension-only (no stem/identity) → near-zero
        if id_hits == 0 and goal_ext and (goal_ext & skill_ext):
            art_score = 0.15
        elif id_hits == 0:
            art_score = 0.0
    elif contract.artifacts:
        # Artifacts present but no identity tokens (weird paths) — weak
        art_score = 0.2
    else:
        art_score = 0.4  # no artifact demand — neutral

    beh_score = _behavior_score(skill, contract, blob)
    outcome_score = _outcome_score(skill, contract, blob)

    # Distinctive domain overlap
    if skill_toks and goal_toks:
        overlap = skill_toks & goal_toks
        denom = max(1, min(len(skill_toks), len(goal_toks)))
        domain_score = len(overlap) / denom
    elif not skill_toks and not goal_toks:
        domain_score = 0.2
    else:
        domain_score = 0.0

    content_score = 0.5
    if contract.content_requirements:
        content_hits = 0
        for c in contract.content_requirements:
            cl = str(c).lower()
            if len(cl) >= 4 and cl in blob.lower():
                content_hits += 1
            elif _tokens(cl) & skill_toks:
                content_hits += 1
        content_score = content_hits / max(1, len(contract.content_requirements))

    # Outcomes/behaviors/artifacts dominate; bare keywords/extensions do not
    score = (
        0.25 * domain_score
        + 0.30 * art_score
        + 0.15 * action_score
        + 0.10 * content_score
        + 0.10 * beh_score
        + 0.10 * outcome_score
    )

    reasons: list[str] = []
    compatible = True

    if skill_toks and goal_toks and len(skill_toks & goal_toks) == 0:
        if goal_id and art_score < 0.34:
            compatible = False
            reasons.append("domain tokens disjoint and artifacts uncovered")
        elif not goal_id and domain_score < 0.34 and action_score < 0.5:
            compatible = False
            reasons.append("domain tokens disjoint; semantic/language overlap only")

    if goal_id and art_score < 0.34 and domain_score < 0.5:
        compatible = False
        reasons.append("TaskContract artifacts not covered by skill")

    # Behavior targets required but skill shows no awareness
    if contract.behaviors and beh_score < 0.25 and art_score < 0.34:
        compatible = False
        reasons.append("TaskContract behaviors not covered by skill")

    if score < _COMPAT_SCORE:
        compatible = False
        if not reasons:
            reasons.append(f"score {score:.2f} below {_COMPAT_SCORE:.2f}")

    strong = (
        art_score >= 0.5
        or domain_score >= 0.5
        or beh_score >= 0.5
        or (action_score >= 0.5 and domain_score >= 0.34)
    )
    if compatible and not strong:
        compatible = False
        reasons.append("no strong action/artifact/behavior/domain signal")

    if compatible and not reasons:
        reasons.append("outcomes/behaviors/artifacts/domain align with TaskContract")

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
            "behavior_score": round(beh_score, 3),
            "outcome_score": round(outcome_score, 3),
            "skill_tokens": sorted(skill_toks)[:12],
            "goal_tokens": sorted(goal_toks)[:12],
            "goal_artifacts": list(contract.artifacts)[:8],
            "required_outcomes": list(contract.required_outcomes or ())[:8],
        },
    }


def outcome_capability_mismatch(
    skill: Optional[dict[str, Any]],
    contract: TaskContract,
) -> bool:
    """
    True when a selected skill cannot cover locked TaskContract outcomes/artifacts.

    Requires an inspectable skill (name + some meta/path/capabilities). Without
    that evidence, DIAGNOSE must not invent capability_mismatch — content/VERIFY
    failures fall through to skill_code / unknown with rewrite gates.
    """
    if not skill or not contract:
        return False
    name = str(skill.get("name") or "").strip()
    if not name:
        return False
    has_signals = bool(
        skill.get("capabilities")
        or skill.get("file_path")
        or skill.get("pending_path")
        or str(skill.get("description") or "").strip()
    )
    if not has_signals:
        return False
    match = evaluate_capability(skill, contract)
    return not bool(match.get("compatible"))


def verify_implies_capability_mismatch(
    verification: dict[str, Any],
    skill: dict[str, Any],
    contract: TaskContract,
) -> bool:
    """
    If VERIFY failed because required artifacts/content were not produced,
    re-check capability fit before REPAIR. Incompatible → BUILD_NEW.
    """
    match = evaluate_capability(skill, contract)
    if not match["compatible"]:
        return True
    signals = match.get("signals") or {}
    domain = float(signals.get("domain_score") or 0.0)
    art = float(signals.get("artifact_score") or 0.0)
    beh = float(signals.get("behavior_score") or 0.0)
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
    if missing_artifact and beh < 0.25 and art < 0.34:
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
