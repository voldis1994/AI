"""
Immutable TaskGoal — preserves the original USER REQUEST for the full cycle.

VERIFY and context mapping always ground against this object so learning/repair
cannot drift away from what the user asked for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# Generic placeholder / invented values — not task-specific file names.
_INVENTED_RE = re.compile(
    r"(?i)\b(?:user[_-]?provided(?:[_-]?\w+)?|"
    r"default(?:[_-]?(?:path|file|name|content|value|dir|output))?|"
    r"placeholder|changeme|your[_-]?name|todo|tbd|xxx+|dummy|"
    r"sample[_-]?(?:path|file)?|example[_-]?(?:path|file)?|"
    r"temp[_-]?file|untitled)\b"
)

_PATH_LIKE = re.compile(
    r"^(?:[A-Za-z]:)?[A-Za-z0-9_./\\-]+\.[A-Za-z0-9]{1,12}$"
)

# Function words / verbs — never treated as file paths or payload by themselves.
_STOP = {
    "a", "an", "the", "and", "or", "to", "for", "with", "from", "into",
    "in", "on", "at", "of", "by", "is", "are", "be", "as", "it", "this",
    "that", "create", "write", "make", "build", "run", "file", "files",
    "containing", "contains", "named", "called", "please", "jarvis",
    "text", "content", "contents", "data", "value", "values",
    "izveido", "uzraksti", "failu", "faila", "ar", "saturu", "satur",
    "nosaukumu", "tekstu", "teksts", "lūdzu", "ludzu",
}

# Canonical DIAGNOSE layers (legacy aliases accepted + normalized).
FAULT_LAYERS = frozenset({
    "goal_parsing",
    "context_mapping",
    "skill_code",
    "execution",
    "environment",
    "verifier",
})
_LEGACY_LAYER = {
    "context_args": "context_mapping",
    "test_harness": "execution",
    "dependency": "environment",
}


@dataclass(frozen=True)
class TaskGoal:
    """
    Frozen goal for one user task.

    user_request — exact original text (never mutated)
    goal         — planner/intent summary (may be shorter; VERIFY prefers user_request)
    constraints  — structured expectations derived only from the request (+ grounded args)
    """

    user_request: str
    goal: str
    constraints: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_request(
        cls,
        user_request: str,
        goal: Optional[str] = None,
        args: Optional[dict[str, Any]] = None,
    ) -> "TaskGoal":
        req = (user_request or "").strip()
        g = (goal or req).strip() or req
        grounded_args = cls.ground_args(args or {}, req)
        constraints = cls.derive_constraints(req, grounded_args)
        return cls(user_request=req, goal=g, constraints=constraints)

    def with_args(self, args: Optional[dict[str, Any]]) -> "TaskGoal":
        """Return a new TaskGoal with constraints refreshed from grounded args."""
        grounded = self.ground_args(args or {}, self.user_request)
        return TaskGoal(
            user_request=self.user_request,
            goal=self.goal,
            constraints=self.derive_constraints(self.user_request, grounded),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_request": self.user_request,
            "goal": self.goal,
            "constraints": self.constraints,
        }

    # ── Fault-layer helpers (universal) ─────────────────────────────────

    @staticmethod
    def normalize_fault_layer(layer: Any) -> str:
        raw = str(layer or "skill_code").strip().lower()
        raw = _LEGACY_LAYER.get(raw, raw)
        if raw not in FAULT_LAYERS:
            return "skill_code"
        return raw

    @classmethod
    def rewrite_skill_for_layer(cls, layer: Any) -> bool:
        """Skill rewrite is allowed ONLY when the fault is skill_code."""
        return cls.normalize_fault_layer(layer) == "skill_code"

    # ── Grounding helpers (universal) ───────────────────────────────────

    @classmethod
    def is_invented_default(cls, value: Any, request: str) -> bool:
        if value in (None, ""):
            return False
        sv = str(value)
        if cls.is_grounded(sv, request):
            return False
        return bool(_INVENTED_RE.search(sv))

    @staticmethod
    def is_grounded(value: Any, request: str) -> bool:
        if value in (None, ""):
            return False
        sv = str(value)
        req = request or ""
        if sv in req:
            return True
        base = Path(sv).name
        if base and base in req:
            return True
        # Case-insensitive containment for short tokens
        if len(sv) >= 2 and sv.lower() in req.lower():
            return True
        if base and len(base) >= 2 and base.lower() in req.lower():
            return True
        return False

    @classmethod
    def ground_args(cls, args: dict[str, Any], request: str) -> dict[str, Any]:
        """Keep only arg values present in / clearly derived from the USER REQUEST."""
        out: dict[str, Any] = {}
        for k, v in (args or {}).items():
            if v in (None, ""):
                continue
            if cls.is_invented_default(v, request):
                continue
            if not cls.is_grounded(v, request):
                # Allow numbers/bools always; strings must be grounded
                if isinstance(v, (int, float, bool)):
                    out[str(k)] = v
                continue
            out[str(k)] = v
        return out

    @staticmethod
    def looks_like_url(value: str) -> bool:
        return str(value).strip().lower().startswith(("http://", "https://"))

    @classmethod
    def looks_like_path(cls, value: str) -> bool:
        sv = str(value).strip()
        if not sv or cls.looks_like_url(sv):
            return False
        if _PATH_LIKE.match(sv):
            return True
        if ("/" in sv or "\\" in sv) and " " not in sv:
            return True
        return False

    @staticmethod
    def looks_like_content(value: str) -> bool:
        sv = str(value).strip()
        if not sv:
            return False
        if " " in sv or "\n" in sv or "\t" in sv:
            return True
        if len(sv) > 48:
            return True
        return False

    @classmethod
    def derive_constraints(
        cls, request: str, args: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """
        Conservative constraints from USER REQUEST + grounded args only.

        Does NOT treat every word in the sentence as a file path.
        Prefer: quoted strings, extension-bearing filenames, grounded args.
        """
        args = dict(args or {})
        expect: dict[str, Any] = {
            "files": [],
            "directories": [],
            "imports": [],
            "http": [],
            "must_contain": [],
            "check_process": True,
            "source": "task_goal",
        }

        path_vals: list[str] = []
        text_vals: list[str] = []

        # 1) Grounded structured args
        for _k, val in args.items():
            sv = str(val)
            if cls.looks_like_url(sv):
                expect["http"].append({"url": sv})
            elif cls.looks_like_path(sv):
                if sv not in path_vals:
                    path_vals.append(sv)
            elif cls.looks_like_content(sv):
                if sv not in text_vals:
                    text_vals.append(sv)
            else:
                # Bare token only if grounded (already filtered) — prefer content
                # when spaces absent but it's the payload; path only with extension
                if sv not in text_vals and sv not in path_vals:
                    text_vals.append(sv)

        # 2) Quoted strings from the request (highest-confidence literals)
        for a, b in re.findall(r"\"([^\"]+)\"|'([^']+)'", request or ""):
            tok = (a or b).strip()
            if not tok or cls.is_invented_default(tok, request):
                continue
            if cls.looks_like_url(tok):
                expect["http"].append({"url": tok})
            elif cls.looks_like_path(tok):
                if tok not in path_vals:
                    path_vals.append(tok)
            else:
                if tok not in text_vals:
                    text_vals.append(tok)

        # 3) Bare filenames with extensions in the request (not every word)
        for m in re.finditer(
            r"(?<![A-Za-z0-9_\"'])([A-Za-z0-9_./\\-]+\.[A-Za-z0-9]{1,12})\b",
            request or "",
        ):
            tok = m.group(1)
            if cls.is_invented_default(tok, request):
                continue
            if tok not in path_vals:
                path_vals.append(tok)

        # 4) key=value pairs in the request
        for km in re.finditer(
            r"(?P<k>[A-Za-z_][\w]*)\s*[:=]\s*(?P<v>\"[^\"]*\"|'[^']*'|[^\s,;]+)",
            request or "",
        ):
            raw = km.group("v")
            if (raw.startswith('"') and raw.endswith('"')) or (
                raw.startswith("'") and raw.endswith("'")
            ):
                raw = raw[1:-1]
            if cls.is_invented_default(raw, request):
                continue
            if cls.looks_like_path(raw) and raw not in path_vals:
                path_vals.append(raw)
            elif cls.looks_like_content(raw) and raw not in text_vals:
                text_vals.append(raw)
            elif raw not in text_vals and raw not in path_vals:
                # key=value is explicit — trust as content unless path-shaped
                text_vals.append(raw)

        # 5) Remaining significant tokens as CONTENT only (never as invented files).
        #    Extension-bearing / slash paths already captured above.
        for tok in re.findall(r"[A-Za-z0-9_./\\-]{2,}", request or ""):
            low = tok.lower()
            if low in _STOP:
                continue
            if cls.is_invented_default(tok, request):
                continue
            if cls.looks_like_url(tok) or cls.looks_like_path(tok):
                continue
            if tok in path_vals or tok in text_vals:
                continue
            # Bare identifier left in the sentence after path extraction → payload
            text_vals.append(tok)

        primary = text_vals[0] if text_vals else None
        for i, p in enumerate(path_vals):
            entry: dict[str, Any] = {"path": p, "min_bytes": 1}
            if i == 0 and primary is not None:
                entry["contains"] = primary
            expect["files"].append(entry)

        if primary is not None:
            for extra in text_vals[1:]:
                if len(str(extra)) >= 2:
                    expect["must_contain"].append(str(extra))
        elif text_vals and not path_vals:
            expect["must_contain"].extend(
                str(t) for t in text_vals if len(str(t)) >= 2
            )

        low = (request or "").lower()
        if any(w in low for w in ("directory", "folder", "katalog", "mapi")):
            for tok in re.findall(r"[A-Za-z0-9_./\\-]{2,}", request or ""):
                if ("/" in tok or "\\" in tok) and not Path(tok).suffix:
                    if not cls.is_invented_default(tok, request):
                        expect["directories"].append({"path": tok})

        return expect
