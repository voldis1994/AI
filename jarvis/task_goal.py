"""
Immutable TaskGoal — preserves the original USER REQUEST for the full cycle.

VERIFY and context mapping always ground against this object so learning/repair
cannot drift away from what the user asked for.

Structured fields (actions / artifacts / content requirements / success criteria)
are derived from the request + grounded schema args — never by splitting the
sentence into dozens of word-args.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from jarvis.request_items import (
    classify_request_items,
    path_segment_tokens,
    sanitize_libraries,
)


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
    "nosaukumu", "tekstu", "teksts", "lūdzu", "ludzu", "ka", "kā",
    "par", "paradi", "parādi", "man", "lai", "un", "vai", "bet",
    "jo", "ja", "ko", "kas", "kur", "kad", "tik", "tai", "tos", "tas",
    "šo", "so", "ti", "tu", "es", "mēs", "mes", "jūs", "jus",
    "example", "examples", "demo", "show", "how", "works", "working",
    "python", "code", "script", "program",
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

# Action verbs used only to label high-level actions (not as args).
_ACTION_VERBS = re.compile(
    r"(?i)\b(create|write|make|build|run|fetch|download|install|learn|"
    r"research|execute|test|verify|izveido|uzraksti|palaid|iemācies|"
    r"iemacies|paradi|parādi)\b"
)


@dataclass(frozen=True)
class TaskGoal:
    """
    Frozen goal for one user task.

    user_request — exact original text (never mutated)
    goal         — planner/intent summary (may be shorter; VERIFY prefers user_request)
    constraints  — structured expectations derived only from the request (+ grounded args)
    actions / artifacts / content_requirements / success_criteria — structured views
    """

    user_request: str
    goal: str
    constraints: dict[str, Any] = field(default_factory=dict)
    actions: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    content_requirements: tuple[str, ...] = ()
    success_criteria: tuple[str, ...] = ()

    @classmethod
    def from_request(
        cls,
        user_request: str,
        goal: Optional[str] = None,
        args: Optional[dict[str, Any]] = None,
        skill_meta: Optional[dict[str, Any]] = None,
    ) -> "TaskGoal":
        req = (user_request or "").strip()
        g = (goal or req).strip() or req
        grounded_args = cls.sanitize_args(args or {}, req, skill_meta=skill_meta)
        constraints = cls.derive_constraints(req, grounded_args)
        return cls(
            user_request=req,
            goal=g,
            constraints=constraints,
            actions=cls._derive_actions(req),
            artifacts=cls._derive_artifacts(constraints),
            content_requirements=cls._derive_content_requirements(constraints),
            success_criteria=cls._derive_success_criteria(constraints),
        )

    def with_args(
        self,
        args: Optional[dict[str, Any]],
        skill_meta: Optional[dict[str, Any]] = None,
    ) -> "TaskGoal":
        """Return a new TaskGoal with constraints refreshed from grounded args."""
        grounded = self.sanitize_args(
            args or {}, self.user_request, skill_meta=skill_meta
        )
        constraints = self.derive_constraints(self.user_request, grounded)
        return TaskGoal(
            user_request=self.user_request,
            goal=self.goal,
            constraints=constraints,
            actions=self.actions or self._derive_actions(self.user_request),
            artifacts=self._derive_artifacts(constraints),
            content_requirements=self._derive_content_requirements(constraints),
            success_criteria=self._derive_success_criteria(constraints),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_request": self.user_request,
            "goal": self.goal,
            "constraints": self.constraints,
            "actions": list(self.actions),
            "artifacts": list(self.artifacts),
            "content_requirements": list(self.content_requirements),
            "success_criteria": list(self.success_criteria),
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
    def request_tokens(cls, request: str) -> set[str]:
        return {
            t.lower()
            for t in re.findall(r"[A-Za-z0-9_./\\-]{2,}", request or "")
            if t.lower() not in _STOP
        }

    @classmethod
    def path_segment_tokens(
        cls,
        request: str,
        artifacts: Optional[tuple[str, ...] | list[str]] = None,
    ) -> set[str]:
        """Stems/dirs from path-like request tokens — never deps or arg keys."""
        return path_segment_tokens(request, artifacts)

    @classmethod
    def classify_items(
        cls,
        request: str,
        *,
        args: Optional[dict[str, Any]] = None,
        artifacts: Optional[tuple[str, ...] | list[str]] = None,
    ) -> list:
        """PATH/FILE/LIBRARY/COMMAND/CONTENT/REQUIREMENT classification."""
        return classify_request_items(
            request, grounded_args=args, artifacts=artifacts
        )

    @classmethod
    def sanitize_libraries(
        cls,
        names: Optional[list],
        request: str,
        *,
        artifacts: Optional[tuple[str, ...] | list[str]] = None,
        skill_code: Optional[str] = None,
    ) -> list[str]:
        return sanitize_libraries(
            names, request, artifacts=artifacts, skill_code=skill_code
        )

    @classmethod
    def is_polluted_arg_key(
        cls,
        key: str,
        request: str,
        *,
        schema_keys: Optional[set[str]] = None,
    ) -> bool:
        """
        True when a key looks like a request content-token promoted to an arg name.

        Schema keys and explicit key=value names in the request are never polluted.
        Path stems (e.g. calculator from workspace/calculator.py) are always polluted.
        """
        k = str(key or "").strip()
        if not k:
            return True
        if schema_keys and k in schema_keys:
            return False
        if re.search(rf"(?i)\b{re.escape(k)}\s*[:=]", request or ""):
            return False
        # Path segments of output files must never become arg keys / deps
        if k.lower() in cls.path_segment_tokens(request):
            return True
        # Very short keys that appear as bare words in the request are needles
        tokens = cls.request_tokens(request)
        # Also include stop-filtered scan so 'ti' (stopword) is still caught
        bare = {t.lower() for t in re.findall(r"[A-Za-z0-9_./\\-]{2,}", request or "")}
        if k.lower() in bare and k.lower() not in {s.lower() for s in (schema_keys or set())}:
            # Allow only if it is an explicit schema-shaped name not from request tokens
            # Keys that appear in the request as words are polluted unless schema-declared.
            return True
        if k.lower() in tokens:
            return True
        return False

    @classmethod
    def sanitize_args(
        cls,
        args: dict[str, Any],
        request: str,
        *,
        skill_meta: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Ground values AND drop polluted word-keys from the request sentence."""
        schema: set[str] = set()
        if isinstance(skill_meta, dict):
            for src in (
                skill_meta.get("required_args"),
                skill_meta.get("input_schema"),
                list((skill_meta.get("args_schema") or {}).keys())
                if isinstance(skill_meta.get("args_schema"), dict)
                else [],
            ):
                if isinstance(src, list):
                    schema.update(str(x) for x in src if x)
                elif isinstance(src, dict):
                    schema.update(str(x) for x in src.keys())
        grounded = cls.ground_args(args or {}, request)
        out: dict[str, Any] = {}
        for k, v in grounded.items():
            if cls.is_polluted_arg_key(k, request, schema_keys=schema):
                continue
            out[str(k)] = v
        return out

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

        Does NOT treat every word in the sentence as a file path or must_contain
        needle. Prefer: quoted strings, extension-bearing filenames, grounded
        schema args, explicit key=value pairs.
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

        # 1) Grounded structured args (schema keys / explicit mappings)
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
                # Bare token from a schema arg — treat as content payload only
                # when paired with a path, otherwise keep as soft content.
                if sv not in text_vals and sv not in path_vals and len(sv) >= 2:
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
                text_vals.append(raw)

        # 5) Content after explicit cue words (NOT every sentence token).
        #    Prevents "ti"/"Python"/… must_contain pollution while still
        #    capturing payloads like: containing HELLO / ar tekstu DARBOJAS.
        if not text_vals:
            for m in re.finditer(
                r"(?i)(?:containing|contains|with\s+text|text|content|"
                r"tekstu|saturu|ar\s+tekstu)\s+[\"']?"
                r"([A-Za-z0-9_./\\-]{2,})",
                request or "",
            ):
                tok = m.group(1)
                if tok.lower() in _STOP:
                    continue
                if cls.is_invented_default(tok, request) or cls.looks_like_path(tok):
                    continue
                if tok not in text_vals:
                    text_vals.append(tok)
                    break

        # 6) If we have a path but still no payload, take at most ONE remaining
        #    significant token (≥3 chars) that is not a stopword/path.
        if path_vals and not text_vals:
            leftovers: list[str] = []
            for tok in re.findall(r"[A-Za-z0-9_./\\-]{3,}", request or ""):
                low = tok.lower()
                if low in _STOP:
                    continue
                if cls.is_invented_default(tok, request) or cls.looks_like_path(tok):
                    continue
                if tok in path_vals:
                    continue
                leftovers.append(tok)
            if len(leftovers) == 1:
                text_vals.append(leftovers[0])

        primary = text_vals[0] if text_vals else None
        for i, p in enumerate(path_vals):
            entry: dict[str, Any] = {"path": p, "min_bytes": 1}
            if i == 0 and primary is not None:
                entry["contains"] = primary
            expect["files"].append(entry)

        if primary is not None:
            for extra in text_vals[1:]:
                # Only multi-word / substantial extras become must_contain
                if cls.looks_like_content(str(extra)) or len(str(extra)) >= 4:
                    expect["must_contain"].append(str(extra))
        elif text_vals and not path_vals:
            expect["must_contain"].extend(
                str(t) for t in text_vals
                if cls.looks_like_content(str(t)) or len(str(t)) >= 4
            )

        low = (request or "").lower()
        if any(w in low for w in ("directory", "folder", "katalog", "mapi")):
            for tok in re.findall(r"[A-Za-z0-9_./\\-]{2,}", request or ""):
                if ("/" in tok or "\\" in tok) and not Path(tok).suffix:
                    if not cls.is_invented_default(tok, request):
                        expect["directories"].append({"path": tok})

        return expect

    # ── Structured views ────────────────────────────────────────────────

    @classmethod
    def _derive_actions(cls, request: str) -> tuple[str, ...]:
        found = [m.group(1).lower() for m in _ACTION_VERBS.finditer(request or "")]
        # Preserve order, unique
        out: list[str] = []
        for a in found:
            if a not in out:
                out.append(a)
        if not out:
            out = ["execute"]
        return tuple(out[:8])

    @staticmethod
    def _derive_artifacts(constraints: dict[str, Any]) -> tuple[str, ...]:
        arts: list[str] = []
        for f in constraints.get("files") or []:
            p = str((f or {}).get("path") or "").strip()
            if p and p not in arts:
                arts.append(p)
        for d in constraints.get("directories") or []:
            p = str((d or {}).get("path") or "").strip()
            if p and p not in arts:
                arts.append(p)
        for h in constraints.get("http") or []:
            u = str((h or {}).get("url") or "").strip()
            if u and u not in arts:
                arts.append(u)
        return tuple(arts)

    @staticmethod
    def _derive_content_requirements(constraints: dict[str, Any]) -> tuple[str, ...]:
        reqs: list[str] = []
        for f in constraints.get("files") or []:
            c = (f or {}).get("contains")
            if c is not None and str(c).strip() and str(c) not in reqs:
                reqs.append(str(c))
        for m in constraints.get("must_contain") or []:
            if m is not None and str(m).strip() and str(m) not in reqs:
                reqs.append(str(m))
        return tuple(reqs)

    @classmethod
    def _derive_success_criteria(cls, constraints: dict[str, Any]) -> tuple[str, ...]:
        crit: list[str] = []
        for p in cls._derive_artifacts(constraints):
            crit.append(f"artifact_exists:{p}")
        for c in cls._derive_content_requirements(constraints):
            crit.append(f"content_present:{c}")
        if not crit:
            crit.append("skill_ok_and_verified")
        return tuple(crit[:16])
