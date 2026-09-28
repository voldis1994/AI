"""
Immutable TaskGoal — preserves the original USER REQUEST for the full cycle.

VERIFY and context mapping always ground against this object so learning/repair
cannot drift away from what the user asked for.

Structured fields (action / target / subject / content_requirements /
content_source / constraints / success_criteria) are derived from the request
+ grounded schema args — never by inventing a disambiguated sense for a
polysemous word, and never by splitting the sentence into dozens of word-args.
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

# Lean stop set for SUBJECT extraction — keep domain words (python/java/…)
# so polysemous surface tokens remain part of the topic.
_SUBJECT_STOP = {
    "a", "an", "the", "and", "or", "to", "for", "with", "from", "into",
    "in", "on", "at", "of", "by", "is", "are", "be", "as", "it", "this",
    "that", "create", "write", "make", "build", "run", "file", "files",
    "containing", "contains", "named", "called", "please", "jarvis",
    "text", "content", "contents", "data", "value", "values",
    "about", "regarding", "concerning", "summary", "summarize", "notes",
    "note", "using", "based", "according", "source", "learn", "learned",
    "knowledge", "previously", "already", "prior", "memory",
    "izveido", "uzraksti", "failu", "faila", "ar", "saturu", "satur",
    "nosaukumu", "tekstu", "teksts", "lūdzu", "ludzu", "ka", "kā",
    "par", "paradi", "parādi", "man", "lai", "un", "vai", "bet",
    "jo", "ja", "ko", "kas", "kur", "kad", "tik", "tai", "tos", "tas",
    "šo", "so", "ti", "tu", "es", "mēs", "mes", "jūs", "jus",
    "example", "examples", "demo", "show", "how", "works", "working",
}

# Canonical DIAGNOSE layers (legacy aliases accepted + normalized).
# "unknown" = unrecognized / insufficient evidence — NEVER treated as skill_code.
FAULT_LAYERS = frozenset({
    "goal_parsing",
    "context_mapping",
    "skill_code",
    "execution",
    "environment",
    "verifier",
    "unknown",
})
_LEGACY_LAYER = {
    "context_args": "context_mapping",
    "test_harness": "execution",
    "dependency": "environment",
    "unknown": "unknown",
    "unrecognized": "unknown",
    "none": "unknown",
    "null": "unknown",
    "": "unknown",
}

# Action verbs used only to label high-level actions (not as args).
_ACTION_VERBS = re.compile(
    r"(?i)\b(create|write|make|build|run|fetch|download|install|learn|"
    r"research|execute|test|verify|izveido|uzraksti|palaid|iemācies|"
    r"iemacies|paradi|parādi|summarize|summary|note|notes)\b"
)

# Topic cues — capture the surface subject without inventing a sense.
_ABOUT_CUE = re.compile(
    r"(?i)\b(?:about|regarding|concerning|on\s+the\s+topic\s+of|"
    r"par(?:\s+tēmu)?|par\s+tematu)\s+"
    r"(.+?)(?=\s+(?:containing|contains|with\s+text|ar\s+tekstu|saturu|"
    r"from|using|via|based\s+on|create|write|make|build|"
    r"izveido|uzraksti)\b|[,\"']|$)"
)

# Explicit content origins grounded in the request (never invented).
# Prefer URL / memory / path-like tokens — not bare adverbs after "using".
_SOURCE_CUE = re.compile(
    r"(?i)\b(?:from|via|based\s+on|according\s+to|source)\s+"
    r"[\"']?((?:https?://\S+)|(?:memory|prior\s+knowledge)|"
    r"[A-Za-z0-9_./\\-]{2,}\.[A-Za-z0-9]{1,12}|[A-Za-z0-9_./\\-]{3,})[\"']?"
)

_MEMORY_CUE = re.compile(
    r"(?i)\b(?:previously\s+learned|already\s+(?:know|learned|researched)|"
    r"from\s+(?:memory|prior\s+knowledge)|what\s+you\s+(?:know|learned|"
    r"researched)|prior\s+knowledge|saved\s+knowledge|"
    r"apgūtaj\w*|iemācīt\w*|no\s+atmiņ\w*)\b"
)

# Sentinel subject values — never invent a disambiguated meaning.
SUBJECT_UNKNOWN = "unknown"
SUBJECT_AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class TaskGoal:
    """
    Frozen goal for one user task.

    user_request — exact original text (never mutated; immutable source of truth)
    goal         — planner/intent summary (may be shorter; VERIFY prefers user_request)
    constraints  — structured expectations derived only from the request (+ grounded args)
    actions / artifacts / subject / content_requirements / content_source /
    success_criteria — structured views for PLAN / MEMORY / RESEARCH / BUILD
    """

    user_request: str
    goal: str
    constraints: dict[str, Any] = field(default_factory=dict)
    actions: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    subject: str = SUBJECT_UNKNOWN
    content_requirements: tuple[str, ...] = ()
    content_source: tuple[str, ...] = ()
    success_criteria: tuple[str, ...] = ()

    @property
    def topic(self) -> str:
        """Alias for subject — MEMORY retrieval key surface."""
        return self.subject

    @property
    def action(self) -> str:
        """Primary action label (first derived verb)."""
        return self.actions[0] if self.actions else "execute"

    @property
    def target(self) -> tuple[str, ...]:
        """Alias for artifacts (output targets)."""
        return self.artifacts

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
        artifacts = cls._derive_artifacts(constraints)
        content_reqs = cls._derive_content_requirements(constraints)
        content_source = cls._derive_content_source(req, constraints)
        subject = cls._derive_subject(req, artifacts, content_reqs, content_source)
        return cls(
            user_request=req,
            goal=g,
            constraints=constraints,
            actions=cls._derive_actions(req),
            artifacts=artifacts,
            subject=subject,
            content_requirements=content_reqs,
            content_source=content_source,
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
        artifacts = self._derive_artifacts(constraints)
        content_reqs = self._derive_content_requirements(constraints)
        # Preserve request-derived subject/source; refresh source with new URLs.
        content_source = self._derive_content_source(self.user_request, constraints)
        # Subject is request-grounded — only re-derive if previously unknown.
        if self.subject in ("", SUBJECT_UNKNOWN, SUBJECT_AMBIGUOUS):
            subject = self._derive_subject(
                self.user_request, artifacts, content_reqs, content_source
            )
        else:
            subject = self.subject
        return TaskGoal(
            user_request=self.user_request,
            goal=self.goal,
            constraints=constraints,
            actions=self.actions or self._derive_actions(self.user_request),
            artifacts=artifacts,
            subject=subject,
            content_requirements=content_reqs,
            content_source=content_source or self.content_source,
            success_criteria=self._derive_success_criteria(constraints),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_request": self.user_request,
            "goal": self.goal,
            "constraints": self.constraints,
            "actions": list(self.actions),
            "action": self.action,
            "artifacts": list(self.artifacts),
            "target": list(self.artifacts),
            "subject": self.subject,
            "topic": self.subject,
            "content_requirements": list(self.content_requirements),
            "content_source": list(self.content_source),
            "success_criteria": list(self.success_criteria),
        }

    # ── Fault-layer helpers (universal) ─────────────────────────────────

    @staticmethod
    def normalize_fault_layer(layer: Any) -> str:
        """
        Map diagnosis fault_layer to a canonical value.

        Missing / UNKNOWN / unrecognized layers stay ``unknown``.
        They must NEVER silently become ``skill_code`` (no rewrite without evidence).
        """
        if layer is None:
            return "unknown"
        raw = str(layer).strip().lower()
        if not raw or raw in ("none", "null", "unknown", "unrecognized", "?"):
            return "unknown"
        raw = _LEGACY_LAYER.get(raw, raw)
        if raw not in FAULT_LAYERS:
            return "unknown"
        return raw

    @classmethod
    def rewrite_skill_for_layer(cls, layer: Any) -> bool:
        """
        Skill rewrite allowed ONLY for an explicit skill_code attribution.

        unknown / unrecognized / other layers → False (never auto-rewrite).
        """
        return cls.normalize_fault_layer(layer) == "skill_code"

    @classmethod
    def apply_rewrite_gate(cls, diagnosis: Optional[dict[str, Any]]) -> dict[str, Any]:
        """
        Normalize fault_layer; rewrite_skill=True only for skill_code.

        Diagnosers must attribute skill_code only with evidence. unknown /
        unrecognized layers never unlock a skill rewrite.
        """
        out = dict(diagnosis or {})
        layer = cls.normalize_fault_layer(out.get("fault_layer"))
        out["fault_layer"] = layer
        out["rewrite_skill"] = layer == "skill_code"
        return out

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

        # 2) Bare URLs in the request (content_source; never path artifacts)
        for m in re.finditer(r"https?://\S+", request or ""):
            u = m.group(0).rstrip(".,);]\"'")
            if u and not cls.is_invented_default(u, request):
                expect["http"].append({"url": u})

        # 3) Quoted strings from the request (highest-confidence literals)
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

        # 4) Bare filenames with extensions in the request (not every word)
        #    Skip tokens that are substrings of a captured URL host/path.
        url_blobs = " ".join(
            str((h or {}).get("url") or "") for h in expect["http"]
        ).lower()
        for m in re.finditer(
            r"(?<![A-Za-z0-9_\"'])([A-Za-z0-9_./\\-]+\.[A-Za-z0-9]{1,12})\b",
            request or "",
        ):
            tok = m.group(1)
            if cls.is_invented_default(tok, request):
                continue
            if cls.looks_like_url(tok):
                continue
            if tok.lower() in url_blobs or any(
                tok.lower() in str((h or {}).get("url") or "").lower()
                for h in expect["http"]
            ):
                continue
            if tok not in path_vals:
                path_vals.append(tok)

        # 5) key=value pairs in the request (never split https:// into key=value)
        for km in re.finditer(
            r"(?P<k>[A-Za-z_][\w]*)\s*[:=]\s*(?P<v>\"[^\"]*\"|'[^']*'|[^\s,;]+)",
            request or "",
        ):
            key = (km.group("k") or "").lower()
            if key in ("http", "https"):
                continue
            raw = km.group("v")
            if (raw.startswith('"') and raw.endswith('"')) or (
                raw.startswith("'") and raw.endswith("'")
            ):
                raw = raw[1:-1]
            # URL fragment left after https: split — treat as URL, not path
            if raw.startswith("//") and key in ("http", "https"):
                continue
            if raw.startswith("//"):
                continue
            if cls.is_invented_default(raw, request):
                continue
            if cls.looks_like_url(raw):
                expect["http"].append({"url": raw})
            elif cls.looks_like_path(raw) and raw not in path_vals:
                if raw.lower() in url_blobs:
                    continue
                path_vals.append(raw)
            elif cls.looks_like_content(raw) and raw not in text_vals:
                text_vals.append(raw)
            elif raw not in text_vals and raw not in path_vals:
                text_vals.append(raw)

        # 6) Content after explicit cue words (NOT every sentence token).
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

        # 7) If we have a path but still no payload, take at most ONE remaining
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
        # URLs are content_source, not output artifacts, unless also a file target.
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
    def _derive_content_source(
        cls, request: str, constraints: dict[str, Any]
    ) -> tuple[str, ...]:
        """
        Origins of content grounded in the request only.

        URLs, explicit from/using/via cues, and memory references.
        Empty when the request does not name a source — never invent one.
        """
        sources: list[str] = []
        for h in constraints.get("http") or []:
            u = str((h or {}).get("url") or "").strip()
            if u and u not in sources:
                sources.append(u)
        for m in re.finditer(r"https?://\S+", request or ""):
            u = m.group(0).rstrip(".,);]")
            if u and u not in sources:
                sources.append(u)
        for m in _SOURCE_CUE.finditer(request or ""):
            tok = (m.group(1) or "").strip()
            if not tok:
                continue
            low = tok.lower()
            if low in ("memory", "prior knowledge"):
                tok = "memory"
            elif low in _SUBJECT_STOP or low in _STOP:
                continue
            if tok not in sources and not cls.is_invented_default(tok, request):
                sources.append(tok)
        if _MEMORY_CUE.search(request or "") and "memory" not in sources:
            sources.append("memory")
        return tuple(sources[:8])

    @classmethod
    def _derive_subject(
        cls,
        request: str,
        artifacts: tuple[str, ...] | list[str],
        content_requirements: tuple[str, ...] | list[str],
        content_source: tuple[str, ...] | list[str],
    ) -> str:
        """
        Surface subject/topic from the request — never a guessed sense.

        Prefer explicit about/regarding cues; otherwise leftover content tokens
        after stripping actions, artifacts, payloads, and sources.
        Ambiguous single-token leftovers stay ``ambiguous``; empty → ``unknown``.
        """
        req = request or ""
        # 1) Explicit topic cue — keep surface phrase (do not disambiguate)
        m = _ABOUT_CUE.search(req)
        if m:
            phrase = " ".join((m.group(1) or "").split()).strip(" .,;:-")
            # Strip trailing artifact mentions accidentally captured
            for art in artifacts:
                if art and art in phrase:
                    phrase = phrase.replace(art, " ")
            cleaned: list[str] = []
            for tok in re.findall(r"[A-Za-z0-9_]{3,}", phrase):
                low = tok.lower()
                if low in _SUBJECT_STOP:
                    continue
                if cls.looks_like_path(tok):
                    continue
                if tok not in cleaned:
                    cleaned.append(tok)
            if len(cleaned) >= 2:
                return " ".join(cleaned[:10])
            if len(cleaned) == 1:
                return SUBJECT_AMBIGUOUS
            # fall through if cue yielded nothing useful

        exclude: set[str] = set()
        for art in artifacts:
            exclude.add(str(art).lower())
            exclude.add(Path(str(art)).stem.lower())
            for part in re.findall(r"[A-Za-z0-9_]{2,}", str(art)):
                exclude.add(part.lower())
        for c in content_requirements:
            exclude.add(str(c).lower())
            for part in re.findall(r"[A-Za-z0-9_]{2,}", str(c)):
                exclude.add(part.lower())
        for src in content_source:
            exclude.add(str(src).lower())
            for part in re.findall(r"[A-Za-z0-9_]{2,}", str(src)):
                if part.lower() not in ("http", "https", "www"):
                    exclude.add(part.lower())
        exclude.update(path_segment_tokens(req, artifacts))
        for a in cls._derive_actions(req):
            exclude.add(a.lower())

        tokens: list[str] = []
        for tok in re.findall(r"[A-Za-z0-9_]{3,}", req):
            low = tok.lower()
            if low in _SUBJECT_STOP or low in exclude:
                continue
            if cls.looks_like_path(tok) or cls.looks_like_url(tok):
                continue
            if tok not in tokens:
                tokens.append(tok)

        if not tokens:
            return SUBJECT_UNKNOWN
        # Single leftover token with no supporting context → do not invent sense
        if len(tokens) == 1:
            return SUBJECT_AMBIGUOUS
        return " ".join(tokens[:10])

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

    # ── Query / MEMORY grounding ────────────────────────────────────────

    def grounding_bundle(self) -> str:
        """Text used to score research queries and memory relevance."""
        parts = [
            self.subject
            if self.subject not in ("", SUBJECT_UNKNOWN, SUBJECT_AMBIGUOUS)
            else "",
            " ".join(self.content_source),
            " ".join(self.artifacts),
            " ".join(self.content_requirements),
            self.user_request,
        ]
        return " ".join(p for p in parts if p).strip()

    def memory_topic(self) -> str:
        """Slug for MEMORY retrieval — subject first, never invent a sense."""
        from jarvis.intent import IntentClassifier

        if self.subject not in ("", SUBJECT_UNKNOWN, SUBJECT_AMBIGUOUS):
            return IntentClassifier.topic_slug(self.subject)
        return IntentClassifier.topic_slug(self.user_request)

    def prefers_memory_first(self) -> bool:
        """True when the request points at prior/learned knowledge as source."""
        if "memory" in {s.lower() for s in self.content_source}:
            return True
        return bool(_MEMORY_CUE.search(self.user_request or ""))

    def subject_tokens(self) -> set[str]:
        anchor = (
            self.subject
            if self.subject not in ("", SUBJECT_UNKNOWN, SUBJECT_AMBIGUOUS)
            else ""
        )
        if not anchor:
            return set()
        return {
            t.lower()
            for t in re.findall(r"[A-Za-z0-9_]{3,}", anchor)
            if t.lower() not in _SUBJECT_STOP
        }

    def query_is_grounded(self, query: str) -> bool:
        """
        True when a research query stays semantically tied to this TaskGoal.

        Rejects queries that share only a single short token with a multi-token
        subject (classic polysemy drift) while inventing an unrelated domain.
        Soft relatedness alone is not enough when subject-token overlap is weak.
        """
        q = (query or "").strip()
        if not q:
            return False
        # Whole query literally present in request/subject
        if q in (self.user_request or "") or (
            self.subject
            and self.subject not in (SUBJECT_UNKNOWN, SUBJECT_AMBIGUOUS)
            and q in self.subject
        ):
            return True

        subj = (
            self.subject
            if self.subject not in ("", SUBJECT_UNKNOWN, SUBJECT_AMBIGUOUS)
            else ""
        )
        for src in self.content_source:
            src_l = str(src).lower()
            if src_l == "memory":
                continue
            if src and src_l in q.lower():
                return True

        subj_tokens = self.subject_tokens()
        q_tokens = {
            t.lower()
            for t in re.findall(r"[A-Za-z0-9_]{3,}", q)
            if t.lower() not in _SUBJECT_STOP
        }
        overlap = subj_tokens & q_tokens

        # Multi-token subject: require ≥2 subject tokens in the query.
        # A single shared token is classic polysemy drift (python→pip, java→JDK).
        if len(subj_tokens) >= 2:
            if len(overlap) < 2:
                return False
            try:
                from jarvis.learning_verify import soft_relatedness

                return soft_relatedness(subj or self.grounding_bundle(), q) >= 0.18
            except Exception:
                return True

        if overlap:
            return True

        # unknown/ambiguous subject — require request-token overlap (not actions)
        req_tokens = {
            t.lower()
            for t in re.findall(r"[A-Za-z0-9_]{3,}", self.user_request or "")
            if t.lower() not in _SUBJECT_STOP
        }
        actions = {a.lower() for a in self.actions}
        art_stems = {Path(a).stem.lower() for a in self.artifacts}
        useful = {
            t for t in (req_tokens & q_tokens)
            if t not in actions and t not in art_stems
        }
        if len(useful) >= 2:
            return True
        return False

    def default_research_queries(self) -> list[str]:
        """Regenerate queries strictly from TaskGoal fields (no invented sense)."""
        out: list[str] = []
        subj = (
            self.subject
            if self.subject not in ("", SUBJECT_UNKNOWN, SUBJECT_AMBIGUOUS)
            else ""
        )
        if subj:
            out.append(subj)
            if self.content_requirements:
                out.append(f"{subj} {' '.join(self.content_requirements[:2])}")
        for src in self.content_source[:2]:
            if str(src).lower() == "memory":
                continue
            if subj:
                out.append(f"{subj} {src}")
            else:
                out.append(str(src))
        if not out:
            # Fall back to full immutable request — never invent a topic
            out = [self.user_request]
        # Dedupe
        seen: list[str] = []
        for q in out:
            q = " ".join(str(q).split())
            if q and q not in seen:
                seen.append(q)
        return seen[:6]

    def ground_research_queries(
        self,
        queries: Optional[list] = None,
        *,
        regenerate: bool = True,
    ) -> list[str]:
        """
        Keep only TaskGoal-grounded queries; regenerate from TaskGoal on total drift.
        """
        raw = [str(q).strip() for q in (queries or []) if str(q).strip()]
        kept = [q for q in raw if self.query_is_grounded(q)]
        # Preserve order, unique
        out: list[str] = []
        for q in kept:
            if q not in out:
                out.append(q)
        if out:
            return out[:6]
        if regenerate:
            return self.default_research_queries()
        return []
