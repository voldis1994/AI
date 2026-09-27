"""
JARVIS context/args builder — universal task → structured context.

Does not hardcode task-specific argument names (path, content, etc.).
Ollama (or a generic offline extractor) decides the arg schema from the goal.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional


_STOP = {
    "a", "an", "the", "and", "or", "to", "for", "with", "from", "into",
    "in", "on", "at", "of", "by", "is", "are", "be", "as", "it", "this",
    "that", "create", "write", "make", "build", "run", "file", "files",
    "containing", "contains", "named", "called", "please", "jarvis",
    "izveido", "uzraksti", "failu", "ar", "saturu", "satur", "nosaukumu",
}


class ContextBuilder:
    """Turn a user goal into context dict for skill.run(context)."""

    def __init__(
        self,
        brain: Any = None,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.brain = brain
        self.on_log = on_log or (lambda _m: None)

    def build(
        self,
        goal: str,
        workspace: str,
        *,
        mode: str = "execute",
        skill_meta: Optional[dict] = None,
        prior_args: Optional[dict] = None,
        diagnosis: Optional[dict] = None,
        extra_args: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Returns full context: {goal, args, workspace, mode}.

        Args are merged: prior → extracted → diagnosis.suggested_args →
        offline fill for still-missing keys → extra.
        """
        args: dict[str, Any] = {}
        if isinstance(prior_args, dict):
            args.update(prior_args)

        extracted = self.extract_args(
            goal,
            skill_meta=skill_meta,
            prior_args=args,
            diagnosis=diagnosis,
        )
        if isinstance(extracted, dict):
            args.update({k: v for k, v in extracted.items() if v is not None})

        if diagnosis and isinstance(diagnosis.get("suggested_args"), dict):
            args.update(
                {k: v for k, v in diagnosis["suggested_args"].items() if v is not None}
            )

        # Offline fill for keys still missing (brain may have returned {})
        needed = self._needed_keys(diagnosis, skill_meta)
        still_missing = [
            k for k in needed
            if k not in args or args.get(k) in (None, "")
        ]
        if still_missing or (diagnosis and not args):
            offline = self._offline_extract(
                goal,
                diagnosis={
                    **(diagnosis or {}),
                    "missing_args": still_missing or list(
                        (diagnosis or {}).get("missing_args") or []
                    ),
                    "required_args": list(
                        (diagnosis or {}).get("required_args") or needed
                    ),
                },
            )
            for k, v in offline.items():
                if k not in args or args.get(k) in (None, ""):
                    args[k] = v

        if isinstance(extra_args, dict):
            args.update({k: v for k, v in extra_args.items() if v is not None})

        context = {
            "goal": goal,
            "args": args,
            "workspace": str(workspace),
            "mode": mode,
        }
        self.on_log(
            f"CONTEXT: mode={mode} args_keys={list(args.keys())} "
            f"arg_count={len(args)}"
        )
        return context

    def extract_args(
        self,
        goal: str,
        skill_meta: Optional[dict] = None,
        prior_args: Optional[dict] = None,
        diagnosis: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Universal arg extraction — brain when available, else generic offline."""
        if self.brain is not None and getattr(self.brain, "is_available", lambda: False)():
            try:
                result = self.brain.extract_task_args(
                    goal,
                    skill_meta=skill_meta,
                    prior_args=prior_args,
                    diagnosis=diagnosis,
                )
                if isinstance(result, dict) and result:
                    return result
            except Exception as exc:
                self.on_log(f"CONTEXT: brain extract failed ({exc}); offline fallback")
        return self._offline_extract(goal, diagnosis=diagnosis)

    @staticmethod
    def _needed_keys(
        diagnosis: Optional[dict], skill_meta: Optional[dict]
    ) -> list[str]:
        needed: list[str] = []
        if diagnosis:
            for key in ("required_args", "missing_args"):
                val = diagnosis.get(key)
                if isinstance(val, list):
                    needed.extend(str(x) for x in val)
            suggested = diagnosis.get("suggested_args")
            if isinstance(suggested, dict):
                for k in suggested:
                    if k not in needed:
                        needed.append(str(k))
        if skill_meta:
            req = skill_meta.get("required_args")
            if isinstance(req, list):
                for k in req:
                    if str(k) not in needed:
                        needed.append(str(k))
        # preserve order, drop empties
        out: list[str] = []
        for k in needed:
            if k and k not in out:
                out.append(k)
        return out

    @staticmethod
    def parse_missing_arg_names(error: str) -> list[str]:
        """
        Extract argument names from skill/harness error text.

        Handles messages like:
          missing required arguments 'path' and 'content'
          missing required arguments: ['target', 'payload']
          KeyError: 'url'
          required arg foo not provided
        """
        if not error:
            return []
        names: list[str] = []

        # list / tuple literals in the message
        for m in re.finditer(r"\[([^\[\]]+)\]", error):
            inner = m.group(1)
            for part in re.findall(r"['\"]([A-Za-z_][\w]*)['\"]", inner):
                names.append(part)

        # quoted names near missing/required/argument/key
        low = error.lower()
        if any(
            tok in low
            for tok in ("missing", "required", "argument", "args", "keyerror", "key error")
        ):
            for part in re.findall(r"['\"]([A-Za-z_][\w]*)['\"]", error):
                names.append(part)

        # KeyError: 'name'
        for m in re.finditer(r"KeyError\s*:\s*['\"]([A-Za-z_][\w]*)['\"]", error, re.I):
            names.append(m.group(1))

        out: list[str] = []
        for n in names:
            if n not in out:
                out.append(n)
        return out

    @staticmethod
    def goal_value_candidates(goal: str) -> list[str]:
        """
        Ordered value candidates from a goal — no key-name hardcoding.

        Prefer path/url-like tokens first, then other quoted strings, then
        remaining significant tokens. This keeps (file, payload) style goals
        aligned when diagnosis supplies ordered missing arg names.
        """
        path_like = re.compile(
            r"^(?:[A-Za-z]:)?[A-Za-z0-9_./\\-]+\.[A-Za-z0-9]{1,12}$"
            r"|^https?://\S+$",
            re.I,
        )

        quoted: list[str] = []
        for a, b in re.findall(r"\"([^\"]+)\"|'([^']+)'", goal):
            val = a or b
            if val and val not in quoted:
                quoted.append(val)

        bare_paths: list[str] = []
        # Word-boundary start — avoid matching "rgs_e2e_out.txt" inside "args_e2e_out.txt"
        for m in re.finditer(
            r"(?<![A-Za-z0-9_\"'])([A-Za-z0-9_./\\-]+\.[A-Za-z0-9]{1,12})\b",
            goal,
        ):
            val = m.group(1)
            if val not in bare_paths and val not in quoted:
                bare_paths.append(val)

        values: list[str] = []
        # 1) path-like quoted
        for val in quoted:
            if path_like.match(val) and val not in values:
                values.append(val)
        # 2) bare path-like
        for val in bare_paths:
            if val not in values:
                values.append(val)
        # 3) remaining quoted (payloads, text, etc.)
        for val in quoted:
            if val not in values:
                values.append(val)
        # 4) other significant tokens
        for tok in re.findall(r"[A-Za-z0-9_./\\-]{2,}", goal):
            if tok.lower() in _STOP or tok in values:
                continue
            values.append(tok)

        return values

    @classmethod
    def _offline_extract(
        cls,
        goal: str,
        diagnosis: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Generic offline extractor — no task-specific hardcoding.

        Supports:
          - JSON object embedded in the goal
          - key=value / key: value pairs (any keys)
          - diagnosis.suggested_args / required_args / missing_args
            filled from ordered goal value candidates
        """
        args: dict[str, Any] = {}

        # Embedded JSON object
        m = re.search(r"\{[^{}]+\}", goal)
        if m:
            try:
                data = json.loads(m.group(0))
                if isinstance(data, dict):
                    args.update(data)
            except Exception:
                pass

        # Universal key=value or key: value
        for km in re.finditer(
            r"(?P<k>[A-Za-z_][\w]*)\s*[:=]\s*(?P<v>\"[^\"]*\"|'[^']*'|[^\s,;]+)",
            goal,
        ):
            key = km.group("k")
            raw = km.group("v")
            if (raw.startswith('"') and raw.endswith('"')) or (
                raw.startswith("'") and raw.endswith("'")
            ):
                raw = raw[1:-1]
            args[key] = raw

        needed = cls._needed_keys(diagnosis, None)
        if needed:
            candidates = cls.goal_value_candidates(goal)
            for key, val in zip(needed, candidates):
                if key not in args or args.get(key) in (None, ""):
                    args[key] = val

        return args

    @staticmethod
    def merge_args(*parts: Optional[dict]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for p in parts:
            if isinstance(p, dict):
                out.update({k: v for k, v in p.items() if v is not None})
        return out

    @classmethod
    def enrich_diagnosis_args(
        cls,
        diagnosis: dict[str, Any],
        observation: dict[str, Any],
        goal: str,
    ) -> dict[str, Any]:
        """
        Ensure diagnosis carries missing_args / suggested_args when the
        fault is (or looks like) context/args preparation.
        """
        err = str(
            observation.get("exception")
            or observation.get("error")
            or ""
        )
        ctx = observation.get("context") or {}
        ctx_args = ctx.get("args") if isinstance(ctx, dict) else {}
        if not isinstance(ctx_args, dict):
            ctx_args = {}

        parsed = cls.parse_missing_arg_names(err)
        missing = list(diagnosis.get("missing_args") or [])
        for name in parsed:
            if name not in missing:
                missing.append(name)
        if missing:
            diagnosis["missing_args"] = missing
            req = list(diagnosis.get("required_args") or [])
            for name in missing:
                if name not in req:
                    req.append(name)
            diagnosis["required_args"] = req

        empty_or_partial = (not ctx_args) or bool(missing and any(
            m not in ctx_args or ctx_args.get(m) in (None, "") for m in missing
        ))
        args_signal = bool(parsed) or any(
            tok in err.lower()
            for tok in ("argument", "args", "missing", "required", "keyerror")
        )
        if empty_or_partial and args_signal:
            diagnosis["fault_layer"] = "context_args"
            diagnosis["rewrite_skill"] = False
            if not diagnosis.get("approach") or diagnosis.get("approach") in (
                "",
                "initial",
            ):
                diagnosis["approach"] = "context_args_prep"
            if not diagnosis.get("what_to_change"):
                diagnosis["what_to_change"] = (
                    "Prepare structured context['args'] from the user goal and retest"
                )

        suggested = dict(diagnosis.get("suggested_args") or {})
        if diagnosis.get("fault_layer") == "context_args" or missing:
            filled = cls._offline_extract(goal, diagnosis=diagnosis)
            for k, v in filled.items():
                if k not in suggested or suggested.get(k) in (None, ""):
                    suggested[k] = v
            # Prefer values already present in context when valid
            for k, v in ctx_args.items():
                if v not in (None, "") and (k not in suggested or suggested.get(k) in (None, "")):
                    suggested[k] = v
        diagnosis["suggested_args"] = suggested
        return diagnosis
