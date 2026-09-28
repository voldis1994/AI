"""
JARVIS verifier — independent factual checks against the USER REQUEST.

PASS is allowed only when the real world / real tool behavior satisfies the
user's goal and TaskContract success_criteria. Never trusts:
  - skill self-proof / evidence text
  - skill.ok alone as proof of success
  - default / placeholder values invented by a skill

Behavior criteria (skill_output_contains / skill_result_equals /
skill_result_type) inspect real stdout and the returned result payload —
not narrated evidence.
"""

from __future__ import annotations

import importlib
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

from jarvis.task_contract import TaskContract


# Generic placeholder markers — not task-specific paths/content keys.
_DEFAULT_RE = re.compile(
    r"(?i)\b(?:user[_-]?provided(?:[_-]?\w+)?|"
    r"default(?:[_-]?(?:path|file|name|content|value|dir|output))?|"
    r"placeholder|changeme|your[_-]?name|todo|tbd|xxx+|dummy|"
    r"sample[_-]?(?:path|file)?|example[_-]?(?:path|file)?|"
    r"temp[_-]?file|untitled)\b"
)

_PATH_RE = re.compile(
    r"^(?:[A-Za-z]:)?[A-Za-z0-9_./\\-]+\.[A-Za-z0-9]{1,12}$"
    r"|^https?://\S+$",
    re.I,
)


class Verifier:
    """Independently validates the world against the user request."""

    def __init__(
        self,
        workspace: str | Path,
        brain: Any = None,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.workspace = Path(workspace)
        self.brain = brain
        self.on_log = on_log or (lambda _m: None)

    def verify(
        self,
        goal: str,
        skill_result: dict[str, Any],
        require_brain_confirm: bool = False,
        expect: Optional[dict[str, Any]] = None,
        args: Optional[dict[str, Any]] = None,
        user_request: Optional[str] = None,
        constraints: Optional[dict[str, Any]] = None,
        contract: Optional[TaskContract] = None,
    ) -> dict[str, Any]:
        """
        Returns VERIFIER RESULT:
          verified (bool), reason, checks, skill_result (echo), verifier_result

        Expectations come from the immutable TaskContract (original USER REQUEST) —
        never from the skill's claimed result or invented defaults.
        """
        self.on_log(
            "VERIFY: independent checks against original TaskContract / USER REQUEST "
            "(not trusting skill self-proof / defaults)"
        )

        sr = (
            skill_result.get("skill_result")
            if isinstance(skill_result.get("skill_result"), dict)
            else skill_result
        )
        checks: list[dict[str, Any]] = []
        if contract is not None:
            request = contract.original_request
        else:
            request = (user_request or goal or "").strip()
        # Ground args against the original request before building constraints
        task_args = TaskContract.ground_args(dict(args or {}), request)

        if sr.get("timed_out"):
            return self._fail("Skill subprocess timed out", checks, sr, [
                {"name": "subprocess_timeout", "ok": False, "detail": "timed_out"}
            ])
        if sr.get("crash") and not sr.get("result") and not sr.get("ok"):
            return self._fail(
                f"Skill subprocess crash rc={sr.get('returncode')}",
                checks,
                sr,
                [{"name": "subprocess_crash", "ok": False, "detail": (sr.get("stderr") or "")[:300]}],
            )

        result = sr.get("result")
        evidence = str(sr.get("evidence") or "")

        # Constraints from immutable TaskContract only — never re-parse USER REQUEST
        # when a contract is provided. Skill result is not a source of expectations.
        if constraints:
            built = constraints
        elif contract is not None:
            built = contract.constraints_for_verify(task_args)
        else:
            built = self.extract_constraints(request, task_args)
        if expect and constraints is None and contract is None and args is None:
            # Legacy callers may pass expect — still run default/alignment guards
            built = self._normalize_expect(expect, request, task_args)

        # Refuse empty/unverifiable TaskContract — never fall through to skill self-proof
        if contract is not None and contract.needs_refine():
            return self._fail(
                "TaskContract has no verifiable success_criteria — refine before VERIFY",
                checks,
                sr,
                [{
                    "name": "user_constraints",
                    "ok": False,
                    "detail": (
                        "needs_refine: empty/unverifiable constraints from "
                        "USER REQUEST; refusing PASS on skill self-proof"
                    ),
                }],
            )

        # ── Process signal (not proof of goal) ──────────────────────────
        rc = sr.get("returncode")
        proc_ok = (not sr.get("timed_out")) and (rc is not None) and bool(sr.get("ok"))
        checks.append({
            "name": "python_process",
            "ok": proc_ok,
            "detail": (
                f"returncode={rc} ok={sr.get('ok')} "
                f"(process signal only — not goal proof)"
            ),
        })

        # ── Reject defaults / placeholders not present in the user request ──
        default_hits = self._find_untrusted_defaults(result, evidence, request, task_args)
        checks.append({
            "name": "reject_defaults",
            "ok": not default_hits,
            "detail": (
                "no untrusted default/placeholder values"
                if not default_hits
                else f"untrusted defaults: {default_hits[:6]}"
            ),
        })

        # ── Goal-derived file constraints ───────────────────────────────
        file_specs = list(built.get("files") or [])
        for spec in file_specs:
            path = self._resolve(spec.get("path"))
            exists = path.exists() and path.is_file()
            detail = f"{path} exists={exists}"
            content_ok = True
            if not exists:
                content_ok = False
            else:
                if spec.get("contains") is not None:
                    try:
                        text = path.read_text(encoding="utf-8", errors="replace")
                    except Exception as exc:
                        text = ""
                        content_ok = False
                        detail += f" read_error={exc}"
                    else:
                        needle = str(spec["contains"])
                        content_ok = needle in text
                        detail += f" contains({needle!r})={content_ok}"
                if spec.get("min_bytes") is not None:
                    size = path.stat().st_size
                    size_ok = size >= int(spec["min_bytes"])
                    content_ok = content_ok and size_ok
                    detail += f" size={size}>={spec['min_bytes']}={size_ok}"
                # Reject default-looking filenames that user never requested
                if self._is_untrusted_default(path.name, request, task_args):
                    content_ok = False
                    detail += " untrusted_default_filename=True"
            checks.append({
                "name": "file",
                "ok": exists and content_ok,
                "detail": detail,
            })

        # ── Standalone content constraints ──────────────────────────────
        # With file targets → workspace content. Pathless → real skill output.
        for needle in built.get("must_contain") or []:
            if file_specs:
                ok, detail = self._content_present(str(needle), file_specs)
            else:
                ok, detail = self._skill_output_contains(str(needle), sr)
            checks.append({"name": "content_constraint", "ok": ok, "detail": detail})

        # ── Directory constraints ───────────────────────────────────────
        for spec in built.get("directories") or []:
            path = self._resolve(spec.get("path"))
            ok = path.exists() and path.is_dir()
            if self._is_untrusted_default(path.name, request, task_args):
                ok = False
            checks.append({
                "name": "directory",
                "ok": ok,
                "detail": f"{path} is_dir={ok}",
            })

        # ── Import / HTTP constraints from user side ────────────────────
        for mod_name in built.get("imports") or []:
            ok, detail = self._check_import(str(mod_name))
            checks.append({"name": "dependency_import", "ok": ok, "detail": detail})

        for spec in built.get("http") or []:
            ok, detail = self._check_http(spec)
            checks.append({"name": "http", "ok": ok, "detail": detail})

        # ── Immutable acceptance_criteria / verification_plan vs observations ──
        # When TaskContract is present, do not rebuild criteria from the request.
        criteria = []
        if contract is not None:
            criteria = list(contract.acceptance_criteria or ())
            if not criteria:
                for ch in contract.verification_plan or ():
                    if isinstance(ch, dict) and ch.get("kind") and ch.get("target"):
                        how = ch.get("how") or ""
                        criteria.append(
                            f"{ch['kind']}:{ch['target']}"
                            + (f"@{how}" if how else "")
                        )
        elif isinstance(built.get("checks"), list):
            for ch in built["checks"]:
                if isinstance(ch, dict) and ch.get("kind") and ch.get("target"):
                    how = ch.get("how") or ""
                    criteria.append(
                        f"{ch['kind']}:{ch['target']}"
                        + (f"@{how}" if how else "")
                    )
        for crit in criteria:
            ok, detail = self._check_success_criterion(str(crit), file_specs, sr)
            checks.append({
                "name": "success_criterion",
                "ok": ok,
                "detail": detail,
            })

        # ── Alignment: skill-claimed artifacts must match user constraints ──
        # Inspect claimed paths only to REJECT mismatches — never invent PASS.
        # claim_aligns_with_request is NEVER a substitute for user-derived checks.
        claimed = self._collect_claimed_paths(result, evidence)
        user_paths = {
            self._resolve(s.get("path")).resolve()
            for s in file_specs
            if s.get("path")
        }
        if claimed and user_paths:
            mismatched = [
                str(p) for p in claimed
                if p.resolve() not in user_paths
                and p.name.lower() not in {u.name.lower() for u in user_paths}
            ]
            for p in claimed:
                if self._is_untrusted_default(p.name, request, task_args):
                    if str(p) not in mismatched:
                        mismatched.append(str(p))
            checks.append({
                "name": "claim_aligns_with_request",
                "ok": not mismatched,
                "detail": (
                    "claimed artifacts align with user constraints"
                    if not mismatched
                    else f"claimed path not in user request: {mismatched[:4]}"
                ),
            })
        elif claimed and not user_paths:
            # No user file constraints — skill claims cannot earn PASS
            bad = [
                str(p) for p in claimed
                if self._is_untrusted_default(p.name, request, task_args)
                or not self._token_supported_by_request(p.name, request, task_args)
            ]
            claim_list = [str(p) for p in claimed][:4]
            checks.append({
                "name": "claim_aligns_with_request",
                "ok": False,
                "detail": (
                    "skill-claimed artifacts without user file constraints — "
                    f"not accepted as proof (claims={claim_list}"
                    + (f"; bad={bad[:4]}" if bad else "")
                    + ")"
                ),
            })

        # ── Must have at least one user-derived constraint check ────────
        # claim_aligns_with_request does NOT count — only TaskContract-derived checks.
        constraint_checks = [
            c for c in checks
            if c["name"] in (
                "file", "directory", "content_constraint", "http",
                "dependency_import", "success_criterion",
            )
        ]
        if not constraint_checks:
            checks.append({
                "name": "user_constraints",
                "ok": False,
                "detail": (
                    "No user-derived constraints extracted from REQUEST/args — "
                    "refusing PASS based on skill self-proof alone"
                ),
            })
        elif all(
            c.get("name") == "success_criterion"
            and str(c.get("detail") or "").startswith("needs_refine")
            for c in constraint_checks
        ):
            checks.append({
                "name": "user_constraints",
                "ok": False,
                "detail": "success_criteria=needs_refine — TaskContract not verifiable",
            })

        # Skill evidence is recorded but never counts toward PASS
        checks.append({
            "name": "skill_self_evidence_ignored_as_proof",
            "ok": True,
            "detail": f"skill.evidence recorded ({len(evidence)} chars) but not trusted",
        })

        if require_brain_confirm and self.brain is not None:
            judgment = self.brain.verify_claim(goal, {"result": result}, evidence)
            checks.append({
                "name": "brain_advisory",
                "ok": bool(judgment.get("achieved")),
                "detail": str(judgment.get("reason") or ""),
            })

        substantive = [
            c for c in checks
            if c["name"] not in ("skill_self_evidence_ignored_as_proof", "brain_advisory")
        ]
        verified = all(c["ok"] for c in substantive) and bool(substantive)

        reason = (
            "VERIFIER PASS — user constraints satisfied by independent checks"
            if verified
            else "VERIFIER FAIL — " + "; ".join(
                f"{c['name']}:{c.get('detail')}" for c in substantive if not c["ok"]
            )
        )
        self.on_log(f"VERIFY: {'PASS' if verified else 'FAIL'} — {reason[:220]}")

        verifier_result = {
            "verified": verified,
            "pass": verified,
            "reason": reason,
            "checks": checks,
            "constraints": built,
            "expectations": built,
        }
        return {
            "verified": verified,
            "reason": reason,
            "checks": checks,
            "confidence": 0.9 if verified else 0.1,
            "skill_result": {
                "ok": sr.get("ok"),
                "result": result,
                "error": sr.get("error"),
                "evidence": evidence,
                "returncode": sr.get("returncode"),
                "stdout": (sr.get("stdout") or "")[:1000],
                "stderr": (sr.get("stderr") or "")[:1000],
            },
            "verifier_result": verifier_result,
        }

    # ── Constraint extraction (USER REQUEST only) ───────────────────────

    def extract_constraints(
        self,
        request: str,
        args: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Legacy constraint extract when no TaskContract is supplied.

        Prefer ``contract=`` so VERIFY uses immutable acceptance_criteria.
        Uses grounded offline semantic draft — does not invent requirements.
        """
        from jarvis.contract_semantics import (
            ground_semantic_draft,
            offline_semantic_draft,
        )

        grounded = TaskContract.ground_args(dict(args or {}), request or "")
        draft = ground_semantic_draft(
            offline_semantic_draft(request or "", grounded), request or ""
        )
        return dict(draft.get("constraints") or {})

    def _check_success_criterion(
        self,
        criterion: str,
        file_specs: list[dict[str, Any]],
        skill_result: Optional[dict[str, Any]] = None,
    ) -> tuple[bool, str]:
        """
        Evaluate one TaskContract success_criterion against the real world.

        Format: ``kind:target`` or ``kind:target@how``.
        Artifact/network kinds check the workspace/network. Behavior kinds
        (skill_output_contains / skill_result_equals / skill_result_type)
        check the real tool return value and stdout — never skill.evidence
        or skill.ok self-proof.
        """
        raw = (criterion or "").strip()
        if not raw or raw in ("skill_ok_and_verified", "needs_refine"):
            return False, f"needs_refine:{raw or 'empty'}"
        how = ""
        body = raw
        if "@" in raw:
            body, how = raw.rsplit("@", 1)
        if ":" not in body:
            return False, f"unparseable_criterion:{raw}"
        kind, target = body.split(":", 1)
        kind = kind.strip()
        target = target.strip()
        if not kind or not target:
            return False, f"unparseable_criterion:{raw}"

        sr = skill_result if isinstance(skill_result, dict) else {}

        if kind == "artifact_exists":
            path = self._resolve(target)
            ok = path.exists() and path.is_file()
            return ok, f"artifact_exists:{path} exists={ok} how={how or 'workspace_stat'}"
        if kind == "directory_exists":
            path = self._resolve(target)
            ok = path.exists() and path.is_dir()
            return ok, f"directory_exists:{path} is_dir={ok} how={how or 'directory_stat'}"
        if kind == "content_present":
            ok, detail = self._content_present(target, file_specs)
            return ok, f"content_present:{detail} how={how or 'workspace_read'}"
        if kind == "http_ok":
            ok, detail = self._check_http({"url": target})
            return ok, f"http_ok:{detail} how={how or 'http_get'}"
        if kind == "import_ok":
            ok, detail = self._check_import(target)
            return ok, f"import_ok:{detail} how={how or 'importlib'}"
        if kind == "skill_output_contains":
            ok, detail = self._skill_output_contains(target, sr)
            return ok, (
                f"skill_output_contains:{detail} "
                f"how={how or 'skill_stdout_or_result'}"
            )
        if kind == "skill_result_equals":
            ok, detail = self._skill_result_equals(target, sr)
            return ok, f"skill_result_equals:{detail} how={how or 'skill_result'}"
        if kind == "skill_result_type":
            ok, detail = self._skill_result_type(target, sr)
            return ok, f"skill_result_type:{detail} how={how or 'skill_result'}"
        return False, f"unknown_criterion_kind:{kind}"

    def _skill_observable_text(self, sr: dict[str, Any]) -> str:
        """
        Real tool observables used for behavior checks.

        Includes stdout and the returned result payload. Excludes evidence
        (self-narration) and error strings used as proof.
        """
        parts: list[str] = []
        stdout = sr.get("stdout")
        if stdout:
            parts.append(str(stdout))
        result = sr.get("result")
        if result is not None:
            parts.append(self._result_surface(result))
        return "\n".join(parts)

    @staticmethod
    def _result_surface(result: Any) -> str:
        """Flatten a skill result into searchable / comparable text."""
        if result is None:
            return ""
        if isinstance(result, (str, int, float, bool)):
            return str(result)
        if isinstance(result, (list, tuple)):
            return repr(list(result) if isinstance(result, tuple) else result)
        if isinstance(result, dict):
            # Prefer common value-bearing keys, then full dict repr
            preferred = []
            for key in (
                "value", "result", "answer", "output", "text", "message",
                "says", "content", "data", "return", "returned",
            ):
                if key in result and result[key] is not None:
                    preferred.append(str(result[key]))
            if preferred:
                return "\n".join(preferred) + "\n" + repr(result)
            return repr(result)
        return repr(result)

    def _skill_output_contains(
        self, needle: str, sr: dict[str, Any]
    ) -> tuple[bool, str]:
        if not needle:
            return True, "empty needle"
        blob = self._skill_observable_text(sr)
        if not blob:
            return False, f"{needle!r} missing — no skill stdout/result"
        # Case-sensitive first (grounded literal), then case-insensitive fallback
        if needle in blob:
            return True, f"{needle!r} found in skill stdout/result"
        if needle.lower() in blob.lower():
            return True, f"{needle!r} found in skill stdout/result (case-insensitive)"
        return False, f"{needle!r} not in skill stdout/result"

    def _skill_result_equals(
        self, expected: str, sr: dict[str, Any]
    ) -> tuple[bool, str]:
        exp_raw = (expected or "").strip()
        if not exp_raw:
            return True, "empty expected"
        result = sr.get("result")
        # Unwrap common envelopes
        candidate = result
        if isinstance(result, dict):
            for key in (
                "value", "result", "answer", "output", "text", "data", "return",
            ):
                if key in result and result[key] is not None:
                    candidate = result[key]
                    break
        if self._values_equal(candidate, exp_raw):
            return True, f"result={candidate!r} equals {exp_raw!r}"
        # Also accept exact match against stdout (trimmed)
        stdout = str(sr.get("stdout") or "").strip()
        if stdout and self._values_equal(stdout, exp_raw):
            return True, f"stdout={stdout!r} equals {exp_raw!r}"
        return False, f"result={candidate!r} != expected {exp_raw!r}"

    def _skill_result_type(
        self, expected_type: str, sr: dict[str, Any]
    ) -> tuple[bool, str]:
        want = (expected_type or "").strip().lower()
        result = sr.get("result")
        candidate = result
        if isinstance(result, dict):
            # If envelope only, peek at value-bearing keys; else type is dict
            for key in ("value", "result", "answer", "output", "data", "return"):
                if key in result and result[key] is not None:
                    candidate = result[key]
                    break
        actual = type(candidate).__name__.lower()
        if candidate is None:
            actual = "none"
        aliases = {
            "int": {"int"},
            "str": {"str"},
            "list": {"list"},
            "dict": {"dict"},
            "bool": {"bool"},
            "float": {"float"},
            "tuple": {"tuple"},
            "none": {"none", "nonetype"},
        }
        ok = actual in aliases.get(want, {want})
        # int satisfies float ask? No. float with .0 satisfying int? allow int==int only.
        if not ok and want == "float" and actual == "int":
            ok = True
        return ok, f"type(result)={actual} expect={want} ok={ok}"

    @staticmethod
    def _values_equal(produced: Any, expected: str) -> bool:
        """Compare produced skill value to a request-grounded expected literal."""
        exp = (expected or "").strip()
        if produced is None:
            return exp.lower() in ("none", "null")
        # Numeric
        try:
            if re.fullmatch(r"-?\d+", exp):
                return int(produced) == int(exp)
            if re.fullmatch(r"-?\d+\.\d+", exp):
                return abs(float(produced) - float(exp)) < 1e-9
        except (TypeError, ValueError):
            pass
        # Sequence / literal via repr or ast
        if exp[:1] in "[(" and exp[-1:] in "])":
            try:
                import ast

                exp_val = ast.literal_eval(exp)
                if isinstance(produced, tuple) and isinstance(exp_val, list):
                    return list(produced) == exp_val
                if isinstance(produced, list) and isinstance(exp_val, tuple):
                    return produced == list(exp_val)
                if produced == exp_val:
                    return True
                if isinstance(produced, (list, tuple)) and isinstance(
                    exp_val, (list, tuple)
                ):
                    return list(produced) == list(exp_val)
            except Exception:
                pass
            return repr(produced).replace(" ", "") == exp.replace(" ", "")
        # Bool
        if exp.lower() in ("true", "false"):
            return str(produced).lower() == exp.lower()
        # String / generic
        if isinstance(produced, str):
            return produced.strip() == exp or produced == exp
        return str(produced).strip() == exp

    def _normalize_expect(
        self,
        expect: dict[str, Any],
        request: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge legacy expect with user-request constraints; never drop user side."""
        user = self.extract_constraints(request, args)
        out = dict(user)
        # Allow explicit expect to add checks, but user files win for paths
        user_paths = {str(f.get("path")) for f in user.get("files") or []}
        for f in expect.get("files") or []:
            p = str(f.get("path") or "")
            if p and p not in user_paths and not self._is_untrusted_default(p, request, args):
                # Only accept extra file specs that are supported by the request
                if self._token_supported_by_request(Path(p).name, request, args):
                    out.setdefault("files", []).append(f)
        for key in ("directories", "imports", "http", "must_contain"):
            for item in expect.get(key) or []:
                if item not in out.get(key, []):
                    out.setdefault(key, []).append(item)
        return out

    # ── Default / alignment helpers ─────────────────────────────────────

    def _find_untrusted_defaults(
        self,
        result: Any,
        evidence: str,
        request: str,
        args: dict[str, Any],
    ) -> list[str]:
        hits: list[str] = []
        values: list[str] = []
        if isinstance(result, dict):
            for v in result.values():
                if isinstance(v, (str, int, float)):
                    values.append(str(v))
                elif isinstance(v, (list, tuple)):
                    values.extend(str(x) for x in v if isinstance(x, (str, int, float)))
        elif isinstance(result, str):
            values.append(result)
        # Scan evidence for default-looking filenames
        for m in re.finditer(
            r"[A-Za-z0-9_./\\-]*default[A-Za-z0-9_./\\-]*",
            evidence or "",
            re.I,
        ):
            values.append(m.group(0))
        for v in values:
            if self._is_untrusted_default(v, request, args):
                if v not in hits:
                    hits.append(v)
        return hits

    def _is_untrusted_default(
        self,
        value: str,
        request: str,
        args: dict[str, Any],
    ) -> bool:
        if not value:
            return False
        sv = str(value)
        # If the user literally asked for this string, it is not "default abuse"
        if self._token_supported_by_request(sv, request, args):
            return False
        return bool(_DEFAULT_RE.search(sv))

    @staticmethod
    def _token_supported_by_request(
        token: str,
        request: str,
        args: dict[str, Any],
    ) -> bool:
        if not token:
            return False
        t = str(token)
        if t in (request or ""):
            return True
        # basename match for paths
        base = Path(t).name
        if base and base in (request or ""):
            return True
        for v in (args or {}).values():
            if v is None:
                continue
            sv = str(v)
            if t == sv or base == Path(sv).name or t in sv or sv in t:
                return True
        # Case-insensitive contains for short tokens
        low_req = (request or "").lower()
        if base.lower() in low_req or t.lower() in low_req:
            return True
        return False

    def _content_present(
        self, needle: str, file_specs: list[dict[str, Any]]
    ) -> tuple[bool, str]:
        if not needle:
            return True, "empty needle"
        paths = [self._resolve(s.get("path")) for s in file_specs if s.get("path")]
        if not paths:
            # No file targets — search workspace shallowly for the needle
            try:
                for p in self.workspace.iterdir():
                    if p.is_file() and p.stat().st_size < 2_000_000:
                        try:
                            if needle in p.read_text(encoding="utf-8", errors="replace"):
                                return True, f"found in {p.name}"
                        except Exception:
                            continue
            except Exception as exc:
                return False, f"workspace scan failed: {exc}"
            return False, f"content {needle!r} not found in workspace"
        for p in paths:
            if p.exists() and p.is_file():
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                if needle in text:
                    return True, f"{needle!r} in {p}"
        return False, f"{needle!r} missing from expected files {[str(p) for p in paths]}"

    # ── Type heuristics ─────────────────────────────────────────────────

    @staticmethod
    def _looks_like_path(value: str) -> bool:
        sv = str(value).strip()
        if not sv or "://" in sv:
            return False
        if _PATH_RE.match(sv):
            return True
        # Windows / POSIX path with separator
        if ("/" in sv or "\\" in sv) and not sv.startswith("http"):
            return True
        return False

    @staticmethod
    def _looks_like_content(value: str) -> bool:
        """Heuristic: prose / payloads are content; bare identifiers are not."""
        sv = str(value).strip()
        if not sv:
            return False
        if " " in sv or "\n" in sv or "\t" in sv:
            return True
        if len(sv) > 48:
            return True
        return False

    @staticmethod
    def _looks_like_url(value: str) -> bool:
        return str(value).strip().lower().startswith(("http://", "https://"))

    # ── Path / network helpers ──────────────────────────────────────────

    def _collect_claimed_paths(self, result: Any, evidence: str) -> list[Path]:
        paths = self._extract_paths(evidence + "\n" + repr(result))
        if isinstance(result, dict):
            for key in ("path", "file", "output", "output_path", "filepath", "directory", "dir"):
                if result.get(key):
                    paths.append(self._resolve(result[key]))
        out, seen = [], set()
        for p in paths:
            s = str(p)
            if s not in seen:
                seen.add(s)
                out.append(p)
        return out

    def _resolve(self, raw: Any) -> Path:
        p = Path(str(raw))
        if not p.is_absolute():
            p = self.workspace / p
        return p

    def _check_import(self, name: str) -> tuple[bool, str]:
        mod = name.replace("-", "_").split("[")[0]
        try:
            importlib.import_module(mod)
            return True, f"import {mod} OK"
        except Exception as exc:
            return False, f"import {mod} FAILED: {exc}"

    def _check_http(self, spec: dict) -> tuple[bool, str]:
        url = str(spec.get("url") or "")
        if not url:
            return False, "missing url"
        expect_status = spec.get("expect_status", 200)
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "JARVIS-Verifier/1.0"}, method="GET"
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                status = getattr(resp, "status", 200)
                body = resp.read(200)
            ok = int(status) == int(expect_status)
            return ok, f"HTTP {status} (expect {expect_status}) bytes={len(body)}"
        except urllib.error.HTTPError as exc:
            ok = int(exc.code) == int(expect_status)
            return ok, f"HTTP {exc.code} (expect {expect_status})"
        except Exception as exc:
            return False, f"HTTP failed: {exc}"

    def _extract_paths(self, text: str) -> list[Path]:
        found: list[Path] = []
        patterns = [
            r"(?:^|[\s\"'=])(/[^\s\"']+)",
            r"(?:^|[\s\"'=])((?:\./|\.\./)?(?:[A-Za-z0-9_\-./]+/[A-Za-z0-9_\-./]+))",
            r"(?:^|[\s\"'=])([A-Za-z0-9_\-./]+\.(?:txt|json|csv|py|md|png|jpg|pdf|html|xml|zip|log))",
        ]
        for pat in patterns:
            for m in re.finditer(pat, text):
                raw = m.group(1).rstrip(".,;:)")
                if len(raw) < 3:
                    continue
                found.append(self._resolve(raw))
        return found

    def _fail(
        self,
        reason: str,
        checks: list,
        sr: dict,
        extra: list,
    ) -> dict[str, Any]:
        checks = list(checks) + list(extra)
        self.on_log(f"VERIFY: FAIL — {reason[:200]}")
        verifier_result = {
            "verified": False,
            "pass": False,
            "reason": reason,
            "checks": checks,
        }
        return {
            "verified": False,
            "reason": reason,
            "checks": checks,
            "confidence": 0.0,
            "skill_result": {
                "ok": sr.get("ok"),
                "result": sr.get("result"),
                "error": sr.get("error"),
                "evidence": sr.get("evidence"),
                "returncode": sr.get("returncode"),
            },
            "verifier_result": verifier_result,
        }
