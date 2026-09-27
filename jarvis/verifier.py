"""
JARVIS verifier — independent factual checks against the USER REQUEST.

PASS is allowed only when the real world satisfies the user's goal/constraints.
Never trusts:
  - skill self-proof / evidence text
  - skill result fields as proof of success
  - default / placeholder values invented by a skill
"""

from __future__ import annotations

import importlib
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

from jarvis.context_builder import ContextBuilder


# Generic placeholder markers — not task-specific paths/content keys.
_DEFAULT_RE = re.compile(
    r"(?i)\b(?:default(?:[_-]?(?:path|file|name|content|value|dir|output))?|"
    r"placeholder|changeme|your[_-]?name|todo|tbd|xxx+|dummy|sample[_-]?(?:path|file)?|"
    r"example[_-]?(?:path|file)?|temp[_-]?file|untitled)\b"
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
    ) -> dict[str, Any]:
        """
        Returns VERIFIER RESULT:
          verified (bool), reason, checks, skill_result (echo), verifier_result

        Expectations come from USER REQUEST + structured args/constraints —
        never from the skill's claimed result.
        """
        self.on_log(
            "VERIFY: independent checks against USER REQUEST "
            "(not trusting skill self-proof / defaults)"
        )

        sr = (
            skill_result.get("skill_result")
            if isinstance(skill_result.get("skill_result"), dict)
            else skill_result
        )
        checks: list[dict[str, Any]] = []
        request = (user_request or goal or "").strip()
        task_args = dict(args or {})

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

        # Constraints from USER side only (goal + args). Skill result is not a source.
        built = constraints or expect or self.extract_constraints(request, task_args)
        if expect and constraints is None and args is None:
            # Legacy callers may pass expect — still run default/alignment guards
            built = self._normalize_expect(expect, request, task_args)

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

        # ── Standalone content constraints (must appear in expected files) ──
        for needle in built.get("must_contain") or []:
            ok, detail = self._content_present(str(needle), file_specs)
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

        # ── Alignment: skill-claimed artifacts must match user constraints ──
        # We inspect claimed paths only to REJECT mismatches — never to invent PASS.
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
            # Also treat claimed path as mismatch if it's a default not in request
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
            # Skill claims files but user request had no file constraint — still
            # reject defaults; otherwise require at least one claimed path exists
            # AND every claimed value token appears in the request.
            bad = [
                str(p) for p in claimed
                if self._is_untrusted_default(p.name, request, task_args)
                or not self._token_supported_by_request(p.name, request, task_args)
            ]
            checks.append({
                "name": "claim_aligns_with_request",
                "ok": not bad and all(p.exists() for p in claimed),
                "detail": (
                    "claimed artifacts supported by user request"
                    if not bad
                    else f"unsupported/default claims: {bad[:4]}"
                ),
            })

        # ── Must have at least one user-derived constraint check ────────
        constraint_checks = [
            c for c in checks
            if c["name"] in (
                "file", "directory", "content_constraint", "http",
                "dependency_import", "claim_aligns_with_request",
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
        Build verification constraints from the user request + structured args.

        Does not read skill results. Does not hardcode task-specific keys —
        values are classified by shape (path-like / url / text).
        """
        args = dict(args or {})
        expect: dict[str, Any] = {
            "files": [],
            "directories": [],
            "imports": [],
            "http": [],
            "must_contain": [],
            "check_process": True,
            "source": "user_request",
        }

        # Structured args → constraints (key names are whatever the planner used)
        path_vals: list[str] = []
        text_vals: list[str] = []
        for _key, val in args.items():
            if val in (None, ""):
                continue
            sv = str(val)
            if self._looks_like_url(sv):
                expect["http"].append({"url": sv})
            elif self._looks_like_path(sv):
                path_vals.append(sv)
            else:
                text_vals.append(sv)

        # Goal/request tokens (quoted, path-like, key=value via ContextBuilder)
        offline = ContextBuilder._offline_extract(request)
        for _key, val in offline.items():
            if val in (None, ""):
                continue
            sv = str(val)
            if self._looks_like_url(sv):
                expect["http"].append({"url": sv})
            elif self._looks_like_path(sv):
                if sv not in path_vals:
                    path_vals.append(sv)
            else:
                if sv not in text_vals:
                    text_vals.append(sv)

        known_text = set(text_vals)
        for tok in ContextBuilder.goal_value_candidates(request):
            if self._looks_like_path(tok) and tok not in path_vals:
                path_vals.append(tok)
            elif (
                not self._looks_like_path(tok)
                and not self._looks_like_url(tok)
                and len(tok) >= 2
                and tok not in text_vals
            ):
                # Skip fragments already covered by a longer arg/quoted value
                if any(tok in k for k in known_text if k != tok):
                    continue
                text_vals.append(tok)
                known_text.add(tok)

        # Pair path + content when both present (universal: first path gets contents)
        primary_contains = text_vals[0] if text_vals else None
        for i, p in enumerate(path_vals):
            entry: dict[str, Any] = {"path": p, "min_bytes": 1}
            if i == 0 and primary_contains is not None:
                entry["contains"] = primary_contains
            expect["files"].append(entry)

        # Extra text constraints beyond the primary paired content
        if primary_contains is not None:
            for extra in text_vals[1:]:
                if len(str(extra)) >= 2:
                    expect["must_contain"].append(str(extra))
        elif text_vals and not path_vals:
            expect["must_contain"].extend(str(t) for t in text_vals if len(str(t)) >= 2)

        # Directory hints from request words + slash tokens without extension
        low = request.lower()
        if any(w in low for w in ("directory", "folder", "mapi", "katalog")):
            for tok in re.findall(r"[A-Za-z0-9_./\\-]{2,}", request):
                if ("/" in tok or "\\" in tok or tok.endswith("_dir")) and not Path(tok).suffix:
                    expect["directories"].append({"path": tok})

        return expect

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
