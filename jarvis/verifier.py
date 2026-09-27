"""
JARVIS verifier — independent factual checks.

SKILL RESULT (what the skill claimed) is never enough for DONE.
VERIFIER RESULT is produced here by re-checking the world.
"""

from __future__ import annotations

import importlib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional


class Verifier:
    """Independently validates claims. Never trusts skill self-evidence alone."""

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
    ) -> dict[str, Any]:
        """
        Returns VERIFIER RESULT:
          verified (bool), reason, checks, skill_result (echo), verifier_result
        """
        self.on_log("VERIFY: independent checks (not trusting skill self-proof)")

        # Normalize: accept either wrapped test output or raw skill_result
        sr = skill_result.get("skill_result") if isinstance(skill_result.get("skill_result"), dict) else skill_result

        checks: list[dict[str, Any]] = []

        # Hard fail on timeout / crash — nothing to verify
        if sr.get("timed_out"):
            return self._fail("Skill subprocess timed out", checks, sr, [
                {"name": "subprocess_timeout", "ok": False, "detail": "timed_out"}
            ])
        if sr.get("crash") and not sr.get("result") and not sr.get("ok"):
            return self._fail(
                f"Skill subprocess crash rc={sr.get('returncode')}",
                checks,
                sr,
                [{"name": "subprocess_crash", "ok": False, "detail": sr.get("stderr", "")[:300]}],
            )

        result = sr.get("result")
        evidence = str(sr.get("evidence") or "")
        expect = expect or self._infer_expectations(goal, result, evidence)

        # ── File checks ────────────────────────────────────────────────
        for spec in expect.get("files") or []:
            path = self._resolve(spec.get("path"))
            exists = path.exists() and path.is_file()
            detail = f"{path} exists={exists}"
            content_ok = True
            if exists and spec.get("contains") is not None:
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
            if exists and spec.get("min_bytes") is not None:
                size = path.stat().st_size
                size_ok = size >= int(spec["min_bytes"])
                content_ok = content_ok and size_ok
                detail += f" size={size}>={spec['min_bytes']}={size_ok}"
            checks.append({
                "name": "file",
                "ok": exists and content_ok,
                "detail": detail,
            })

        # ── Directory checks ───────────────────────────────────────────
        for spec in expect.get("directories") or []:
            path = self._resolve(spec.get("path"))
            ok = path.exists() and path.is_dir()
            checks.append({
                "name": "directory",
                "ok": ok,
                "detail": f"{path} is_dir={ok}",
            })

        # ── Python process checks (from skill subprocess) ──────────────
        if expect.get("check_process", True):
            rc = sr.get("returncode")
            # Skill may return ok with rc=0; failure path rc=1 is fine if we're verifying failure
            # For success verification we need skill ok AND no timeout
            proc_ok = (not sr.get("timed_out")) and (rc is not None)
            checks.append({
                "name": "python_process",
                "ok": proc_ok and bool(sr.get("ok")),
                "detail": (
                    f"returncode={rc} stdout_len={len(sr.get('stdout') or '')} "
                    f"stderr_len={len(sr.get('stderr') or '')} ok={sr.get('ok')}"
                ),
            })

        # ── Dependency / import checks ─────────────────────────────────
        for mod_name in expect.get("imports") or []:
            ok, detail = self._check_import(str(mod_name))
            checks.append({"name": "dependency_import", "ok": ok, "detail": detail})

        # ── HTTP checks ────────────────────────────────────────────────
        for spec in expect.get("http") or []:
            ok, detail = self._check_http(spec)
            checks.append({"name": "http", "ok": ok, "detail": detail})

        # ── Generic claimed paths — re-stat independently ───────────────
        claimed_paths = self._collect_claimed_paths(result, evidence)
        for p in claimed_paths[:12]:
            exists = p.exists()
            kind = "dir" if p.is_dir() else ("file" if p.is_file() else "missing")
            checks.append({
                "name": "claimed_path",
                "ok": exists,
                "detail": f"{p} ({kind})",
            })

        # If skill claims success but provides nothing independently checkable → fail
        if not checks:
            checks.append({
                "name": "checkable_evidence",
                "ok": False,
                "detail": "No independently verifiable artifacts (file/dir/http/import)",
            })

        # Skill self-evidence text is recorded but NEVER sufficient alone
        checks.append({
            "name": "skill_self_evidence_ignored_as_proof",
            "ok": True,
            "detail": f"skill.evidence recorded ({len(evidence)} chars) but not trusted alone",
        })

        if require_brain_confirm and self.brain is not None:
            judgment = self.brain.verify_claim(goal, {"result": result}, evidence)
            checks.append({
                "name": "brain_advisory",
                "ok": bool(judgment.get("achieved")),
                "detail": str(judgment.get("reason") or ""),
            })

        # PASS only if every substantive check ok (ignore advisory naming)
        substantive = [
            c for c in checks
            if c["name"] not in ("skill_self_evidence_ignored_as_proof", "brain_advisory")
        ]
        verified = all(c["ok"] for c in substantive) and bool(substantive)

        reason = (
            "VERIFIER PASS — independent checks succeeded"
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
            "expectations": expect,
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

    # ── Expectation inference ───────────────────────────────────────────

    def _infer_expectations(
        self, goal: str, result: Any, evidence: str
    ) -> dict[str, Any]:
        expect: dict[str, Any] = {
            "files": [],
            "directories": [],
            "imports": [],
            "http": [],
            "check_process": True,
        }
        # From structured result
        if isinstance(result, dict):
            for key in ("path", "file", "output", "output_path", "filepath"):
                if result.get(key):
                    entry: dict[str, Any] = {"path": str(result[key]), "min_bytes": 1}
                    if result.get("contains"):
                        entry["contains"] = result["contains"]
                    if result.get("content_preview"):
                        entry["contains"] = str(result["content_preview"])[:40]
                    expect["files"].append(entry)
            for key in ("directory", "dir", "folder"):
                if result.get(key):
                    expect["directories"].append({"path": str(result[key])})
            if result.get("import"):
                expect["imports"].append(result["import"])
            if result.get("imports"):
                expect["imports"].extend(result["imports"])
            if result.get("url"):
                http_spec = {"url": str(result["url"])}
                if result.get("expect_status"):
                    http_spec["expect_status"] = int(result["expect_status"])
                expect["http"].append(http_spec)

        # From goal heuristics (only if nothing structured)
        g = goal.lower()
        if not expect["files"] and not expect["directories"] and not expect["http"]:
            # Look for explicit filenames in goal
            for m in re.finditer(
                r"([A-Za-z0-9_\-./]+\.(?:txt|json|csv|py|md|log|html|xml))", goal
            ):
                expect["files"].append({"path": m.group(1), "min_bytes": 1})
            if "directory" in g or "folder" in g or "mapi" in g:
                for m in re.finditer(r"([A-Za-z0-9_\-./]+)", goal):
                    token = m.group(1)
                    if "/" in token or token.endswith("_dir"):
                        expect["directories"].append({"path": token})

        # Paths mentioned in evidence string
        if not expect["files"] and not expect["directories"]:
            for p in self._extract_paths(evidence + "\n" + repr(result))[:5]:
                if p.suffix:
                    expect["files"].append({"path": str(p), "min_bytes": 0})
                else:
                    # could be file without suffix or dir — check both later via claimed_paths
                    expect["files"].append({"path": str(p)})

        return expect

    def _collect_claimed_paths(self, result: Any, evidence: str) -> list[Path]:
        paths = self._extract_paths(evidence + "\n" + repr(result))
        if isinstance(result, dict):
            for key in ("path", "file", "output", "output_path", "filepath", "directory", "dir"):
                if result.get(key):
                    paths.append(self._resolve(result[key]))
        # de-dupe
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
            # try common aliases
            aliases = {"bs4": "bs4", "PIL": "PIL", "yaml": "yaml", "cv2": "cv2"}
            alt = aliases.get(name, mod)
            if alt != mod:
                try:
                    importlib.import_module(alt)
                    return True, f"import {alt} OK"
                except Exception as exc2:
                    return False, f"import {mod} FAILED: {exc2}"
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
