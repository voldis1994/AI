"""
JARVIS verifier — DONE only after factual verification.

Ollama must not simply declare success. This module checks evidence on disk /
in returned payloads before allowing ACTIVE / DONE.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable, Optional


class Verifier:
    """Validates that a skill's claimed result is backed by real evidence."""

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
        test_result: dict[str, Any],
        require_brain_confirm: bool = False,
    ) -> dict[str, Any]:
        self.on_log("VERIFY: checking evidence")
        checks: list[dict[str, Any]] = []

        if not test_result.get("passed") and not test_result.get("ok"):
            return {
                "verified": False,
                "reason": f"Test did not pass: {test_result.get('error')}",
                "checks": [{"name": "test_passed", "ok": False}],
                "confidence": 0.0,
            }

        evidence = str(test_result.get("evidence") or "")
        result = test_result.get("result")

        # 1) Evidence non-empty
        checks.append({
            "name": "evidence_present",
            "ok": bool(evidence.strip()),
            "detail": evidence[:300],
        })

        # 2) Detect and confirm file paths mentioned in evidence/result
        paths = self._extract_paths(evidence, result)
        path_ok = True
        path_details = []
        for p in paths[:10]:
            exists = p.exists()
            path_details.append(f"{p}: {'EXISTS' if exists else 'MISSING'}")
            if not exists:
                path_ok = False
        if paths:
            checks.append({
                "name": "referenced_paths_exist",
                "ok": path_ok,
                "detail": "; ".join(path_details),
            })

        # 3) If result claims a file, it must exist
        if isinstance(result, dict):
            for key in ("path", "file", "output", "output_path", "filepath"):
                if key in result and result[key]:
                    rp = Path(str(result[key]))
                    if not rp.is_absolute():
                        rp = self.workspace / rp
                    exists = rp.exists()
                    checks.append({
                        "name": f"result.{key}_exists",
                        "ok": exists,
                        "detail": str(rp),
                    })

        # 4) Error must be empty/None when ok
        err = test_result.get("error")
        checks.append({
            "name": "no_error_when_ok",
            "ok": not err,
            "detail": str(err) if err else "",
        })

        # 5) Optional brain judgment — advisory only; cannot override hard fails
        brain_ok = True
        brain_reason = ""
        confidence = 0.7
        if require_brain_confirm and self.brain is not None:
            judgment = self.brain.verify_claim(goal, {"result": result}, evidence)
            brain_ok = bool(judgment.get("achieved"))
            brain_reason = str(judgment.get("reason") or "")
            try:
                confidence = float(judgment.get("confidence") or 0.5)
            except Exception:
                confidence = 0.5
            checks.append({
                "name": "brain_judgment",
                "ok": brain_ok,
                "detail": brain_reason,
            })

        hard_ok = all(c["ok"] for c in checks if c["name"] != "brain_judgment")
        verified = hard_ok and (brain_ok if require_brain_confirm else True)

        # Extra: if no paths referenced and result is None and evidence is vague → fail
        if verified and not paths and result in (None, "", {}, []):
            if len(evidence.strip()) < 10:
                verified = False
                checks.append({
                    "name": "substance",
                    "ok": False,
                    "detail": "Evidence too weak / empty result",
                })

        reason = (
            "Verified with concrete evidence"
            if verified
            else "; ".join(
                f"{c['name']}: {c.get('detail') or 'fail'}"
                for c in checks
                if not c["ok"]
            )
            or "Verification failed"
        )
        self.on_log(f"VERIFY: {'PASS' if verified else 'FAIL'} — {reason[:200]}")
        return {
            "verified": verified,
            "reason": reason,
            "checks": checks,
            "confidence": confidence if verified else min(confidence, 0.3),
            "paths_checked": [str(p) for p in paths],
        }

    def _extract_paths(self, evidence: str, result: Any) -> list[Path]:
        found: list[Path] = []
        text = evidence + "\n" + repr(result)
        # Unix-like and relative paths with extensions or directories
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
                p = Path(raw)
                if not p.is_absolute():
                    p = self.workspace / p
                found.append(p)
        # de-dupe
        uniq: list[Path] = []
        seen = set()
        for p in found:
            key = str(p)
            if key not in seen:
                seen.add(key)
                uniq.append(p)
        return uniq
