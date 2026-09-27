"""
JARVIS observer — universal failure observation package.

Collects everything the brain needs to diagnose and repair,
without encoding task-specific solutions.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Callable, Optional


class Observer:
    """Build structured observations after BUILD/TEST/VERIFY failures."""

    def __init__(
        self,
        workspace: str | Path,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.workspace = Path(workspace)
        self.on_log = on_log or (lambda _m: None)

    def observe_failure(
        self,
        *,
        goal: str,
        skill_name: str,
        version: int,
        phase: str,
        skill_code: Optional[str],
        context: Optional[dict] = None,
        dependencies: Optional[list] = None,
        test_result: Optional[dict] = None,
        verification: Optional[dict] = None,
        build_error: Optional[str] = None,
        deps_result: Optional[dict] = None,
        prior_approaches: Optional[list] = None,
        prior_diagnoses: Optional[list] = None,
    ) -> dict[str, Any]:
        """
        Assemble a full observation dict:

        exception/traceback, stdout/stderr, return code, skill code,
        context, dependencies, artifacts (+ prior failed approaches).
        """
        tr = test_result or {}
        sr = tr.get("skill_result") if isinstance(tr.get("skill_result"), dict) else tr

        exception = (
            build_error
            or sr.get("error")
            or tr.get("error")
            or (verification or {}).get("reason")
            or ""
        )
        tb = str(sr.get("traceback") or tr.get("traceback") or "")
        stdout = str(sr.get("stdout") or tr.get("stdout") or "")
        stderr = str(sr.get("stderr") or tr.get("stderr") or "")
        returncode = sr.get("returncode", tr.get("returncode"))

        ctx = dict(context or {})
        ctx.setdefault("goal", goal)
        ctx.setdefault("workspace", str(self.workspace))
        ctx.setdefault("mode", tr.get("mode") or "test")

        artifacts = self._scan_artifacts(sr.get("result"), sr.get("evidence") or tr.get("evidence"))

        observation = {
            "goal": goal,
            "skill_name": skill_name,
            "version": version,
            "phase": phase,
            "exception": str(exception)[:4000],
            "traceback": tb[:8000],
            "stdout": stdout[:4000],
            "stderr": stderr[:4000],
            "returncode": returncode,
            "timed_out": bool(sr.get("timed_out") or tr.get("timed_out")),
            "crash": bool(sr.get("crash") or tr.get("crash")),
            "killed": bool(sr.get("killed") or tr.get("killed")),
            "skill_code": skill_code or "",
            "context": ctx,
            "dependencies": list(dependencies or []),
            "deps_result": deps_result or {},
            "artifacts": artifacts,
            "skill_result": {
                "ok": sr.get("ok"),
                "result": sr.get("result"),
                "error": sr.get("error"),
                "evidence": sr.get("evidence"),
            },
            "verifier_result": (verification or {}).get("verifier_result")
            or (verification if verification else None),
            "prior_approaches": list(prior_approaches or [])[:10],
            "prior_diagnoses": list(prior_diagnoses or [])[:10],
            "code_fingerprint": self.fingerprint_code(skill_code or ""),
        }
        self.on_log(
            f"OBSERVE: phase={phase} rc={returncode} "
            f"artifacts={len(artifacts)} code_fp={observation['code_fingerprint'][:8]}"
        )
        return observation

    def _scan_artifacts(self, result: Any, evidence: Any) -> list[dict[str, Any]]:
        """Independently list workspace artifacts relevant to the attempt."""
        found: list[dict[str, Any]] = []
        seen: set[str] = set()

        candidates: list[Path] = []
        if isinstance(result, dict):
            for key in ("path", "file", "output", "output_path", "filepath", "directory", "dir"):
                if result.get(key):
                    candidates.append(Path(str(result[key])))

        # Recent files in workspace (bounded)
        try:
            if self.workspace.exists():
                for root, _dirs, files in os.walk(self.workspace):
                    for name in files[:50]:
                        candidates.append(Path(root) / name)
                    break  # top level only for breadth; walk one level deep
                for child in list(self.workspace.iterdir())[:40]:
                    if child.is_dir():
                        for f in list(child.iterdir())[:20]:
                            if f.is_file():
                                candidates.append(f)
        except Exception:
            pass

        for raw in candidates:
            p = raw if raw.is_absolute() else self.workspace / raw
            key = str(p.resolve()) if p.exists() else str(p)
            if key in seen:
                continue
            seen.add(key)
            info: dict[str, Any] = {"path": str(p), "exists": p.exists()}
            if p.exists():
                try:
                    st = p.stat()
                    info["is_file"] = p.is_file()
                    info["is_dir"] = p.is_dir()
                    info["size"] = st.st_size if p.is_file() else None
                    if p.is_file() and st.st_size <= 2048:
                        try:
                            info["preview"] = p.read_text(encoding="utf-8", errors="replace")[:200]
                        except Exception:
                            info["preview"] = None
                except Exception as exc:
                    info["stat_error"] = str(exc)
            found.append(info)
            if len(found) >= 30:
                break

        # Always note evidence string length (not trusted as proof)
        if evidence:
            found.append({
                "path": "(skill_evidence)",
                "exists": False,
                "evidence_len": len(str(evidence)),
                "evidence_preview": str(evidence)[:200],
            })
        return found

    @staticmethod
    def fingerprint_code(code: str) -> str:
        normalized = "\n".join(
            line.rstrip()
            for line in (code or "").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def fingerprint_approach(approach: str) -> str:
        text = " ".join((approach or "").lower().split())
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
