"""
JARVIS skill loader — executes ACTIVE skills in an isolated subprocess.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from jarvis.skill_runner import DEFAULT_TIMEOUT, run_skill_subprocess


class SkillLoader:
    """Execute trusted (ACTIVE) skills via subprocess — never in-process."""

    def __init__(
        self,
        skills_dir: str | Path,
        workspace: str | Path,
        timeout: float = DEFAULT_TIMEOUT,
        on_event: Optional[Callable[[str, dict], None]] = None,
    ) -> None:
        self.skills_dir = Path(skills_dir)
        self.workspace = Path(workspace)
        self.timeout = timeout
        self.on_event = on_event or (lambda _k, _p: None)

    def execute(
        self,
        skill_name: str,
        goal: str,
        args: Optional[dict] = None,
        file_path: Optional[str] = None,
        allow_non_active: bool = False,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        path = Path(file_path) if file_path else self.skills_dir / f"{skill_name}.py"
        skill_result = run_skill_subprocess(
            skill_path=path,
            goal=goal,
            workspace=self.workspace,
            args=args,
            mode="execute",
            timeout=float(timeout if timeout is not None else self.timeout),
        )
        try:
            self.on_event(
                "subprocess",
                {
                    "mode": "execute",
                    "command": f"python -c skill:{path.name}",
                    "stdout": str(skill_result.get("stdout") or "")[:8000],
                    "stderr": str(skill_result.get("stderr") or "")[:8000],
                    "traceback": str(skill_result.get("traceback") or "")[:8000],
                    "returncode": skill_result.get("returncode"),
                    "timed_out": bool(skill_result.get("timed_out")),
                    "ok": bool(skill_result.get("ok")),
                    "error": str(skill_result.get("error") or "")[:500],
                    "skill_path": str(path),
                },
            )
        except Exception:
            pass
        return {
            "ok": bool(skill_result.get("ok")),
            "result": skill_result.get("result"),
            "error": skill_result.get("error"),
            "evidence": str(skill_result.get("evidence") or ""),
            "returncode": skill_result.get("returncode"),
            "stdout": skill_result.get("stdout") or "",
            "stderr": skill_result.get("stderr") or "",
            "timed_out": bool(skill_result.get("timed_out")),
            "killed": bool(skill_result.get("killed")),
            "crash": bool(skill_result.get("crash")),
            "traceback": skill_result.get("traceback") or "",
            "raw": skill_result.get("raw"),
            "skill_result": skill_result,
            "allow_non_active": allow_non_active,
            "skill_path": str(path),
        }
