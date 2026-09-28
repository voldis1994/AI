"""
JARVIS skill tester — runs candidate skills in an isolated Python subprocess.

Never imports/executes skill code inside the main JARVIS process.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from jarvis.skill_runner import DEFAULT_TIMEOUT, run_skill_subprocess


class SkillTester:
    """Test skills via subprocess with timeout / stdout / stderr / returncode."""

    def __init__(
        self,
        workspace: str | Path,
        on_log: Optional[Callable[[str], None]] = None,
        on_event: Optional[Callable[[str, dict], None]] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.on_log = on_log or (lambda _m: None)
        self.on_event = on_event or (lambda _k, _p: None)
        self.timeout = timeout

    def test(
        self,
        skill_path: str | Path,
        goal: str,
        args: Optional[dict] = None,
        timeout_hint: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Execute skill in a subprocess. Returns SKILL RESULT only.

        Fields:
          ok, passed (legacy alias for process-level success signal),
          result, error, evidence, returncode, stdout, stderr,
          timed_out, killed, crash, traceback

        Verifier must be called separately — this does NOT prove the goal.
        """
        path = Path(skill_path)
        timeout = float(timeout_hint if timeout_hint is not None else self.timeout)
        self.on_log(f"TEST: subprocess {path.name} (timeout={timeout}s)")

        skill_result = run_skill_subprocess(
            skill_path=path,
            goal=goal,
            workspace=self.workspace,
            args=args,
            mode="test",
            timeout=timeout,
        )

        # Process-level "passed" = subprocess finished without timeout/crash
        # and skill reported ok. Real goal proof is VERIFIER RESULT only.
        process_ok = (
            not skill_result.get("timed_out")
            and not skill_result.get("crash")
            and skill_result.get("returncode") in (0, 1)
        )
        skill_ok = bool(skill_result.get("ok"))
        passed = process_ok and skill_ok

        self.on_log(
            f"TEST: rc={skill_result.get('returncode')} "
            f"ok={skill_ok} timed_out={skill_result.get('timed_out')} "
            f"crash={skill_result.get('crash')}"
        )
        try:
            self.on_event(
                "subprocess",
                {
                    "mode": "test",
                    "command": f"python -c skill:{path.name}",
                    "stdout": str(skill_result.get("stdout") or "")[:8000],
                    "stderr": str(skill_result.get("stderr") or "")[:8000],
                    "traceback": str(skill_result.get("traceback") or "")[:8000],
                    "returncode": skill_result.get("returncode"),
                    "timed_out": bool(skill_result.get("timed_out")),
                    "ok": skill_ok,
                    "error": str(skill_result.get("error") or "")[:500],
                    "skill_path": str(path),
                },
            )
        except Exception:
            pass

        return {
            "ok": skill_ok,
            "passed": passed,
            "result": skill_result.get("result"),
            "error": skill_result.get("error"),
            "evidence": skill_result.get("evidence") or "",
            "traceback": skill_result.get("traceback") or "",
            "returncode": skill_result.get("returncode"),
            "stdout": skill_result.get("stdout") or "",
            "stderr": skill_result.get("stderr") or "",
            "timed_out": bool(skill_result.get("timed_out")),
            "killed": bool(skill_result.get("killed")),
            "crash": bool(skill_result.get("crash")),
            "raw": skill_result.get("raw"),
            "skill_path": str(path),
            "timeout": timeout,
            # Explicit separation for callers / logs
            "skill_result": skill_result,
        }
