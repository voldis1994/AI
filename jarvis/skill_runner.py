"""
Shared subprocess runner for JARVIS skills.

Skills never execute inside the main JARVIS process.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Optional


DEFAULT_TIMEOUT = 60.0


def run_skill_subprocess(
    skill_path: str | Path,
    goal: str,
    workspace: str | Path,
    args: Optional[dict] = None,
    mode: str = "test",
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """
    Execute skill.run(context) in a fresh Python subprocess.

    Returns skill_result dict with:
      ok, result, error, evidence, returncode, stdout, stderr,
      timed_out, killed, crash
    """
    path = Path(skill_path).resolve()
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        return {
            "ok": False,
            "result": None,
            "error": f"Skill file missing: {path}",
            "evidence": "",
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
            "killed": False,
            "crash": False,
        }

    context = {
        "goal": goal,
        "args": args or {},
        "workspace": str(workspace),
        "mode": mode,
    }

    # Marker files so binary stdout pollution cannot hide the result JSON
    with tempfile.TemporaryDirectory(prefix="jarvis_skill_") as tmp:
        tmp_path = Path(tmp)
        ctx_file = tmp_path / "context.json"
        out_file = tmp_path / "result.json"
        ctx_file.write_text(json.dumps(context, default=str), encoding="utf-8")

        runner = textwrap.dedent(
            f"""
            import json, importlib.util, sys, traceback
            from pathlib import Path

            skill_path = Path({str(path)!r})
            ctx_path = Path({str(ctx_file)!r})
            out_path = Path({str(out_file)!r})

            def write(payload):
                out_path.write_text(json.dumps(payload, default=str), encoding="utf-8")

            try:
                context = json.loads(ctx_path.read_text(encoding="utf-8"))
                spec = importlib.util.spec_from_file_location("jarvis_skill_sub", skill_path)
                if spec is None or spec.loader is None:
                    write({{"ok": False, "result": None, "error": "Cannot load skill spec", "evidence": ""}})
                    sys.exit(2)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if not hasattr(mod, "run") or not callable(mod.run):
                    write({{"ok": False, "result": None, "error": "Skill has no callable run(context)", "evidence": ""}})
                    sys.exit(2)
                raw = mod.run(context)
                if not isinstance(raw, dict):
                    write({{"ok": False, "result": raw, "error": "run() must return a dict", "evidence": ""}})
                    sys.exit(2)
                write({{
                    "ok": bool(raw.get("ok")),
                    "result": raw.get("result"),
                    "error": raw.get("error"),
                    "evidence": str(raw.get("evidence") or ""),
                    "raw": raw,
                }})
                sys.exit(0 if raw.get("ok") else 1)
            except Exception as exc:
                write({{
                    "ok": False,
                    "result": None,
                    "error": f"Runtime error: {{exc}}",
                    "evidence": "",
                    "traceback": traceback.format_exc(),
                }})
                sys.exit(1)
            """
        )

        proc = subprocess.Popen(
            [sys.executable, "-c", runner],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(workspace),
            start_new_session=True,
        )
        timed_out = False
        killed = False
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            stdout, stderr = _kill_process_group(proc)
        returncode = proc.returncode

        skill_payload: dict[str, Any] = {}
        if out_file.exists():
            try:
                skill_payload = json.loads(out_file.read_text(encoding="utf-8"))
            except Exception as exc:
                skill_payload = {
                    "ok": False,
                    "result": None,
                    "error": f"Failed to parse skill result file: {exc}",
                    "evidence": "",
                }

        crash = (returncode not in (0, 1, None) and not skill_payload) or (
            timed_out
        )
        if timed_out:
            error = f"Skill subprocess timed out after {timeout}s"
        elif not skill_payload:
            error = f"Skill subprocess produced no result (rc={returncode})"
            crash = True
        else:
            error = skill_payload.get("error")

        # Display command points at the skill module actually loaded (real path).
        # Isolation still uses python -c runner; argv kept for diagnostics.
        display_cmd = f"{sys.executable} {path}"
        return {
            "ok": bool(skill_payload.get("ok")) if skill_payload else False,
            "result": skill_payload.get("result") if skill_payload else None,
            "error": error,
            "evidence": str(skill_payload.get("evidence") or "") if skill_payload else "",
            "raw": skill_payload.get("raw") if skill_payload else None,
            "traceback": skill_payload.get("traceback", ""),
            "returncode": returncode,
            "stdout": stdout or "",
            "stderr": stderr or "",
            "timed_out": timed_out,
            "killed": killed or timed_out,
            "crash": crash,
            "skill_path": str(path),
            "command": display_cmd,
            "argv": [sys.executable, "-c", "<skill_runner>"],
        }


def _kill_process_group(proc: subprocess.Popen) -> tuple[str, str]:
    """Terminate, then kill, the subprocess process group."""
    killed_stdout, killed_stderr = "", ""
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        pgid = None

    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGTERM)
        else:
            proc.terminate()
    except Exception:
        pass

    try:
        killed_stdout, killed_stderr = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            pass
        try:
            killed_stdout, killed_stderr = proc.communicate(timeout=3)
        except Exception:
            killed_stdout, killed_stderr = "", ""
    return killed_stdout or "", killed_stderr or ""
