"""
JARVIS skill tester — runs candidate skills in isolation and reports evidence.
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Optional


class SkillTester:
    """Loads a skill module from disk and invokes run(context)."""

    def __init__(
        self,
        workspace: str | Path,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.on_log = on_log or (lambda _m: None)

    def test(
        self,
        skill_path: str | Path,
        goal: str,
        args: Optional[dict] = None,
        timeout_hint: float = 60.0,
    ) -> dict[str, Any]:
        """
        Execute skill.run. Returns structured test result.
        Does NOT mark the skill ACTIVE — that is the verifier/registry's job.
        """
        path = Path(skill_path)
        self.on_log(f"TEST: loading {path.name}")
        if not path.exists():
            return {
                "ok": False,
                "passed": False,
                "result": None,
                "error": f"Skill file missing: {path}",
                "evidence": "",
                "traceback": "",
            }

        try:
            module = self._load_module(path)
        except Exception as exc:
            tb = traceback.format_exc()
            return {
                "ok": False,
                "passed": False,
                "result": None,
                "error": f"Import failed: {exc}",
                "evidence": "",
                "traceback": tb,
            }

        if not hasattr(module, "run") or not callable(module.run):
            return {
                "ok": False,
                "passed": False,
                "result": None,
                "error": "Skill has no callable run(context)",
                "evidence": "",
                "traceback": "",
            }

        context = {
            "goal": goal,
            "args": args or {},
            "workspace": str(self.workspace),
            "mode": "test",
        }
        self.on_log(f"TEST: executing run() for goal={goal!r}")
        try:
            raw = module.run(context)
        except Exception as exc:
            tb = traceback.format_exc()
            return {
                "ok": False,
                "passed": False,
                "result": None,
                "error": f"Runtime error: {exc}",
                "evidence": "",
                "traceback": tb,
            }

        if not isinstance(raw, dict):
            return {
                "ok": False,
                "passed": False,
                "result": raw,
                "error": "run() must return a dict",
                "evidence": "",
                "traceback": "",
            }

        ok = bool(raw.get("ok"))
        evidence = str(raw.get("evidence") or "")
        error = raw.get("error")
        # Minimal bar: ok True AND non-empty evidence
        passed = ok and bool(evidence.strip())
        if ok and not evidence.strip():
            error = (str(error) + "; " if error else "") + "ok=True but evidence empty"
            passed = False
            ok = False

        return {
            "ok": ok,
            "passed": passed,
            "result": raw.get("result"),
            "error": error,
            "evidence": evidence,
            "traceback": "",
            "raw": raw,
            "timeout_hint": timeout_hint,
        }

    def _load_module(self, path: Path):
        mod_name = f"jarvis_skill_{path.stem}_{abs(hash(str(path.resolve()))) % 10**8}"
        # Drop cached module so repairs reload
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        spec = importlib.util.spec_from_file_location(mod_name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        return module
