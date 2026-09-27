"""
JARVIS skill loader — loads and executes ACTIVE skills only (trusted path).
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path
from typing import Any, Optional


class SkillLoader:
    """Execute trusted (ACTIVE) skills from the skills directory."""

    def __init__(self, skills_dir: str | Path, workspace: str | Path) -> None:
        self.skills_dir = Path(skills_dir)
        self.workspace = Path(workspace)

    def execute(
        self,
        skill_name: str,
        goal: str,
        args: Optional[dict] = None,
        file_path: Optional[str] = None,
        allow_non_active: bool = False,
    ) -> dict[str, Any]:
        """
        Load skill by name and run it.
        Caller must ensure skill is ACTIVE unless allow_non_active=True.
        """
        path = Path(file_path) if file_path else self.skills_dir / f"{skill_name}.py"
        if not path.exists():
            return {
                "ok": False,
                "result": None,
                "error": f"Skill file not found: {path}",
                "evidence": "",
            }
        try:
            module = self._load(path)
            if not hasattr(module, "run"):
                return {
                    "ok": False,
                    "result": None,
                    "error": "No run() in skill",
                    "evidence": "",
                }
            context = {
                "goal": goal,
                "args": args or {},
                "workspace": str(self.workspace),
                "mode": "execute",
                "allow_non_active": allow_non_active,
            }
            raw = module.run(context)
            if not isinstance(raw, dict):
                return {
                    "ok": False,
                    "result": raw,
                    "error": "Skill returned non-dict",
                    "evidence": "",
                }
            return {
                "ok": bool(raw.get("ok")),
                "result": raw.get("result"),
                "error": raw.get("error"),
                "evidence": str(raw.get("evidence") or ""),
                "raw": raw,
            }
        except Exception as exc:
            return {
                "ok": False,
                "result": None,
                "error": str(exc),
                "evidence": "",
                "traceback": traceback.format_exc(),
            }

    def _load(self, path: Path):
        mod_name = f"jarvis_active_{path.stem}"
        if mod_name in sys.modules:
            # Reload to pick up repairs
            del sys.modules[mod_name]
        spec = importlib.util.spec_from_file_location(mod_name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        return module
