"""
JARVIS skill builder — writes Python skill modules to skills/.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Callable, Optional


SKILL_TEMPLATE = '''"""
JARVIS skill: {name}
Auto-generated candidate skill. Status managed by capability registry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


SKILL_META = {{
    "name": "{name}",
    "description": "{description}",
    "capabilities": {capabilities!r},
    "dependencies": {dependencies!r},
    "version": {version},
}}


def run(context: dict) -> dict:
    """Execute the skill.

    context keys: goal (str), args (dict), workspace (str)
    Must return: ok (bool), result (Any), error (str|None), evidence (str)
    """
    goal = context.get("goal", "")
    args = context.get("args") or {{}}
    workspace = Path(context.get("workspace") or ".")
    try:
        # TODO: replace with learned implementation
        return {{
            "ok": False,
            "result": None,
            "error": "Skill body not implemented yet",
            "evidence": f"workspace={{workspace}} goal={{goal!r}} args={{args!r}}",
        }}
    except Exception as exc:
        return {{
            "ok": False,
            "result": None,
            "error": str(exc),
            "evidence": "",
        }}
'''


class SkillBuilder:
    """Creates and updates skill Python files under skills/."""

    def __init__(
        self,
        skills_dir: str | Path,
        brain: Any = None,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.brain = brain
        self.on_log = on_log or (lambda _m: None)
        # Ensure skills is a package
        init = self.skills_dir / "__init__.py"
        if not init.exists():
            init.write_text('"""JARVIS learned skills."""\n', encoding="utf-8")

    def build(
        self,
        skill_name: str,
        description: str,
        research: dict[str, Any],
        version: int = 1,
        previous_code: Optional[str] = None,
        error_log: Optional[str] = None,
        protect_active_path: Optional[str | Path] = None,
        diagnosis: Optional[dict[str, Any]] = None,
        failed_approaches: Optional[list] = None,
        test_plan: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Build a skill file from research + optional diagnosis.

        Core stays universal — no task-specific code paths here.
        If protect_active_path is set, write a candidate file only.
        """
        name = self._safe_name(skill_name)
        self.on_log(f"BUILD: generating skill '{name}' v{version}")

        research_for_brain = dict(research)
        if research.get("results"):
            research_for_brain["research_results"] = research["results"][:10]

        code = ""
        if self.brain is not None:
            code = self.brain.write_skill_code(
                skill_name=name,
                description=description,
                research=research_for_brain,
                previous_code=previous_code,
                error_log=error_log,
                diagnosis=diagnosis,
                failed_approaches=failed_approaches,
                test_plan=test_plan or (diagnosis or {}).get("test_plan"),
            )

        if not code or "def run" not in code:
            caps = research.get("key_apis") or [description]
            deps = research.get("libraries") or []
            code = SKILL_TEMPLATE.format(
                name=name,
                description=description.replace('"', "'")[:200],
                capabilities=list(caps)[:8],
                dependencies=list(deps)[:8],
                version=version,
            )

        code = self._ensure_meta(code, name, description, research, version)
        ok, err = self._validate_syntax(code)
        if not ok:
            self.on_log("BUILD: syntax error — attempting repair wrap")
            if self.brain is not None and not error_log:
                code2 = self.brain.write_skill_code(
                    skill_name=name,
                    description=description,
                    research=research_for_brain,
                    previous_code=code,
                    error_log=f"SyntaxError: {err}",
                    diagnosis=diagnosis,
                    failed_approaches=failed_approaches,
                    test_plan=test_plan,
                )
                code2 = self._ensure_meta(code2, name, description, research, version)
                ok2, err2 = self._validate_syntax(code2)
                if ok2:
                    code = code2
                    ok, err = True, ""
                else:
                    return {
                        "ok": False,
                        "name": name,
                        "path": None,
                        "code": code2,
                        "error": err2,
                        "meta": {},
                        "protected_active": bool(protect_active_path),
                    }
            else:
                return {
                    "ok": False,
                    "name": name,
                    "path": None,
                    "code": code,
                    "error": err,
                    "meta": {},
                    "protected_active": bool(protect_active_path),
                }

        active_path = Path(protect_active_path) if protect_active_path else None
        if active_path and active_path.exists():
            # Never overwrite ACTIVE before PASS — write candidate beside it
            path = self.skills_dir / f"{name}.v{version}.candidate.py"
            protected = True
        else:
            path = self.skills_dir / f"{name}.py"
            protected = False

        path.write_text(code, encoding="utf-8")
        meta = self._extract_meta(code, name, description, research, version)
        self.on_log(
            f"BUILD: wrote {path}"
            + (" (ACTIVE protected)" if protected else "")
        )
        return {
            "ok": True,
            "name": name,
            "path": str(path),
            "code": code,
            "error": None,
            "meta": meta,
            "protected_active": protected,
            "active_path": str(active_path) if active_path else None,
        }

    def read_skill_code(self, skill_name: str, path: Optional[str | Path] = None) -> Optional[str]:
        if path:
            p = Path(path)
            if p.exists():
                return p.read_text(encoding="utf-8")
        p = self.skills_dir / f"{self._safe_name(skill_name)}.py"
        if p.exists():
            return p.read_text(encoding="utf-8")
        return None

    def _ensure_meta(
        self,
        code: str,
        name: str,
        description: str,
        research: dict,
        version: int,
    ) -> str:
        if "SKILL_META" not in code:
            caps = research.get("key_apis") or [description]
            deps = research.get("libraries") or []
            meta_block = (
                f"\nSKILL_META = {{\n"
                f'    "name": "{name}",\n'
                f'    "description": """{description[:300]}""",\n'
                f'    "capabilities": {list(caps)[:8]!r},\n'
                f'    "dependencies": {list(deps)[:8]!r},\n'
                f'    "version": {version},\n'
                f"}}\n\n"
            )
            code = meta_block + code
        if "def run(" not in code:
            code += (
                "\n\ndef run(context: dict) -> dict:\n"
                "    return {'ok': False, 'result': None, "
                "'error': 'run() missing', 'evidence': ''}\n"
            )
        return code

    def _extract_meta(
        self,
        code: str,
        name: str,
        description: str,
        research: dict,
        version: int,
    ) -> dict[str, Any]:
        meta = {
            "name": name,
            "description": description,
            "capabilities": list(research.get("key_apis") or [description])[:8],
            "dependencies": list(research.get("libraries") or [])[:8],
            "version": version,
        }
        try:
            tree = ast.parse(code)
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name) and t.id == "SKILL_META":
                            meta.update(ast.literal_eval(node.value))
        except Exception:
            pass
        meta["name"] = name
        meta["version"] = version
        return meta

    @staticmethod
    def _validate_syntax(code: str) -> tuple[bool, str]:
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as exc:
            return False, f"{exc.msg} at line {exc.lineno}"

    @staticmethod
    def _safe_name(name: str) -> str:
        s = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip().lower())
        s = re.sub(r"_+", "_", s).strip("_")
        if not s:
            s = "skill"
        if s[0].isdigit():
            s = "skill_" + s
        return s[:64]
