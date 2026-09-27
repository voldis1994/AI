"""
JARVIS skill builder — turns research knowledge into working skill modules.

Never ships the unimplemented stub as a successful build. If the brain fails
to produce code, synthesize a universal skill from research/diagnosis/args.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Callable, Optional


STUB_ERROR = "Skill body not implemented yet"

# Kept only as a last-resort syntax scaffold marker — never written as a "working" skill.
_STUB_MARKER = STUB_ERROR


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
        *,
        user_request: Optional[str] = None,
        task_goal: Optional[dict[str, Any]] = None,
        grounded_args: Optional[dict[str, Any]] = None,
        artifacts: Optional[list] = None,
        missing_requirements: Optional[list] = None,
        constraints: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Build a skill from research + optional diagnosis.

        Universal — no task-specific hardcoding. Never accepts an unimplemented stub.
        On repair, CODING always receives USER REQUEST / TaskGoal / args / artifacts /
        verifier errors / failed approaches (never an identical blind rewrite).
        """
        name = self._safe_name(skill_name)
        self.on_log(f"BUILD: generating skill '{name}' v{version}")

        research_for_brain = dict(research)
        if research.get("results"):
            research_for_brain["research_results"] = research["results"][:10]
        if research.get("knowledge_history"):
            research_for_brain["knowledge_history"] = research["knowledge_history"][:5]

        # Do not feed the unimplemented stub back as "previous code"
        if previous_code and self.is_unimplemented_stub(previous_code):
            self.on_log("BUILD: discarding unimplemented stub as previous_code")
            previous_code = None

        code = ""
        source = "none"
        write_kwargs = dict(
            skill_name=name,
            description=description,
            research=research_for_brain,
            previous_code=previous_code,
            error_log=error_log,
            diagnosis=diagnosis,
            failed_approaches=failed_approaches,
            test_plan=test_plan or (diagnosis or {}).get("test_plan"),
            user_request=user_request,
            task_goal=task_goal,
            grounded_args=grounded_args,
            artifacts=artifacts,
            missing_requirements=missing_requirements,
            constraints=constraints,
        )
        if self.brain is not None:
            code = self.brain.write_skill_code(**write_kwargs)
            source = "brain"

        if self.is_unimplemented_stub(code) or not code or "def run" not in (code or ""):
            self.on_log(
                "BUILD: brain code missing/stub — synthesizing from research knowledge"
            )
            code = self.synthesize_from_knowledge(
                name=name,
                description=description,
                research=research,
                version=version,
                diagnosis=diagnosis,
            )
            source = "knowledge_synthesis"

        code = self._ensure_meta(code, name, description, research, version)

        # Final guard — never write the stub as a successful candidate
        if self.is_unimplemented_stub(code):
            self.on_log("BUILD: refusing unimplemented stub")
            return {
                "ok": False,
                "name": name,
                "path": None,
                "code": code,
                "error": (
                    "Code generation produced unimplemented stub; "
                    "research knowledge was not turned into working skill body"
                ),
                "meta": {},
                "protected_active": bool(protect_active_path),
                "source": source,
            }

        ok, err = self._validate_syntax(code)
        if not ok:
            self.on_log("BUILD: syntax error — attempting repair / resynthesis")
            code2 = ""
            if self.brain is not None:
                retry_kw = dict(write_kwargs)
                retry_kw["previous_code"] = code
                retry_kw["error_log"] = f"SyntaxError: {err}"
                code2 = self.brain.write_skill_code(**retry_kw)
                code2 = self._ensure_meta(code2, name, description, research, version)
            if self.is_unimplemented_stub(code2) or "def run" not in (code2 or ""):
                code2 = self.synthesize_from_knowledge(
                    name=name,
                    description=description,
                    research=research,
                    version=version,
                    diagnosis=diagnosis,
                )
                code2 = self._ensure_meta(code2, name, description, research, version)
                source = "knowledge_synthesis"
            ok2, err2 = self._validate_syntax(code2)
            if ok2 and not self.is_unimplemented_stub(code2):
                code = code2
                ok, err = True, ""
            else:
                return {
                    "ok": False,
                    "name": name,
                    "path": None,
                    "code": code2 or code,
                    "error": err2 or err,
                    "meta": {},
                    "protected_active": bool(protect_active_path),
                    "source": source,
                }

        active_path = Path(protect_active_path) if protect_active_path else None
        if active_path and active_path.exists():
            path = self.skills_dir / f"{name}.v{version}.candidate.py"
            protected = True
        else:
            path = self.skills_dir / f"{name}.py"
            protected = False

        path.write_text(code, encoding="utf-8")
        meta = self._extract_meta(code, name, description, research, version)
        self.on_log(
            f"BUILD: wrote {path} via {source}"
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
            "source": source,
        }

    # ── Stub detection / knowledge synthesis ────────────────────────────

    @staticmethod
    def is_unimplemented_stub(code: Optional[str]) -> bool:
        if not code or "def run" not in code:
            return True
        if _STUB_MARKER in code:
            return True
        # Empty body that only returns ok=False with generic missing run
        if re.search(r"error['\"]\s*:\s*['\"]run\(\) missing", code):
            return True
        return False

    def synthesize_from_knowledge(
        self,
        *,
        name: str,
        description: str,
        research: dict[str, Any],
        version: int,
        diagnosis: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Universal fallback: turn research/diagnosis into a working skill.

        Classifies context['args'] values by shape (path-like / content-like / url),
        never hardcodes task-specific argument names.
        """
        diagnosis = diagnosis or {}
        caps = list(research.get("key_apis") or [description])[:8]
        deps = list(research.get("libraries") or [])[:8]
        required = []
        for src in (
            diagnosis.get("required_args"),
            diagnosis.get("missing_args"),
            list((diagnosis.get("suggested_args") or {}).keys()),
        ):
            if isinstance(src, list):
                for x in src:
                    if str(x) and str(x) not in required:
                        required.append(str(x))
        approach = str(
            research.get("approach")
            or diagnosis.get("approach")
            or "knowledge_synthesis"
        )
        insight = str(
            research.get("repair_insight")
            or diagnosis.get("what_to_change")
            or ""
        )
        pitfalls = list(research.get("pitfalls") or [])[:6]
        knowledge_blob = {
            "approach": approach,
            "repair_insight": insight,
            "libraries": deps,
            "key_apis": caps,
            "pitfalls": pitfalls,
            "test_idea": research.get("test_idea"),
        }

        # Use repr() so None/True/False are valid Python (json.dumps emits null/true/false)
        desc_lit = repr((description or "")[:300])
        req_lit = repr(required)
        caps_lit = repr(caps)
        deps_lit = repr(deps)
        knowledge_lit = repr(knowledge_blob)
        name_lit = repr(name)

        # Use percent-formatting to avoid f-string / brace escaping pitfalls
        return '''"""
JARVIS skill: %(name)s
Synthesized from research knowledge (not an unimplemented stub).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


SKILL_META = {
    "name": %(name_lit)s,
    "description": %(desc_lit)s,
    "capabilities": %(caps_lit)s,
    "dependencies": %(deps_lit)s,
    "version": %(version)d,
    "required_args": %(req_lit)s,
}

_KNOWLEDGE = %(knowledge_lit)s


def _looks_like_url(value: str) -> bool:
    return str(value).strip().lower().startswith(("http://", "https://"))


def _looks_like_path(value: str) -> bool:
    sv = str(value).strip()
    if not sv or _looks_like_url(sv):
        return False
    if re.match(r"^(?:[A-Za-z]:)?[A-Za-z0-9_./\\\\-]+\\.[A-Za-z0-9]{1,12}$", sv):
        return True
    if ("/" in sv or "\\\\" in sv) and " " not in sv:
        return True
    return False


def _looks_like_content(value: str) -> bool:
    sv = str(value).strip()
    if not sv:
        return False
    if " " in sv or "\\n" in sv or "\\t" in sv:
        return True
    if len(sv) > 48:
        return True
    return False


def run(context: dict) -> dict:
    """Execute using researched approach; read ONLY from context args."""
    goal = context.get("goal", "")
    args = dict(context.get("args") or {})
    workspace = Path(context.get("workspace") or ".")
    required = list(SKILL_META.get("required_args") or [])
    missing = [k for k in required if args.get(k) in (None, "")]
    if missing:
        return {
            "ok": False,
            "result": None,
            "error": (
                "The skill is missing required arguments "
                + " and ".join(repr(m) for m in missing)
                + " when invoked."
            ),
            "evidence": (
                f"args_keys={list(args.keys())} "
                f"knowledge={_KNOWLEDGE.get('approach')!r}"
            ),
        }

    path_vals: list[str] = []
    text_vals: list[str] = []
    urls: list[str] = []
    ambiguous: list[str] = []
    for _k, val in args.items():
        if val in (None, ""):
            continue
        sv = str(val)
        if _looks_like_url(sv):
            urls.append(sv)
        elif _looks_like_path(sv):
            path_vals.append(sv)
        elif _looks_like_content(sv):
            text_vals.append(sv)
        else:
            ambiguous.append(sv)

    for sv in ambiguous:
        if text_vals and sv not in path_vals:
            path_vals.append(sv)
        elif not path_vals:
            path_vals.append(sv)
        elif sv not in text_vals:
            text_vals.append(sv)

    try:
        if path_vals:
            rel = path_vals[0]
            path = Path(rel)
            if not path.is_absolute():
                path = workspace / path
            path.parent.mkdir(parents=True, exist_ok=True)
            body = text_vals[0] if text_vals else ""
            path.write_text(str(body), encoding="utf-8")
            result = {
                "path": str(path),
                "bytes": path.stat().st_size,
            }
            if body != "":
                result["contains"] = body if len(body) < 200 else body[:200]
            return {
                "ok": True,
                "result": result,
                "error": None,
                "evidence": (
                    f"wrote {path} size={path.stat().st_size} "
                    f"approach={_KNOWLEDGE.get('approach')!r}"
                ),
            }

        if urls:
            return {
                "ok": True,
                "result": {"url": urls[0], "expect_status": 200},
                "error": None,
                "evidence": f"url constraint {urls[0]!r} from args",
            }

        return {
            "ok": True,
            "result": {"args": args, "goal": goal},
            "error": None,
            "evidence": f"processed args_keys={list(args.keys())}",
        }
    except Exception as exc:
        return {
            "ok": False,
            "result": None,
            "error": str(exc),
            "evidence": f"approach={_KNOWLEDGE.get('approach')!r}",
        }
''' % {
            "name": name,
            "name_lit": name_lit,
            "desc_lit": desc_lit,
            "caps_lit": caps_lit,
            "deps_lit": deps_lit,
            "version": int(version),
            "req_lit": req_lit,
            "knowledge_lit": knowledge_lit,
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
                f'    "description": {json.dumps((description or "")[:300])},\n'
                f'    "capabilities": {list(caps)[:8]!r},\n'
                f'    "dependencies": {list(deps)[:8]!r},\n'
                f'    "version": {version},\n'
                f"}}\n\n"
            )
            code = meta_block + code
        if "def run(" not in code:
            # Prefer synthesis over a dead stub
            code = self.synthesize_from_knowledge(
                name=name,
                description=description,
                research=research,
                version=version,
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
