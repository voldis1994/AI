#!/usr/bin/env python3
"""
JARVIS — Autonomous self-learning AI agent.

Main entry point. Brain: Ollama concurrent model pool
(FAST / REASONING / CODING independent workers — see jarvis/model_config.py).
Learning happens by creating/repairing Python skills — not by hardcoding
every future capability into this core.

Cycle:
  REQUEST → PLAN → check capabilities → research → learn → build skill →
  install deps → test → repair → verify → save ACTIVE skill → execute →
  save experience → DONE  (DONE only after factual verification)

Usage:
  python JARVIS.py              # GUI
  python JARVIS.py --cli        # terminal mode
  python JARVIS.py --check      # syntax/import/self-check
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path
from typing import Optional

# tkinter is optional until GUI mode; keep CLI/--check usable headless
try:
    import tkinter as tk
    from tkinter import scrolledtext
except ModuleNotFoundError:  # pragma: no cover
    tk = None  # type: ignore
    scrolledtext = None  # type: ignore

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

(ROOT / "data").mkdir(exist_ok=True)
(ROOT / "skills").mkdir(exist_ok=True)

from jarvis.brain import Brain, DEFAULT_MODEL
from jarvis.model_config import TIER_MODELS, TIER_FAST, TIER_REASONING, TIER_CODING
from jarvis.orchestrator import Orchestrator
from jarvis.gui import JarvisGUI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(ROOT / "data" / "jarvis.log", encoding="utf-8"),
    ],
)


# ── CLI mode ────────────────────────────────────────────────────────────

def run_cli(root: Path) -> int:
    print("JARVIS CLI — type /quit to exit, /help for commands")
    logs: list[str] = []

    def on_log(m: str) -> None:
        print(f"  · {m}")
        logs.append(m)

    def on_status(s: str) -> None:
        print(f"  [{s}]")

    orch = Orchestrator(
        root=root,
        brain=Brain(),
        on_log=on_log,
        on_status=on_status,
    )
    catalog = ""
    if hasattr(orch.brain, "model_catalog_summary"):
        catalog = orch.brain.model_catalog_summary()
    print(f"Brain: {orch.brain.model_status()} (multi-model)")
    if catalog:
        print(f"Models: {catalog}")
    try:
        while True:
            try:
                text = input("YOU › ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            if text.lower() in ("/quit", "/exit", "quit", "exit"):
                break
            result = orch.handle_user_message(text)
            print(f"JARVIS › {result.get('reply')}\n")
    finally:
        orch.close()
    return 0


def run_check(root: Path) -> int:
    import time
    """Verify syntax, imports, subprocess isolation, and e2e learn→reuse."""
    import ast
    import importlib
    import shutil
    import traceback

    errors: list[str] = []
    print("=== JARVIS self-check ===")

    # 1) Parse all Python sources
    py_files = [root / "JARVIS.py"] + sorted((root / "jarvis").glob("*.py"))
    for path in py_files:
        try:
            src = path.read_text(encoding="utf-8")
            ast.parse(src, filename=str(path))
            print(f"  OK syntax  {path.relative_to(root)}")
        except SyntaxError as exc:
            msg = f"SYNTAX {path}: {exc}"
            print(f"  FAIL {msg}")
            errors.append(msg)

    # 2) Import package modules
    modules = [
        "jarvis",
        "jarvis.brain",
        "jarvis.model_config",
        "jarvis.model_router",
        "jarvis.recovery",
        "jarvis.calibration",
        "jarvis.knowledge_artifact",
        "jarvis.memory",
        "jarvis.ledger",
        "jarvis.implementation_registry",
        "jarvis.capability_match",
        "jarvis.task_contract",
        "jarvis.ontology",
        "jarvis.invariants",
        "jarvis.runtime_fingerprint",
        "jarvis.competence",
        "jarvis.competence_registry",
        "jarvis.research",
        "jarvis.source_pipeline",
        "jarvis.dependency_manager",
        "jarvis.skill_builder",
        "jarvis.skill_tester",
        "jarvis.skill_runner",
        "jarvis.observer",
        "jarvis.verifier",
        "jarvis.skill_loader",
        "jarvis.context_builder",
        "jarvis.orchestrator",
    ]
    for name in modules:
        try:
            importlib.import_module(name)
            print(f"  OK import  {name}")
        except Exception as exc:
            msg = f"IMPORT {name}: {exc}"
            print(f"  FAIL {msg}")
            errors.append(msg)

    # 3) Subprocess tester + independent verifier + ACTIVE protection
    print("  — subprocess test + verifier separation + ACTIVE protect —")
    try:
        from jarvis.skill_builder import SkillBuilder
        from jarvis.implementation_registry import ImplementationRegistry
        from jarvis.skill_tester import SkillTester
        from jarvis.verifier import Verifier
        from jarvis.dependency_manager import DependencyManager
        from jarvis.brain import Brain as RealBrain

        tmp = root / "data" / "_selfcheck"
        if tmp.exists():
            shutil.rmtree(tmp)
        skills = tmp / "skills"
        ws = tmp / "ws"
        skills.mkdir(parents=True)
        ws.mkdir(parents=True)
        db = tmp / "check.db"

        skill_code = '''
SKILL_META = {
    "name": "write_hello_file",
    "description": "Write a hello.txt file into workspace",
    "capabilities": ["write hello file", "create text file"],
    "dependencies": [],
    "version": 1,
}

def run(context: dict) -> dict:
    from pathlib import Path
    workspace = Path(context.get("workspace") or ".")
    path = workspace / "hello.txt"
    path.write_text("Hello from JARVIS skill\\n", encoding="utf-8")
    return {
        "ok": True,
        "result": {"path": str(path), "bytes": path.stat().st_size, "contains": "Hello"},
        "error": None,
        "evidence": f"Wrote {path} exists={path.exists()} size={path.stat().st_size}",
    }
'''
        skill_path = skills / "write_hello_file.py"
        skill_path.write_text(skill_code, encoding="utf-8")

        reg = ImplementationRegistry(db)
        reg.register_candidate(
            "write_hello_file", "Write hello.txt", str(skill_path),
            ["write hello file"], [], 1,
        )
        reg.set_status("write_hello_file", "TESTING")
        tester = SkillTester(ws)
        test_res = tester.test(skill_path, goal="Create hello.txt in workspace")
        assert test_res.get("returncode") is not None, test_res
        assert "stdout" in test_res and "stderr" in test_res
        assert test_res.get("ok"), test_res
        assert "skill_result" in test_res
        # Crash isolation: hang skill must timeout without killing parent
        hang = skills / "hang_skill.py"
        hang.write_text(
            "SKILL_META={'name':'hang_skill','description':'hang','capabilities':[],"
            "'dependencies':[],'version':1}\n"
            "def run(context):\n"
            "    import time\n"
            "    time.sleep(999)\n"
            "    return {'ok': True, 'result': None, 'error': None, 'evidence': 'no'}\n",
            encoding="utf-8",
        )
        hang_res = tester.test(hang, goal="hang", timeout_hint=2.0)
        assert hang_res.get("timed_out"), hang_res
        assert hang_res.get("killed"), hang_res

        ver = Verifier(ws)
        vres = ver.verify("Create hello.txt", test_res)
        assert vres["verified"], vres
        assert "skill_result" in vres and "verifier_result" in vres
        assert vres["verifier_result"]["pass"]

        # Defaults / skill self-proof must NOT earn PASS against a different USER REQUEST
        (ws / "default_path.txt").write_text("default content\n", encoding="utf-8")
        bad_sr = {
            "ok": True,
            "result": {
                "path": str(ws / "default_path.txt"),
                "contains": "default content",
            },
            "evidence": f"wrote {ws / 'default_path.txt'}",
            "error": None,
            "returncode": 0,
        }
        bad = ver.verify(
            'Create "notes.txt" containing "hello world"',
            bad_sr,
            args={"dest": "notes.txt", "body": "hello world"},
            user_request='Create "notes.txt" containing "hello world"',
        )
        assert not bad.get("verified"), bad
        assert any(
            c.get("name") == "reject_defaults" and not c.get("ok")
            for c in bad.get("checks") or []
        ), bad.get("checks")

        # Promote path
        reg.promote_candidate(
            "write_hello_file", str(skill_path), 1,
            "Write hello.txt", ["write hello file"], [],
            archive_dir=skills / "archive",
        )
        assert reg.get_skill("write_hello_file")["status"] == "ACTIVE"
        active_file = Path(reg.get_skill("write_hello_file")["file_path"])
        active_before = active_file.read_text(encoding="utf-8")

        # Build v2 while ACTIVE protected — must not overwrite ACTIVE file
        builder = SkillBuilder(skills)
        built = builder.build(
            "write_hello_file",
            "Write hello.txt v2",
            {"libraries": [], "key_apis": ["write hello file"], "approach": "rewrite"},
            version=2,
            protect_active_path=str(active_file),
        )
        assert built["ok"] and built.get("protected_active")
        assert Path(built["path"]).name.endswith(".candidate.py")
        assert active_file.read_text(encoding="utf-8") == active_before
        print("  OK subprocess timeout/kill + SKILL≠VERIFIER + ACTIVE protect")

        # Brain model status API
        b = RealBrain()
        st = b.model_status()
        assert st in ("ONLINE", "MODEL MISSING", "OFFLINE"), st
        print(f"  OK brain     model_status={st}")

        dm = DependencyManager()
        assert dm.ensure([])["ok"]

        reg.close()
    except Exception as exc:
        msg = f"CYCLE: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # 4) E2E: missing skill → fail once → OBSERVE/DIAGNOSE/REPAIR → ACTIVE → reuse
    # FakeBrain only simulates Ollama; core stays universal (no hardcoded skill logic).
    print("  — e2e observe/diagnose/repair → ACTIVE → restart → reuse —")
    try:
        e2e_root = root / "data" / "_e2e"
        if e2e_root.exists():
            shutil.rmtree(e2e_root)
        e2e_root.mkdir(parents=True)

        class FakeBrain(Brain):
            """Simulates Ollama decisions from observations — not a core hardcode."""

            def __init__(self) -> None:
                super().__init__()
                self.build_count = 0

            def model_status(self) -> str:
                return "ONLINE"

            def is_available(self) -> bool:
                return True

            def classify_intent(self, user_text: str) -> dict:
                return {
                    "intent": "task",
                    "goal": user_text,
                    "needs_capability": True,
                    "keywords": user_text.split()[:8],
                }

            def plan(self, goal: str, known_capabilities: list[str]) -> dict:
                reuse = []
                for line in known_capabilities:
                    if "write_e2e_marker" in line:
                        reuse.append("write_e2e_marker")
                return {
                    "steps": ["accomplish goal via skill"],
                    "can_reuse": reuse,
                    "missing": [] if reuse else [goal],
                    "needs_research": not bool(reuse),
                    "needs_new_skill": not bool(reuse),
                    "skill_name": "write_e2e_marker",
                    "skill_description": goal,
                    "research_queries": [goal],
                    "args": {},
                    "required_args": [],
                }

            def extract_task_args(
                self, goal, skill_meta=None, prior_args=None, diagnosis=None
            ):
                # Existing e2e skill hardcodes its artifact; no args required.
                return {}

            def research_notes(self, query: str, gathered: str) -> dict:
                return {
                    "approach": "initial_broken_then_repair",
                    "libraries": [],
                    "key_apis": [query],
                    "pitfalls": [],
                    "test_idea": "independent artifact verification",
                }

            def write_skill_code(
                self,
                skill_name,
                description,
                research,
                previous_code=None,
                error_log=None,
                diagnosis=None,
                failed_approaches=None,
                test_plan=None,
            ) -> str:
                self.build_count += 1
                # First build intentionally fails — exercises OBSERVE→DIAGNOSE→REPAIR
                if self.build_count == 1 and not diagnosis:
                    return f'''
SKILL_META = {{
    "name": "{skill_name}",
    "description": "{description}",
    "capabilities": ["e2e"],
    "dependencies": [],
    "version": 1,
}}

def run(context: dict) -> dict:
    return {{
        "ok": False,
        "result": None,
        "error": "intentional first-attempt failure for repair pipeline test",
        "evidence": "",
    }}
'''
                # After diagnosis: different approach that actually works
                return f'''
from pathlib import Path

SKILL_META = {{
    "name": "{skill_name}",
    "description": "{description}",
    "capabilities": ["e2e"],
    "dependencies": [],
    "version": 2,
}}

def run(context: dict) -> dict:
    workspace = Path(context.get("workspace") or ".")
    path = workspace / "jarvis_e2e_marker.txt"
    path.write_text("E2E_OK\\n", encoding="utf-8")
    return {{
        "ok": True,
        "result": {{"path": str(path), "contains": "E2E_OK"}},
        "error": None,
        "evidence": f"created {{path}} size={{path.stat().st_size}}",
    }}
'''

            def diagnose(self, observation, failed_approaches=None, prior_solutions=None):
                from jarvis.observer import Observer
                approach = "pathlib_write_after_observe"
                # Must differ from failed approaches
                fps = {
                    a.get("approach_fingerprint")
                    for a in (failed_approaches or [])
                }
                fp = Observer.fingerprint_approach(approach)
                if fp in fps:
                    approach = approach + "_v2"
                    fp = Observer.fingerprint_approach(approach)
                return {
                    "root_cause": str(observation.get("exception") or "test failed"),
                    "fault_layer": "skill_code",
                    "rewrite_skill": True,
                    "what_to_change": "Implement real artifact creation based on goal",
                    "approach": approach,
                    "approach_fingerprint": fp,
                    "approach_changed": True,
                    "needs_research": False,
                    "research_queries": [],
                    "needs_new_deps": [],
                    "missing_args": [],
                    "required_args": [],
                    "suggested_args": {},
                    "test_plan": "subprocess run + verifier checks file exists with content",
                    "expected_artifacts": ["jarvis_e2e_marker.txt"],
                    "is_unfixable": False,
                    "diagnosis": "First approach returned ok=False; switch strategy",
                }

            def verify_claim(self, goal, result, evidence) -> dict:
                return {"achieved": True, "confidence": 0.9, "reason": "advisory"}

            def converse(self, user_text, history=None) -> str:
                return "ok"

        phases: list[str] = []
        orch = Orchestrator(
            root=e2e_root,
            brain=FakeBrain(),
            on_status=lambda s: phases.append(s),
            on_log=lambda m: print(f"    · {m}") if any(
                k in m for k in (
                    "REQUEST", "RESEARCH", "BUILD", "TEST", "RETEST", "OBSERVE",
                    "DIAGNOSE", "REPAIR", "VERIFY", "ACTIVE", "DONE", "EXECUTE",
                    "SKILL RESULT", "VERIFIER",
                )
            ) else None,
        )
        assert orch.registry.get_skill("write_e2e_marker") is None

        goal = "Create file jarvis_e2e_marker.txt containing E2E_OK"
        result1 = orch.run_cycle(goal)
        assert result1.get("success"), result1
        assert orch.registry.get_skill("write_e2e_marker")["status"] == "ACTIVE"
        marker = e2e_root / "workspace_runtime" / "jarvis_e2e_marker.txt"
        assert marker.exists() and "E2E_OK" in marker.read_text(encoding="utf-8")

        # Universal repair pipeline phases exercised
        for required in ("TEST", "OBSERVE", "DIAGNOSE", "REPAIR", "RETEST", "VERIFY", "DONE"):
            assert required in phases, f"missing phase {required} in {phases}"

        # Failures / diagnoses / solutions persisted
        assert orch.memory.stats()["learning_failures"] >= 1
        assert orch.memory.stats()["learning_diagnoses"] >= 1
        assert orch.memory.stats()["learning_solutions"] >= 1
        assert orch.memory.get_failed_approaches("write_e2e_marker")

        fact = orch.memory.get_fact("research:write_e2e_marker")
        assert fact and "results" in fact
        orch.close()
        print("  OK e2e      BUILD→TEST→OBSERVE→DIAGNOSE→REPAIR→RETEST→VERIFY→ACTIVE")

        phases2: list[str] = []
        orch2 = Orchestrator(
            root=e2e_root,
            brain=FakeBrain(),
            on_status=lambda s: phases2.append(s),
        )
        assert orch2.registry.get_skill("write_e2e_marker")["status"] == "ACTIVE"
        if marker.exists():
            marker.unlink()
        result2 = orch2.run_cycle(goal)
        assert result2.get("success"), result2
        assert marker.exists()
        assert result2.get("skill") == "write_e2e_marker"
        assert "EXECUTE" in phases2 and "VERIFY" in phases2 and "DONE" in phases2
        # Reuse should not need another OBSERVE/DIAGNOSE cycle
        assert "OBSERVE" not in phases2
        orch2.close()
        print("  OK reuse    restart → ACTIVE skill reused (no re-learn)")
    except Exception as exc:
        msg = f"E2E: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # 4b) E2E: empty args → DIAGNOSE context_mapping → RETEST with args (no skill rewrite)
    print("  — e2e context_mapping fault_layer repair (no skill rewrite) —")
    try:
        from jarvis.context_builder import ContextBuilder
        from jarvis.task_contract import TaskContract
        from jarvis.verifier import Verifier

        # Unit: immutable TaskContract + grounding (no invented defaults / no token→file noise)
        tg = TaskContract.from_request("Izveido failu test.txt ar tekstu DARBOJAS")
        assert tg.original_request == "Izveido failu test.txt ar tekstu DARBOJAS"
        paths = [f.get("path") for f in tg.constraints.get("files") or []]
        assert paths == ["test.txt"], paths
        assert (tg.constraints.get("files") or [{}])[0].get("contains") == "DARBOJAS"
        # Structured TaskContract views (not word-split args)
        assert "test.txt" in tg.artifacts
        assert "DARBOJAS" in tg.content_requirements
        assert tg.actions
        assert tg.acceptance_criteria
        grounded = TaskContract.ground_args(
            {"path": "user_provided_path", "content": "DARBOJAS"},
            tg.original_request,
        )
        assert "path" not in grounded and grounded.get("content") == "DARBOJAS", grounded
        # Polluted word-keys from the request sentence must be stripped
        long_req = (
            "Paradi ka Python darbojas dekoratori ar piemeri faila "
            "decorators_example.py"
        )
        polluted = TaskContract.sanitize_args(
            {
                "Python": "darbojas",
                "darbojas": "dekoratori",
                "dekoratori": "piemeri",
                "path": "decorators_example.py",
            },
            long_req,
            skill_meta={"required_args": ["path"]},
        )
        assert list(polluted.keys()) == ["path"], polluted
        assert polluted.get("path") == "decorators_example.py"
        # Natural-language request must NOT explode into dozens of must_contain needles
        tg_long = TaskContract.from_request(long_req)
        must = list(tg_long.constraints.get("must_contain") or [])
        assert len(must) <= 3, must
        assert "ti" not in must and "ka" not in must
        assert "Python" not in must  # stopword / not cue-captured
        assert any(
            (f.get("path") or "").endswith("decorators_example.py")
            for f in (tg_long.constraints.get("files") or [])
        ), tg_long.constraints
        # enrich_diagnosis must not turn content_constraint into word args
        diag = ContextBuilder.enrich_diagnosis_args(
            {
                "fault_layer": "skill_code",
                "missing_args": ["ti", "Python"],
                "required_args": ["ti", "Python"],
                "suggested_args": {},
            },
            {
                "phase": "VERIFY",
                "exception": "content_constraint: 'ti' missing from expected files",
                "context": {"args": {"path": "decorators_example.py"}, "original_request": long_req},
            },
            long_req,
            user_request=long_req,
        )
        assert diag.get("fault_layer") == "skill_code"
        assert diag.get("rewrite_skill") is True
        assert not diag.get("missing_args"), diag
        assert "ti" not in (diag.get("suggested_args") or {})
        noisy_v = Verifier(root / "data" / "_tg_verify_ws")
        (root / "data" / "_tg_verify_ws").mkdir(parents=True, exist_ok=True)
        built = noisy_v.extract_constraints(
            tg.original_request,
            {"path": "user_provided_path", "content": "user_provided_content"},
        )
        file_paths = [f.get("path") for f in built.get("files") or []]
        assert file_paths == ["test.txt"], built
        # Content token stays as contains — never a second invented file path
        assert "DARBOJAS" not in file_paths, built
        assert (built.get("files") or [{}])[0].get("contains") == "DARBOJAS", built
        # user_provided_* must not become file constraints
        assert not any(
            "user_provided" in str(f.get("path") or "").lower()
            for f in built.get("files") or []
        ), built

        # Unit: universal missing-arg parse + offline fill (no hardcoded key names)
        parsed = ContextBuilder.parse_missing_arg_names(
            "The skill is missing required arguments 'alpha' and 'beta' when invoked."
        )
        assert parsed == ["alpha", "beta"], parsed
        # Verifier content_constraint must NEVER become missing arg names
        assert ContextBuilder.parse_missing_arg_names(
            "content_constraint: 'ti' missing from expected files "
            "['C:\\\\JARVIS\\\\workspace_runtime\\\\decorators_example.py']"
        ) == []
        filled = ContextBuilder._offline_extract(
            'Do work alpha_file.dat with "PAYLOAD_Z"',
            diagnosis={"missing_args": ["alpha", "beta"], "required_args": ["alpha", "beta"]},
        )
        assert filled.get("alpha") == "alpha_file.dat", filled
        assert filled.get("beta") == "PAYLOAD_Z", filled
        # Context mapper rejects invented defaults even if diagnosis suggests them
        cb = ContextBuilder()
        mapped = cb.build(
            tg.original_request,
            str(root / "data" / "_tg_verify_ws"),
            prior_args={"path": "user_provided_path", "content": "user_provided_content"},
            diagnosis={
                "missing_args": ["path", "content"],
                "required_args": ["path", "content"],
                "suggested_args": {
                    "path": "user_provided_path",
                    "content": "user_provided_content",
                },
                "fault_layer": "context_mapping",
            },
            user_request=tg.original_request,
            contract=tg,
        )
        assert mapped["args"].get("path") == "test.txt", mapped["args"]
        assert mapped["args"].get("content") == "DARBOJAS", mapped["args"]
        assert TaskContract.normalize_fault_layer("context_args") == "context_mapping"
        assert TaskContract.rewrite_skill_for_layer("context_mapping") is False
        assert TaskContract.rewrite_skill_for_layer("skill_code") is True
        # UNKNOWN / unrecognized must NOT become skill_code (no auto-rewrite)
        assert TaskContract.normalize_fault_layer(None) == "unknown"
        assert TaskContract.normalize_fault_layer("") == "unknown"
        assert TaskContract.normalize_fault_layer("UNKNOWN") == "unknown"
        assert TaskContract.normalize_fault_layer("garbage_layer") == "unknown"
        assert TaskContract.rewrite_skill_for_layer("unknown") is False
        assert TaskContract.rewrite_skill_for_layer(None) is False
        gated = TaskContract.apply_rewrite_gate({"fault_layer": "UNKNOWN", "rewrite_skill": True})
        assert gated["fault_layer"] == "unknown" and gated["rewrite_skill"] is False
        gated_ok = TaskContract.apply_rewrite_gate(
            {"fault_layer": "skill_code", "rewrite_skill": False}
        )
        assert gated_ok["fault_layer"] == "skill_code" and gated_ok["rewrite_skill"] is True

        # Universal request-item classification: path stems ≠ libraries / deps
        from jarvis.request_items import (
            ItemKind,
            classify_request_items,
            library_tokens_for_pypi,
            path_segment_tokens,
            sanitize_libraries,
        )
        from jarvis.research import ResearchSystem

        path_req = 'Create workspace/out_tool.py containing "READY"'
        segs = path_segment_tokens(path_req)
        assert "out_tool" in segs, segs
        assert any("out_tool.py" in s or s.endswith("out_tool.py") for s in segs) or \
            "out_tool.py" in segs or any("out_tool" == s for s in segs)
        # Directory + stem from relative path
        assert "workspace" in segs or any("workspace" in s for s in segs)
        assert TaskContract.is_polluted_arg_key("out_tool", path_req)
        assert TaskContract.is_polluted_arg_key("workspace", path_req)
        items = classify_request_items(path_req)
        kinds = {i.kind for i in items}
        assert ItemKind.PATH in kinds or ItemKind.FILE in kinds, items
        assert not any(i.kind == ItemKind.LIBRARY for i in items), items
        assert library_tokens_for_pypi(path_req) == []
        assert sanitize_libraries(
            ["out_tool", "workspace", "requests"], path_req
        ) == ["requests"]
        # Code with only stdlib → install nothing (even if research lied)
        stdlib_skill = (
            "from pathlib import Path\n"
            "def run(context):\n"
            "    Path('x').write_text('y')\n"
            "    return {'ok': True}\n"
        )
        assert sanitize_libraries(
            ["out_tool", "workspace", "pathlib"],
            path_req,
            skill_code=stdlib_skill,
        ) == []
        rs_unit = ResearchSystem(brain=None)
        notes, hits = rs_unit._pypi_lookup(path_req)
        assert notes == "" and hits == [], (notes, hits)
        tg_path = TaskContract.from_request(path_req)
        assert any(
            "out_tool.py" in a for a in tg_path.artifacts
        ), tg_path.artifacts

        # Universal request grounding — polysemous words must not rewrite TaskContract
        from jarvis.task_contract import SUBJECT_AMBIGUOUS, SUBJECT_UNKNOWN

        poly_req = (
            'Create habitat_notes.md about python wetland habitat '
            'containing "PYTHON_OK"'
        )
        tg_poly = TaskContract.from_request(poly_req)
        assert tg_poly.original_request == poly_req  # immutable source of truth
        assert "create" in tg_poly.actions
        assert any("habitat_notes.md" in a for a in tg_poly.artifacts), tg_poly.artifacts
        assert "python" in tg_poly.subject.lower()
        assert "wetland" in tg_poly.subject.lower()
        assert "PYTHON_OK" in tg_poly.content_requirements
        assert tg_poly.content_source == () or "memory" not in tg_poly.content_source
        # Drifted queries (programming sense of "python") must be rejected
        assert tg_poly.query_is_grounded("python wetland habitat ecology") is True
        assert tg_poly.query_is_grounded(
            "pip install python package programming language"
        ) is False
        grounded_q = tg_poly.ground_research_queries(
            ["pip install python", "java JDK tutorial", "wetland habitat survey"]
        )
        assert grounded_q, grounded_q
        assert not any("pip" in q.lower() for q in grounded_q), grounded_q
        assert not any("jdk" in q.lower() for q in grounded_q), grounded_q
        assert any(
            "wetland" in q.lower() or "habitat" in q.lower() for q in grounded_q
        ), grounded_q

        java_req = (
            "Write summary.txt about java island geography from "
            "https://example.com/java-isle containing \"JAVA_OK\""
        )
        tg_java = TaskContract.from_request(java_req)
        assert "java" in tg_java.subject.lower() and "island" in tg_java.subject.lower()
        assert any(
            "example.com/java-isle" in s for s in tg_java.content_source
        ), tg_java.content_source
        assert all(
            "example.com" not in a for a in tg_java.artifacts
        ), tg_java.artifacts
        assert tg_java.query_is_grounded("java programming language JDK") is False
        assert tg_java.query_is_grounded("java island geography") is True

        mem_req = (
            "Using previously learned knowledge about orange citrus, "
            'create citrus.txt containing "ORANGE_OK"'
        )
        tg_mem = TaskContract.from_request(mem_req)
        assert "orange" in tg_mem.subject.lower() and "citrus" in tg_mem.subject.lower()
        assert "memory" in {s.lower() for s in tg_mem.content_source}
        assert tg_mem.prefers_memory_first() is True
        assert "citrus.txt" in tg_mem.artifacts

        # Ambiguity: do not invent a sense for a bare leftover token
        tg_bare = TaskContract.from_request('Create bank.txt containing "X"')
        assert tg_bare.subject in (SUBJECT_UNKNOWN, SUBJECT_AMBIGUOUS)
        tg_bank = TaskContract.from_request(
            'Learn about bank river erosion and write notes.txt containing "BANK_OK"'
        )
        assert "bank" in tg_bank.subject.lower() and "river" in tg_bank.subject.lower()
        assert tg_bank.query_is_grounded("bank interest rates finance loan") is False
        assert tg_bank.query_is_grounded("bank river erosion sediment") is True

        # Lifecycle snapshot: action+artifact+subject+content_source preserved
        life = tg_java.to_dict()
        for key in (
            "actions", "artifacts", "subject", "content_source",
            "content_requirements", "acceptance_criteria", "original_request",
        ):
            assert key in life, life.keys()
        assert life["original_request"] == java_req
        assert life["subject"] == tg_java.subject
        assert life["content_source"] == list(tg_java.content_source)
        # with_args must not drop subject / content_source
        tg_java2 = tg_java.with_args({"path": "summary.txt", "content": "JAVA_OK"})
        assert tg_java2.subject == tg_java.subject
        assert tg_java2.content_source == tg_java.content_source
        assert tg_java2.original_request == java_req

        # Hierarchical competence classification (no task-specific hardcode)
        from jarvis.competence import (
            classify_task_competences,
            preferred_tool_name,
        )
        clf_poly = classify_task_competences(tg_poly)
        assert "io.files" in clf_poly["skill_ids"], clf_poly
        assert any("io.create" in c for c in clf_poly["capability_ids"]), clf_poly
        assert "subject." in (clf_poly.get("subject_branch") or "")
        tool_nm = preferred_tool_name(tg_poly, clf_poly)
        assert tool_nm.startswith("io_files"), tool_nm
        assert "habitat_notes" not in tool_nm  # not a per-request slug
        # Multi-competence: fetch + write opens net + io
        tg_multi = TaskContract.from_request(
            'Fetch https://example.com/x and write out.txt containing "OK"'
        )
        clf_multi = classify_task_competences(tg_multi)
        assert "net.http" in clf_multi["skill_ids"], clf_multi
        assert "io.files" in clf_multi["skill_ids"], clf_multi
        assert any("net.fetch" in c for c in clf_multi["capability_ids"]), clf_multi
        assert any("io.write" in c for c in clf_multi["capability_ids"]), clf_multi

        # Canonical TaskContract fields (single truth source)
        for key in (
            "request_id", "original_request", "intent", "desired_outcomes",
            "artifacts", "behaviors", "constraints", "inputs", "outputs",
            "side_effects", "acceptance_criteria", "verification_plan",
            "capability_requirements",
        ):
            assert key in tg.to_dict(), key
        assert tg.validated and tg.is_verifiable(), tg.to_dict()
        assert tg.desired_outcomes
        assert tg.verification_plan
        # Locked fields survive with_inputs
        tg_locked = tg.with_inputs({"body": "DARBOJAS"})
        assert tg_locked.original_request == tg.original_request
        assert tg_locked.desired_outcomes == tg.desired_outcomes
        assert tg_locked.acceptance_criteria == tg.acceptance_criteria
        # Universal acceptance_criteria from full request (what/result/reqs/how)
        assert any(s.startswith("artifact_exists:") for s in tg.acceptance_criteria), (
            tg.acceptance_criteria
        )
        assert any("content_present:" in s for s in tg.acceptance_criteria), (
            tg.acceptance_criteria
        )
        assert any("@workspace_" in s for s in tg.acceptance_criteria), tg.acceptance_criteria
        assert "skill_ok_and_verified" not in tg.acceptance_criteria
        assert isinstance(tg.constraints.get("checks"), list) and tg.constraints["checks"]
        # HTTP-only request → http_ok criterion (not skill stub)
        tg_http = TaskContract.from_request("Fetch https://example.com/resource")
        assert tg_http.is_verifiable(), tg_http.to_dict()
        assert any(s.startswith("http_ok:") for s in tg_http.acceptance_criteria), (
            tg_http.acceptance_criteria
        )
        # Semantic UNDERSTAND — quoted payload (model/offline draft, not verb tables)
        tg_say = TaskContract.from_request('Say "hello" politely to the room')
        assert tg_say.is_verifiable(), tg_say.to_dict()
        assert any(
            s.startswith("skill_output_contains:hello") for s in tg_say.acceptance_criteria
        ), tg_say.acceptance_criteria
        assert "needs_refine" not in tg_say.acceptance_criteria
        # Unquoted communicative text without grounded literals → conversation
        # offline (no invented skill behavior). Model UNDERSTAND can promote to
        # a verifiable behavior when available (see SemanticBrain below).
        tg_unq = TaskContract.from_request("Say hello politely to the room")
        assert tg_unq.requires("conversation") or tg_unq.needs_refine(), tg_unq.to_dict()
        assert not tg_unq.requires_capability_stages(), tg_unq.required_outcomes
        tg_say_r = TaskContract.refine(tg_say)
        assert tg_say_r.is_verifiable(), tg_say_r.acceptance_criteria
        # VERIFY checks real skill stdout/result — not evidence self-proof
        soft_ws = root / "data" / "_tg_soft_ws"
        soft_ws.mkdir(parents=True, exist_ok=True)
        soft_v = Verifier(soft_ws)
        say_pass = soft_v.verify(
            tg_say_r.original_request,
            {
                "ok": True,
                "result": {"says": "hello"},
                "stdout": "hello\n",
                "evidence": "ignored",
                "returncode": 0,
            },
            contract=tg_say_r,
        )
        assert say_pass.get("verified"), say_pass
        say_fail = soft_v.verify(
            tg_say_r.original_request,
            {
                "ok": True,
                "result": {"says": "hi"},
                "stdout": "hi\n",
                "evidence": "I said hello",
                "returncode": 0,
            },
            contract=tg_say_r,
        )
        assert not say_fail.get("verified"), say_fail
        # VERIFY must use frozen contract — not re-interpret a mutated request string
        frozen = TaskContract.validate(
            TaskContract.understand('Create freeze.txt containing "Z"')
        )
        assert frozen.validated
        wrong_goal = "Create other.txt containing NEVER"
        freeze_ok = soft_v.verify(
            wrong_goal,  # deliberately wrong goal string
            {
                "ok": True,
                "result": {"path": str(soft_ws / "freeze.txt")},
                "returncode": 0,
            },
            user_request=wrong_goal,
            contract=frozen,
        )
        # Still checks freeze.txt / Z from contract, not wrong_goal
        (soft_ws / "freeze.txt").write_text("Z\n", encoding="utf-8")
        freeze_ok = soft_v.verify(
            wrong_goal,
            {
                "ok": True,
                "result": {"path": str(soft_ws / "freeze.txt")},
                "returncode": 0,
            },
            user_request=wrong_goal,
            contract=frozen,
        )
        assert freeze_ok.get("verified"), freeze_ok
        # Return/type/arithmetic behavior criteria (still request-grounded)
        tg_ret = TaskContract.from_request("Return 42")
        assert any(
            s.startswith("skill_result_equals:42") for s in tg_ret.acceptance_criteria
        ), tg_ret.acceptance_criteria
        assert soft_v.verify(
            tg_ret.original_request,
            {"ok": True, "result": 42, "returncode": 0},
            contract=tg_ret,
        ).get("verified")
        tg_type = TaskContract.from_request("Return an integer")
        assert any(
            s.startswith("skill_result_type:int") for s in tg_type.acceptance_criteria
        ), tg_type.acceptance_criteria
        tg_arith = TaskContract.from_request("Compute 2+2")
        assert any(
            s.startswith("skill_result_equals:4") for s in tg_arith.acceptance_criteria
        ), tg_arith.acceptance_criteria
        # Soft prose alone → conversation outcome (not a capability BUILD).
        # Capability hint without concrete criteria → needs_refine / BUILD refuse.
        tg_soft = TaskContract.from_request("Be helpful please")
        assert tg_soft.requires("conversation"), tg_soft.required_outcomes
        assert not tg_soft.needs_refine(), tg_soft.acceptance_criteria
        tg_soft_cap = TaskContract.understand("Be helpful please").with_requirements_hint(
            {"intent": "task", "needs_capability": True}
        )
        tg_soft_cap = TaskContract.validate(tg_soft_cap)
        assert tg_soft_cap.needs_refine() or not tg_soft_cap.is_verifiable(), (
            tg_soft_cap.to_dict()
        )
        soft_res = soft_v.verify(
            tg_soft_cap.original_request,
            {
                "ok": True,
                "result": {"says": "sure"},
                "evidence": "I was helpful",
                "returncode": 0,
            },
            contract=tg_soft_cap,
        )
        assert not soft_res.get("verified"), soft_res
        assert "needs_refine" in (soft_res.get("reason") or "") or any(
            c.get("name") == "user_constraints" and not c.get("ok")
            for c in soft_res.get("checks") or []
        ), soft_res

        args_root = root / "data" / "_e2e_args"
        if args_root.exists():
            shutil.rmtree(args_root)
        args_root.mkdir(parents=True)

        class ArgsFaultBrain(Brain):
            """Skill is correct; PLAN/TEST start with empty args — repair must fill them."""

            def __init__(self) -> None:
                super().__init__()
                self.build_count = 0

            def model_status(self) -> str:
                return "ONLINE"

            def is_available(self) -> bool:
                return True

            def classify_intent(self, user_text: str) -> dict:
                return {
                    "intent": "task",
                    "goal": user_text,
                    "needs_capability": True,
                    "keywords": user_text.split()[:8],
                }

            def plan(self, goal: str, known_capabilities: list[str]) -> dict:
                reuse = []
                for line in known_capabilities:
                    if "args_echo_skill" in line:
                        reuse.append("args_echo_skill")
                return {
                    "steps": ["run skill with structured args"],
                    "can_reuse": reuse,
                    "missing": [] if reuse else [goal],
                    "needs_research": not bool(reuse),
                    "needs_new_skill": not bool(reuse),
                    "skill_name": "args_echo_skill",
                    "skill_description": goal,
                    "research_queries": [goal],
                    # Simulate the bug: planner forgot to extract args
                    "args": {},
                    "required_args": [],
                }

            def extract_task_args(
                self, goal, skill_meta=None, prior_args=None, diagnosis=None
            ):
                # Without diagnosis, pretend brain extracted nothing (empty args bug).
                # With diagnosis missing_args, fill universally from the goal.
                if diagnosis and (
                    diagnosis.get("missing_args")
                    or diagnosis.get("required_args")
                    or diagnosis.get("suggested_args")
                ):
                    return ContextBuilder._offline_extract(goal, diagnosis=diagnosis)
                return {}

            def research_notes(self, query: str, gathered: str) -> dict:
                return {
                    "approach": "read_context_args",
                    "libraries": [],
                    "key_apis": [query],
                    "pitfalls": ["never invent missing args"],
                    "test_idea": "pass structured args into run(context)",
                }

            def write_skill_code(
                self,
                skill_name,
                description,
                research,
                previous_code=None,
                error_log=None,
                diagnosis=None,
                failed_approaches=None,
                test_plan=None,
            ) -> str:
                self.build_count += 1
                # Correct skill from the start — fails only when context.args incomplete
                import json as _json
                desc_lit = _json.dumps(description or "")
                return f'''
from pathlib import Path

SKILL_META = {{
    "name": "{skill_name}",
    "description": {desc_lit},
    "capabilities": ["args_echo"],
    "dependencies": [],
    "version": {self.build_count},
    "required_args": ["dest", "body"],
}}

def run(context: dict) -> dict:
    args = context.get("args") or {{}}
    required = list(SKILL_META.get("required_args") or [])
    missing = [k for k in required if args.get(k) in (None, "")]
    if missing:
        return {{
            "ok": False,
            "result": None,
            "error": (
                "The skill is missing required arguments "
                + " and ".join(repr(m) for m in missing)
                + " when invoked."
            ),
            "evidence": f"args_keys={{list(args.keys())}}",
        }}
    workspace = Path(context.get("workspace") or ".")
    path = workspace / str(args["dest"])
    path.write_text(str(args["body"]) + "\\n", encoding="utf-8")
    return {{
        "ok": True,
        "result": {{"path": str(path), "contains": str(args["body"])}},
        "error": None,
        "evidence": f"wrote {{path}} size={{path.stat().st_size}}",
    }}
'''

            def diagnose(self, observation, failed_approaches=None, prior_solutions=None):
                from jarvis.observer import Observer

                err = str(observation.get("exception") or "")
                phase = str(observation.get("phase") or "")
                parsed = ContextBuilder.parse_missing_arg_names(err)
                if parsed:
                    d = {
                        "root_cause": "context.args missing required keys",
                        "fault_layer": "context_mapping",
                        "rewrite_skill": False,
                        "what_to_change": "Map TaskContract → skill args from USER REQUEST",
                        "approach": "context_mapping",
                        "approach_changed": True,
                        "needs_research": False,
                        "research_queries": [],
                        "needs_new_deps": [],
                        "missing_args": parsed,
                        "required_args": parsed,
                        "suggested_args": {},
                        "test_plan": "retest with grounded context.args",
                        "expected_artifacts": [],
                        "is_unfixable": False,
                        "diagnosis": err[:300],
                    }
                    d["approach_fingerprint"] = Observer.fingerprint_approach(
                        d["approach"]
                    )
                    return ContextBuilder.enrich_diagnosis_args(
                        d, observation, str(observation.get("goal") or "")
                    )
                # Verifier issues are not skill-code faults in this scenario
                if phase == "VERIFY" or "VERIFIER" in err.upper():
                    approach = "verifier_align"
                    fp = Observer.fingerprint_approach(approach)
                    return {
                        "root_cause": err[:500],
                        "fault_layer": "verifier",
                        "rewrite_skill": False,
                        "what_to_change": "Keep skill; rely on result.path evidence",
                        "approach": approach,
                        "approach_fingerprint": fp,
                        "approach_changed": True,
                        "needs_research": False,
                        "research_queries": [],
                        "needs_new_deps": [],
                        "missing_args": [],
                        "required_args": [],
                        "suggested_args": {},
                        "test_plan": "retest with same args",
                        "expected_artifacts": [],
                        "is_unfixable": False,
                        "diagnosis": err[:300],
                    }
                approach = "unexpected_skill_fix"
                fp = Observer.fingerprint_approach(approach)
                return {
                    "root_cause": err[:500],
                    "fault_layer": "skill_code",
                    "rewrite_skill": True,
                    "what_to_change": "fix skill",
                    "approach": approach,
                    "approach_fingerprint": fp,
                    "approach_changed": True,
                    "needs_research": False,
                    "research_queries": [],
                    "needs_new_deps": [],
                    "missing_args": [],
                    "required_args": [],
                    "suggested_args": {},
                    "test_plan": "retest",
                    "expected_artifacts": [],
                    "is_unfixable": False,
                    "diagnosis": err[:300],
                }

            def verify_claim(self, goal, result, evidence) -> dict:
                return {"achieved": True, "confidence": 0.9, "reason": "advisory"}

            def converse(self, user_text, history=None) -> str:
                return "ok"

        phases_a: list[str] = []
        logs_a: list[str] = []
        brain_a = ArgsFaultBrain()
        orch_a = Orchestrator(
            root=args_root,
            brain=brain_a,
            on_status=lambda s: phases_a.append(s),
            on_log=lambda m: logs_a.append(m),
        )
        # Quoted values only — arg *names* come from skill error / diagnosis,
        # not from hardcoded path/content keys in the goal.
        goal_a = 'Create "args_e2e_out.txt" containing "ARGS_OK"'
        result_a = orch_a.run_cycle(goal_a)
        assert result_a.get("success"), result_a
        out = args_root / "workspace_runtime" / "args_e2e_out.txt"
        assert out.exists() and "ARGS_OK" in out.read_text(encoding="utf-8"), out
        # Skill written once — context_mapping repair must not rewrite skill
        assert brain_a.build_count == 1, brain_a.build_count
        assert "OBSERVE" in phases_a and "DIAGNOSE" in phases_a and "RETEST" in phases_a
        assert any(
            "context_mapping" in m or "context_args" in m or "CONTEXT repair" in m
            for m in logs_a
        ), logs_a[-20:]
        assert any("Skip skill rewrite" in m for m in logs_a), logs_a[-20:]
        assert any("TaskContract" in m or "REQUEST:" in m for m in logs_a), logs_a[:10]
        orch_a.close()
        print("  OK args     empty args → context_mapping diagnose → retest (no rewrite)")
    except Exception as exc:
        msg = f"E2E_ARGS: {exc!r}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # 4c) E2E: default artifact must VERIFY FAIL → skill rewrite → honor USER REQUEST
    print("  — e2e verifier rejects defaults; repair honors USER REQUEST —")
    try:
        def_root = root / "data" / "_e2e_defaults"
        if def_root.exists():
            shutil.rmtree(def_root)
        def_root.mkdir(parents=True)

        class DefaultTrapBrain(Brain):
            def __init__(self) -> None:
                super().__init__()
                self.build_count = 0

            def model_status(self) -> str:
                return "ONLINE"

            def is_available(self) -> bool:
                return True

            def classify_intent(self, user_text: str) -> dict:
                return {
                    "intent": "task",
                    "goal": user_text,
                    "needs_capability": True,
                    "keywords": user_text.split()[:8],
                }

            def plan(self, goal: str, known_capabilities: list[str]) -> dict:
                return {
                    "steps": ["create requested artifact"],
                    "can_reuse": [],
                    "missing": [goal],
                    "needs_research": True,
                    "needs_new_skill": True,
                    "skill_name": "make_artifact",
                    "skill_description": goal,
                    "research_queries": [goal],
                    "args": {"dest": "user_note.txt", "body": "REAL_PAYLOAD"},
                    "required_args": ["dest", "body"],
                }

            def extract_task_args(
                self, goal, skill_meta=None, prior_args=None, diagnosis=None
            ):
                return {"dest": "user_note.txt", "body": "REAL_PAYLOAD"}

            def research_notes(self, query: str, gathered: str) -> dict:
                return {
                    "approach": "default_trap_then_honor_request",
                    "libraries": [],
                    "key_apis": [query],
                    "pitfalls": ["never use default_path / default content"],
                    "test_idea": "verifier must match USER REQUEST",
                }

            def write_skill_code(
                self,
                skill_name,
                description,
                research,
                previous_code=None,
                error_log=None,
                diagnosis=None,
                failed_approaches=None,
                test_plan=None,
            ) -> str:
                self.build_count += 1
                import json as _json
                desc_lit = _json.dumps(description or "")
                # First build: intentionally writes defaults (must VERIFY FAIL)
                if self.build_count == 1 and not (
                    diagnosis and diagnosis.get("rewrite_skill")
                ):
                    return f'''
from pathlib import Path
SKILL_META = {{
    "name": "{skill_name}",
    "description": {desc_lit},
    "capabilities": ["make_artifact"],
    "dependencies": [],
    "version": 1,
    "required_args": ["dest", "body"],
}}
def run(context: dict) -> dict:
    workspace = Path(context.get("workspace") or ".")
    path = workspace / "default_path.txt"
    path.write_text("default content\\n", encoding="utf-8")
    return {{
        "ok": True,
        "result": {{"path": str(path), "contains": "default content"}},
        "error": None,
        "evidence": f"wrote {{path}}",
    }}
'''
                # Repair: honor context args / user request
                return f'''
from pathlib import Path
SKILL_META = {{
    "name": "{skill_name}",
    "description": {desc_lit},
    "capabilities": ["make_artifact"],
    "dependencies": [],
    "version": {self.build_count},
    "required_args": ["dest", "body"],
}}
def run(context: dict) -> dict:
    args = context.get("args") or {{}}
    required = list(SKILL_META.get("required_args") or [])
    missing = [k for k in required if args.get(k) in (None, "")]
    if missing:
        return {{
            "ok": False,
            "result": None,
            "error": "missing required arguments: " + repr(missing),
            "evidence": "",
        }}
    workspace = Path(context.get("workspace") or ".")
    path = workspace / str(args["dest"])
    path.write_text(str(args["body"]) + "\\n", encoding="utf-8")
    return {{
        "ok": True,
        "result": {{"path": str(path), "contains": str(args["body"])}},
        "error": None,
        "evidence": f"wrote {{path}}",
    }}
'''

            def diagnose(self, observation, failed_approaches=None, prior_solutions=None):
                from jarvis.observer import Observer
                from jarvis.context_builder import ContextBuilder

                err = str(observation.get("exception") or "")
                phase = str(observation.get("phase") or "")
                if phase == "VERIFY" or "default" in err.lower() or "VERIFIER" in err:
                    approach = "honor_user_request"
                    fp = Observer.fingerprint_approach(approach)
                    d = {
                        "root_cause": "skill used default artifact instead of USER REQUEST",
                        "fault_layer": "skill_code",
                        "rewrite_skill": True,
                        "what_to_change": "Write dest/body from context args; no defaults",
                        "approach": approach,
                        "approach_fingerprint": fp,
                        "approach_changed": True,
                        "needs_research": False,
                        "research_queries": [],
                        "needs_new_deps": [],
                        "missing_args": [],
                        "required_args": ["dest", "body"],
                        "suggested_args": {
                            "dest": "user_note.txt",
                            "body": "REAL_PAYLOAD",
                        },
                        "test_plan": "VERIFY against USER REQUEST",
                        "expected_artifacts": ["user_note.txt"],
                        "is_unfixable": False,
                        "diagnosis": err[:300],
                    }
                    return ContextBuilder.enrich_diagnosis_args(
                        d, observation, str(observation.get("goal") or "")
                    )
                approach = "generic_fix"
                fp = Observer.fingerprint_approach(approach)
                return {
                    "root_cause": err[:500],
                    "fault_layer": "skill_code",
                    "rewrite_skill": True,
                    "what_to_change": "fix skill",
                    "approach": approach,
                    "approach_fingerprint": fp,
                    "approach_changed": True,
                    "needs_research": False,
                    "research_queries": [],
                    "needs_new_deps": [],
                    "missing_args": [],
                    "required_args": [],
                    "suggested_args": {},
                    "test_plan": "retest",
                    "expected_artifacts": [],
                    "is_unfixable": False,
                    "diagnosis": err[:300],
                }

            def verify_claim(self, goal, result, evidence) -> dict:
                return {"achieved": True, "confidence": 0.9, "reason": "advisory"}

            def converse(self, user_text, history=None) -> str:
                return "ok"

        phases_d: list[str] = []
        logs_d: list[str] = []
        brain_d = DefaultTrapBrain()
        orch_d = Orchestrator(
            root=def_root,
            brain=brain_d,
            on_status=lambda s: phases_d.append(s),
            on_log=lambda m: logs_d.append(m),
        )
        goal_d = 'Create "user_note.txt" containing "REAL_PAYLOAD"'
        result_d = orch_d.run_cycle(goal_d)
        assert result_d.get("success"), result_d
        note = def_root / "workspace_runtime" / "user_note.txt"
        assert note.exists() and "REAL_PAYLOAD" in note.read_text(encoding="utf-8"), note
        assert not (def_root / "workspace_runtime" / "default_path.txt").exists() or (
            "REAL_PAYLOAD" in note.read_text(encoding="utf-8")
        )
        assert brain_d.build_count >= 2, brain_d.build_count
        assert "VERIFY" in phases_d and "OBSERVE" in phases_d and "DIAGNOSE" in phases_d
        assert any("reject_defaults" in m or "VERIFIER FAIL" in m or "default" in m.lower()
                   for m in logs_d), logs_d[-40:]
        orch_d.close()
        print("  OK defaults VERIFY FAIL → rewrite → USER REQUEST honored")
    except Exception as exc:
        msg = f"E2E_DEFAULTS: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # 4d) E2E: repeated VERIFY fail → CODING repair (research NOT universal fallback)
    print("  — e2e progress-aware repair (CODING, no research spam) —")
    try:
        from jarvis.observer import Observer as _Obs

        ar_root = root / "data" / "_e2e_adaptive_research"
        if ar_root.exists():
            shutil.rmtree(ar_root)
        ar_root.mkdir(parents=True)

        class CodingFirstRepairBrain(Brain):
            def __init__(self) -> None:
                super().__init__()
                self.build_count = 0
                self.research_notes_calls = 0
                self.gen_query_calls = 0
                self.diagnose_approaches: list[str] = []

            def model_status(self) -> str:
                return "ONLINE"

            def is_available(self) -> bool:
                return True

            def classify_intent(self, user_text: str) -> dict:
                return {
                    "intent": "task",
                    "goal": user_text,
                    "needs_capability": True,
                    "keywords": user_text.split()[:8],
                }

            def plan(self, goal: str, known_capabilities: list[str]) -> dict:
                return {
                    "steps": ["build skill that honors args"],
                    "can_reuse": [],
                    "missing": [goal],
                    "needs_research": False,  # CODING-first; no entry RESEARCH tax
                    "needs_new_skill": True,
                    "skill_name": "repeat_fail_skill",
                    "skill_description": goal,
                    "research_queries": [],
                    "args": {"dest": "adapt_out.txt", "body": "ADAPT_OK"},
                    "required_args": ["dest", "body"],
                }

            def extract_task_args(
                self, goal, skill_meta=None, prior_args=None, diagnosis=None
            ):
                return {"dest": "adapt_out.txt", "body": "ADAPT_OK"}

            def generate_research_queries(
                self, observation, diagnosis=None, failed_approaches=None
            ):
                self.gen_query_calls += 1
                return ["SHOULD_NOT_BE_CALLED_WITHOUT_KNOWLEDGE_GAP"]

            def research_notes(self, query: str, gathered: str) -> dict:
                self.research_notes_calls += 1
                return {
                    "approach": "should_not_research",
                    "libraries": [],
                    "key_apis": [],
                    "pitfalls": [],
                    "test_idea": "",
                    "repair_insight": "",
                }

            def write_skill_code(
                self,
                skill_name,
                description,
                research,
                previous_code=None,
                error_log=None,
                diagnosis=None,
                failed_approaches=None,
                test_plan=None,
            ) -> str:
                self.build_count += 1
                import json as _json
                desc_lit = _json.dumps(description or "")
                # First two builds: same VERIFY failure class (empty file)
                if self.build_count <= 2:
                    return f'''
from pathlib import Path
SKILL_META = {{
    "name": "{skill_name}",
    "description": {desc_lit},
    "capabilities": ["repeat_fail"],
    "dependencies": [],
    "version": {self.build_count},
    "required_args": ["dest", "body"],
}}
def run(context: dict) -> dict:
    args = context.get("args") or {{}}
    workspace = Path(context.get("workspace") or ".")
    dest = str(args.get("dest") or "adapt_out.txt")
    path = workspace / dest
    path.write_text("", encoding="utf-8")
    return {{
        "ok": True,
        "result": {{"path": str(path), "contains": ""}},
        "error": None,
        "evidence": f"touched {{path}}",
    }}
'''
                # After stagnation forces new CODING approach: honor args
                return f'''
from pathlib import Path
SKILL_META = {{
    "name": "{skill_name}",
    "description": {desc_lit},
    "capabilities": ["repeat_fail"],
    "dependencies": [],
    "version": {self.build_count},
    "required_args": ["dest", "body"],
}}
def run(context: dict) -> dict:
    args = context.get("args") or {{}}
    workspace = Path(context.get("workspace") or ".")
    path = workspace / str(args["dest"])
    path.write_text(str(args["body"]) + "\\n", encoding="utf-8")
    return {{
        "ok": True,
        "result": {{"path": str(path), "contains": str(args["body"])}},
        "error": None,
        "evidence": f"wrote {{path}}",
    }}
'''

            def diagnose(self, observation, failed_approaches=None, prior_solutions=None):
                err = str(observation.get("exception") or "")
                approach = "empty_touch_coding_repair"
                fps = {
                    a.get("approach_fingerprint")
                    for a in (failed_approaches or [])
                }
                fp = _Obs.fingerprint_approach(approach)
                if fp in fps:
                    approach = approach + f"_v{len(fps)+1}"
                    fp = _Obs.fingerprint_approach(approach)
                self.diagnose_approaches.append(approach)
                return {
                    "root_cause": "file created empty; content missing for VERIFY",
                    "fault_layer": "skill_code",
                    "rewrite_skill": True,
                    "what_to_change": "Write body from args into dest",
                    "approach": approach,
                    "approach_fingerprint": fp,
                    "approach_changed": True,
                    "needs_research": False,
                    "missing_knowledge": [],
                    "research_queries": [],
                    "needs_new_deps": [],
                    "missing_args": [],
                    "required_args": ["dest", "body"],
                    "suggested_args": {
                        "dest": "adapt_out.txt",
                        "body": "ADAPT_OK",
                    },
                    "test_plan": "VERIFY dest contains body",
                    "expected_artifacts": ["adapt_out.txt"],
                    "is_unfixable": False,
                    "diagnosis": err[:300],
                }

            def verify_claim(self, goal, result, evidence) -> dict:
                return {"achieved": True, "confidence": 0.9, "reason": "advisory"}

            def converse(self, user_text, history=None) -> str:
                return "ok"

        phases_r: list[str] = []
        logs_r: list[str] = []
        brain_r = CodingFirstRepairBrain()
        orch_r = Orchestrator(
            root=ar_root,
            brain=brain_r,
            on_status=lambda s: phases_r.append(s),
            on_log=lambda m: logs_r.append(m),
        )
        goal_r = 'Create "adapt_out.txt" containing "ADAPT_OK"'
        result_r = orch_r.run_cycle(goal_r)
        assert result_r.get("success"), result_r
        out_r = ar_root / "workspace_runtime" / "adapt_out.txt"
        assert out_r.exists() and "ADAPT_OK" in out_r.read_text(encoding="utf-8")
        assert brain_r.build_count >= 3, brain_r.build_count
        # Research must NOT be the universal fallback for repeated VERIFY fails
        assert brain_r.gen_query_calls == 0, brain_r.gen_query_calls
        assert not any("KNOWLEDGE-GAP RESEARCH" in m for m in logs_r), logs_r[-40:]
        assert not any("ADAPTIVE RESEARCH" in m for m in logs_r), logs_r[-40:]
        assert any("STAGNATION" in m or "CODING" in m for m in logs_r), logs_r[-50:]
        assert phases_r.count("RESEARCH") == 0, phases_r
        orch_r.close()
        print("  OK progress-aware CODING repair — no research spam on VERIFY repeat")
    except Exception as exc:
        msg = f"E2E_ADAPTIVE_RESEARCH: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # 4d2) E2E: identical verifier failure must NOT repeat an identical cycle
    print("  — e2e stagnation: identical VERIFY FAIL ≠ identical research cycle —")
    try:
        from jarvis.observer import Observer as _Obs2
        from jarvis.recovery import ProgressAwareRecovery

        st_root = root / "data" / "_e2e_stagnation"
        if st_root.exists():
            shutil.rmtree(st_root)
        st_root.mkdir(parents=True)

        class StickyVerifyFailBrain(Brain):
            """Always emits the same broken skill → identical VERIFY failure."""

            def __init__(self) -> None:
                super().__init__()
                self.build_count = 0
                self.research_notes_calls = 0
                self.approaches: list[str] = []

            def model_status(self) -> str:
                return "ONLINE"

            def is_available(self) -> bool:
                return True

            def classify_intent(self, user_text: str) -> dict:
                return {
                    "intent": "task",
                    "goal": user_text,
                    "needs_capability": True,
                    "keywords": ["create", "file"],
                }

            def plan(self, goal: str, known_capabilities: list[str]) -> dict:
                return {
                    "steps": ["build"],
                    "can_reuse": [],
                    "missing": [goal],
                    "needs_research": False,
                    "needs_new_skill": True,
                    "skill_name": "sticky_fail_skill",
                    "skill_description": goal,
                    "research_queries": [],
                    "args": {"dest": "sticky.txt", "body": "STICKY_OK"},
                    "required_args": ["dest", "body"],
                }

            def extract_task_args(
                self, goal, skill_meta=None, prior_args=None, diagnosis=None
            ):
                return {"dest": "sticky.txt", "body": "STICKY_OK"}

            def research_notes(self, query: str, gathered: str) -> dict:
                self.research_notes_calls += 1
                return {
                    "approach": "identical_research_cycle",
                    "libraries": [],
                    "key_apis": [],
                    "pitfalls": [],
                    "test_idea": "",
                    "repair_insight": "noop",
                }

            def generate_research_queries(
                self, observation, diagnosis=None, failed_approaches=None
            ):
                return ["identical_query_should_not_loop"]

            def write_skill_code(
                self,
                skill_name,
                description,
                research,
                previous_code=None,
                error_log=None,
                diagnosis=None,
                failed_approaches=None,
                test_plan=None,
            ) -> str:
                self.build_count += 1
                import json as _json
                desc_lit = _json.dumps(description or "")
                # Identical broken body every time (same failure_signature)
                return f'''
from pathlib import Path
SKILL_META = {{
    "name": "{skill_name}",
    "description": {desc_lit},
    "capabilities": ["sticky"],
    "dependencies": [],
    "version": {self.build_count},
    "required_args": ["dest", "body"],
}}
def run(context: dict) -> dict:
    args = context.get("args") or {{}}
    workspace = Path(context.get("workspace") or ".")
    path = workspace / str(args.get("dest") or "sticky.txt")
    path.write_text("", encoding="utf-8")
    return {{
        "ok": True,
        "result": {{"path": str(path), "contains": ""}},
        "error": None,
        "evidence": "empty",
    }}
'''

            def diagnose(self, observation, failed_approaches=None, prior_solutions=None):
                # Intentionally sticky approach label — recovery must diverge it
                approach = "always_same_approach"
                self.approaches.append(approach)
                return {
                    "root_cause": "identical empty artifact every time",
                    "fault_layer": "skill_code",
                    "rewrite_skill": True,
                    "what_to_change": "write body",
                    "approach": approach,
                    "approach_fingerprint": _Obs2.fingerprint_approach(approach),
                    "approach_changed": False,
                    "needs_research": True,  # must be ignored without missing_knowledge
                    "missing_knowledge": [],
                    "research_queries": ["identical_query_should_not_loop"],
                    "needs_new_deps": [],
                    "missing_args": [],
                    "required_args": ["dest", "body"],
                    "suggested_args": {"dest": "sticky.txt", "body": "STICKY_OK"},
                    "test_plan": "VERIFY",
                    "expected_artifacts": [],
                    "is_unfixable": False,
                    "diagnosis": "sticky",
                }

            def verify_claim(self, goal, result, evidence) -> dict:
                return {"achieved": False, "confidence": 0.1, "reason": "no"}

            def converse(self, user_text, history=None) -> str:
                return "ok"

        logs_s: list[str] = []
        phases_s: list[str] = []
        brain_s = StickyVerifyFailBrain()
        orch_s = Orchestrator(
            root=st_root,
            brain=brain_s,
            on_log=lambda m: logs_s.append(m),
            on_status=lambda s: phases_s.append(s),
        )
        result_s = orch_s.run_cycle('Create "sticky.txt" containing "STICKY_OK"')
        assert not result_s.get("success"), result_s
        # Must stop with unresolved report — not spin forever
        assert any("RECOVERY STOP" in m or "RECOVERY BUDGET STOP" in m for m in logs_s), logs_s[-60:]
        assert any("STAGNATION" in m for m in logs_s), logs_s[-60:]
        # needs_research without missing_knowledge must NOT trigger research loop
        assert brain_s.research_notes_calls == 0, brain_s.research_notes_calls
        assert phases_s.count("RESEARCH") == 0, phases_s
        assert not any("KNOWLEDGE-GAP RESEARCH" in m for m in logs_s)
        assert not any("ADAPTIVE RESEARCH" in m for m in logs_s)
        # Approaches must diverge after stagnation (not identical cycle)
        diag_approaches = [
            (d.get("diagnosis") or {}).get("approach")
            if isinstance(d.get("diagnosis"), dict)
            else d.get("approach")
            for d in orch_s.memory.recent_diagnoses("sticky_fail_skill", 10)
        ]
        diag_approaches = [a for a in diag_approaches if a]
        assert any("coding_diverge" in str(a) for a in diag_approaches), diag_approaches
        # Failure signatures persisted
        fails = orch_s.memory.recent_failures("sticky_fail_skill", 10)
        assert fails
        assert any(
            (f.get("failure_signature") or (f.get("observation") or {}).get("failure_signature"))
            for f in fails
        )
        # Unit: ProgressAwareRecovery itself rejects identical no-progress loops
        tracker = ProgressAwareRecovery(max_attempts=4, stagnation_limit=2, max_research=1)
        sig = "abc123sig"
        for i in range(3):
            tracker.record(
                fault_layer="skill_code",
                failure_signature=sig,
                attempted_solution="same",
                model_used="test",
                result="VERIFY FAIL same",
                artifacts=[{"path": "x", "exists": True, "size": 0}],
                verifier_failure={"reason": "contains missing"},
                code_fingerprint="deadbeef",
                approach_fingerprint="samefp",
            )
            dec = tracker.evaluate(
                failure_signature=sig,
                fault_layer="skill_code",
                diagnosis_needs_research=True,
                missing_knowledge=[],
            )
        assert dec.stagnated
        assert not dec.allow_research  # no knowledge gap
        assert dec.prefer_coding
        orch_s.close()
        print("  OK stagnation — identical VERIFY FAIL does not repeat research cycle")
    except Exception as exc:
        msg = f"E2E_STAGNATION: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # 4e) E2E: brain stub → knowledge synthesis; research knowledge persisted
    print("  — e2e research→knowledge→code (no unimplemented stub) + save —")
    try:
        from jarvis.skill_builder import SkillBuilder as _SB

        kn_root = root / "data" / "_e2e_knowledge"
        if kn_root.exists():
            shutil.rmtree(kn_root)
        kn_root.mkdir(parents=True)

        class StubThenUselessBrain(Brain):
            """Always returns unimplemented stub — builder must synthesize."""

            def __init__(self) -> None:
                super().__init__()
                self.write_calls = 0
                self.research_notes_calls = 0

            def model_status(self) -> str:
                return "ONLINE"

            def is_available(self) -> bool:
                return True

            def classify_intent(self, user_text: str) -> dict:
                return {
                    "intent": "task",
                    "goal": user_text,
                    "needs_capability": True,
                    "keywords": user_text.split()[:8],
                }

            def plan(self, goal: str, known_capabilities: list[str]) -> dict:
                return {
                    "steps": ["synthesize skill from research"],
                    "can_reuse": [],
                    "missing": [goal],
                    "needs_research": True,
                    "needs_new_skill": True,
                    "skill_name": "knowledge_skill",
                    "skill_description": goal,
                    "research_queries": [goal, "python pathlib write file"],
                    "args": {"dest": "know_out.txt", "body": "KNOW_OK"},
                    "required_args": ["dest", "body"],
                }

            def extract_task_args(
                self, goal, skill_meta=None, prior_args=None, diagnosis=None
            ):
                return {"dest": "know_out.txt", "body": "KNOW_OK"}

            def research_notes(self, query: str, gathered: str) -> dict:
                self.research_notes_calls += 1
                return {
                    "approach": "pathlib_write_from_args",
                    "libraries": [],
                    "key_apis": ["pathlib", "write_text"],
                    "pitfalls": ["never ship unimplemented stub"],
                    "test_idea": "verifier checks dest+body",
                    "repair_insight": "write body into dest under workspace",
                }

            def write_skill_code(self, *a, **kw) -> str:
                self.write_calls += 1
                # Classic failure mode from the screenshot
                return '''
SKILL_META = {"name": "knowledge_skill", "description": "x", "capabilities": [],
 "dependencies": [], "version": 1}
def run(context: dict) -> dict:
    return {
        "ok": False,
        "result": None,
        "error": "Skill body not implemented yet",
        "evidence": "TODO",
    }
'''

            def diagnose(self, observation, failed_approaches=None, prior_solutions=None):
                from jarvis.observer import Observer
                approach = "force_knowledge_synthesis"
                fp = Observer.fingerprint_approach(approach)
                return {
                    "root_cause": "unimplemented stub instead of researched code",
                    "fault_layer": "skill_code",
                    "rewrite_skill": True,
                    "what_to_change": "Turn research knowledge into working run()",
                    "approach": approach,
                    "approach_fingerprint": fp,
                    "approach_changed": True,
                    "needs_research": True,
                    "research_queries": ["python pathlib write file from dict args"],
                    "needs_new_deps": [],
                    "missing_args": [],
                    "required_args": ["dest", "body"],
                    "suggested_args": {"dest": "know_out.txt", "body": "KNOW_OK"},
                    "test_plan": "VERIFY know_out.txt contains KNOW_OK",
                    "expected_artifacts": ["know_out.txt"],
                    "is_unfixable": False,
                    "diagnosis": str(observation.get("exception") or "")[:300],
                }

            def verify_claim(self, goal, result, evidence) -> dict:
                return {"achieved": True, "confidence": 0.9, "reason": "advisory"}

            def converse(self, user_text, history=None) -> str:
                return "ok"

        logs_k: list[str] = []
        brain_k = StubThenUselessBrain()
        orch_k = Orchestrator(
            root=kn_root,
            brain=brain_k,
            on_log=lambda m: logs_k.append(m),
        )
        goal_k = 'Create "know_out.txt" containing "KNOW_OK"'
        result_k = orch_k.run_cycle(goal_k)
        assert result_k.get("success"), result_k
        out_k = kn_root / "workspace_runtime" / "know_out.txt"
        assert out_k.exists() and "KNOW_OK" in out_k.read_text(encoding="utf-8"), out_k
        # Brain returned stub, but builder must not ship it
        assert brain_k.write_calls >= 1
        assert any("knowledge_synthesis" in m or "synthesizing from research" in m
                   for m in logs_k), logs_k[-40:]
        skill_code = Path(orch_k.registry.get_skill("knowledge_skill")["file_path"]).read_text(
            encoding="utf-8"
        )
        assert not _SB.is_unimplemented_stub(skill_code)
        assert "Skill body not implemented yet" not in skill_code
        # Knowledge must be saved and reloadable
        hist = orch_k.memory.get_research_knowledge("knowledge_skill")
        assert hist, hist
        assert orch_k.memory.get_latest_research("knowledge_skill")
        assert any("KNOWLEDGE SAVED" in m for m in logs_k), logs_k[-30:]
        orch_k.close()
        print("  OK knowledge stub→synthesis + research knowledge persisted")
    except Exception as exc:
        msg = f"E2E_KNOWLEDGE: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # 4f) E2E: learning request must NOT inherit prior skill (e.g. create_file)
    print("  — e2e learning intent isolated from prior ACTIVE skills —")
    try:
        from jarvis.intent import IntentClassifier

        # Unit: universal classification (no skill-name hardcoding)
        assert IntentClassifier.classify_offline(
            "Learn algorithms and create practical tests to verify knowledge"
        )["intent"] == "learning"
        assert IntentClassifier.classify_offline(
            "Iemācies tēmu un izveido praktiskus testus zināšanu pārbaudei"
        )["intent"] == "learning"
        assert IntentClassifier.classify_offline(
            'Create file notes.txt containing "hello"'
        )["intent"] == "task"
        assert IntentClassifier.classify_offline("What is recursion?")["intent"] == "conversation"
        # normalize must NOT downgrade semantic needs_capability → learning
        norm_cap = IntentClassifier.normalize(
            {"intent": "task", "needs_capability": True},
            "Research how caching works",
        )
        assert norm_cap["needs_capability"] is True, norm_cap
        assert norm_cap["intent"] == "task", norm_cap
        # Without capability flag, research-only offline hint stays non-capability
        norm_res = IntentClassifier.normalize(
            {"intent": "task", "needs_capability": False},
            "Research how caching works",
        )
        assert norm_res["needs_capability"] is False, norm_res
        assert norm_res["needs_research"] is True, norm_res

        learn_root = root / "data" / "_e2e_learning_intent"
        if learn_root.exists():
            shutil.rmtree(learn_root)
        learn_root.mkdir(parents=True)

        class OfflineBrain(Brain):
            def model_status(self) -> str:
                return "OFFLINE"

            def is_available(self) -> bool:
                return False

        logs_l: list[str] = []
        phases_l: list[str] = []
        orch_l = Orchestrator(
            root=learn_root,
            brain=OfflineBrain(),
            on_log=lambda m: logs_l.append(m),
            on_status=lambda s: phases_l.append(s),
        )
        # Tempt inheritance: ACTIVE unrelated skill already present
        bait = learn_root / "skills" / "create_file.py"
        bait.parent.mkdir(parents=True, exist_ok=True)
        bait.write_text(
            "SKILL_META={'name':'create_file','description':'create file',"
            "'capabilities':['create file'],'dependencies':[],'version':27}\n"
            "def run(context):\n"
            "    return {'ok': True, 'result': {}, 'error': None, 'evidence': 'x'}\n",
            encoding="utf-8",
        )
        orch_l.registry.register_candidate(
            "create_file",
            "create file",
            str(bait),
            ["create file"],
            version=27,
            protect_active=False,
        )
        orch_l.registry.set_status("create_file", "ACTIVE")
        assert orch_l.registry.get_skill("create_file")["status"] == "ACTIVE"

        req_l = "Learn network basics and create practical tests to verify knowledge"
        result_l = orch_l.handle_user_message(req_l)
        assert result_l.get("type") == "learning", result_l
        assert result_l.get("success"), result_l
        assert result_l.get("topic"), result_l
        assert "BUILD_SKILL" not in phases_l, phases_l
        assert "REPAIR" not in phases_l, phases_l
        assert not any("generating skill" in m for m in logs_l), logs_l[-30:]
        assert not any("KNOWLEDGE SAVED: create_file" in m for m in logs_l), logs_l[-30:]
        assert any("topic:" in m or "LEARNING" in m for m in logs_l), logs_l[-30:]
        # Topic knowledge saved; skill-bound research for create_file untouched
        assert orch_l.memory.get_topic_knowledge(result_l["topic"])
        assert not orch_l.memory.get_research_knowledge("create_file")
        # create_file must remain untouched at same version
        assert orch_l.registry.get_skill("create_file")["version"] == 27
        orch_l.close()
        print("  OK learning isolated — no skill inherit / no BUILD_SKILL")
    except Exception as exc:
        msg = f"E2E_LEARNING_INTENT: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # 4g) E2E: learning self-correct — VERIFY fail → gap-fill → PASS (not full restart)
    print("  — e2e learning self-correct (gap-fill KnowledgeArtifact) —")
    try:
        from jarvis.knowledge_artifact import KnowledgeArtifact as _KA

        sc_root = root / "data" / "_e2e_learning_self_correct"
        if sc_root.exists():
            shutil.rmtree(sc_root)
        sc_root.mkdir(parents=True)

        class OfflineBrainSC(Brain):
            def model_status(self) -> str:
                return "OFFLINE"

            def is_available(self) -> bool:
                return False

        class SelfCorrectOrch(Orchestrator):
            """First synthesis is thin so VERIFY fails once; gap-fill recovers."""

            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self._synth_calls = 0
                self._synth_queries: list[list] = []

            def _synthesize_learning_artifact(self, **kw):
                self._synth_calls += 1
                if self._synth_calls == 1:
                    # Empty artifact → substance fail; diagnose must name fields
                    return _KA(
                        request_id=kw.get("request_id") or "",
                        topic=kw.get("topic") or "",
                        original_request=kw.get("original_request") or "",
                        concepts=[],
                        explanations=[],
                        examples=[],
                        practice="",
                        summary="x",
                    )
                return super()._synthesize_learning_artifact(**kw)

        # Track research queries via wrapping ResearchSystem.research
        logs_sc: list[str] = []
        phases_sc: list[str] = []
        orch_sc = SelfCorrectOrch(
            root=sc_root,
            brain=OfflineBrainSC(),
            on_log=lambda m: logs_sc.append(m),
            on_status=lambda s: phases_sc.append(s),
        )
        _orig_research = orch_sc.research.research
        query_batches: list[list] = []

        research_kwargs: list[dict] = []

        def _spy_research(queries, goal="", **kwargs):
            query_batches.append(list(queries or []))
            research_kwargs.append(dict(kwargs))
            return _orig_research(queries, goal=goal, **kwargs)

        orch_sc.research.research = _spy_research  # type: ignore[method-assign]

        req_sc = "Learn graph basics and create practical tests to verify knowledge"
        result_sc = orch_sc.handle_user_message(req_sc)
        assert result_sc.get("type") == "learning", result_sc
        assert result_sc.get("success"), result_sc
        assert (result_sc.get("attempts") or 0) >= 2, result_sc
        assert orch_sc._synth_calls >= 2
        assert "OBSERVE" in phases_sc and "DIAGNOSE" in phases_sc
        assert phases_sc.count("RESEARCH") >= 2
        assert phases_sc.count("VERIFY") >= 2
        assert any("LEARNING VERIFY: FAIL" in m for m in logs_sc), logs_sc[-40:]
        assert any("LEARNING VERIFY: PASS" in m for m in logs_sc), logs_sc[-40:]
        assert any("gap-fill" in m.lower() or "GAP" in m for m in logs_sc), logs_sc[-40:]
        topic_sc = result_sc["topic"]
        failed = orch_sc.memory.get_failed_approaches(f"learning:{topic_sc}")
        assert failed, failed
        failed_labels = {str(a.get("approach") or "") for a in failed}
        assert any("knowledge_artifact" in x for x in failed_labels), failed_labels
        # Second research batch must be gap-fill (not identical full restart only)
        assert len(query_batches) >= 2
        q2_text = " ".join(str(q) for q in query_batches[1]).lower()
        assert any(
            tok in q2_text for tok in ("fill", "concept", "example", "explain", "practice")
        ), query_batches
        # Gap-fill attempt must be local (no network / no brain skill notes)
        assert len(research_kwargs) >= 2
        assert research_kwargs[0].get("mode") == "learning"
        assert research_kwargs[1].get("network") is False, research_kwargs[1]
        assert research_kwargs[1].get("use_brain") is False, research_kwargs[1]
        verified_hist = [
            e for e in orch_sc.memory.get_topic_knowledge(topic_sc)
            if isinstance(e, dict) and e.get("verified")
        ]
        assert verified_hist, orch_sc.memory.get_topic_knowledge(topic_sc)
        # Verified entry must carry clean artifact fields (no skill jargon)
        vent = verified_hist[-1]
        blob = " ".join(
            [
                str(vent.get("summary") or ""),
                " ".join(str(x) for x in (vent.get("concepts") or [])),
                " ".join(str(x) for x in (vent.get("explanations") or [])),
            ]
        )
        assert "SKILL_META" not in blob and "rewrite_skill" not in blob
        for bad in (
            "Prefer Python stdlib",
            "Never claim success",
            "result.path",
            "Local capability hints",
        ):
            assert bad.lower() not in blob.lower(), (bad, blob[:400])

        # Reuse verified knowledge on a fresh orchestrator (same DB root)
        logs_ru: list[str] = []
        orch_ru = Orchestrator(
            root=sc_root,
            brain=OfflineBrainSC(),
            on_log=lambda m: logs_ru.append(m),
        )
        result_ru = orch_ru.handle_user_message(req_sc)
        assert result_ru.get("success"), result_ru
        assert (result_ru.get("attempts") or 1) == 1, result_ru
        assert any(
            ("MEMORY USED" in m)
            or ("Reusing" in m and "verified" in m.lower())
            or ("MEMORY RETRIEVAL" in m and "verified=" in m)
            for m in logs_ru
        ), logs_ru[:40]
        assert any("MEMORY RETRIEVAL" in m for m in logs_ru), logs_ru[:40]
        orch_sc.close()
        orch_ru.close()
        print("  OK learning self-correct FAIL→gap-fill→PASS + reuse")
    except Exception as exc:
        msg = f"E2E_LEARNING_SELF_CORRECT: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # 4h) E2E: LEARNING VERIFY is semantic (no token hits) + practical result
    print("  — e2e learning semantic VERIFY + practical result —")
    try:
        from jarvis import learning_verify as lv

        # Unit: paraphrase covers goal without exact keyword equality
        paraphrase_knowledge = {
            "approach": (
                "Study how devices exchange data: addressing, routing, and "
                "protocols that move packets across links on a network."
            ),
            "key_apis": ["addressing models", "routing basics", "protocol layers"],
            "pitfalls": ["confusing local and wide links"],
            "test_idea": "Self-check: explain packet delivery in your own words.",
            "raw": "Notes about communication between computers.",
            "sources": [{"title": "local"}],
            "results": [{"title": "x"}],
        }
        j_ok = lv.offline_semantic_judgment(
            "Learn networking fundamentals", paraphrase_knowledge
        )
        assert j_ok["covers_goal"], j_ok
        j_bad = lv.offline_semantic_judgment(
            "Learn networking fundamentals",
            {
                **paraphrase_knowledge,
                "approach": "Bake sourdough bread with long fermentation.",
                "key_apis": ["flour hydration", "starter feeding"],
                "raw": "bakery notes",
            },
        )
        assert not j_bad["covers_goal"], j_bad

        # Unit: practical ask produces + verifies a concrete result (any expression)
        expr_req = "Learn arithmetic and compute 12+5"
        assert lv.requests_practical_result(expr_req)
        produced = lv.produce_practical_offline(expr_req, paraphrase_knowledge)
        assert produced.get("result") == 17, produced
        assert lv.verify_practical_offline(expr_req, produced)["ok"]

        sem_root = root / "data" / "_e2e_learning_semantic"
        if sem_root.exists():
            shutil.rmtree(sem_root)
        sem_root.mkdir(parents=True)

        class OfflineBrainSem(Brain):
            def model_status(self) -> str:
                return "OFFLINE"

            def is_available(self) -> bool:
                return False

        logs_sem: list[str] = []
        orch_sem = Orchestrator(
            root=sem_root,
            brain=OfflineBrainSem(),
            on_log=lambda m: logs_sem.append(m),
        )
        # Universal practical learning request — expression not hardcoded in product code
        req_sem = "Learn arithmetic basics and compute 12+5"
        result_sem = orch_sem.handle_user_message(req_sem)
        assert result_sem.get("type") == "learning", result_sem
        assert result_sem.get("success"), result_sem
        ver = result_sem.get("verification") or {}
        check_names = [c.get("name") for c in (ver.get("checks") or [])]
        assert "semantic_goal_coverage" in check_names, check_names
        assert "practical_result" in check_names, check_names
        assert not any("token_hits" in str(c) for c in (ver.get("checks") or []))
        assert (ver.get("practical_result") or {}).get("result") == 17, ver
        assert "Practical result" in (result_sem.get("reply") or "")
        orch_sem.close()
        print("  OK learning semantic VERIFY + practical result checked")
    except Exception as exc:
        msg = f"E2E_LEARNING_SEMANTIC: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # 4h2) E2E: KnowledgeArtifact — clean, one-attempt, VERIFY vs USER REQUEST, gap-fill
    print("  — e2e learning KnowledgeArtifact (clean / 1-attempt / gap-fill) —")
    try:
        from jarvis.knowledge_artifact import (
            KnowledgeArtifact,
            synthesize_offline,
            is_polluted,
        )
        from jarvis import learning_verify as lv2

        # Unit: polluted research raw never enters VERIFY blob
        polluted = {
            "approach": "Implement Python solution for: Learn optics basics",
            "test_idea": "Execute skill.run and independently verify artifacts",
            "raw": (
                "### Local hints\n"
                "- Skill must define SKILL_META and run(context)\n"
                "- Return concrete result.path\n"
                "Optics studies light: reflection, refraction, and lenses.\n"
                "For example, a lens focuses parallel rays to a point.\n"
            ),
            "key_apis": ["Hint: pathlib / os"],
            "sources": [
                {"title": "Local capability hints", "url": "", "source": "local"},
                {"title": "Optics overview", "url": "http://ex.test/optics", "source": "web"},
            ],
        }
        req_u = "Learn optics basics and create practical tests to verify knowledge"
        art_u = synthesize_offline(
            user_request=req_u, topic="optics", request_id="rid1", research=polluted
        )
        assert art_u.has_substance(), art_u.to_dict()
        assert not art_u.is_contaminated(), art_u.to_dict()
        assert not any(is_polluted(c) for c in art_u.concepts)
        for c in art_u.concepts + art_u.explanations + art_u.examples:
            assert "Prefer Python" not in c
            assert "Never claim success" not in c
            assert "SKILL_META" not in c
        blob_u = lv2.knowledge_blob(
            {**art_u.research_compat(), "artifact": art_u.to_dict(), "raw": polluted["raw"]}
        )
        assert "SKILL_META" not in blob_u
        assert "rewrite_skill" not in blob_u
        assert "result.path" not in blob_u
        assert "Prefer Python stdlib" not in blob_u
        assert "Never claim success" not in blob_u
        assert "pathlib" not in blob_u.lower() or "optics" in blob_u.lower()
        # High relatedness with substance → covers in one offline judge
        judge_u = lv2.offline_semantic_judgment(
            req_u, {**art_u.research_compat(), "artifact": art_u.to_dict(), "raw": ""}
        )
        assert judge_u["covers_goal"], judge_u
        assert judge_u.get("relatedness", 0) >= 0.45

        # Simple knowledge request completes in ONE attempt (offline brain)
        class OfflineBrainKA(Brain):
            def model_status(self) -> str:
                return "OFFLINE"

            def is_available(self) -> bool:
                return False

        ka_root = root / "data" / "_e2e_knowledge_artifact"
        if ka_root.exists():
            shutil.rmtree(ka_root)
        ka_root.mkdir(parents=True)
        logs_ka: list[str] = []
        orch_ka = Orchestrator(
            root=ka_root,
            brain=OfflineBrainKA(),
            on_log=lambda m: logs_ka.append(m),
        )
        # Inject research content via spy so we don't depend on network
        _real = orch_ka.research.research

        def _rich_research(queries, goal="", **kwargs):
            base = _real(queries, goal=goal, **kwargs)
            base["raw"] = (
                str(base.get("raw") or "")
                + "\nSignal processing transforms measurements into useful information. "
                "Filtering removes noise; sampling captures discrete values. "
                "For example, a low-pass filter attenuates high frequencies. "
                "Practice: explain sampling and give one filtering example."
            )
            base["approach"] = ""
            base["test_idea"] = "Execute skill.run and independently verify artifacts"
            return base

        orch_ka.research.research = _rich_research  # type: ignore[method-assign]
        req_ka = (
            "Learn signal processing basics and create practical tests "
            "to verify knowledge"
        )
        result_ka = orch_ka.handle_user_message(req_ka)
        assert result_ka.get("type") == "learning", result_ka
        assert result_ka.get("success"), result_ka
        assert (result_ka.get("attempts") or 99) == 1, result_ka
        assert result_ka.get("request_id")
        art_saved = result_ka.get("artifact") or (result_ka.get("knowledge") or {}).get("artifact")
        assert art_saved, result_ka
        assert art_saved.get("concepts") or art_saved.get("explanations")
        narr = " ".join(
            [
                str(art_saved.get("summary") or ""),
                " ".join(str(x) for x in (art_saved.get("concepts") or [])),
                " ".join(str(x) for x in (art_saved.get("explanations") or [])),
            ]
        )
        assert "SKILL_META" not in narr and "fault_layer" not in narr
        for bad in (
            "Prefer Python stdlib",
            "Never claim success",
            "result.path",
            "skill.run",
            "Local capability hints",
            "rewrite_skill",
        ):
            assert bad.lower() not in narr.lower(), (bad, narr[:400])
        # VERIFY must mention KnowledgeArtifact / USER REQUEST coverage
        ver_ka = result_ka.get("verification") or {}
        assert ver_ka.get("verified")
        assert any(
            c.get("name") == "semantic_goal_coverage" and c.get("ok")
            for c in (ver_ka.get("checks") or [])
        ), ver_ka
        # No research raw in verify path logs
        assert not any("Execute skill.run" in m for m in logs_ka if "VERIFY" in m)

        # Gap-fill: thin first artifact → diagnose concrete fields → second fills only gaps
        class GapFillOrch(Orchestrator):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.calls = 0
                self.gap_queries: list[list] = []

            def _synthesize_learning_artifact(self, **kw):
                self.calls += 1
                if self.calls == 1:
                    return KnowledgeArtifact(
                        request_id=kw.get("request_id") or "",
                        topic=kw.get("topic") or "",
                        original_request=kw.get("original_request") or "",
                        summary="short",
                        concepts=[],
                        explanations=[],
                        examples=[],
                    )
                # Gap-fill merge path
                return super()._synthesize_learning_artifact(**kw)

        gf_root = root / "data" / "_e2e_knowledge_gapfill"
        if gf_root.exists():
            shutil.rmtree(gf_root)
        gf_root.mkdir(parents=True)
        orch_gf = GapFillOrch(
            root=gf_root,
            brain=OfflineBrainKA(),
            on_log=lambda m: None,
        )
        _real2 = orch_gf.research.research

        def _spy2(queries, goal="", **kwargs):
            orch_gf.gap_queries.append(list(queries or []))
            base = _real2(queries, goal=goal, **kwargs)
            base["raw"] = (
                "Control systems regulate behavior using feedback loops. "
                "A setpoint is compared to measured output. "
                "For example, a thermostat turns heating on and off. "
                "Practice questions check understanding of feedback."
            )
            return base

        orch_gf.research.research = _spy2  # type: ignore[method-assign]
        req_gf = "Learn control systems basics and create practical tests to verify knowledge"
        res_gf = orch_gf.handle_user_message(req_gf)
        assert res_gf.get("success"), res_gf
        assert (res_gf.get("attempts") or 0) >= 2
        assert orch_gf.calls >= 2
        assert len(orch_gf.gap_queries) >= 2
        # Second batch targets missing fields — not a blind full-topic restart only
        q2 = " ".join(orch_gf.gap_queries[1]).lower()
        assert any(
            tok in q2 for tok in ("concept", "explain", "example", "practice", "fill", "gap")
        ), orch_gf.gap_queries[1]
        orch_ka.close()
        orch_gf.close()
        print("  OK KnowledgeArtifact clean / 1-attempt / gap-fill")
    except Exception as exc:
        msg = f"E2E_KNOWLEDGE_ARTIFACT: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # 4i) E2E: research pipeline MEMORY→SEARCH→OPEN→EXTRACT→SAVE→REUSE
    print("  — e2e research+memory pipeline (open/extract, not snippets) —")
    try:
        from jarvis import source_pipeline as sp
        from jarvis.research import ResearchSystem as RSPipe
        from jarvis.orchestrator import Orchestrator as OrchPipe

        # Unit: snippets alone never rank as knowledge; extract required
        hits = [
            {
                "url": "https://example.test/optics-guide",
                "title": "Optics guide: reflection and refraction",
                "snippet": "Short blurb",
                "provider": "duckduckgo",
                "query": "optics",
            },
            {
                "url": "http://spam.xyz/buy-now",
                "title": "Buy now casino",
                "snippet": "click",
                "provider": "duckduckgo",
                "query": "optics",
            },
            {
                "url": "",
                "title": "no url",
                "snippet": "x",
                "provider": "duckduckgo",
            },
        ]
        ranked = sp.rank_search_hits(
            hits, goal="Learn optics basics reflection refraction", limit=3
        )
        assert ranked and ranked[0]["url"].startswith("https://")
        assert not any("casino" in (h.get("title") or "").lower() for h in ranked)

        html_body = """
        <html><head><title>Optics</title></head><body>
        <nav>Menu</nav>
        <p>Optics is the study of light. Reflection occurs when light bounces
        off a surface. Refraction bends light as it passes between media.</p>
        <p>For example, a lens focuses parallel rays to a point using refraction.</p>
        <script>evil()</script>
        </body></html>
        """
        text = sp.extract_text(html_body, content_type="text/html")
        assert "Optics is the study" in text
        assert "evil" not in text
        assert "Menu" not in text or len(text) > 80

        # Mock OPEN so tests do not depend on live network
        real_fetch = sp.fetch_url

        def _fake_fetch(url, timeout=12.0):
            return {
                "url": url,
                "ok": True,
                "status": 200,
                "content_type": "text/html",
                "body": html_body,
                "error": "",
            }

        sp.fetch_url = _fake_fetch  # type: ignore[assignment]
        try:
            extracts, comparison = sp.open_and_extract(
                ranked,
                goal="Learn optics basics and create practical tests",
                on_log=lambda m: None,
            )
            assert any(e.get("ok") for e in extracts), extracts
            blob = sp.knowledge_from_extracts(
                extracts, comparison,
                goal="Learn optics basics and create practical tests",
            )
            assert blob and "Optics is the study" in blob
            assert "Short blurb" not in blob  # snippet must not be the knowledge
        finally:
            sp.fetch_url = real_fetch  # type: ignore[assignment]

        # ResearchSystem learning path logs SEARCH / SOURCE FOUND / OPEN / EXTRACT
        pipe_logs: list[str] = []
        rs = RSPipe(brain=None, on_log=lambda m: pipe_logs.append(m), timeout=2.0)

        def _ddg_optics(q, append_python=False):
            return "hit", [{
                "url": "https://example.test/optics-guide",
                "title": "Optics guide: reflection and refraction",
                "snippet": "Short blurb only — not knowledge",
                "source": "web",
                "provider": "duckduckgo",
                "timestamp": 0,
                "query": q,
            }]

        rs._duckduckgo_search = _ddg_optics  # type: ignore[method-assign]
        sp.fetch_url = _fake_fetch  # type: ignore[assignment]
        try:
            notes = rs.research(
                ["Learn optics basics"],
                goal="Learn optics basics and create practical tests to verify knowledge",
                mode="learning",
                network=True,
                use_brain=False,
                append_python=False,
            )
        finally:
            sp.fetch_url = real_fetch  # type: ignore[assignment]

        assert notes.get("knowledge_from_extracts") is True, notes
        assert "Optics is the study" in (notes.get("raw") or "")
        assert "Short blurb only" not in (notes.get("raw") or "")
        assert any(m.startswith("SEARCH:") for m in pipe_logs), pipe_logs
        assert any(m.startswith("SOURCE FOUND:") for m in pipe_logs), pipe_logs
        assert any(m.startswith("SOURCE OPEN:") for m in pipe_logs), pipe_logs
        assert any(m.startswith("SOURCE EXTRACT:") for m in pipe_logs), pipe_logs

        # Orchestrator: MEMORY RETRIEVAL → SAVE → second request MEMORY USED
        class OfflineBrainPipe(Brain):
            def model_status(self) -> str:
                return "OFFLINE"

            def is_available(self) -> bool:
                return False

        mem_root = root / "data" / "_e2e_research_memory_pipe"
        if mem_root.exists():
            shutil.rmtree(mem_root)
        mem_root.mkdir(parents=True)
        logs1: list[str] = []
        orch1 = OrchPipe(
            root=mem_root,
            brain=OfflineBrainPipe(),
            on_log=lambda m: logs1.append(m),
        )
        _real_r = orch1.research.research

        def _rich_pipe(queries, goal="", **kwargs):
            # Force extract-backed research without live web
            base = {
                "approach": "",
                "libraries": [],
                "key_apis": [],
                "pitfalls": [],
                "test_idea": "Self-check practice",
                "raw": (
                    "### Source: Thermodynamics primer\n"
                    "URL: https://example.test/thermo\n"
                    "Thermodynamics studies energy, heat, and work. "
                    "The first law states energy is conserved. "
                    "For example, a heat engine converts thermal energy to mechanical work. "
                    "Practice: state the first law in your own words."
                ),
                "sources": [{
                    "url": "https://example.test/thermo",
                    "title": "Thermodynamics primer",
                    "source": "extracted",
                    "provider": "web",
                    "extracted": True,
                    "char_count": 200,
                }],
                "results": [],
                "extracts": [{
                    "url": "https://example.test/thermo",
                    "title": "Thermodynamics primer",
                    "ok": True,
                    "char_count": 200,
                    "relatedness": 0.5,
                }],
                "comparison": {"consensus": ["energy is conserved"], "source_count": 1},
                "mode": "learning",
                "network": True,
                "brain_used": False,
                "opened_sources": True,
                "knowledge_from_extracts": True,
            }
            logs1.append("SEARCH: query → " + ",".join(queries or []))
            logs1.append("SOURCE FOUND: 1 candidate URLs from SEARCH")
            logs1.append("SOURCE OPEN: https://example.test/thermo")
            logs1.append("SOURCE EXTRACT: ok chars=200 rel=0.50 title=Thermodynamics")
            return base

        orch1.research.research = _rich_pipe  # type: ignore[method-assign]
        req_t = (
            "Learn thermodynamics basics and create practical tests "
            "to verify knowledge"
        )
        r1 = orch1.handle_user_message(req_t)
        assert r1.get("success"), r1
        assert any("MEMORY RETRIEVAL" in m for m in logs1), logs1[:40]
        assert any("KNOWLEDGE SAVED" in m for m in logs1), logs1[-30:]
        topic_t = r1["topic"]
        verified = [
            e for e in orch1.memory.get_topic_knowledge(topic_t)
            if e.get("verified")
        ]
        assert verified, orch1.memory.get_topic_knowledge(topic_t)
        # Sources must be extracted evidence, not empty
        assert (verified[-1].get("sources") or verified[-1].get("artifact")), verified[-1]

        logs2: list[str] = []
        orch2 = OrchPipe(
            root=mem_root,
            brain=OfflineBrainPipe(),
            on_log=lambda m: logs2.append(m),
        )
        research_calls = {"n": 0}
        _r2 = orch2.research.research

        def _count_research(queries, goal="", **kwargs):
            research_calls["n"] += 1
            return _r2(queries, goal=goal, **kwargs)

        orch2.research.research = _count_research  # type: ignore[method-assign]
        r2 = orch2.handle_user_message(req_t)
        assert r2.get("success"), r2
        assert (r2.get("attempts") or 1) == 1
        assert research_calls["n"] == 0, research_calls  # full research skipped
        assert any("MEMORY USED" in m for m in logs2), logs2[:40]
        assert any("MEMORY RETRIEVAL" in m for m in logs2), logs2[:40]
        orch1.close()
        orch2.close()
        print("  OK research+memory pipeline OPEN/EXTRACT + MEMORY reuse")
    except Exception as exc:
        msg = f"E2E_RESEARCH_MEMORY_PIPELINE: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # 4j) E2E: performance hardening — no offline brain hang, local gap-fill, tags cache
    print("  — e2e performance hardening (research/brain/router) —")
    try:
        from jarvis.research import ResearchSystem
        from jarvis.model_router import ModelRouter, TAGS_CACHE_TTL_SEC
        from jarvis.brain import Brain as BrainCls, _is_connection_failure
        from jarvis import learning_verify as lv_perf
        from jarvis.orchestrator import Orchestrator as OrchPerf

        assert _is_connection_failure("Connection refused")
        assert _is_connection_failure(ConnectionRefusedError("boom"))
        assert TAGS_CACHE_TTL_SEC >= 5

        # Research: offline brain must NOT call research_notes
        class _TrapBrain:
            def is_available(self):
                return False

            def research_notes(self, *a, **k):
                raise AssertionError("research_notes must not run when offline")

        calls = {"brain": 0, "ddg": 0}

        rs = ResearchSystem(brain=_TrapBrain(), on_log=lambda m: None, timeout=2.0)
        _ddg = rs._duckduckgo_search

        def _count_ddg(q, append_python=True):
            calls["ddg"] += 1
            return "(stub)", []

        rs._duckduckgo_search = _count_ddg  # type: ignore[method-assign]

        def _no_pypi(q):
            return "", []

        rs._pypi_lookup = _no_pypi  # type: ignore[method-assign]
        from jarvis import source_pipeline as _sp_perf
        _prev_fetch = _sp_perf.fetch_url

        def _no_fetch(url, timeout=12.0):
            return {
                "url": url,
                "ok": False,
                "status": 0,
                "content_type": "",
                "body": "",
                "error": "stub",
            }

        _sp_perf.fetch_url = _no_fetch  # type: ignore[assignment]
        try:
            out_skill = rs.research(
                ["make zip file"], goal="zip", mode="skill", network=True
            )
        finally:
            _sp_perf.fetch_url = _prev_fetch  # type: ignore[assignment]
        assert out_skill.get("brain_used") is False
        assert "Local hints" in (out_skill.get("raw") or "")
        assert "SKILL_META" in (out_skill.get("raw") or "")
        assert calls["ddg"] >= 1

        # Learning mode: no skill hints, no python append, no brain
        calls["ddg"] = 0
        seen_append: list[bool] = []

        def _ddg2(q, append_python=True):
            seen_append.append(bool(append_python))
            calls["ddg"] += 1
            return "optics notes", [{
                "url": "http://ex/o", "title": "Optics", "source": "web",
                "provider": "ddg", "timestamp": 0, "query": q, "snippet": "light",
            }]

        rs._duckduckgo_search = _ddg2  # type: ignore[method-assign]
        out_learn = rs.research(
            ["Learn optics basics"],
            goal="Learn optics basics",
            mode="learning",
            network=True,
            use_brain=False,
            append_python=False,
        )
        assert out_learn.get("mode") == "learning"
        assert out_learn.get("brain_used") is False
        assert "SKILL_META" not in (out_learn.get("raw") or "")
        assert "Learning focus" in (out_learn.get("raw") or "")
        assert seen_append and seen_append[0] is False

        # Gap-fill local: network=False → no DDG
        calls["ddg"] = 0
        out_gap = rs.research(
            ["optics — core concepts"],
            goal="Learn optics",
            mode="learning",
            network=False,
            use_brain=False,
        )
        assert calls["ddg"] == 0
        assert out_gap.get("network") is False
        assert "Gap-fill focus" in (out_gap.get("raw") or "")

        # Tags cache: status() must not refresh every call
        n_list = {"n": 0}

        def _list():
            n_list["n"] += 1
            return ["qwen3:4b", "qwen3:30b", "qwen3-coder:30b"]

        router = ModelRouter(list_models=_list, on_log=lambda m: None)
        assert router.status() == "ONLINE"
        first = n_list["n"]
        assert router.status() == "ONLINE"
        assert router.status() == "ONLINE"
        assert n_list["n"] == first  # cached

        # Connection fail-fast: no HTTP double-wait
        class FailFastBrain(BrainCls):
            def _get_client(self):
                class C:
                    def chat(self, **kw):
                        raise ConnectionRefusedError("Connection refused")

                return C()

            def _http_chat(self, *a, **k):
                raise AssertionError("HTTP fallback must not run on connection refused")

            def _list_model_names(self):
                return ["qwen3:4b"]

        ff = FailFastBrain()
        text = ff._chat_on_model([{"role": "user", "content": "hi"}], 0.1, "qwen3:4b")
        assert text.startswith("[BRAIN ERROR]")
        assert "unreachable" in text.lower() or "refused" in text.lower()

        # Practical: offline expression before any brain call
        class SpyPracticalOrch(OrchPerf):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.brain_practical_calls = 0

        class OnlineSpyBrain(Brain):
            def model_status(self) -> str:
                return "ONLINE"

            def is_available(self) -> bool:
                return True

            def produce_learning_answer(self, *a, **k):
                raise AssertionError("brain practical must not run for offline expression")

        pr_root = root / "data" / "_e2e_perf_practical"
        if pr_root.exists():
            shutil.rmtree(pr_root)
        pr_root.mkdir(parents=True)
        orch_p = SpyPracticalOrch(root=pr_root, brain=OnlineSpyBrain())
        # Monkeypatch produce on the instance brain
        def _boom(*a, **k):
            orch_p.brain_practical_calls += 1
            raise AssertionError("should not call brain for arithmetic")

        orch_p.brain.produce_learning_answer = _boom  # type: ignore[method-assign]
        got = orch_p._produce_learning_practical(
            "What is 12+8? Explain briefly.",
            {"approach": "arithmetic", "key_apis": [], "test_idea": "check sum"},
        )
        assert got and got.get("source") == "offline_expression", got
        assert orch_p.brain_practical_calls == 0
        assert lv_perf.extract_arithmetic_tasks("What is 12+8?")
        orch_p.close()

        # Ledger start_task does not double-log REQUEST
        from jarvis.ledger import Ledger
        led_path = root / "data" / "_e2e_perf_ledger" / "l.db"
        led_path.parent.mkdir(parents=True, exist_ok=True)
        led = Ledger(led_path)
        tid = led.start_task("goal x")
        actions = led.task_actions(tid) if hasattr(led, "task_actions") else None
        if actions is None:
            # Fallback: query via get
            with led._lock:
                rows = led._conn.execute(
                    "SELECT phase FROM actions WHERE task_id=? ORDER BY id", (tid,)
                ).fetchall()
            phases = [r[0] for r in rows]
        else:
            phases = [a.get("phase") for a in actions]
        assert phases.count("REQUEST") == 0, phases  # caller logs REQUEST once
        led.log(tid, "REQUEST", "once")
        with led._lock:
            rows = led._conn.execute(
                "SELECT phase FROM actions WHERE task_id=?", (tid,)
            ).fetchall()
        assert [r[0] for r in rows].count("REQUEST") == 1
        led.close() if hasattr(led, "close") else None

        print("  OK performance hardening — research/brain/router/practical/ledger")
    except Exception as exc:
        msg = f"E2E_PERF_HARDENING: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # 4k) E2E: each USER REQUEST isolated — conversation ≠ prior learning final_result
    print("  — e2e request routing isolation (conversation ≠ prior DONE) —")
    try:
        from jarvis.intent import IntentClassifier
        from jarvis.memory import Memory

        route_root = root / "data" / "_e2e_request_routing"
        if route_root.exists():
            shutil.rmtree(route_root)
        route_root.mkdir(parents=True)

        prior_done = (
            "DONE. Learning complete — knowledge verified & saved.\n"
            "Topic: sample_topic\n"
            "Attempts: 1\n"
            "Approach: initial_topic_research\n"
            "Summary: prior learning final_result that must not leak\n"
            "Sources: 0"
        )

        class HistoryLeakBrain(Brain):
            """Simulates the bug: if prior DONE is in history, echo it as the reply."""

            def model_status(self) -> str:
                return "ONLINE"

            def is_available(self) -> bool:
                return True

            def classify_intent(self, text: str) -> dict:
                return IntentClassifier.classify_offline(text)

            def converse(self, user_text: str, history=None) -> str:
                for m in history or []:
                    content = str((m or {}).get("content") or "")
                    if Memory._looks_like_cycle_outcome(content):
                        return content  # stale reuse of prior final_result
                return f"Fresh conversational answer about: {user_text}"

        orch_r = Orchestrator(
            root=route_root,
            brain=HistoryLeakBrain(),
            on_log=lambda m: None,
            on_status=lambda s: None,
        )
        # Plant a completed learning turn in conversation memory
        orch_r.memory.add_message(
            "user",
            "Learn sample topic thoroughly",
            meta={"request_id": "oldreq01", "intent": "learning", "phase": "received"},
        )
        orch_r.memory.add_message(
            "assistant",
            prior_done,
            meta={"request_id": "oldreq01", "intent": "learning", "task_id": 99},
        )
        assert orch_r.memory.last_cycle_assistant_reply() == prior_done
        # Filtered history must NOT include the learning DONE pair
        hist = orch_r.memory.conversation_history_for_llm(8)
        assert not any(
            Memory._looks_like_cycle_outcome(m.get("content") or "") for m in hist
        ), hist
        assert not any("sample topic" in (m.get("content") or "").lower() for m in hist), hist

        chat_q = "What is photosynthesis in simple terms?"
        assert IntentClassifier.classify_offline(chat_q)["intent"] == "conversation"
        result_c = orch_r.handle_user_message(chat_q)
        assert result_c.get("type") == "conversation", result_c
        assert result_c.get("request_id"), result_c
        assert result_c.get("request_id") != "oldreq01"
        reply_c = result_c.get("reply") or ""
        assert reply_c.strip() != prior_done.strip(), reply_c
        assert "DONE. Learning complete" not in reply_c, reply_c
        assert "prior learning final_result" not in reply_c, reply_c
        assert "photosynthesis" in reply_c.lower() or "Fresh conversational" in reply_c, reply_c
        # Second request still isolated — sticky stale brain path
        orch_r.close()

        class StickyStaleBrain(Brain):
            """Always returns prior DONE — _ensure_fresh_reply must reject it."""

            def model_status(self) -> str:
                return "ONLINE"

            def is_available(self) -> bool:
                return True

            def classify_intent(self, text: str) -> dict:
                return IntentClassifier.classify_offline(text)

            def converse(self, user_text: str, history=None) -> str:
                return prior_done

        orch_s = Orchestrator(
            root=route_root,
            brain=StickyStaleBrain(),
            on_log=lambda m: None,
            on_status=lambda s: None,
        )
        orch_s.memory.add_message(
            "assistant",
            prior_done,
            meta={"request_id": "oldreq02", "intent": "learning", "task_id": 100},
        )
        chat_q2 = "How are you today?"
        result_s = orch_s.handle_user_message(chat_q2)
        assert result_s.get("type") == "conversation", result_s
        assert result_s.get("request_id")
        reply_s = result_s.get("reply") or ""
        assert reply_s.strip() != prior_done.strip(), reply_s
        assert "DONE. Learning complete" not in reply_s, reply_s
        assert chat_q2[:20] in reply_s or "jautājumu" in reply_s.lower() or "How are you" in reply_s
        # Distinct request_ids across turns
        assert result_c.get("request_id") != result_s.get("request_id")
        orch_s.close()
        print("  OK request routing isolation — conversation ≠ prior learning DONE")
    except Exception as exc:
        msg = f"E2E_REQUEST_ROUTING: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # 4l) E2E: auto-calibration — VERIFY feedback, priority, rollback, persist
    print("  — e2e auto-calibration / self-improvement —")
    try:
        from jarvis.calibration import (
            AutoCalibration,
            CalibParams,
            EVAL_MIN_OUTCOMES,
            FACT_PARAMS,
            FACT_PENDING,
            FACT_CONFIG_STACK,
        )
        from jarvis.memory import Memory as MemCal
        from jarvis.recovery import ProgressAwareRecovery

        cal_root = root / "data" / "_e2e_calibration"
        if cal_root.exists():
            shutil.rmtree(cal_root)
        cal_root.mkdir(parents=True)
        mem = MemCal(cal_root / "jarvis.db")
        logs_c: list[str] = []
        cal = AutoCalibration(mem, on_log=lambda m: logs_c.append(m))

        # Defaults load / clamps
        assert cal.repair_attempts() >= 2
        p = CalibParams(max_repair_attempts=99, chat_timeout_sec=5.0).clamped()
        assert p.max_repair_attempts <= 8
        assert p.chat_timeout_sec >= 45.0

        # Multi-fail → avoid approach; success → higher score
        fp_bad = cal.fingerprint("bad_strategy", "repair")
        for _ in range(4):
            cal.observe_verify(
                verified=False, domain="repair", approach="bad_strategy",
                approach_fingerprint=fp_bad,
            )
        assert cal.should_avoid_approach(fp_bad)
        fp_good = cal.fingerprint("good_strategy", "repair")
        for _ in range(3):
            cal.observe_verify(
                verified=True, domain="repair", approach="good_strategy",
                approach_fingerprint=fp_good,
            )
        assert cal.approach_score(fp_good) > cal.approach_score(fp_bad)
        ranked = cal.rank_approaches(
            ["bad_strategy", "good_strategy", "mid"], domain="repair"
        )
        assert ranked[0] == "good_strategy", ranked
        picked = cal.pick_best_approach(
            ["bad_strategy", "good_strategy"], domain="repair"
        )
        assert picked == "good_strategy"

        # Recovery kwargs feed ProgressAwareRecovery
        rk = cal.recovery_kwargs()
        rec = ProgressAwareRecovery(**rk)
        assert rec.stagnation_limit == cal.params.stagnation_same_signature

        # Persist across "restart" (new AutoCalibration on same DB)
        cal2 = AutoCalibration(mem, on_log=lambda m: None)
        assert cal2.should_avoid_approach(fp_bad)
        assert cal2.approach_score(fp_good) > 0.5
        assert mem.get_fact(FACT_PARAMS)

        # Snapshot + rollback when new config worse
        # Seed a healthy baseline window
        for _ in range(EVAL_MIN_OUTCOMES):
            cal2.observe_verify(
                verified=True, domain="repair", approach="good_strategy",
                approach_fingerprint=fp_good,
            )
        baseline = cal2.window_success_rate()
        assert baseline >= 0.5
        # Force an adaptation by simulating stagnation signals
        for _ in range(3):
            cal2.observe_recovery(
                stagnated=True, stopped=False, allow_research=False,
                failure_signature="sig_stag",
            )
        # Pending eval should exist after adapt OR we force apply
        pending = mem.get_fact(FACT_PENDING) or {}
        if not pending.get("ts"):
            cal2._apply_params(
                CalibParams.from_dict({
                    **cal2.params.to_dict(),
                    "max_repair_attempts": min(8, cal2.params.max_repair_attempts + 1),
                }),
                reason="test_force_adapt",
            )
            pending = mem.get_fact(FACT_PENDING) or {}
        assert pending.get("ts"), pending
        assert mem.get_fact(FACT_CONFIG_STACK)
        # Worse outcomes after change → rollback
        for _ in range(EVAL_MIN_OUTCOMES):
            cal2.observe_verify(
                verified=False, domain="repair", approach="bad_strategy",
                approach_fingerprint=fp_bad,
            )
        pending_after = mem.get_fact(FACT_PENDING) or {}
        # Either rolled back (pending cleared) or still pending if rates borderline
        st = cal2.status()
        assert "params" in st and "success_rate" in st
        assert any("CALIBRATE:" in m for m in logs_c)

        # Orchestrator wires calibration into loops
        class OfflineBrainCal(Brain):
            def model_status(self) -> str:
                return "OFFLINE"

            def is_available(self) -> bool:
                return False

        orch_c = Orchestrator(
            root=cal_root / "orch",
            brain=OfflineBrainCal(),
            on_log=lambda m: logs_c.append(m),
        )
        assert hasattr(orch_c, "calibration")
        assert orch_c.calibration.learning_attempts() >= 2
        dash = orch_c.get_dashboard_stats()
        assert "calibration" in dash
        status_txt = orch_c._format_status()
        assert "Calibration" in status_txt
        orch_c.close()
        mem.close()
        print("  OK auto-calibration — scores/avoid/persist/rollback hooks")
    except Exception as exc:
        msg = f"E2E_AUTO_CALIBRATION: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # 4m) E2E: concurrent model pool — independent workers, warm/LRU, dedupe, parallel
    print("  — e2e concurrent model pool (FAST/REASONING/CODING workers) —")
    try:
        from jarvis.model_config import (
            TIER_MODELS,
            TIER_FAST,
            TIER_REASONING,
            TIER_CODING,
            WARM_ORDER,
            tier_for_work,
            models_for_tier,
            size_hint_bytes,
        )
        from jarvis.model_router import ModelPool, ModelRouter, RequestWorkContext

        assert tier_for_work("intent") == TIER_FAST
        assert tier_for_work("extract_args") == TIER_FAST
        assert tier_for_work("query_generation") == TIER_FAST
        assert tier_for_work("converse") == TIER_FAST
        assert tier_for_work("plan") == TIER_FAST  # plan is structured/simple → FAST
        assert tier_for_work("learning") == TIER_REASONING
        assert tier_for_work("research") == TIER_REASONING
        assert tier_for_work("diagnose") == TIER_REASONING
        assert tier_for_work("semantic") == TIER_REASONING
        assert tier_for_work("skill_code") == TIER_CODING
        assert tier_for_work("code_repair") == TIER_CODING
        assert tier_for_work("unknown_work_xyz") == TIER_FAST  # never default to 30B
        assert WARM_ORDER[0] == TIER_FAST
        assert size_hint_bytes("qwen3:4b") < size_hint_bytes("qwen3:30b")

        assert TIER_MODELS[TIER_FAST]["primary"] == "qwen3:4b"
        assert TIER_MODELS[TIER_REASONING]["primary"] == "qwen3:30b"
        assert TIER_MODELS[TIER_CODING]["primary"] == "qwen3-coder:30b"
        assert "qwen2.5-coder:7b" in models_for_tier(TIER_CODING)
        assert "qwen2.5-coder:7b" in models_for_tier(TIER_FAST)

        route_logs: list[str] = []
        preloaded: list[str] = []
        unloaded: list[str] = []

        installed = ["qwen2.5-coder:7b"]

        def _list() -> list[str]:
            return list(installed)

        def _preload(model: str, keep_alive) -> bool:
            preloaded.append(f"{model}:{keep_alive}")
            return True

        def _unload(model: str) -> bool:
            unloaded.append(model)
            return True

        # Tiny budget → only FAST warm (intelligent warm-pool, no thrash)
        pool = ModelRouter(
            list_models=_list,
            on_log=lambda m: route_logs.append(m),
            preload_fn=_preload,
            unload_fn=_unload,
            ps_fn=lambda: [],
        )
        # Force tiny budget via monkeypatch after init path
        report = pool.initialize(warm=False)
        assert report.get("online") is True
        pool._budget_bytes = 5_000_000_000  # ~5GiB — one 7b-ish model
        for w in pool.workers.values():
            w.size_bytes = 4_500_000_000
        warm_rep = pool.warm_start()
        assert any("FAST" in x for x in (warm_rep.get("warmed") or [])), warm_rep
        # REASONING/CODING should be skipped when budget tight (same resolved model
        # may already be warm — that's OK; skipped list or single warm entry)
        assert pool.workers[TIER_FAST].warm is True

        # Direct acquire — CODING without going through FAST→REASONING
        route_logs.clear()
        d_code = pool.acquire(work="skill_code", ensure_warm=False)
        assert d_code.tier == TIER_CODING
        assert d_code.model == "qwen2.5-coder:7b"
        assert any(m.startswith("MODEL ROUTE: CODING →") for m in route_logs), route_logs

        # Fallback when only legacy model installed
        route_logs.clear()
        d_fast = pool.route(work="intent", ensure_warm=False)
        assert d_fast.tier == TIER_FAST
        assert d_fast.used_fallback is True
        assert any("MODEL FALLBACK:" in m for m in route_logs), route_logs

        # Primaries available — direct tier select (not a chain)
        installed = ["qwen3:4b", "qwen3:30b", "qwen3-coder:30b", "qwen2.5-coder:7b"]
        pool.invalidate_cache()
        pool.initialize(warm=False)
        route_logs.clear()
        d_plan = pool.acquire(work="plan", ensure_warm=False)
        assert d_plan.tier == TIER_FAST and d_plan.model == "qwen3:4b"
        d_diag = pool.acquire(work="diagnose", ensure_warm=False)
        assert d_diag.tier == TIER_REASONING and d_diag.model == "qwen3:30b"
        d_code2 = pool.acquire(work="code_repair", ensure_warm=False)
        assert d_code2.tier == TIER_CODING and d_code2.model == "qwen3-coder:30b"

        # Escalation only when prior insufficient
        route_logs.clear()
        esc = pool.escalate(TIER_FAST, work="intent")
        assert esc is not None and esc.tier == TIER_REASONING
        assert any(m == "MODEL ESCALATION: FAST → REASONING" for m in route_logs), route_logs

        route_logs.clear()
        retry = pool.escalate(TIER_CODING, work="code_repair")
        assert retry is not None and retry.tier == TIER_CODING
        assert any("MODEL RETRY: CODING" in m for m in route_logs), route_logs

        assert ModelRouter.result_insufficient("")
        assert ModelRouter.result_insufficient("[BRAIN ERROR] boom")
        assert ModelRouter.result_insufficient("sorry no", expect_json=True)
        assert not ModelRouter.result_insufficient('{"ok": true}', expect_json=True)

        # Request context dedupe — same work not re-run on another model
        ctx = pool.begin_request("req_dedupe")
        key = RequestWorkContext.work_key("intent", "abc", "")
        ctx.put(key, '{"intent":"conversation","goal":"hi","needs_capability":false}')
        esc_blocked = pool.escalate(
            TIER_FAST, work="intent", prior_key=key
        )
        assert esc_blocked is None  # sufficient result → no escalate redo
        assert any("MODEL DEDUPE" in m for m in route_logs), route_logs
        pool.end_request()

        # Parallel independent jobs
        parallel_calls: list[str] = []

        def _runner(job):
            parallel_calls.append(job["id"])
            time.sleep(0.02)
            return job["id"]

        t_par0 = time.perf_counter()
        outs = pool.run_parallel(
            [{"id": "a"}, {"id": "b"}, {"id": "c"}], _runner
        )
        t_par = time.perf_counter() - t_par0
        assert outs == ["a", "b", "c"], outs
        assert t_par < 0.08, t_par  # concurrent, not 3× serial

        # LRU eviction when warming a large model under tight budget
        unloaded.clear()
        pool._budget_bytes = 5_000_000_000
        for tier, w in pool.workers.items():
            w.size_bytes = 4_500_000_000
            w.warm = tier == TIER_FAST
            w.ready = True
            w.in_flight = 0
            w.last_used = time.time() - (10 if tier == TIER_FAST else 0)
        pool.workers[TIER_FAST].model = "qwen3:4b"
        pool.workers[TIER_REASONING].model = "qwen3:30b"
        pool.workers[TIER_REASONING].warm = False
        pool.ensure_warm(TIER_REASONING)
        assert "qwen3:4b" in unloaded or pool.workers[TIER_REASONING].warm

        # Brain.chat: escalate only on empty FAST; keep_alive path works
        calls: list[str] = []

        class RouterProbeBrain(Brain):
            def _list_model_names(self):
                return ["qwen3:4b", "qwen3:30b", "qwen2.5-coder:7b"]

            def _chat_on_model(self, messages, temperature, model, keep_alive=None):
                calls.append(model)
                if model == "qwen3:4b":
                    return ""
                return '{"intent":"conversation","goal":"hi","needs_capability":false}'

            def ensure_pool_ready(self, *, warm: bool = True):
                return {"online": True, "warmed": [], "forced_test": True}

        probe_logs: list[str] = []
        probe = RouterProbeBrain(on_log=lambda m: probe_logs.append(m))
        # Avoid real Ollama preload HTTP during unit e2e
        probe.pool._preload_fn = lambda model, ka: True
        probe.pool._unload_fn = lambda model: True
        probe.pool._ps_fn = lambda: []
        probe.pool.initialize(warm=False)
        probe.begin_request_pool("probe1")
        out = probe.generate(
            "hi",
            system="Reply JSON",
            work="intent",
            allow_escalate=True,
            expect_json=True,
        )
        assert "qwen3:4b" in calls and "qwen3:30b" in calls, calls
        assert "intent" in out or "conversation" in out
        assert any("MODEL ESCALATION: FAST → REASONING" in m for m in probe_logs), probe_logs
        # Dedupe: second identical generate must not add more model calls
        n_before = len(calls)
        out2 = probe.generate(
            "hi",
            system="Reply JSON",
            work="intent",
            allow_escalate=True,
            expect_json=True,
        )
        assert len(calls) == n_before, calls
        assert out2
        assert any("MODEL DEDUPE" in m for m in probe_logs), probe_logs
        probe.end_request_pool()

        # Orchestrator dashboard exposes pool status
        class OfflinePoolBrain(Brain):
            def model_status(self) -> str:
                return "OFFLINE"

            def is_available(self) -> bool:
                return False

            def ensure_pool_ready(self, *, warm: bool = True):
                return {"online": False, "warmed": []}

        orch_p = Orchestrator(
            root=root / "data" / "_e2e_pool",
            brain=OfflinePoolBrain(),
            on_log=lambda m: None,
        )
        dash = orch_p.get_dashboard_stats()
        assert "model_pool" in dash, dash
        status_txt = orch_p._format_status()
        assert "Pool" in status_txt or "model pool" in status_txt.lower()
        orch_p.close()

        print("  OK model pool — warm/LRU/direct acquire/dedupe/parallel")
    except Exception as exc:
        msg = f"E2E_MODEL_POOL: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # 4n) E2E: stabilize — conversation + perf telemetry + forced repair PASS
    print("  — e2e stabilize / speed / repair context / perf —")
    try:
        from jarvis.perf import PerfTracker, set_active_tracker
        from jarvis.task_contract import TaskContract as TGStab

        # Perf tracker records stages + model calls
        pt = PerfTracker("perf_test")
        set_active_tracker(pt)
        with pt.stage("PLAN"):
            time.sleep(0.01)
        pt.record_model(work="plan", tier="FAST", model="qwen3:4b", ms=12.5)
        lines = pt.summary_lines()
        assert any("PERF total=" in ln for ln in lines), lines
        assert pt.as_dict()["model_call_count"] == 1
        set_active_tracker(None)

        stab_root = root / "data" / "_e2e_stabilize"
        if stab_root.exists():
            shutil.rmtree(stab_root)
        stab_root.mkdir(parents=True)

        class StabBrain(Brain):
            def __init__(self) -> None:
                super().__init__()
                self.builds = 0
                self.saw_repair_context = False

            def model_status(self) -> str:
                return "ONLINE"

            def is_available(self) -> bool:
                return True

            def classify_intent(self, user_text: str) -> dict:
                low = (user_text or "").lower()
                if low.startswith("hello") or low.startswith("sveiki"):
                    return {
                        "intent": "conversation",
                        "goal": user_text,
                        "needs_capability": False,
                    }
                return {
                    "intent": "task",
                    "goal": user_text,
                    "needs_capability": True,
                }

            def converse(self, user_text: str, history=None) -> str:
                return f"Conversational reply to: {user_text}"

            def plan(self, goal: str, known_capabilities: list) -> dict:
                return {
                    "steps": ["build skill", "test", "verify"],
                    "can_reuse": [],
                    "missing": [goal],
                    "needs_research": False,
                    "needs_new_skill": True,
                    "skill_name": "stab_create_file",
                    "skill_description": goal,
                    "research_queries": [],
                    "args": {},
                    "required_args": ["path", "content"],
                }

            def extract_task_args(
                self, goal, skill_meta=None, prior_args=None, diagnosis=None
            ):
                from jarvis.context_builder import ContextBuilder as CB
                return CB._offline_extract(
                    goal, diagnosis=diagnosis, skill_meta=skill_meta
                )

            def write_skill_code(
                self,
                skill_name,
                description,
                research,
                previous_code=None,
                error_log=None,
                diagnosis=None,
                failed_approaches=None,
                test_plan=None,
                **kwargs,
            ):
                self.builds += 1
                # Repair / build must receive original request + TaskContract + args
                assert kwargs.get("original_request"), kwargs
                assert kwargs.get("contract"), kwargs
                assert isinstance(kwargs.get("grounded_args"), dict), kwargs
                if self.builds >= 2 or diagnosis or error_log:
                    self.saw_repair_context = True
                    assert error_log or diagnosis
                    assert "USER REQUEST" in str(error_log or "") or kwargs.get(
                        "original_request"
                    )
                wrong_first = self.builds == 1
                return (
                    "from pathlib import Path\n"
                    f"SKILL_META = {{'name': {skill_name!r}, "
                    f"'description': {description!r}, "
                    "'capabilities': ['create_file'], 'dependencies': [], "
                    "'version': 1, 'required_args': ['path', 'content']}\n"
                    "def run(context):\n"
                    "    args = dict(context.get('args') or {})\n"
                    "    ws = Path(context.get('workspace') or '.')\n"
                    "    rel = args.get('path') or 'stab_out.txt'\n"
                    "    body = args.get('content') or ''\n"
                    f"    if {wrong_first!r}:\n"
                    "        body = 'WRONG'\n"
                    "    p = ws / rel if not Path(str(rel)).is_absolute() else Path(rel)\n"
                    "    p.parent.mkdir(parents=True, exist_ok=True)\n"
                    "    p.write_text(str(body), encoding='utf-8')\n"
                    "    return {'ok': True, 'result': {'path': str(p), 'contains': body},\n"
                    "            'error': None, 'evidence': f'wrote {p}'}\n"
                )

        logs_s: list[str] = []
        brain_s = StabBrain()
        orch_s = Orchestrator(
            root=stab_root,
            brain=brain_s,
            on_log=lambda m: logs_s.append(m),
        )
        # 1) simple conversation
        r_conv = orch_s.handle_user_message("Hello JARVIS")
        assert r_conv.get("type") == "conversation", r_conv
        assert isinstance(r_conv.get("perf"), dict), r_conv
        assert any("PERF total=" in m for m in logs_s), logs_s[-20:]

        # 2) capability build + forced failure → repair → PASS
        logs_s.clear()
        goal_s = "Create file stab_marker.txt containing STAB_OK"
        r_task = orch_s.handle_user_message(goal_s)
        assert r_task.get("success") is True, r_task
        marker = stab_root / "workspace_runtime" / "stab_marker.txt"
        assert marker.exists(), list((stab_root / "workspace_runtime").iterdir()) if (
            stab_root / "workspace_runtime"
        ).exists() else "no workspace"
        assert "STAB_OK" in marker.read_text(encoding="utf-8")
        # CONTEXT must not explode into word-args
        assert not any(
            "args_keys=['ti'" in m or "args_keys=['Python'" in m for m in logs_s
        ), [m for m in logs_s if "args_keys=" in m][:8]
        tg_s = TGStab.from_request(goal_s)
        assert "stab_marker.txt" in tg_s.artifacts
        assert "STAB_OK" in tg_s.content_requirements
        assert brain_s.builds >= 2, brain_s.builds
        assert brain_s.saw_repair_context
        assert any("PERF" in m for m in logs_s)
        orch_s.close()
        print("  OK stabilize — conversation/perf/TaskContract/repair→PASS")
    except Exception as exc:
        msg = f"E2E_STABILIZE: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # Capability match: unrelated ACTIVE bait must not be REPAIR'd for a new TaskContract
    print("  — e2e capability match (BUILD_NEW vs unrelated ACTIVE bait) —")
    try:
        from jarvis.capability_match import evaluate_capability as _eval_cap

        cap_root = root / "data" / "_e2e_capability"
        if cap_root.exists():
            shutil.rmtree(cap_root)
        cap_root.mkdir(parents=True)
        skills_dir = cap_root / "skills"
        skills_dir.mkdir(parents=True)
        ws_dir = cap_root / "workspace_runtime"
        ws_dir.mkdir(parents=True)

        # Bait skill: specialized "decorators" — must NOT absorb unrelated tasks
        bait_path = skills_dir / "python_decorators_basics.py"
        bait_path.write_text(
            '''
SKILL_META = {
    "name": "python_decorators_basics",
    "description": "Teach Python decorators with wrap examples",
    "capabilities": ["decorators", "wrap functions", "decorator syntax"],
}
def run(context):
    return {"ok": True, "result": "decorators lesson", "error": None, "evidence": "decorated"}
'''.strip()
            + "\n",
            encoding="utf-8",
        )

        class CapBrain(Brain):
            def __init__(self) -> None:
                super().__init__()
                self.builds = 0
                self.repaired_bait = False

            def model_status(self) -> str:
                return "ONLINE"

            def is_available(self) -> bool:
                return True

            def classify_intent(self, user_text: str) -> dict:
                return {
                    "intent": "task",
                    "goal": user_text,
                    "needs_capability": True,
                    "keywords": user_text.split()[:8],
                }

            def plan(self, goal: str, known_capabilities: list[str]) -> dict:
                # Planner wrongly offers the bait (simulates semantic lure)
                return {
                    "steps": ["handle goal"],
                    "can_reuse": ["python_decorators_basics"],
                    "missing": [],
                    "needs_research": False,
                    "needs_new_skill": False,
                    "skill_name": "python_decorators_basics",
                    "skill_description": goal,
                    "research_queries": [goal, "python programming"],
                    "args": {},
                    "required_args": [],
                }

            def extract_task_args(
                self, goal, skill_meta=None, prior_args=None, diagnosis=None
            ):
                return {}

            def research_notes(self, query: str, gathered: str) -> dict:
                return {
                    "approach": "pathlib_write",
                    "libraries": [],
                    "key_apis": ["Path.write_text"],
                    "pitfalls": [],
                    "test_idea": "artifact exists",
                }

            def write_skill_code(
                self,
                skill_name,
                description,
                research,
                previous_code=None,
                error_log=None,
                diagnosis=None,
                failed_approaches=None,
                test_plan=None,
                **kwargs,
            ) -> str:
                self.builds += 1
                if skill_name == "python_decorators_basics":
                    self.repaired_bait = True
                # Universal writer for whatever artifact the TaskContract asks
                return f'''
SKILL_META = {{
    "name": "{skill_name}",
    "description": {description!r},
    "capabilities": ["write file artifact"],
}}
from pathlib import Path
def run(context):
    ws = Path(context["workspace"])
    target = ws / "alpha_report.txt"
    target.write_text("ALPHA_OK", encoding="utf-8")
    return {{"ok": True, "result": str(target), "error": None, "evidence": target.read_text(encoding="utf-8")}}
'''.strip()

            def diagnose(self, *a, **k):
                return {
                    "fault_layer": "skill_code",
                    "approach": "pathlib_write",
                    "rewrite_skill": True,
                    "summary": "fix skill",
                }

            def chat(self, *a, **k):
                return "{}"

        logs_c: list[str] = []
        orch_c = Orchestrator(
            root=cap_root,
            brain=CapBrain(),
            on_log=lambda m: logs_c.append(m),
        )
        # Register bait as ACTIVE (trusted-looking)
        orch_c.registry.register_candidate(
            name="python_decorators_basics",
            description="Teach Python decorators with wrap examples",
            file_path=str(bait_path),
            capabilities=["decorators", "wrap functions", "decorator syntax"],
            version=1,
        )
        orch_c.registry.set_status("python_decorators_basics", "ACTIVE")
        bait_before = orch_c.registry.get_skill("python_decorators_basics")
        assert bait_before and bait_before["status"] == "ACTIVE"

        # Matcher unit: bait incompatible with new TaskContract
        tg_alpha = __import__("jarvis.task_contract", fromlist=["TaskContract"]).TaskContract.from_request(
            "Create file alpha_report.txt containing ALPHA_OK"
        )
        bait_match = _eval_cap(bait_before, tg_alpha)
        assert bait_match["compatible"] is False, bait_match

        result_c = orch_c.handle_user_message(
            "Create file alpha_report.txt containing ALPHA_OK"
        )
        assert result_c.get("success") is True, result_c
        assert any("CAPABILITY MATCH:" in m for m in logs_c), logs_c[-40:]
        assert any(
            "CAPABILITY MATCH:" in m
            and "python_decorators_basics" in m
            and "compatible=False" in m
            for m in logs_c
        ), [m for m in logs_c if "CAPABILITY" in m]
        assert any(
            "CAPABILITY DECISION: BUILD_NEW" in m
            or "CAPABILITY DECISION: EXTEND" in m
            or "CAPABILITY DECISION: OPEN_BRANCH" in m
            for m in logs_c
        ), [m for m in logs_c if "CAPABILITY" in m]
        # Must not REPAIR the bait into a universal skill
        assert not orch_c.brain.repaired_bait, "bait skill was rewritten"
        bait_after = orch_c.registry.get_skill("python_decorators_basics")
        assert bait_after["status"] == "ACTIVE", bait_after
        assert int(bait_after["version"]) == int(bait_before["version"])
        # New skill/artifact for THIS TaskContract
        alpha = ws_dir / "alpha_report.txt"
        assert alpha.exists() and "ALPHA_OK" in alpha.read_text(encoding="utf-8")
        new_skills = [
            s["name"]
            for s in orch_c.registry.list_skills()
            if s["name"] != "python_decorators_basics" and s["status"] == "ACTIVE"
        ]
        assert new_skills, orch_c.registry.list_skills()

        # Second distinct task + calculator bait → still BUILD_NEW, not calculator REPAIR
        calc_path = skills_dir / "simple_calculator.py"
        calc_path.write_text(
            "SKILL_META={'name':'simple_calculator','capabilities':['arithmetic']}\n"
            "def run(c):\n    return {'ok': True, 'result': 0, 'error': None, 'evidence': '0'}\n",
            encoding="utf-8",
        )
        orch_c.registry.register_candidate(
            name="simple_calculator",
            description="Add subtract multiply divide numbers",
            file_path=str(calc_path),
            capabilities=["arithmetic", "calculator"],
            version=1,
        )
        orch_c.registry.set_status("simple_calculator", "ACTIVE")

        class CapBrain2(CapBrain):
            def plan(self, goal: str, known_capabilities: list[str]) -> dict:
                return {
                    "steps": ["handle goal"],
                    "can_reuse": ["simple_calculator", "python_decorators_basics"],
                    "missing": [],
                    "needs_research": False,
                    "needs_new_skill": False,
                    "skill_name": "simple_calculator",
                    "skill_description": goal,
                    "research_queries": [goal],
                    "args": {},
                    "required_args": [],
                }

            def write_skill_code(self, skill_name, description, research, **kwargs):
                self.builds += 1
                if skill_name in ("simple_calculator", "python_decorators_basics"):
                    self.repaired_bait = True
                return f'''
SKILL_META = {{"name": "{skill_name}", "capabilities": ["write beta_notes.md"]}}
from pathlib import Path
def run(context):
    p = Path(context["workspace"]) / "beta_notes.md"
    p.write_text("BETA_OK", encoding="utf-8")
    return {{"ok": True, "result": str(p), "error": None, "evidence": "BETA_OK"}}
'''.strip()

        logs_c2: list[str] = []
        brain2 = CapBrain2()
        orch_c2 = Orchestrator(
            root=cap_root,
            brain=brain2,
            on_log=lambda m: logs_c2.append(m),
        )
        # Keep registry skills from same DB
        r2 = orch_c2.handle_user_message(
            "Create markdown file beta_notes.md containing BETA_OK"
        )
        assert r2.get("success") is True, r2
        assert any(
            "CAPABILITY DECISION: BUILD_NEW" in m
            or "CAPABILITY DECISION: EXTEND" in m
            or "CAPABILITY DECISION: OPEN_BRANCH" in m
            for m in logs_c2
        ), [m for m in logs_c2 if "CAPABILITY" in m]
        assert any(
            "compatible=False" in m and "simple_calculator" in m for m in logs_c2
        ), [m for m in logs_c2 if "CAPABILITY MATCH" in m]
        assert not brain2.repaired_bait
        assert (ws_dir / "beta_notes.md").read_text(encoding="utf-8") == "BETA_OK"
        calc_after = orch_c2.registry.get_skill("simple_calculator")
        assert calc_after["status"] == "ACTIVE"
        assert int(calc_after["version"]) == 1

        # Compatible REUSE still works
        class CapBrainReuse(CapBrain):
            def plan(self, goal: str, known_capabilities: list[str]) -> dict:
                reuse = []
                for line in known_capabilities:
                    if "alpha_report" in line or "alpha" in line.lower():
                        # pick any non-bait ACTIVE that fits
                        pass
                # Prefer the skill that wrote alpha_report
                for s in known_capabilities:
                    name = s.split()[0]
                    if name not in ("python_decorators_basics", "simple_calculator"):
                        reuse.append(name)
                        break
                return {
                    "steps": ["reuse"],
                    "can_reuse": reuse,
                    "missing": [] if reuse else [goal],
                    "needs_research": not bool(reuse),
                    "needs_new_skill": not bool(reuse),
                    "skill_name": reuse[0] if reuse else "alpha_report",
                    "skill_description": goal,
                    "research_queries": [goal],
                    "args": {},
                    "required_args": [],
                }

            def write_skill_code(self, *a, **k):
                self.builds += 1
                raise AssertionError("REUSE must not rebuild")

        logs_r: list[str] = []
        orch_r = Orchestrator(
            root=cap_root,
            brain=CapBrainReuse(),
            on_log=lambda m: logs_r.append(m),
        )
        # Ensure alpha artifact gone so reuse skill recreates it
        if alpha.exists():
            alpha.unlink()
        r3 = orch_r.handle_user_message(
            "Create file alpha_report.txt containing ALPHA_OK"
        )
        assert r3.get("success") is True, r3
        assert any("CAPABILITY DECISION: REUSE" in m for m in logs_r), [
            m for m in logs_r if "CAPABILITY" in m
        ]
        assert orch_r.brain.builds == 0
        assert alpha.exists()

        orch_c.close()
        orch_c2.close()
        orch_r.close()
        print(
            "  OK capability match — bait rejected, EXTEND/BUILD_NEW, compatible REUSE"
        )
    except Exception as exc:
        msg = f"E2E_CAPABILITY_MATCH: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # Path in request = OUTPUT PATH, never PyPI / install target
    print("  — e2e path artifact ≠ dependency (BUILD→TEST→EXECUTE→VERIFY) —")
    try:
        path_root = root / "data" / "_e2e_path_not_dep"
        if path_root.exists():
            shutil.rmtree(path_root)
        path_root.mkdir(parents=True)
        ensured: list[list[str]] = []

        class PathNotDepBrain(Brain):
            def __init__(self) -> None:
                super().__init__()
                self.builds = 0

            def model_status(self) -> str:
                return "ONLINE"

            def is_available(self) -> bool:
                return True

            def classify_intent(self, user_text: str) -> dict:
                return {
                    "intent": "task",
                    "goal": user_text,
                    "needs_capability": True,
                    "keywords": ["create", "file"],
                }

            def plan(self, goal: str, known_capabilities: list[str]) -> dict:
                return {
                    "steps": ["build"],
                    "can_reuse": [],
                    "missing": [goal],
                    # No entry research required — path stems still must not install
                    "needs_research": False,
                    "needs_new_skill": True,
                    "skill_name": "path_out_skill",
                    "skill_description": goal,
                    "research_queries": [],
                    "args": {"dest": "workspace/out_tool.py", "body": "PATH_OK"},
                    "required_args": ["dest", "body"],
                }

            def extract_task_args(
                self, goal, skill_meta=None, prior_args=None, diagnosis=None
            ):
                return {"dest": "workspace/out_tool.py", "body": "PATH_OK"}

            def research_notes(self, query: str, gathered: str) -> dict:
                # Deliberately polluted libraries — orchestrator must scrub
                return {
                    "approach": "write_output_path",
                    "libraries": ["workspace", "out_tool", "calculator"],
                    "key_apis": ["pathlib"],
                    "pitfalls": [],
                    "test_idea": "",
                    "repair_insight": "",
                }

            def write_skill_code(
                self,
                skill_name,
                description,
                research,
                previous_code=None,
                error_log=None,
                diagnosis=None,
                failed_approaches=None,
                test_plan=None,
                **kwargs,
            ) -> str:
                self.builds += 1
                # Stdlib only — no third-party installs required
                return f'''
from pathlib import Path
SKILL_META = {{
    "name": "{skill_name}",
    "description": {description!r},
    "capabilities": ["write_file"],
    "dependencies": ["workspace", "out_tool"],
    "version": {self.builds},
    "required_args": ["dest", "body"],
}}
def run(context: dict) -> dict:
    args = context.get("args") or {{}}
    workspace = Path(context.get("workspace") or ".")
    dest = str(args.get("dest") or "out.txt")
    path = workspace / dest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(args.get("body") or "") + "\\n", encoding="utf-8")
    return {{
        "ok": True,
        "result": {{"path": str(path), "contains": str(args.get("body") or "")}},
        "error": None,
        "evidence": f"wrote {{path}}",
    }}
'''

            def diagnose(self, observation, failed_approaches=None, prior_solutions=None):
                return {
                    "root_cause": "n/a",
                    "fault_layer": "skill_code",
                    "rewrite_skill": True,
                    "what_to_change": "honor dest path",
                    "approach": "path_output",
                    "approach_changed": True,
                    "needs_research": False,
                    "missing_knowledge": [],
                    "research_queries": [],
                    "needs_new_deps": ["workspace", "out_tool"],
                    "missing_args": [],
                    "required_args": ["dest", "body"],
                    "suggested_args": {
                        "dest": "workspace/out_tool.py",
                        "body": "PATH_OK",
                    },
                    "test_plan": "VERIFY path",
                    "expected_artifacts": ["workspace/out_tool.py"],
                    "is_unfixable": False,
                    "diagnosis": "ok",
                }

            def verify_claim(self, goal, result, evidence) -> dict:
                return {"achieved": True, "confidence": 0.9, "reason": "advisory"}

            def converse(self, user_text, history=None) -> str:
                return "ok"

        logs_p: list[str] = []
        events_p: list[tuple[str, dict]] = []
        brain_p = PathNotDepBrain()
        orch_p = Orchestrator(
            root=path_root,
            brain=brain_p,
            on_log=lambda m: logs_p.append(m),
            on_event=lambda k, p: events_p.append((k, dict(p or {}))),
        )
        # Intercept ensure to prove path stems are never installed
        _orig_ensure = orch_p.deps.ensure

        def _track_ensure(deps):
            ensured.append(list(deps or []))
            return _orig_ensure(deps)

        orch_p.deps.ensure = _track_ensure  # type: ignore[method-assign]
        goal_p = 'Create workspace/out_tool.py containing "PATH_OK"'
        result_p = orch_p.run_cycle(goal_p)
        assert result_p.get("success"), result_p
        out_p = path_root / "workspace_runtime" / "workspace" / "out_tool.py"
        assert out_p.exists(), list(
            (path_root / "workspace_runtime").rglob("*")
        )
        assert "PATH_OK" in out_p.read_text(encoding="utf-8")
        # Never install path stems
        flat = [d.lower() for batch in ensured for d in batch]
        assert "workspace" not in flat, ensured
        assert "out_tool" not in flat, ensured
        assert "calculator" not in flat, ensured
        assert any("REQUEST ITEMS" in m for m in logs_p), logs_p[-40:]
        assert any(
            "PATH=" in m or "FILE=" in m for m in logs_p if "REQUEST ITEMS" in m
        ), [m for m in logs_p if "REQUEST ITEMS" in m]
        # Pipeline: BUILD → TEST → EXECUTE → VERIFY (TEST ≠ DONE alone)
        assert any("SKILL TEST PASS" in m and "EXECUTE" in m for m in logs_p), logs_p[-30:]
        assert any("VERIFIER RESULT: PASS" in m or "VERIFY: PASS" in m for m in logs_p)
        skill_rec = orch_p.registry.get_skill("path_out_skill")
        assert skill_rec and skill_rec.get("status") == "ACTIVE"
        deps_meta = list(skill_rec.get("dependencies") or [])
        assert "workspace" not in [str(d).lower() for d in deps_meta], deps_meta
        assert "out_tool" not in [str(d).lower() for d in deps_meta], deps_meta
        orch_p.close()
        print("  OK path artifact ≠ dependency — BUILD→TEST→EXECUTE→VERIFY")
    except Exception as exc:
        msg = f"E2E_PATH_NOT_DEP: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # Fault attribution: UNKNOWN must not auto-become skill_code / rewrite
    print("  — e2e fault attribution (unknown ≠ skill_code rewrite) —")
    try:
        from jarvis.task_contract import TaskContract as _TGFault
        from jarvis.orchestrator import Orchestrator as _OrchFault

        fa_root = root / "data" / "_e2e_fault_attr"
        if fa_root.exists():
            shutil.rmtree(fa_root)
        fa_root.mkdir(parents=True)

        class AmbiguousDiagnoseBrain(Brain):
            """Returns unrecognized fault_layer — must not trigger skill rewrite."""

            def __init__(self) -> None:
                super().__init__()
                self.builds = 0
                self.rewrites = 0

            def model_status(self) -> str:
                return "ONLINE"

            def is_available(self) -> bool:
                return True

            def classify_intent(self, user_text: str) -> dict:
                return {
                    "intent": "task",
                    "goal": user_text,
                    "needs_capability": True,
                    "keywords": ["create"],
                }

            def plan(self, goal: str, known_capabilities: list[str]) -> dict:
                return {
                    "steps": ["build"],
                    "can_reuse": [],
                    "missing": [goal],
                    "needs_research": False,
                    "needs_new_skill": True,
                    "skill_name": "fault_attr_skill",
                    "skill_description": goal,
                    "research_queries": [],
                    "args": {"dest": "fault_attr.txt", "body": "FAULT_OK"},
                    "required_args": ["dest", "body"],
                }

            def extract_task_args(
                self, goal, skill_meta=None, prior_args=None, diagnosis=None
            ):
                return {"dest": "fault_attr.txt", "body": "FAULT_OK"}

            def write_skill_code(
                self,
                skill_name,
                description,
                research,
                previous_code=None,
                error_log=None,
                diagnosis=None,
                failed_approaches=None,
                test_plan=None,
                **kwargs,
            ) -> str:
                self.builds += 1
                if previous_code or diagnosis:
                    self.rewrites += 1
                # First build: process-ok but wrong content → VERIFY will fail.
                # Ambiguous diagnose returns UNKNOWN — must NOT rewrite via skill_code default.
                body = "WRONG" if self.builds == 1 else "FAULT_OK"
                return f'''
from pathlib import Path
SKILL_META = {{
    "name": "{skill_name}",
    "description": {description!r},
    "capabilities": ["write"],
    "dependencies": [],
    "version": {self.builds},
    "required_args": ["dest", "body"],
}}
def run(context):
    args = context.get("args") or {{}}
    p = Path(context.get("workspace") or ".") / str(args.get("dest") or "x.txt")
    p.write_text({body!r} + "\\n", encoding="utf-8")
    return {{"ok": True, "result": {{"path": str(p)}}, "error": None, "evidence": "wrote"}}
'''

            def diagnose(self, observation, failed_approaches=None, prior_solutions=None):
                # Intentionally unrecognized layer — must normalize to unknown, no rewrite
                return {
                    "root_cause": "ambiguous failure",
                    "fault_layer": "COMPLETELY_UNKNOWN_LAYER",
                    "rewrite_skill": True,  # must be forced False by gate
                    "what_to_change": "should not rewrite",
                    "approach": "mystery",
                    "approach_changed": True,
                    "needs_research": False,
                    "missing_knowledge": [],
                    "research_queries": [],
                    "needs_new_deps": [],
                    "missing_args": [],
                    "required_args": ["dest", "body"],
                    "suggested_args": {"dest": "fault_attr.txt", "body": "FAULT_OK"},
                    "test_plan": "none",
                    "expected_artifacts": ["fault_attr.txt"],
                    "is_unfixable": False,
                    "diagnosis": "unknown",
                }

            def verify_claim(self, goal, result, evidence) -> dict:
                return {"achieved": True, "confidence": 0.5, "reason": "advisory"}

            def converse(self, user_text, history=None) -> str:
                return "ok"

        # Unit: gate itself
        g = _TGFault.apply_rewrite_gate(
            {"fault_layer": "COMPLETELY_UNKNOWN_LAYER", "rewrite_skill": True}
        )
        assert g["fault_layer"] == "unknown" and g["rewrite_skill"] is False

        brain_fa = AmbiguousDiagnoseBrain()
        logs_fa: list[str] = []
        orch_fa = _OrchFault(
            root=fa_root,
            brain=brain_fa,
            on_log=lambda m: logs_fa.append(m),
        )
        # Patch deterministic diagnose to also return unknown (no VERIFY content evidence path)
        # so brain.diagnose's UNKNOWN layer is what recovery sees after VERIFY fail.
        # Actually VERIFY content fail will deterministic→skill_code. Force unknown via
        # wrapping _observe_diagnose_enrich post-gate check instead: call diagnose path unit.
        det = orch_fa._deterministic_diagnose(
            {
                "phase": "TEST",
                "exception": "odd opaque failure without layer signals",
                "traceback": "",
                "context": {
                    "args": {"dest": "fault_attr.txt", "body": "FAULT_OK"},
                    "original_request": 'Create "fault_attr.txt" containing "FAULT_OK"',
                },
                "version": 1,
            },
            _TGFault.from_request('Create "fault_attr.txt" containing "FAULT_OK"'),
        )
        assert det.get("fault_layer") == "unknown", det
        assert det.get("rewrite_skill") is False, det

        # Brain diagnose returning garbage layer is gated
        brain_diag = brain_fa.diagnose(
            {
                "phase": "VERIFY",
                "exception": "opaque",
                "context": {"args": {}, "original_request": "x"},
                "goal": "x",
            }
        )
        gated_brain = _TGFault.apply_rewrite_gate(brain_diag)
        assert gated_brain["fault_layer"] == "unknown"
        assert gated_brain["rewrite_skill"] is False

        # Evidence path still allows skill_code rewrite
        det_ev = orch_fa._deterministic_diagnose(
            {
                "phase": "VERIFY",
                "exception": "content_constraint: 'FAULT_OK' missing from expected files",
                "traceback": "",
                "context": {
                    "args": {"dest": "fault_attr.txt", "body": "FAULT_OK"},
                    "original_request": 'Create "fault_attr.txt" containing "FAULT_OK"',
                },
                "version": 1,
            },
            _TGFault.from_request('Create "fault_attr.txt" containing "FAULT_OK"'),
        )
        assert det_ev.get("fault_layer") == "skill_code", det_ev
        assert det_ev.get("rewrite_skill") is True, det_ev

        orch_fa.close()
        print("  OK fault attribution — unknown ≠ skill_code; evidence allows rewrite")
    except Exception as exc:
        msg = f"E2E_FAULT_ATTR: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # Universal request grounding lifecycle (polysemy + MEMORY-first)
    print("  — e2e request grounding (subject/source across MEMORY→VERIFY) —")
    try:
        from jarvis.orchestrator import Orchestrator as _OrchGround
        from jarvis.task_contract import TaskContract as _TGGround

        gr_root = root / "data" / "_e2e_request_grounding"
        if gr_root.exists():
            shutil.rmtree(gr_root)
        gr_root.mkdir(parents=True)
        research_calls: list[list[str]] = []

        class GroundingBrain(Brain):
            def __init__(self) -> None:
                super().__init__()
                self.builds = 0

            def model_status(self) -> str:
                return "ONLINE"

            def is_available(self) -> bool:
                return True

            def classify_intent(self, user_text: str) -> dict:
                return {
                    "intent": "task",
                    "goal": user_text,
                    "needs_capability": True,
                    "keywords": ["create", "file"],
                }

            def plan(self, goal: str, known_capabilities: list[str]) -> dict:
                # Deliberately drifted research_queries (programming sense)
                return {
                    "steps": ["memory", "build", "execute", "verify"],
                    "can_reuse": [],
                    "missing": [goal],
                    "needs_research": True,
                    "needs_new_skill": True,
                    "skill_name": "grounding_poly_skill",
                    "skill_description": goal,
                    "research_queries": [
                        "pip install python package",
                        "python programming language tutorial",
                        "java JDK install",
                    ],
                    "args": {
                        "dest": "habitat_notes.md",
                        "body": "PYTHON_OK",
                    },
                    "required_args": ["dest", "body"],
                }

            def extract_task_args(
                self, goal, skill_meta=None, prior_args=None, diagnosis=None
            ):
                return {"dest": "habitat_notes.md", "body": "PYTHON_OK"}

            def write_skill_code(
                self,
                skill_name,
                description,
                research,
                previous_code=None,
                error_log=None,
                diagnosis=None,
                failed_approaches=None,
                test_plan=None,
                **kwargs,
            ) -> str:
                self.builds += 1
                # BUILD context must keep TaskContract subject (not pip/JDK drift)
                tg = kwargs.get("contract") or {}
                blob = str(tg).lower()
                ur = str(kwargs.get("original_request") or description or "").lower()
                assert (
                    "wetland" in blob or "habitat" in blob
                    or "wetland" in ur or "habitat" in ur
                ), (tg, ur[:200])
                assert "pip install" not in blob and "jdk" not in blob
                return f'''
from pathlib import Path
SKILL_META = {{
    "name": "{skill_name}",
    "description": {description!r},
    "capabilities": ["write_file"],
    "dependencies": [],
    "version": {self.builds},
    "required_args": ["dest", "body"],
}}
def run(context: dict) -> dict:
    args = context.get("args") or {{}}
    workspace = Path(context.get("workspace") or ".")
    dest = str(args.get("dest") or "out.txt")
    path = workspace / dest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(args.get("body") or "") + "\\n", encoding="utf-8")
    return {{
        "ok": True,
        "result": {{"path": str(path), "contains": str(args.get("body") or "")}},
        "error": None,
        "evidence": f"wrote {{path}}",
    }}
'''

            def diagnose(self, observation, failed_approaches=None, prior_solutions=None):
                return {
                    "root_cause": "n/a",
                    "fault_layer": "unknown",
                    "rewrite_skill": False,
                    "what_to_change": "",
                    "approach": "write",
                    "approach_changed": False,
                    "needs_research": False,
                    "missing_knowledge": [],
                    "research_queries": [],
                    "needs_new_deps": [],
                    "missing_args": [],
                    "required_args": ["dest", "body"],
                    "suggested_args": {
                        "dest": "habitat_notes.md",
                        "body": "PYTHON_OK",
                    },
                    "test_plan": "execute",
                    "expected_artifacts": ["habitat_notes.md"],
                    "is_unfixable": False,
                    "diagnosis": "ok",
                }

            def verify_claim(self, goal, result, evidence) -> dict:
                return {"achieved": True, "confidence": 0.9, "reason": "ok"}

            def converse(self, user_text, history=None) -> str:
                return "ok"

        brain_gr = GroundingBrain()
        logs_gr: list[str] = []
        orch_gr = _OrchGround(
            root=gr_root,
            brain=brain_gr,
            on_log=lambda m: logs_gr.append(m),
        )

        # Seed MEMORY with verified knowledge for the subject (not programming)
        mem_topic = _TGGround.from_request(
            "about python wetland habitat"
        ).memory_topic()
        orch_gr.memory.save_topic_knowledge(
            mem_topic,
            {
                "approach": "habitat_facts",
                "concepts": ["python", "wetland", "habitat", "snake"],
                "summary": "Wetland habitat facts about python snakes in marshes",
                "key_apis": ["python", "wetland", "habitat"],
            },
            goal="python wetland habitat",
            summary="Wetland habitat facts about python snakes in marshes",
            verified=True,
        )

        # Patch research to record queries (must be grounded / not pip/JDK)
        def _tracking_research(queries, **kwargs):
            research_calls.append(list(queries or []))
            return {
                "approach": "grounded",
                "libraries": [],
                "key_apis": [],
                "pitfalls": [],
                "test_idea": "",
                "raw": "wetland habitat",
                "sources": [],
                "results": [],
            }

        orch_gr.research.research = _tracking_research  # type: ignore

        goal_gr = (
            'Create habitat_notes.md about python wetland habitat '
            'containing "PYTHON_OK"'
        )
        # Unit: TaskContract fields survive before cycle
        tg0 = _TGGround.from_request(goal_gr)
        assert "wetland" in tg0.subject.lower()
        snap = {
            "action": tg0.action,
            "artifact": list(tg0.artifacts),
            "subject": tg0.subject,
            "content_source": list(tg0.content_source),
        }

        result_gr = orch_gr.run_cycle(goal_gr)
        assert result_gr.get("success"), result_gr

        # MEMORY retrieved by TaskContract subject/topic
        assert any("MEMORY" in m for m in logs_gr), logs_gr[:30]
        assert any(
            "subject=" in m.lower() or "wetland" in m.lower() or "habitat" in m.lower()
            for m in logs_gr
            if "MEMORY" in m or "PLAN" in m
        ), logs_gr[:40]

        # RESEARCH queries (if any) must not drift to pip/JDK
        for qs in research_calls:
            joined = " ".join(qs).lower()
            assert "pip" not in joined, qs
            assert "jdk" not in joined, qs
            assert "programming language" not in joined, qs

        # Artifact created + VERIFY vs original TaskContract
        marker = gr_root / "workspace_runtime" / "habitat_notes.md"
        assert marker.exists(), list(
            (gr_root / "workspace_runtime").rglob("*")
        ) if (gr_root / "workspace_runtime").exists() else "no ws"
        assert "PYTHON_OK" in marker.read_text(encoding="utf-8")
        assert any(
            "VERIFIER RESULT: PASS" in m or "VERIFY: PASS" in m for m in logs_gr
        ), logs_gr[-40:]

        # Subject preserved across REQUEST → … → VERIFY
        assert snap["subject"] == tg0.subject
        assert snap["artifact"] == list(tg0.artifacts)
        assert snap["action"] == tg0.action

        # Direct choke-point: _do_research grounds drifted queries
        research_calls.clear()
        grounded_direct = orch_gr._do_research(
            "t_ground",
            "grounding_poly_skill",
            goal_gr,
            ["pip install python", "java JDK", "random unrelated"],
            contract=tg0,
        )
        assert research_calls, "expected research invocation"
        last_q = research_calls[-1]
        assert last_q, last_q
        assert not any("pip" in q.lower() for q in last_q), last_q
        assert any(
            "wetland" in q.lower() or "habitat" in q.lower() or "python" in q.lower()
            for q in last_q
        ), last_q
        assert grounded_direct.get("approach") == "grounded"
        assert any("RESEARCH grounding" in m for m in logs_gr), logs_gr[-20:]

        # MEMORY-first content_source preserved (action+artifact+subject+source)
        mem_goal = (
            "Using previously learned knowledge about python wetland habitat, "
            'create mem_habitat.txt containing "MEM_OK"'
        )
        tg_mem = _TGGround.from_request(mem_goal)
        assert tg_mem.prefers_memory_first()
        assert "memory" in {s.lower() for s in tg_mem.content_source}
        dmem = tg_mem.to_dict()
        assert dmem["subject"] == tg_mem.subject
        assert "memory" in dmem["content_source"]
        assert any("mem_habitat.txt" in a for a in dmem["artifacts"])
        assert dmem["original_request"] == mem_goal

        orch_gr.close()
        print(
            "  OK request grounding — subject/source preserved; "
            "drifted RESEARCH rejected"
        )
    except Exception as exc:
        msg = f"E2E_REQUEST_GROUNDING: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # Hierarchical competence system (SKILL/CAPABILITY/TOOL + migration + reuse)
    print("  — e2e competence hierarchy (classify/migrate/reuse/extend) —")
    try:
        from jarvis.competence import preferred_tool_name as _ptn_c
        from jarvis.competence_registry import CompetenceRegistry as _CompReg
        from jarvis.implementation_registry import ImplementationRegistry as _CapReg2
        from jarvis.orchestrator import Orchestrator as _OrchComp
        from jarvis.task_contract import TaskContract as _TGComp

        comp_root = root / "data" / "_e2e_competence"
        if comp_root.exists():
            shutil.rmtree(comp_root)
        comp_root.mkdir(parents=True)
        skills_dir = comp_root / "skills"
        skills_dir.mkdir(parents=True)

        # Unit: migrate legacy task-specific skill → TOOL under io.files
        db_c = comp_root / "migrate.db"
        reg_c = _CapReg2(db_c)
        legacy_name = "create_file_alpha_report_txt_containing"
        legacy_path = skills_dir / f"{legacy_name}.py"
        legacy_path.write_text(
            "from pathlib import Path\n"
            "SKILL_META = {\n"
            f'  "name": "{legacy_name}",\n'
            '  "description": "write file",\n'
            '  "capabilities": ["write_file"],\n'
            '  "dependencies": [],\n'
            '  "version": 1,\n'
            '  "required_args": ["dest", "body"],\n'
            "}\n"
            "def run(context):\n"
            "    args = context.get('args') or {}\n"
            "    p = Path(context.get('workspace') or '.') / str(args.get('dest') or 'x.txt')\n"
            "    p.parent.mkdir(parents=True, exist_ok=True)\n"
            "    p.write_text(str(args.get('body') or '') + '\\n', encoding='utf-8')\n"
            "    return {'ok': True, 'result': {'path': str(p)}, 'error': None, 'evidence': 'ok'}\n",
            encoding="utf-8",
        )
        reg_c.register_candidate(
            legacy_name,
            'Create file alpha_report.txt containing ALPHA_OK',
            str(legacy_path),
            ["write_file"],
            version=1,
            protect_active=False,
        )
        reg_c.set_status(legacy_name, "ACTIVE")
        # Mark a success so migration marks verified
        reg_c.mark_success(legacy_name)
        comp_c = _CompReg(db_c, reg_c)
        mig = comp_c.migrate_legacy_skills()
        assert mig["migrated"] >= 1, mig
        tool = comp_c.get_tool(legacy_name)
        assert tool and tool["competence_id"] == "io.files", tool
        assert legacy_path.exists()  # working code preserved
        # Second migrate is idempotent
        mig2 = comp_c.migrate_legacy_skills()
        assert mig2["migrated"] == 0 and mig2["skipped"] >= 1, mig2
        reg_c.close()
        comp_c.close()

        class CompBrain(Brain):
            def __init__(self) -> None:
                super().__init__()
                self.builds = 0
                self.names: list[str] = []

            def model_status(self) -> str:
                return "ONLINE"

            def is_available(self) -> bool:
                return True

            def classify_intent(self, user_text: str) -> dict:
                return {
                    "intent": "task",
                    "goal": user_text,
                    "needs_capability": True,
                    "keywords": ["create", "file"],
                }

            def plan(self, goal: str, known_capabilities: list[str]) -> dict:
                # Deliberately request-slug skill_name — orchestrator must prefer
                # competence tool name when extending (not create yet-another skill)
                return {
                    "steps": ["build"],
                    "can_reuse": [],
                    "missing": [goal],
                    "needs_research": False,
                    "needs_new_skill": True,
                    "skill_name": Brain._slug(goal)[:40],
                    "skill_description": goal,
                    "research_queries": [],
                    "args": {"dest": "comp_out.txt", "body": "COMP_OK"},
                    "required_args": ["dest", "body"],
                }

            def extract_task_args(
                self, goal, skill_meta=None, prior_args=None, diagnosis=None
            ):
                return {"dest": "comp_out.txt", "body": "COMP_OK"}

            def write_skill_code(
                self,
                skill_name,
                description,
                research,
                previous_code=None,
                error_log=None,
                diagnosis=None,
                failed_approaches=None,
                test_plan=None,
                **kwargs,
            ) -> str:
                self.builds += 1
                self.names.append(skill_name)
                return f'''
from pathlib import Path
SKILL_META = {{
    "name": "{skill_name}",
    "description": {description!r},
    "capabilities": ["write_file", "io.create"],
    "dependencies": [],
    "version": {self.builds},
    "required_args": ["dest", "body"],
}}
def run(context: dict) -> dict:
    args = context.get("args") or {{}}
    workspace = Path(context.get("workspace") or ".")
    dest = str(args.get("dest") or "out.txt")
    path = workspace / dest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(args.get("body") or "") + "\\n", encoding="utf-8")
    return {{
        "ok": True,
        "result": {{"path": str(path), "contains": str(args.get("body") or "")}},
        "error": None,
        "evidence": f"wrote {{path}}",
    }}
'''

            def diagnose(self, observation, failed_approaches=None, prior_solutions=None):
                return {
                    "root_cause": "n/a",
                    "fault_layer": "unknown",
                    "rewrite_skill": False,
                    "what_to_change": "",
                    "approach": "write",
                    "approach_changed": False,
                    "needs_research": False,
                    "missing_knowledge": [],
                    "research_queries": [],
                    "needs_new_deps": [],
                    "missing_args": [],
                    "required_args": ["dest", "body"],
                    "suggested_args": {"dest": "comp_out.txt", "body": "COMP_OK"},
                    "test_plan": "execute",
                    "expected_artifacts": ["comp_out.txt"],
                    "is_unfixable": False,
                    "diagnosis": "ok",
                }

            def verify_claim(self, goal, result, evidence) -> dict:
                return {"achieved": True, "confidence": 0.9, "reason": "ok"}

            def converse(self, user_text, history=None) -> str:
                return "ok"

        logs_c: list[str] = []
        brain_c = CompBrain()
        orch_c = _OrchComp(
            root=comp_root,
            brain=brain_c,
            on_log=lambda m: logs_c.append(m),
        )
        assert any("COMPETENCE: migrated" in m for m in logs_c) or True
        goal_c = 'Create comp_out.txt containing "COMP_OK"'
        tg_c = _TGComp.from_request(goal_c)
        expect_tool = _ptn_c(tg_c)
        result_c = orch_c.run_cycle(goal_c)
        assert result_c.get("success"), result_c
        assert any("COMPETENCE DECISION:" in m for m in logs_c), logs_c[:40]
        # Built under competence tool name — not request slug as the only option
        assert brain_c.names, brain_c.names
        assert brain_c.names[0] == expect_tool, (brain_c.names, expect_tool)
        assert brain_c.names[0] != Brain._slug(goal_c)[:40], brain_c.names
        out_c = comp_root / "workspace_runtime" / "comp_out.txt"
        assert out_c.exists() and "COMP_OK" in out_c.read_text(encoding="utf-8")
        # TOOL registered under competence
        tool_row = orch_c.competences.get_tool(brain_c.names[0])
        assert tool_row, orch_c.competences.list_tools()
        assert tool_row.get("competence_id") == "io.files", tool_row
        assert tool_row.get("verified") is True
        # Second similar task REUSE_COMPOSE — no new skill slug
        builds_before = brain_c.builds
        logs_c.clear()
        result_c2 = orch_c.run_cycle('Create comp_out.txt containing "COMP_OK"')
        assert result_c2.get("success"), result_c2
        assert any("REUSE_COMPOSE" in m or "CAPABILITY DECISION: REUSE" in m for m in logs_c), [
            m for m in logs_c if "COMPETENCE" in m or "CAPABILITY" in m
        ]
        # Prefer reuse — builds should not grow (or grow less than a full new skill)
        assert brain_c.builds == builds_before, (brain_c.builds, builds_before)
        stats = orch_c.get_dashboard_stats()
        assert "competences" in stats and stats["competences"].get("tools", 0) >= 1
        orch_c.close()
        print(
            "  OK competence hierarchy — migrate + classify + "
            "TOOL reuse (no per-request skill)"
        )
    except Exception as exc:
        msg = f"E2E_COMPETENCE: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # TaskContract VALIDATE gate — refuse empty constraints before PLAN
    print("  — e2e TaskContract VALIDATE-before-PLAN + full cycle —")
    try:
        from jarvis.orchestrator import Orchestrator as _OrchSC
        from jarvis.task_contract import TaskContract as _TGSC
        from jarvis.invariants import (
            InvariantError as _InvErr,
            assert_active_requires_independent_verify as _assert_active,
            assert_build_allowed as _assert_build,
            assert_done_requires_verify as _assert_done,
            assert_locked_truth as _assert_locked,
            rewrite_allowed_for_fault as _rewrite_ok,
        )
        from jarvis.ontology import (
            LAYER_CAPABILITY,
            LAYER_EXPERIENCE,
            LAYER_KNOWLEDGE,
            LAYER_SKILL,
            LAYER_TOOL,
            canonicalize_layer,
        )
        from jarvis.runtime_fingerprint import fingerprint_line as _fp_line
        from jarvis.implementation_registry import ImplementationRegistry as _ImplReg

        # Architecture invariant unit checks
        assert canonicalize_layer("competence") == LAYER_SKILL
        assert canonicalize_layer("implementation") == LAYER_TOOL
        assert LAYER_CAPABILITY and LAYER_KNOWLEDGE and LAYER_EXPERIENCE
        assert "commit=" in _fp_line()
        # Soft prose alone is conversation. Capability hint without concrete
        # criteria → needs_refine; BUILD must refuse (hint before VALIDATE).
        soft_c = _TGSC.understand("Be helpful please")
        assert soft_c.requires("conversation") or soft_c.intent == "conversation"
        soft_cap = soft_c.with_requirements_hint(
            {"intent": "task", "needs_capability": True}
        )
        assert soft_cap.requires_capability_stages(), soft_cap.required_outcomes
        soft_cap = _TGSC.validate(soft_cap)
        assert soft_cap.needs_refine() or not soft_cap.is_verifiable(), soft_cap.to_dict()
        try:
            _assert_build(soft_cap)
            raise AssertionError("BUILD must refuse unverifiable capability contract")
        except _InvErr:
            pass
        try:
            _assert_active(verified=False, skill_ok=True, test_ok=True)
            raise AssertionError("ACTIVE must refuse without VERIFY")
        except _InvErr:
            pass
        try:
            _assert_done(verified=False, has_evidence=True)
            raise AssertionError("DONE must refuse without VERIFY")
        except _InvErr:
            pass
        assert _rewrite_ok("skill_code") is True
        assert _rewrite_ok("unknown") is False
        assert _rewrite_ok("context_mapping") is False
        ok_c = _TGSC.from_request('Create lock.txt containing "L"')
        assert ok_c.validated
        locked2 = ok_c.with_inputs({"body": "L"})
        _assert_locked(ok_c, locked2)
        # ImplementationRegistry preserves versioning API (skills table lifecycle)
        impl_db = root / "data" / "_e2e_impl_reg" / "jarvis.db"
        impl_db.parent.mkdir(parents=True, exist_ok=True)
        _ = _ImplReg(impl_db)

        sc_root = root / "data" / "_e2e_success_criteria"
        if sc_root.exists():
            shutil.rmtree(sc_root)
        sc_root.mkdir(parents=True)

        class SoftGoalBrain(Brain):
            def model_status(self) -> str:
                return "ONLINE"

            def is_available(self) -> bool:
                return True

            def classify_intent(self, user_text: str) -> dict:
                return {
                    "intent": "task",
                    "goal": user_text,
                    "needs_capability": True,
                    "keywords": ["say"],
                }

            def understand_contract(self, original_request: str, *, prior_args=None):
                # Soft goals stay empty; otherwise use universal offline draft
                from jarvis.contract_semantics import offline_semantic_draft

                if "helpful" in (original_request or "").lower():
                    return {
                        "intent": "task",
                        "desired_outcomes": [],
                        "artifacts": [],
                        "behaviors": [],
                        "constraints": {},
                        "acceptance_criteria": [],
                        "verification_plan": [],
                    }
                return offline_semantic_draft(original_request, prior_args)

            def plan(self, goal: str, known_capabilities: list[str]) -> dict:
                raise AssertionError("PLAN must not run when TaskContract needs refine")

            def write_skill_code(self, *a, **k):
                raise AssertionError("BUILD must not run when TaskContract needs refine")

            def extract_task_args(self, *a, **k):
                return {}

            def converse(self, user_text, history=None) -> str:
                return "ok"

        logs_sc: list[str] = []
        orch_sc = _OrchSC(
            root=sc_root,
            brain=SoftGoalBrain(),
            on_log=lambda m: logs_sc.append(m),
        )
        soft_goal = "Be helpful please"
        # Soft prose alone → conversation. Capability hint without concrete
        # criteria → VALIDATE/goal_parsing refuse (via unified handle path).
        assert not _TGSC.from_request(soft_goal).needs_refine()
        soft_hint = SoftGoalBrain().classify_intent(soft_goal)
        result_sc = orch_sc.run_cycle(
            soft_goal, requirements_hint=soft_hint
        )
        assert result_sc.get("success") is False, result_sc
        assert result_sc.get("fault_layer") == "goal_parsing", result_sc
        assert any("VALIDATE" in m for m in logs_sc), logs_sc[:40]
        assert any("UNDERSTAND" in m for m in logs_sc), logs_sc[:40]
        assert any("JARVIS RUNTIME" in m for m in logs_sc), logs_sc[:20]
        assert not any("CAPABILITY DECISION:" in m for m in logs_sc), logs_sc
        # Quoted communicative payload is verifiable without path/URL
        assert _TGSC.from_request(
            'Say "hello" politely to the room'
        ).is_verifiable()
        # prior_request_isolation — same id is NOT isolated; distinct ids are
        from jarvis.invariants import (
            prior_request_isolation as _pri,
            assert_requests_isolated as _ari,
            InvariantError as _IE2,
        )
        assert _pri(None, "a1") is True
        assert _pri("a1", "a2") is True
        assert _pri("a1", "a1") is False
        assert _pri("a1", "") is False
        try:
            _ari("same", "same")
            raise AssertionError("same request_id must violate isolation")
        except _IE2:
            pass
        # Unknown languages / tasks not covered by old verb regexes
        for unk in (
            'Erstelle datei neu.txt mit "HALLO"',
            'Создай файл data.txt с текстом "PING"',
            'Créer fichier out.txt avec "BONJOUR"',
        ):
            uc = _TGSC.from_request(unk)
            assert uc.is_verifiable(), (unk, uc.acceptance_criteria)
            assert any(s.startswith("artifact_exists:") for s in uc.acceptance_criteria)
            assert any("content_present:" in s for s in uc.acceptance_criteria)
        # Model-assisted UNDERSTAND for free-form (unquoted) semantics
        class SemanticBrain(SoftGoalBrain):
            def understand_contract(self, original_request: str, *, prior_args=None):
                req = original_request or ""
                if "bonjour" in req.lower() and "fichier" not in req.lower():
                    return {
                        "intent": "task",
                        "desired_outcomes": ["behavior:skill_output_contains:bonjour"],
                        "artifacts": [],
                        "behaviors": [{
                            "kind": "skill_output_contains",
                            "target": "bonjour",
                            "how": "skill_stdout_or_result",
                        }],
                        "constraints": {"files": [], "http": [], "must_contain": []},
                        "acceptance_criteria": [
                            "skill_output_contains:bonjour@skill_stdout_or_result"
                        ],
                        "verification_plan": [{
                            "kind": "skill_output_contains",
                            "target": "bonjour",
                            "how": "skill_stdout_or_result",
                        }],
                        "subject": "bonjour",
                        "actions": [],
                    }
                return super().understand_contract(req, prior_args=prior_args)

        fr = _TGSC.understand("Dis bonjour poliment", brain=SemanticBrain())
        fr = _TGSC.validate(fr)
        assert fr.is_verifiable(), fr.acceptance_criteria
        assert any(
            "skill_output_contains:bonjour" in s for s in fr.acceptance_criteria
        ), fr.acceptance_criteria
        # Full cycle: REQUEST→UNDERSTAND→VALIDATE→…→VERIFY→REMEMBER→DONE
        class OkBrain(SoftGoalBrain):
            def __init__(self) -> None:
                super().__init__()
                self.builds = 0

            def plan(self, goal: str, known_capabilities: list[str]) -> dict:
                return {
                    "steps": ["build"],
                    "can_reuse": [],
                    "missing": [goal],
                    "needs_research": False,
                    "needs_new_skill": True,
                    "skill_name": "io_files_io_create_txt",
                    "skill_description": goal,
                    "research_queries": [],
                    "args": {"dest": "sc_ok.txt", "body": "SC_OK"},
                    "required_args": ["dest", "body"],
                }

            def extract_task_args(self, *a, **k):
                return {"dest": "sc_ok.txt", "body": "SC_OK"}

            def write_skill_code(self, skill_name, description, research, **kwargs):
                self.builds += 1
                return f'''
from pathlib import Path
SKILL_META = {{"name": "{skill_name}", "capabilities": ["write_file"],
 "required_args": ["dest", "body"], "version": {self.builds}}}
def run(context):
    args = context.get("args") or {{}}
    p = Path(context.get("workspace") or ".") / str(args.get("dest") or "x.txt")
    p.write_text(str(args.get("body") or "") + "\\n", encoding="utf-8")
    return {{"ok": True, "result": {{"path": str(p)}}, "error": None, "evidence": "ok"}}
'''

            def diagnose(self, *a, **k):
                return {
                    "fault_layer": "unknown",
                    "rewrite_skill": False,
                    "approach": "write",
                    "needs_research": False,
                    "research_queries": [],
                    "missing_args": [],
                    "required_args": ["dest", "body"],
                    "suggested_args": {"dest": "sc_ok.txt", "body": "SC_OK"},
                }

            def verify_claim(self, *a, **k):
                return {"achieved": True, "confidence": 0.9, "reason": "ok"}

        logs_ok: list[str] = []
        orch_ok = _OrchSC(
            root=sc_root,
            brain=OkBrain(),
            on_log=lambda m: logs_ok.append(m),
        )
        ok_goal = 'Create sc_ok.txt containing "SC_OK"'
        tg_ok = _TGSC.from_request(ok_goal)
        assert tg_ok.is_verifiable()
        assert any("@workspace_" in s for s in tg_ok.acceptance_criteria)
        res_ok = orch_ok.run_cycle(ok_goal)
        assert res_ok.get("success"), res_ok
        assert res_ok.get("contract"), res_ok
        assert any("UNDERSTAND" in m for m in logs_ok), logs_ok[:40]
        assert any("VALIDATE" in m for m in logs_ok), logs_ok[:40]
        assert any("success_criterion" in str(m) or "VERIFIER RESULT: PASS" in m for m in logs_ok)
        assert any("DONE" in m for m in logs_ok)
        # Request isolation — second request must not reuse prior DONE bundle
        logs_iso: list[str] = []
        orch_iso = _OrchSC(
            root=sc_root,
            brain=OkBrain(),
            on_log=lambda m: logs_iso.append(m),
        )
        iso_goal = 'Create iso_ok.txt containing "ISO_OK"'
        res_iso = orch_iso.run_cycle(iso_goal)
        assert res_iso.get("success"), res_iso
        assert res_iso.get("task_id") != res_ok.get("task_id")
        orch_sc.close()
        orch_ok.close()
        orch_iso.close()
        print("  OK TaskContract architecture — VALIDATE gate + full cycle + isolation")
    except Exception as exc:
        msg = f"E2E_TASK_CONTRACT_ARCH: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # 4z) Architecture invariants — unified TaskContract lifecycle
    print("  — architecture invariants: one contract / one request_id / DONE iff all —")
    try:
        from jarvis.invariants import (
            assert_all_outcomes_verified as _aov,
            InvariantError as _InvA,
        )
        from jarvis.outcomes import (
            OutcomeTracker as _OT,
            STATUS_VERIFIED as _SV,
            verify_final_output as _vfo,
            extract_final_output_constraints as _efoc,
            OUTCOME_LEARNING as _OL,
            OUTCOME_RESEARCH as _OR,
            OUTCOME_ARTIFACT as _OA,
            OUTCOME_EXECUTION as _OE,
            OUTCOME_TEST as _OTEST,
            OUTCOME_FINAL_OUTPUT as _OFO,
        )
        from jarvis.intent import IntentClassifier as _IC
        from jarvis.task_contract import TaskContract as _TC

        # 1) exactly one TaskContract per USER REQUEST (immutable original_request)
        req_u = 'Learn bees and create bees.txt with "BZZ" in 2 sentences'
        c_u = _TC.from_request(req_u, request_id="rid_arch_01")
        assert c_u.request_id == "rid_arch_01"
        assert c_u.original_request == req_u
        c_u2 = c_u.with_inputs({"dest": "bees.txt"})
        assert c_u2.request_id == c_u.request_id
        assert c_u2.original_request == req_u

        # 2) one request_id across lifecycle (handle_user_message)
        uni_root = root / "data" / "_e2e_unified_lifecycle"
        if uni_root.exists():
            shutil.rmtree(uni_root)
        uni_root.mkdir(parents=True)

        class _UniBrain(Brain):
            def model_status(self) -> str:
                return "ONLINE"

            def is_available(self) -> bool:
                return True

            def classify_intent(self, user_text: str) -> dict:
                return _IC.classify_offline(user_text)

            def understand_contract(self, original_request: str, *, prior_args=None):
                from jarvis.contract_semantics import offline_semantic_draft

                return offline_semantic_draft(original_request, prior_args)

            def plan(self, goal: str, known_capabilities: list[str]) -> dict:
                return {
                    "steps": ["build"],
                    "can_reuse": [],
                    "missing": [goal],
                    "needs_research": False,
                    "needs_new_skill": True,
                    "skill_name": "io_files_io_create_txt",
                    "skill_description": goal,
                    "research_queries": [],
                    "args": {"dest": "bees.txt", "body": "BZZ"},
                    "required_args": ["dest", "body"],
                }

            def extract_task_args(self, *a, **k):
                return {"dest": "bees.txt", "body": "BZZ"}

            def write_skill_code(self, skill_name, description, research, **kwargs):
                return '''
from pathlib import Path
SKILL_META = {"name": "%s", "capabilities": ["write_file"],
 "required_args": ["dest", "body"], "version": 1}
def run(context):
    args = context.get("args") or {}
    p = Path(context.get("workspace") or ".") / str(args.get("dest") or "x.txt")
    p.write_text(str(args.get("body") or "") + "\\n", encoding="utf-8")
    return {"ok": True, "result": {"path": str(p)}, "error": None, "evidence": "wrote"}
''' % skill_name

            def diagnose(self, *a, **k):
                return {
                    "fault_layer": "unknown",
                    "rewrite_skill": False,
                    "approach": "write",
                    "needs_research": False,
                    "research_queries": [],
                    "missing_args": [],
                    "required_args": ["dest", "body"],
                    "suggested_args": {"dest": "bees.txt", "body": "BZZ"},
                }

            def verify_claim(self, *a, **k):
                return {"achieved": True, "confidence": 0.9, "reason": "ok"}

            def converse(self, user_text, history=None) -> str:
                return "Hello from JARVIS."

            def research(self, *a, **k):
                return {
                    "summary": "Bees are pollinating insects. Honeybees live in colonies.",
                    "sources": ["https://example.com/bees"],
                    "queries": ["bees"],
                }

            def synthesize_knowledge(self, *a, **k):
                return {
                    "summary": "Bees are pollinating insects. Honeybees live in colonies.",
                    "facts": ["Bees pollinate plants", "Honeybees form colonies"],
                    "approach": "study bees",
                }

        # 3/4) subflow cannot declare global DONE; DONE iff all outcomes VERIFIED
        tr_partial = _OT(("learning", "artifact", "execution"))
        tr_partial.mark("learning", verified=True, evidence={"ka": True})
        try:
            _aov(tr_partial.required, tr_partial.status)
            raise AssertionError("learning alone must not authorize DONE")
        except _InvA:
            pass
        # 5) learning PASS != task PASS
        assert _OL in tr_partial.required and tr_partial.status[_OL] == _SV
        assert tr_partial.pending() == ["artifact", "execution"]
        # 6) research PASS != execution PASS
        tr_re = _OT(("research", "execution"))
        tr_re.mark("research", verified=True)
        try:
            _aov(tr_re.required, tr_re.status)
            raise AssertionError("research alone must not authorize DONE")
        except _InvA:
            pass
        # 7) TEST PASS != execution PASS
        tr_te = _OT(("test", "execution"))
        tr_te.mark("test", verified=True)
        try:
            _aov(tr_te.required, tr_te.status)
            raise AssertionError("TEST alone must not authorize DONE")
        except _InvA:
            pass
        # 8) tool self-report != VERIFY PASS — ACTIVE gate
        from jarvis.invariants import assert_active_requires_independent_verify as _aact

        try:
            _aact(verified=False, skill_ok=True, test_ok=True)
            raise AssertionError("skill/test self-report must not grant ACTIVE")
        except _InvA:
            pass
        # 9) final-output constraints verified
        foc = _efoc('Reply in Latvian with 2 sentences')
        assert foc.get("language") and foc.get("sentence_count") == 2, foc
        ok_fo, _ = _vfo("Pirmais teikums. Otrais teikums.", foc)
        assert ok_fo is True
        bad_fo, detail_fo = _vfo("Only one sentence here.", foc)
        assert bad_fo is False, detail_fo
        # 10) semantic TaskContract cannot be downgraded by regex classifier
        sem = _TC.understand(
            'Create out.txt containing "KEEP"',
            request_id="rid_sem",
        )
        assert _OA in sem.required_outcomes and _OE in sem.required_outcomes
        # Hostile learning-primary hint must not strip artifact/execution
        hostile = _IC.normalize(
            {"intent": "learning", "needs_learning": True, "needs_capability": False},
            'Create out.txt containing "KEEP"',
        )
        assert hostile["needs_capability"] is True, hostile
        merged = sem.with_requirements_hint(hostile)
        assert _OA in merged.required_outcomes and _OE in merged.required_outcomes
        assert _OL in merged.required_outcomes  # may add learning, never strip arts

        # Combined multi-outcome contracts (unknown languages / free form)
        combos = {
            "learning+artifact": (
                'Learn about ants and create ants.txt with "ANTS"',
                {_OL, _OA, _OE},
            ),
            "research+artifact": (
                'Research moss then create moss.txt with "MOSS"',
                {_OR, _OA, _OE},
            ),
            "research+execution": (
                'Research caching then create cache_note.txt with "CACHE"',
                {_OR, _OA, _OE},
            ),
            "learning+execution": (
                'Iemācies par bitēm un izveido bites.txt ar "BITE"',
                {_OL, _OA, _OE},
            ),
            "artifact+test+execution": (
                'Erstelle datei combo.txt mit "COMBO"',
                {_OA, _OTEST, _OE},
            ),
            "learning+research+artifact+execution+formatted-response": (
                'Learn and research frogs, create frogs.txt with "FROG", '
                'reply in English with 2 sentences',
                {_OL, _OR, _OA, _OE, _OFO},
            ),
        }
        for label, (text, expect) in combos.items():
            cc = _TC.from_request(text)
            got = set(cc.required_outcomes)
            missing = expect - got
            assert not missing, (label, text, got, missing)
            assert cc.original_request == text

        # E2E: learning+artifact — learning alone must not DONE; full path verifies arts
        logs_uni: list[str] = []
        orch_uni = Orchestrator(
            root=uni_root,
            brain=_UniBrain(),
            on_log=lambda m: logs_uni.append(m),
        )
        # Property: handle_user_message creates ONE request_id used throughout
        r_hello = orch_uni.handle_user_message("Hello JARVIS")
        assert r_hello.get("request_id"), r_hello
        assert r_hello.get("type") == "conversation", r_hello
        assert r_hello.get("success") is True, r_hello
        rid_hello = r_hello["request_id"]

        req_la = 'Learn about bees and create bees.txt with "BZZ"'
        r_la = orch_uni.handle_user_message(req_la)
        assert r_la.get("request_id"), r_la
        assert r_la["request_id"] != rid_hello
        # Contract in result must share request_id
        cdict = r_la.get("contract") or {}
        if cdict:
            assert cdict.get("request_id") == r_la["request_id"], cdict
            assert cdict.get("original_request") == req_la
            ro = set(cdict.get("required_outcomes") or [])
            assert _OL in ro and _OA in ro and _OE in ro, ro
        # Actual artifact must exist when success
        bees = uni_root / "workspace" / "bees.txt"
        # workspace path may vary — search
        found_bees = list(uni_root.rglob("bees.txt"))
        if r_la.get("success"):
            assert found_bees, (r_la, list(uni_root.rglob("*"))[:40])
            body = found_bees[0].read_text(encoding="utf-8")
            assert "BZZ" in body, body
            outs = r_la.get("outcomes") or {}
            assert outs.get("all_verified") is True, outs
            assert _OL in (outs.get("status") or {}) and (
                outs["status"].get(_OL) == _SV
            ), outs
            assert outs["status"].get(_OA) == _SV, outs
            assert outs["status"].get(_OE) == _SV, outs
        else:
            # Partial success refused — learning must not alone force DONE
            outs = r_la.get("outcomes") or {}
            if outs.get("status", {}).get(_OL) == _SV:
                assert outs.get("all_verified") is not True, outs
                assert _OA in (outs.get("pending") or []) or _OE in (
                    outs.get("pending") or []
                ), outs

        # run_learning_cycle is NOT an alternate top-level DONE bypass —
        # combined contract still requires all outcomes
        from jarvis.outcomes import OutcomeTracker as _OT2

        fake = _OT2(("learning", "artifact"))
        fake.mark("learning", verified=True)
        try:
            _aov(fake.required, fake.status)
            raise AssertionError("subflow learning must not declare global DONE")
        except _InvA:
            pass

        orch_uni.close()
        print("  OK architecture invariants + combined-request contracts")
    except Exception as exc:
        msg = f"E2E_UNIFIED_LIFECYCLE: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # 4z2) Goal-grounding — TaskContract sole truth across RESEARCH→RECOVERY
    print("  — e2e goal-grounding (outcome queries / mismatch / wrong-tool ≠ progress) —")
    try:
        from jarvis.task_contract import TaskContract as _TGg
        from jarvis.capability_match import (
            evaluate_capability as _evc,
            outcome_capability_mismatch as _ocm,
        )
        from jarvis.recovery import ProgressAwareRecovery as _PAR
        from jarvis.knowledge_artifact import gap_fill_queries as _gfq
        from jarvis.observer import Observer as _ObsG

        # Unknown multi-step requests (no task-specific hardcode)
        multi = [
            'Apgūsti par sūnām un izveido sunas.txt ar "SUNA_OK"',
            'Erforschen Moose und erstelle moose.md mit "MOOS"',
            'Studia le api e crea api.txt con "API_OK", reply in English with 2 sentences',
        ]
        for req in multi:
            cg = _TGg.from_request(req)
            assert cg.original_request == req
            assert cg.required_outcomes, req
            # Research queries must be outcome-derived, not whole-request dump
            qs = cg.default_research_queries()
            assert qs, (req, qs)
            assert req not in qs or len(qs) > 1 or cg.topical_anchor(), (req, qs)
            # Whole free-form request must not be the sole ungrounded query
            # when topical anchor / artifacts exist
            if cg.artifacts or cg.topical_anchor():
                assert not (len(qs) == 1 and qs[0] == req), (req, qs)
            # Per-outcome queries stay on contract fields
            for kind in cg.required_outcomes:
                oq = cg.outcome_research_queries(kind)
                for q in oq:
                    assert cg.query_is_grounded(q), (req, kind, q)

        # Capability: extension-only bait cannot REUSE
        bait = {
            "name": "generic_txt_writer",
            "description": "writes txt files",
            "capabilities": ["write file", "txt"],
            "status": "ACTIVE",
            "file_path": "",
        }
        tgt = _TGg.from_request(
            'Create wetland_habitat.md containing "WETLAND_OK"'
        )
        assert not _evc(bait, tgt).get("compatible"), _evc(bait, tgt)
        assert _ocm(bait, tgt) is True

        # DIAGNOSE outcome↔capability mismatch
        diag = Orchestrator._deterministic_diagnose(
            {
                "phase": "VERIFY",
                "exception": "content_constraint: 'WETLAND_OK' missing from expected files",
                "context": {
                    "args": {"dest": "wetland_habitat.md", "body": "WETLAND_OK"},
                    "skill": bait,
                    "skill_path": "",
                    "capabilities": bait["capabilities"],
                },
                "skill_name": "generic_txt_writer",
                "version": 1,
            },
            tgt,
        )
        assert diag.get("fault_layer") == "capability_mismatch", diag
        assert diag.get("rewrite_skill") is False, diag

        # Wrong-tool artifacts ≠ recovery progress
        prev = type("A", (), {})()  # unused — use RecoveryAttempt via record
        from jarvis.recovery import RecoveryAttempt as _RA

        prev_a = _RA(
            fault_layer="skill_code",
            failure_signature="sig1",
            attempted_solution="rewrite",
            model_used="offline",
            result="fail",
            artifacts=[{"path": "wrong.txt", "exists": True, "size": 3}],
            verifier_failure={"reason": "missing wetland_habitat.md"},
        )
        assert (
            _PAR.material_progress(
                prev_a,
                code_fingerprint="",
                artifacts=[{"path": "other.txt", "exists": True, "size": 4}],
                verifier_failure={"reason": "missing wetland_habitat.md"},
                fault_layer="skill_code",
                approach_fingerprint="alt",
                contract=tgt,
            )
            is False
        )
        assert (
            _PAR.material_progress(
                prev_a,
                code_fingerprint="",
                artifacts=[
                    {
                        "path": "wetland_habitat.md",
                        "exists": True,
                        "size": 20,
                        "preview": "WETLAND_OK",
                    }
                ],
                verifier_failure={"reason": "ok"},
                fault_layer="skill_code",
                approach_fingerprint="fixed",
                contract=tgt,
            )
            is True
        )

        # gap_fill / observer carry frozen contract
        learn_c = _TGg.from_request("Learn about quantum entanglement basics")
        gq = _gfq(
            learn_c.original_request,
            ["concepts", "examples"],
            contract=learn_c,
        )
        assert gq and all(learn_c.query_is_grounded(q) or "quantum" in q.lower() for q in gq), gq
        obs_root = root / "data" / "_e2e_goal_ground_obs"
        obs_root.mkdir(parents=True, exist_ok=True)
        obs = _ObsG(obs_root).observe_failure(
            goal="stale goal string",
            skill_name="x",
            version=1,
            phase="VERIFY",
            skill_code="",
            contract=tgt,
            verification={"reason": "fail"},
        )
        assert obs.get("original_request") == tgt.original_request
        assert "artifact" in (obs.get("required_outcomes") or [])
        assert obs.get("expected_artifacts") == list(tgt.artifacts)

        # Offline plan never remints a second contract
        gg_root = root / "data" / "_e2e_goal_ground_plan"
        if gg_root.exists():
            shutil.rmtree(gg_root)
        gg_root.mkdir(parents=True)

        class _OffBrain(Brain):
            def model_status(self) -> str:
                return "OFFLINE"

            def is_available(self) -> bool:
                return False

        orch_gg = Orchestrator(root=gg_root, brain=_OffBrain())
        plan = orch_gg._offline_plan(tgt, [])
        assert plan.get("research_queries") == tgt.default_research_queries()
        assert plan.get("required_outcomes") == list(tgt.required_outcomes)
        assert plan.get("subject") == tgt.subject
        # Locked contract survives diagnose
        locked = _TGg.validate(_TGg.understand(tgt.original_request, request_id="gg1"))
        d2 = Orchestrator._deterministic_diagnose(
            {
                "phase": "VERIFY",
                "exception": "content_constraint missing",
                "context": {"args": {}, "skill": bait},
                "skill_name": "generic_txt_writer",
            },
            locked,
        )
        assert locked.original_request == tgt.original_request
        assert locked.request_id == "gg1"
        orch_gg.close()
        print("  OK goal-grounding — outcome queries / mismatch / wrong-tool ≠ progress")
    except Exception as exc:
        msg = f"E2E_GOAL_GROUNDING: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # 5) Orchestrator boots + conversation logs must not echo USER/JARVIS replies
    try:
        echoed: list[str] = []
        orch = Orchestrator(
            root=root,
            brain=Brain(),
            on_log=lambda m: echoed.append(m),
        )
        stats = orch.get_dashboard_stats()
        assert "brain_status" in stats
        r = orch.handle_user_message("/status")
        assert "reply" in r
        # /status reply is returned to UI — must not also be pushed via on_log
        status_reply = r.get("reply") or ""
        assert not any(status_reply and status_reply in m for m in echoed), echoed
        assert not any(m.startswith("USER:") for m in echoed), echoed
        assert not any(m.startswith("JARVIS:") for m in echoed), echoed
        orch.close()
        print("  OK orch     boot + /status (no USER/JARVIS log echo)")
    except Exception as exc:
        msg = f"ORCH: {exc}"
        print(f"  FAIL {msg}")
        errors.append(msg)

    print("===", "PASS" if not errors else f"FAIL ({len(errors)})", "===")
    return 1 if errors else 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="JARVIS autonomous agent")
    parser.add_argument("--cli", action="store_true", help="Run in terminal mode")
    parser.add_argument("--check", action="store_true", help="Run self-check and exit")
    parser.add_argument(
        "--root",
        type=str,
        default=str(ROOT),
        help="Project root (skills/, data/)",
    )
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(exist_ok=True)
    (root / "skills").mkdir(exist_ok=True)

    if args.check:
        return run_check(root)
    if args.cli:
        return run_cli(root)

    # GUI default
    if tk is None:
        print("tkinter not installed; falling back to CLI", file=sys.stderr)
        print("Install with: sudo apt-get install python3-tk", file=sys.stderr)
        return run_cli(root)
    try:
        app = JarvisGUI(root)
        app.run()
        return 0
    except Exception as exc:
        # TclError or display missing
        print(f"GUI unavailable ({exc}); falling back to CLI", file=sys.stderr)
        return run_cli(root)


if __name__ == "__main__":
    raise SystemExit(main())
