"""
JARVIS orchestrator — the autonomous learning cycle.

REQUEST → PLAN → check capabilities → research → learn → build skill →
install deps → test → repair loop → verify → save ACTIVE skill →
execute original task → save experience → DONE

DONE is only allowed after factual verification.
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

from jarvis.brain import Brain
from jarvis.memory import Memory
from jarvis.ledger import Ledger
from jarvis.capability_registry import CapabilityRegistry
from jarvis.research import ResearchSystem
from jarvis.dependency_manager import DependencyManager
from jarvis.skill_builder import SkillBuilder
from jarvis.skill_tester import SkillTester
from jarvis.verifier import Verifier
from jarvis.skill_loader import SkillLoader
from jarvis.observer import Observer
from jarvis.context_builder import ContextBuilder

logger = logging.getLogger("jarvis.orchestrator")

MAX_REPAIR_ATTEMPTS = 5


class Orchestrator:
    """Coordinates brain + memory + skills through the full learning cycle."""

    def __init__(
        self,
        root: str | Path,
        brain: Optional[Brain] = None,
        on_log: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.root / "data"
        self.skills_dir = self.root / "skills"
        self.workspace = self.root / "workspace_runtime"
        self.data_dir.mkdir(exist_ok=True)
        self.skills_dir.mkdir(exist_ok=True)
        self.workspace.mkdir(exist_ok=True)

        self.on_log = on_log or (lambda m: None)
        self.on_status = on_status or (lambda s: None)

        db = self.data_dir / "jarvis.db"
        self.brain = brain or Brain()
        self.memory = Memory(db)
        self.ledger = Ledger(db)
        self.registry = CapabilityRegistry(db)
        self.research = ResearchSystem(brain=self.brain, on_log=self._log)
        self.deps = DependencyManager(on_log=self._log)
        self.builder = SkillBuilder(self.skills_dir, brain=self.brain, on_log=self._log)
        self.tester = SkillTester(self.workspace, on_log=self._log)
        self.verifier = Verifier(self.workspace, brain=self.brain, on_log=self._log)
        self.loader = SkillLoader(self.skills_dir, self.workspace)
        self.observer = Observer(self.workspace, on_log=self._log)
        self.contexts = ContextBuilder(brain=self.brain, on_log=self._log)

    # ── Public API ──────────────────────────────────────────────────────

    def handle_user_message(self, text: str) -> dict[str, Any]:
        """Route a user message to conversation or full task cycle."""
        text = (text or "").strip()
        if not text:
            return {"type": "error", "reply": "Empty message."}

        self.memory.add_message("user", text)
        self._status("THINKING")
        # Do not _log USER/JARVIS reply here — GUI/CLI already display them once.

        # Fast-path commands
        low = text.lower().strip()
        if low in ("/status", "status"):
            reply = self._format_status()
            self.memory.add_message("assistant", reply)
            self._status("IDLE")
            return {"type": "status", "reply": reply}
        if low in ("/skills", "skills"):
            reply = self._format_skills()
            self.memory.add_message("assistant", reply)
            self._status("IDLE")
            return {"type": "skills", "reply": reply}
        if low.startswith("/help"):
            reply = (
                "JARVIS commands:\n"
                "  /status  — memory & skill stats\n"
                "  /skills  — list skills & statuses\n"
                "  /help    — this help\n"
                "Or ask anything / give a task to learn & execute."
            )
            self.memory.add_message("assistant", reply)
            self._status("IDLE")
            return {"type": "help", "reply": reply}

        # Classify intent
        model_status = self.brain.model_status()
        brain_up = model_status == "ONLINE"
        if not brain_up:
            intent = self._offline_classify(text)
        else:
            intent = self.brain.classify_intent(text)

        if intent.get("intent") == "conversation" and not intent.get("needs_capability"):
            self._status("CONVERSING")
            history = self.memory.chat_history_for_llm(10)
            history = [m for m in history if not (m["role"] == "user" and m["content"] == text)]
            if brain_up:
                reply = self.brain.converse(text, history=history)
            elif model_status == "MODEL MISSING":
                reply = (
                    f"[MODEL MISSING] Ollama darbojas, bet modelis "
                    f"{self.brain.model} nav atrasts. Palaid: "
                    f"ollama pull {self.brain.model}. Saņēmu: {text}"
                )
            else:
                reply = (
                    "[Ollama OFFLINE] Esmu JARVIS. Palaiž Ollama ar modeli "
                    f"{self.brain.model}, lai runātu un mācītos. "
                    f"Saņēmu: {text}"
                )
            self.memory.add_message("assistant", reply)
            self._status("IDLE")
            return {"type": "conversation", "reply": reply}

        # Task path — full learning cycle
        goal = intent.get("goal") or text
        result = self.run_cycle(goal, original_request=text)
        reply = result.get("reply") or result.get("outcome") or str(result)
        self.memory.add_message("assistant", reply, meta={"task_id": result.get("task_id")})
        self._status("IDLE" if result.get("success") else "ERROR")
        return result

    def run_cycle(self, goal: str, original_request: Optional[str] = None) -> dict[str, Any]:
        """Execute the full REQUEST→…→DONE learning/execution cycle."""
        task_id = self.ledger.start_task(goal)
        self._status("REQUEST")
        self._log(f"[{task_id}] REQUEST: {goal}")

        try:
            # PLAN
            self._status("PLAN")
            caps = self.registry.active_capabilities()
            if self.brain.is_available():
                plan = self.brain.plan(goal, caps)
            else:
                plan = self._offline_plan(goal, caps)
            self.ledger.log(task_id, "PLAN", "Plan created", plan)
            self._log(f"[{task_id}] PLAN: {plan.get('steps')}")

            # Structured args from user goal (universal — not task-hardcoded)
            task_args = {}
            if isinstance(plan.get("args"), dict):
                task_args.update(plan["args"])
            ctx0 = self.contexts.build(
                goal,
                str(self.workspace),
                mode="plan",
                prior_args=task_args,
                diagnosis={
                    "required_args": plan.get("required_args") or [],
                },
            )
            task_args = dict(ctx0.get("args") or {})
            self.ledger.log(
                task_id,
                "PLAN",
                f"Prepared args keys={list(task_args.keys())}",
                {"args": task_args},
            )

            # CHECK CAPABILITIES
            self._status("CHECK_CAPABILITIES")
            reuse = plan.get("can_reuse") or []
            matched = []
            for name in reuse:
                sk = self.registry.get_skill(name)
                if sk and sk["status"] == "ACTIVE":
                    matched.append(sk)
            if not matched:
                keywords = plan.get("research_queries") or [goal]
                tokens = []
                for k in keywords:
                    tokens.extend(str(k).split())
                matched = self.registry.match_skills(tokens, only_active=True)
            self.ledger.log(
                task_id,
                "CHECK_CAPABILITIES",
                f"Matched {len(matched)} active skills",
                {"names": [m["name"] for m in matched]},
            )

            skill_record = None
            exec_result = None

            if matched and not plan.get("needs_new_skill"):
                skill_record = matched[0]
                self._log(f"[{task_id}] Reusing ACTIVE skill: {skill_record['name']}")
                # Refresh args with skill meta hints
                task_args = self.contexts.build(
                    goal,
                    str(self.workspace),
                    mode="execute",
                    skill_meta=skill_record,
                    prior_args=task_args,
                ).get("args") or task_args
                exec_result = self._execute_trusted(
                    task_id, skill_record, goal, args=task_args
                )
                if not exec_result.get("ok"):
                    # Mark broken and fall through to repair/learn
                    self.registry.mark_failure(
                        skill_record["name"],
                        str(exec_result.get("error") or "execution failed"),
                    )
                    self.memory.log_skill_event(
                        skill_record["name"], "broken_detected", "BROKEN", exec_result
                    )
                    self._log(
                        f"[{task_id}] Skill broken — entering repair/learn path"
                    )
                    skill_record, task_args = self._learn_or_repair(
                        task_id, goal, plan, repair_of=skill_record, task_args=task_args
                    )
                    if skill_record and skill_record["status"] == "ACTIVE":
                        exec_result = self._execute_trusted(
                            task_id, skill_record, goal, args=task_args
                        )
            else:
                skill_record, task_args = self._learn_or_repair(
                    task_id, goal, plan, repair_of=None, task_args=task_args
                )
                if skill_record and skill_record["status"] == "ACTIVE":
                    exec_result = self._execute_trusted(
                        task_id, skill_record, goal, args=task_args
                    )

            # VERIFY final execution against USER REQUEST — not skill self-proof
            self._status("VERIFY")
            if not exec_result:
                outcome = "Neizdevās izpildīt uzdevumu: nav skill / nav izpildes rezultāta."
                self.memory.save_experience(
                    goal, outcome, False,
                    skill_name=(skill_record or {}).get("name"),
                    details={"exec": exec_result, "task_id": task_id},
                )
                self.ledger.log(task_id, "SAVE_EXPERIENCE", outcome)
                self.ledger.finish(task_id, False, {"outcome": outcome})
                self._log(f"[{task_id}] FAIL")
                return {
                    "type": "task",
                    "success": False,
                    "task_id": task_id,
                    "reply": outcome,
                    "outcome": outcome,
                }

            self._log(
                f"[{task_id}] SKILL RESULT: ok={exec_result.get('ok')} "
                f"rc={exec_result.get('returncode')} error={exec_result.get('error')}"
            )
            verification = self.verifier.verify(
                goal,
                exec_result,
                require_brain_confirm=False,
                args=task_args,
                user_request=original_request or goal,
            )
            self.ledger.log(
                task_id,
                "VERIFY",
                verification.get("reason"),
                {
                    "skill_result": verification.get("skill_result"),
                    "verifier_result": verification.get("verifier_result"),
                    "args": task_args,
                },
            )
            self._log(
                f"[{task_id}] VERIFIER RESULT: "
                f"{'PASS' if verification.get('verified') else 'FAIL'} — "
                f"{verification.get('reason')}"
            )

            # VERIFY FAIL → OBSERVE → DIAGNOSE → REPAIR → RETEST (then re-execute)
            if not verification.get("verified"):
                if skill_record:
                    self.registry.mark_failure(
                        skill_record["name"],
                        verification.get("reason") or "verify failed",
                    )
                    self.memory.log_skill_event(
                        skill_record["name"],
                        "verifier_failed",
                        "BROKEN",
                        {
                            "reason": verification.get("reason"),
                            "args_keys": list(task_args.keys()),
                        },
                    )
                self._log(
                    f"[{task_id}] VERIFY FAIL vs USER REQUEST — "
                    f"entering OBSERVE→DIAGNOSE→REPAIR→RETEST"
                )
                skill_record, task_args = self._learn_or_repair(
                    task_id,
                    goal,
                    plan,
                    repair_of=skill_record,
                    task_args=task_args,
                    verify_failure=verification,
                )
                if skill_record and skill_record.get("status") == "ACTIVE":
                    exec_result = self._execute_trusted(
                        task_id, skill_record, goal, args=task_args
                    )
                    self._status("VERIFY")
                    verification = self.verifier.verify(
                        goal,
                        exec_result,
                        require_brain_confirm=False,
                        args=task_args,
                        user_request=original_request or goal,
                    )
                    self.ledger.log(
                        task_id,
                        "VERIFY",
                        verification.get("reason"),
                        {
                            "skill_result": verification.get("skill_result"),
                            "verifier_result": verification.get("verifier_result"),
                            "args": task_args,
                            "after_repair": True,
                        },
                    )
                    self._log(
                        f"[{task_id}] VERIFIER RESULT (after repair): "
                        f"{'PASS' if verification.get('verified') else 'FAIL'} — "
                        f"{verification.get('reason')}"
                    )

            if not verification.get("verified"):
                outcome = (
                    f"SKILL RESULT ok={exec_result.get('ok')}, bet VERIFIER FAIL: "
                    f"{verification.get('reason')}. DONE nav atļauts."
                )
                self.memory.save_experience(
                    goal, outcome, False,
                    skill_name=(skill_record or {}).get("name"),
                    details={
                        "skill_result": verification.get("skill_result"),
                        "verifier_result": verification.get("verifier_result"),
                        "task_id": task_id,
                        "args": task_args,
                    },
                )
                self.ledger.log(task_id, "SAVE_EXPERIENCE", outcome)
                self.ledger.finish(task_id, False, {"outcome": outcome})
                return {
                    "type": "task",
                    "success": False,
                    "task_id": task_id,
                    "reply": outcome,
                    "outcome": outcome,
                    "skill_result": verification.get("skill_result"),
                    "verifier_result": verification.get("verifier_result"),
                }

            # SUCCESS path — DONE only after verifier PASS
            if skill_record:
                self.registry.mark_success(skill_record["name"])
            outcome = (
                f"DONE. Uzdevums izpildīts un VERIFIER PASS.\n"
                f"Skill: {(skill_record or {}).get('name')}\n"
                f"SKILL RESULT: {exec_result.get('result')!r}\n"
                f"VERIFIER: {verification.get('reason')}"
            )
            self._status("SAVE_EXPERIENCE")
            self.memory.save_experience(
                goal, outcome, True,
                skill_name=(skill_record or {}).get("name"),
                details={
                    "result": exec_result.get("result"),
                    "evidence": exec_result.get("evidence"),
                    "verification": verification,
                    "task_id": task_id,
                },
            )
            self.ledger.log(task_id, "SAVE_EXPERIENCE", "Experience saved")
            self.ledger.finish(
                task_id,
                True,
                {
                    "result": exec_result.get("result"),
                    "evidence": exec_result.get("evidence"),
                    "skill": (skill_record or {}).get("name"),
                },
            )
            self._status("DONE")
            self._log(f"[{task_id}] DONE")
            return {
                "type": "task",
                "success": True,
                "task_id": task_id,
                "reply": outcome,
                "outcome": outcome,
                "result": exec_result.get("result"),
                "evidence": exec_result.get("evidence"),
                "skill": (skill_record or {}).get("name"),
            }

        except Exception as exc:
            tb = traceback.format_exc()
            outcome = f"Internal error: {exc}"
            self._log(f"[{task_id}] EXCEPTION: {tb}")
            self.ledger.finish(task_id, False, {"error": str(exc), "traceback": tb})
            self.memory.save_experience(goal, outcome, False, details={"traceback": tb})
            return {
                "type": "task",
                "success": False,
                "task_id": task_id,
                "reply": outcome,
                "outcome": outcome,
            }

    # ── Learning / repair ───────────────────────────────────────────────

    def _learn_or_repair(
        self,
        task_id: str,
        goal: str,
        plan: dict,
        repair_of: Optional[dict] = None,
        task_args: Optional[dict] = None,
        verify_failure: Optional[dict] = None,
    ) -> tuple[Optional[dict], dict]:
        """
        Universal learning loop (no task-specific hardcoding):

        BUILD → TEST → OBSERVE → DIAGNOSE → RESEARCH? → REPAIR → RETEST → VERIFY → ACTIVE

        Returns (skill_record, task_args).
        """
        skill_name = (
            (repair_of or {}).get("name")
            or plan.get("skill_name")
            or self.brain._slug(goal)[:40]
        )
        description = plan.get("skill_description") or goal
        version = self.registry.next_version(skill_name)
        args: dict = dict(task_args or {})
        if isinstance(plan.get("args"), dict):
            args = ContextBuilder.merge_args(plan.get("args"), args)

        existing = self.registry.get_skill(skill_name)
        protect_active_path = None
        if existing and existing.get("status") == "ACTIVE":
            protect_active_path = existing.get("file_path")
        elif repair_of and repair_of.get("file_path"):
            protect_active_path = repair_of.get("file_path")
            if existing and existing.get("status") != "REPAIRING":
                self.registry.set_status(skill_name, "REPAIRING")
            self.memory.log_skill_event(skill_name, "repair_start", "REPAIRING")
            self.ledger.log(task_id, "REPAIR", f"Repairing {skill_name}")

        previous_code = None
        if repair_of:
            previous_code = self.builder.read_skill_code(
                skill_name, path=repair_of.get("file_path")
            )

        # Initial research (may be refreshed after DIAGNOSE)
        research = self._do_research(
            task_id, skill_name, goal, plan.get("research_queries") or [goal]
        )

        # Pull prior learning from SQLite
        failed_approaches = self.memory.get_failed_approaches(skill_name)
        prior_solutions = self.memory.recent_solutions(skill_name=skill_name, limit=5)
        if prior_solutions:
            research["prior_solutions"] = [
                {
                    "approach": s.get("approach"),
                    "summary": s.get("diagnosis_summary"),
                    "version": s.get("version"),
                }
                for s in prior_solutions
            ]

        diagnosis: Optional[dict] = None
        last_code = previous_code
        last_error: Optional[str] = (
            (repair_of or {}).get("last_error") if repair_of else None
        )
        current_approach = str(research.get("approach") or "initial")
        built: Optional[dict] = None
        skip_rebuild = False
        meta: dict = {}
        caps: list = [description]
        deps_list: list = []

        # Seed from a final-EXECUTE verifier failure so the first loop iteration
        # OBSERVE→DIAGNOSE that mismatch (defaults / wrong artifact vs USER REQUEST).
        if verify_failure and repair_of:
            seed_obs = self.observer.observe_failure(
                goal=goal,
                skill_name=skill_name,
                version=int(repair_of.get("version") or version),
                phase="VERIFY",
                skill_code=previous_code,
                context={
                    "goal": goal,
                    "args": args,
                    "workspace": str(self.workspace),
                },
                dependencies=list(repair_of.get("dependencies") or []),
                test_result={
                    "ok": True,
                    "error": verify_failure.get("reason"),
                    "result": (verify_failure.get("skill_result") or {}).get("result"),
                    "evidence": (verify_failure.get("skill_result") or {}).get("evidence"),
                    "returncode": 0,
                },
                verification=verify_failure,
                prior_approaches=failed_approaches,
            )
            diagnosis, research, current_approach, last_error, last_code, args = (
                self._observe_diagnose_enrich(
                    task_id, skill_name, goal, seed_obs, research,
                    approach_label=current_approach,
                    code=previous_code,
                    task_args=args,
                )
            )
            failed_approaches = self.memory.get_failed_approaches(skill_name)
            layer = str(diagnosis.get("fault_layer") or "skill_code")
            # Defaults / wrong artifact vs user request → rewrite skill
            skip_rebuild = layer in (
                "context_args", "test_harness", "dependency", "verifier"
            ) and (not diagnosis.get("rewrite_skill", False))
            if existing and existing.get("status") == "ACTIVE":
                protect_active_path = existing.get("file_path")
                self.registry.set_status(skill_name, "REPAIRING")

        for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
            attempt_version = version + (attempt - 1)
            is_retest = attempt > 1
            phase_build = "REPAIR" if is_retest else "BUILD_SKILL"
            phase_test = "RETEST" if is_retest else "TEST"

            # Refresh args each attempt (diagnosis may suggest new ones)
            args = self.contexts.build(
                goal,
                str(self.workspace),
                mode="test",
                skill_meta={"name": skill_name, "description": description, **meta},
                prior_args=args,
                diagnosis=diagnosis,
            ).get("args") or args

            # ── BUILD / REPAIR (skip when fault is only context_args) ───
            if not skip_rebuild or built is None or not built.get("ok"):
                self._status(phase_build)
                built = self.builder.build(
                    skill_name=skill_name,
                    description=description,
                    research=research,
                    version=attempt_version,
                    previous_code=last_code,
                    error_log=last_error,
                    protect_active_path=protect_active_path,
                    diagnosis=diagnosis if (diagnosis or {}).get("rewrite_skill", True) else None,
                    failed_approaches=failed_approaches,
                    test_plan=(diagnosis or {}).get("test_plan") if diagnosis else None,
                )
                self.ledger.log(
                    task_id,
                    phase_build,
                    f"Attempt {attempt}: {'ok' if built['ok'] else built.get('error')}",
                    {
                        "path": built.get("path"),
                        "attempt": attempt,
                        "approach": current_approach,
                        "protected_active": built.get("protected_active"),
                        "args_keys": list(args.keys()),
                    },
                )
                if not built.get("ok"):
                    obs = self.observer.observe_failure(
                        goal=goal,
                        skill_name=skill_name,
                        version=attempt_version,
                        phase="BUILD",
                        skill_code=built.get("code"),
                        context={
                            "goal": goal,
                            "args": args,
                            "workspace": str(self.workspace),
                        },
                        dependencies=list(research.get("libraries") or []),
                        build_error=built.get("error"),
                        prior_approaches=failed_approaches,
                    )
                    diagnosis, research, current_approach, last_error, last_code, args = (
                        self._observe_diagnose_enrich(
                            task_id, skill_name, goal, obs, research,
                            approach_label=current_approach,
                            code=built.get("code"),
                            task_args=args,
                        )
                    )
                    failed_approaches = self.memory.get_failed_approaches(skill_name)
                    skip_rebuild = not diagnosis.get("rewrite_skill", True)
                    continue

                meta = built["meta"]
                caps = meta.get("capabilities") or [description]
                deps_list = list(
                    meta.get("dependencies")
                    or research.get("libraries")
                    or (diagnosis or {}).get("needs_new_deps")
                    or []
                )
                self._register_candidate_safe(
                    skill_name=skill_name,
                    description=meta.get("description") or description,
                    built_path=built["path"],
                    capabilities=caps,
                    deps_list=deps_list,
                    attempt_version=attempt_version,
                    protect_active_path=protect_active_path,
                    existing=existing,
                )
            else:
                self._log(
                    f"[{task_id}] Skip skill rewrite — repairing fault_layer="
                    f"{(diagnosis or {}).get('fault_layer')} with args={list(args.keys())}"
                )
                self.ledger.log(
                    task_id,
                    "REPAIR",
                    f"context/args repair only; args_keys={list(args.keys())}",
                    {"args": args, "fault_layer": (diagnosis or {}).get("fault_layer")},
                )

            # ── INSTALL DEPS ────────────────────────────────────────────
            self._status("INSTALL_DEPS")
            dep_result = self.deps.ensure(deps_list)
            self.ledger.log(task_id, "INSTALL_DEPS", dep_result.get("details"), dep_result)
            if not dep_result.get("ok"):
                obs = self.observer.observe_failure(
                    goal=goal,
                    skill_name=skill_name,
                    version=attempt_version,
                    phase="INSTALL_DEPS",
                    skill_code=(built or {}).get("code"),
                    context={
                        "goal": goal,
                        "args": args,
                        "workspace": str(self.workspace),
                    },
                    dependencies=deps_list,
                    deps_result=dep_result,
                    build_error=f"Dependency install failed: {dep_result.get('failed')}",
                    prior_approaches=failed_approaches,
                )
                diagnosis, research, current_approach, last_error, last_code, args = (
                    self._observe_diagnose_enrich(
                        task_id, skill_name, goal, obs, research,
                        approach_label=current_approach,
                        code=(built or {}).get("code"),
                        task_args=args,
                    )
                )
                failed_approaches = self.memory.get_failed_approaches(skill_name)
                skip_rebuild = diagnosis.get("fault_layer") != "skill_code"
                continue

            # ── TEST / RETEST (subprocess) with prepared args ───────────
            self._status(phase_test)
            protect = bool(protect_active_path) and Path(str(protect_active_path)).exists()
            if not protect:
                self.registry.set_status(skill_name, "TESTING")
            self.memory.log_skill_event(skill_name, "testing", "TESTING")
            test_context = {
                "goal": goal,
                "args": args,
                "workspace": str(self.workspace),
                "mode": "test",
            }
            test_result = self.tester.test(
                built["path"], goal=goal, args=args
            )
            self.ledger.log(
                task_id,
                phase_test,
                f"skill_ok={test_result.get('ok')} rc={test_result.get('returncode')} "
                f"timed_out={test_result.get('timed_out')} "
                f"args_keys={list(args.keys())}",
                {
                    "error": test_result.get("error"),
                    "stdout": (test_result.get("stdout") or "")[:400],
                    "stderr": (test_result.get("stderr") or "")[:400],
                    "returncode": test_result.get("returncode"),
                    "timed_out": test_result.get("timed_out"),
                    "killed": test_result.get("killed"),
                    "args": args,
                },
            )
            self._log(
                f"[{task_id}] SKILL RESULT ({phase_test}): ok={test_result.get('ok')} "
                f"rc={test_result.get('returncode')} args_keys={list(args.keys())}"
            )

            test_failed = (
                test_result.get("timed_out")
                or test_result.get("crash")
                or not test_result.get("ok")
            )
            verification = None
            if not test_failed:
                # ── VERIFY against USER REQUEST / args (not skill claims) ─
                self._status("VERIFY")
                verification = self.verifier.verify(
                    goal,
                    test_result,
                    args=args,
                    user_request=goal,
                )
                self.ledger.log(
                    task_id,
                    "VERIFY",
                    verification.get("reason"),
                    {
                        "skill_result": verification.get("skill_result"),
                        "verifier_result": verification.get("verifier_result"),
                        "args": args,
                    },
                )
                if verification.get("verified"):
                    # ── ACTIVE + remember solution ──────────────────────
                    skill = self._activate_and_remember(
                        task_id=task_id,
                        skill_name=skill_name,
                        goal=goal,
                        built=built,
                        meta=meta,
                        caps=caps,
                        deps_list=deps_list,
                        attempt_version=attempt_version,
                        attempt=attempt,
                        approach=current_approach,
                        diagnosis=diagnosis,
                        description=description,
                    )
                    return skill, args

            # ── OBSERVE → DIAGNOSE → fix the correct layer ─────────────
            obs = self.observer.observe_failure(
                goal=goal,
                skill_name=skill_name,
                version=attempt_version,
                phase="VERIFY" if (not test_failed and verification) else phase_test,
                skill_code=(built or {}).get("code"),
                context=test_context,
                dependencies=deps_list,
                test_result=test_result,
                verification=verification,
                prior_approaches=failed_approaches,
                prior_diagnoses=[
                    d.get("diagnosis") if isinstance(d.get("diagnosis"), dict) else d
                    for d in self.memory.recent_diagnoses(skill_name, 5)
                ],
            )
            if not protect:
                self.registry.set_status(
                    skill_name, "BROKEN", str(obs.get("exception") or "failed")[:2000]
                )
            self.memory.log_skill_event(
                skill_name, "test_or_verify_failed", "BROKEN",
                {
                    "phase": obs["phase"],
                    "exception": obs.get("exception", "")[:500],
                    "args_keys": list(args.keys()),
                },
            )
            diagnosis, research, current_approach, last_error, last_code, args = (
                self._observe_diagnose_enrich(
                    task_id, skill_name, goal, obs, research,
                    approach_label=current_approach,
                    code=(built or {}).get("code"),
                    task_args=args,
                )
            )
            failed_approaches = self.memory.get_failed_approaches(skill_name)
            layer = str(diagnosis.get("fault_layer") or "skill_code")
            # Only rewrite skill when the fault is in skill code (or unknown)
            skip_rebuild = layer in ("context_args", "test_harness", "dependency", "verifier") and (
                not diagnosis.get("rewrite_skill", False)
            )
            if layer == "dependency" and diagnosis.get("needs_new_deps"):
                self.deps.ensure(list(diagnosis["needs_new_deps"]))
            # loop → REPAIR correct layer on next iteration

        self._log(
            f"[{task_id}] Failed to produce ACTIVE skill after "
            f"{MAX_REPAIR_ATTEMPTS} repair attempts"
        )
        return self.registry.get_skill(skill_name), args

    def _do_research(
        self, task_id: str, skill_name: str, goal: str, queries: list
    ) -> dict:
        self._status("RESEARCH")
        if not isinstance(queries, list):
            queries = [str(queries)]
        research = self.research.research(queries, goal=goal)
        self.ledger.log(task_id, "RESEARCH", "Research complete", {
            "libraries": research.get("libraries"),
            "results": len(research.get("results") or []),
            "sources": [
                {
                    "url": s.get("url"),
                    "title": s.get("title"),
                    "source": s.get("source"),
                    "provider": s.get("provider"),
                    "timestamp": s.get("timestamp"),
                    "query": s.get("query"),
                }
                for s in (research.get("sources") or [])[:10]
            ],
        })
        self._status("LEARN")
        self.ledger.log(task_id, "LEARN", (research.get("approach") or "")[:500], {
            "result_count": len(research.get("results") or []),
        })
        self.memory.set_fact(f"research:{skill_name}", research)
        return research

    def _observe_diagnose_enrich(
        self,
        task_id: str,
        skill_name: str,
        goal: str,
        observation: dict,
        research: dict,
        approach_label: str,
        code: Optional[str],
        task_args: Optional[dict] = None,
    ) -> tuple[dict, dict, str, str, Optional[str], dict]:
        """OBSERVE → DIAGNOSE → optional RESEARCH → return repair state + updated args."""
        self._status("OBSERVE")
        # Ensure observation carries the args that were actually used
        if isinstance(observation.get("context"), dict) and task_args is not None:
            observation["context"] = dict(observation["context"])
            observation["context"]["args"] = dict(task_args)

        # Fingerprint this failure for adaptive mid-repair research
        observation["error_fingerprint"] = Observer.fingerprint_error(observation)

        failure_id = self.memory.save_failure(
            skill_name, goal, observation,
            version=observation.get("version"),
            phase=observation.get("phase"),
        )
        self.ledger.log(task_id, "OBSERVE", f"failure_id={failure_id}", {
            "phase": observation.get("phase"),
            "returncode": observation.get("returncode"),
            "exception": (observation.get("exception") or "")[:500],
            "code_fingerprint": observation.get("code_fingerprint"),
            "error_fingerprint": observation.get("error_fingerprint"),
            "artifact_count": len(observation.get("artifacts") or []),
            "args_keys": list((task_args or {}).keys()),
        })

        # Record current approach as failed so next must diverge
        approach_fp = Observer.fingerprint_approach(approach_label)
        self.memory.record_failed_approach(
            skill_name,
            approach_label,
            approach_fp,
            last_error=str(observation.get("exception") or "")[:1000],
        )

        self._status("DIAGNOSE")
        failed = self.memory.get_failed_approaches(skill_name)
        prior_solutions = self.memory.recent_solutions(skill_name=skill_name, limit=5)
        if self.brain.is_available():
            diagnosis = self.brain.diagnose(
                observation,
                failed_approaches=failed,
                prior_solutions=[
                    {
                        "approach": s.get("approach"),
                        "summary": s.get("diagnosis_summary"),
                    }
                    for s in prior_solutions
                ],
            )
        else:
            ctx_args = (observation.get("context") or {}).get("args") or {}
            empty_args = not ctx_args
            err = str(observation.get("exception") or "")
            err_l = err.lower()
            phase = str(observation.get("phase") or "").upper()
            parsed_missing = ContextBuilder.parse_missing_arg_names(err)
            args_fault = (empty_args or bool(parsed_missing)) and any(
                t in err_l
                for t in ("argument", "args", "missing", "required", "keyerror")
            )
            goal_mismatch = phase == "VERIFY" and any(
                t in err_l
                for t in (
                    "default", "placeholder", "untrusted", "claim_aligns",
                    "not in user", "user constraints", "user request",
                    "reject_defaults", "missing from expected",
                )
            )
            if goal_mismatch:
                layer = "skill_code"
                rewrite = True
                approach = "honor_user_request"
                change = (
                    "Rewrite skill to satisfy USER REQUEST "
                    "(no default/placeholder artifacts)"
                )
            elif args_fault:
                layer = "context_args"
                rewrite = False
                approach = "context_args_prep"
                change = "Prepare structured context args from the user goal"
            else:
                layer = "skill_code"
                rewrite = True
                approach = f"offline_alt_v{int(observation.get('version') or 0) + 1}"
                change = "Rebuild with a different strategy using observation data"
            diagnosis = {
                "root_cause": str(observation.get("exception") or "unknown")[:500],
                "fault_layer": layer,
                "rewrite_skill": rewrite,
                "what_to_change": change,
                "approach": approach,
                "approach_changed": True,
                "needs_research": layer == "skill_code",
                "research_queries": Brain._offline_research_queries(
                    observation,
                    {
                        "root_cause": str(observation.get("exception") or "")[:500],
                        "fault_layer": layer,
                    },
                    self.memory.get_failed_approaches(skill_name),
                ),
                "needs_new_deps": [],
                "missing_args": list(parsed_missing),
                "required_args": list(parsed_missing),
                "suggested_args": {},
                "test_plan": (
                    "subprocess retest with prepared args; "
                    "VERIFY against USER REQUEST (reject defaults)"
                ),
                "expected_artifacts": [],
                "is_unfixable": False,
                "diagnosis": str(observation.get("exception") or "failure")[:500],
            }
            diagnosis["approach_fingerprint"] = Observer.fingerprint_approach(
                diagnosis["approach"]
            )

        # Universal enrichment: parse missing arg names + suggest values from goal
        diagnosis = ContextBuilder.enrich_diagnosis_args(diagnosis, observation, goal)
        # Keep VERIFY goal-mismatch as skill_code even if args enrichment runs
        if (
            str(observation.get("phase") or "").upper() == "VERIFY"
            and "default" in str(observation.get("exception") or "").lower()
        ):
            diagnosis["fault_layer"] = "skill_code"
            diagnosis["rewrite_skill"] = True

        # If diagnosis repeats a failed approach fingerprint — force divergence
        # (but do not force-rewrite context_args_prep — args repair is the fix)
        failed_fps = {a.get("approach_fingerprint") for a in failed}
        if (
            diagnosis.get("approach_fingerprint") in failed_fps
            and diagnosis.get("fault_layer") != "context_args"
        ):
            diagnosis["approach"] = (
                f"{diagnosis.get('approach')}::changed::{observation.get('version')}"
            )
            diagnosis["approach_fingerprint"] = Observer.fingerprint_approach(
                diagnosis["approach"]
            )
            diagnosis["approach_changed"] = True

        # Rebuild args when fault is context_args (or suggested_args provided)
        new_args = dict(task_args or {})
        if diagnosis.get("fault_layer") == "context_args" or diagnosis.get("suggested_args"):
            new_args = self.contexts.build(
                goal,
                str(self.workspace),
                mode="test",
                prior_args=new_args,
                diagnosis=diagnosis,
            ).get("args") or new_args
            self._log(
                f"[{task_id}] CONTEXT repair: args_keys={list(new_args.keys())} "
                f"suggested={list((diagnosis.get('suggested_args') or {}).keys())}"
            )

        # Similarity vs prior failures (current observation already persisted)
        prior_obs = [
            f.get("observation")
            for f in self.memory.recent_failures(skill_name, limit=12)
            if isinstance(f.get("observation"), dict)
        ]
        similar_count = Observer.count_similar_errors(observation, prior_obs)
        diagnosis["error_fingerprint"] = observation.get("error_fingerprint")
        diagnosis["similar_failure_count"] = similar_count

        diag_id = self.memory.save_diagnosis(
            skill_name, goal, diagnosis, failure_id=failure_id
        )
        self.ledger.log(
            task_id,
            "DIAGNOSE",
            diagnosis.get("root_cause") or diagnosis.get("diagnosis"),
            {
                "diagnosis_id": diag_id,
                "fault_layer": diagnosis.get("fault_layer"),
                "rewrite_skill": diagnosis.get("rewrite_skill"),
                "approach": diagnosis.get("approach"),
                "approach_changed": diagnosis.get("approach_changed"),
                "needs_research": diagnosis.get("needs_research"),
                "needs_new_deps": diagnosis.get("needs_new_deps"),
                "missing_args": diagnosis.get("missing_args"),
                "suggested_args": diagnosis.get("suggested_args"),
                "args_keys": list(new_args.keys()),
                "test_plan": diagnosis.get("test_plan"),
                "error_fingerprint": diagnosis.get("error_fingerprint"),
                "similar_failure_count": similar_count,
            },
        )
        self._log(
            f"[{task_id}] DIAGNOSE: {diagnosis.get('root_cause', '')[:160]} "
            f"| layer={diagnosis.get('fault_layer')!r} "
            f"rewrite_skill={diagnosis.get('rewrite_skill')} "
            f"| approach={diagnosis.get('approach')!r} "
            f"changed={diagnosis.get('approach_changed')} "
            f"| similar_errors={similar_count}"
        )

        # ── Adaptive RESEARCH inside the repair cycle ───────────────────
        # When the same/similar error repeats, generate fresh queries from
        # observation + diagnosis + traceback + failed approaches + verifier.
        repeated_error = similar_count >= 2  # current save + at least one prior
        many_failed_approaches = len(failed) >= 2
        layer_now = str(diagnosis.get("fault_layer") or "")

        force_research = repeated_error or (
            diagnosis.get("needs_research") and layer_now != "context_args"
        ) or (
            many_failed_approaches and layer_now == "skill_code"
        )
        # Pure first-time context_args prep does not need web research
        if layer_now == "context_args" and not repeated_error:
            force_research = False

        if force_research:
            diagnosis["needs_research"] = True
            diagnosis["adaptive_research"] = bool(repeated_error or many_failed_approaches)
            queries = list(diagnosis.get("research_queries") or [])
            if self.brain.is_available():
                try:
                    generated = self.brain.generate_research_queries(
                        observation, diagnosis, failed
                    )
                except Exception:
                    generated = Brain._offline_research_queries(
                        observation, diagnosis, failed
                    )
            else:
                generated = Brain._offline_research_queries(
                    observation, diagnosis, failed
                )
            for q in generated:
                if q and q not in queries:
                    queries.append(q)
            if not queries:
                queries = [goal, str(observation.get("exception") or "")[:160]]
            diagnosis["research_queries"] = queries[:6]
            self._log(
                f"[{task_id}] ADAPTIVE RESEARCH: similar={similar_count} "
                f"failed_approaches={len(failed)} queries={len(queries)}"
            )
            new_research = self._do_research(task_id, skill_name, goal, list(queries))
            research = self._merge_research(research, new_research)
            # Research must drive a NEW approach for the next repair
            researched_approach = str(research.get("approach") or "").strip()
            if researched_approach:
                candidate = researched_approach.split("\n")[0][:100]
                cand_fp = Observer.fingerprint_approach(candidate)
                failed_fps = {a.get("approach_fingerprint") for a in failed}
                if cand_fp in failed_fps:
                    candidate = (
                        f"{candidate} | researched-v{observation.get('version')}"
                    )
                    cand_fp = Observer.fingerprint_approach(candidate)
                diagnosis["approach"] = candidate
                diagnosis["approach_fingerprint"] = cand_fp
                diagnosis["approach_changed"] = True
                if research.get("repair_insight"):
                    diagnosis["what_to_change"] = (
                        f"{diagnosis.get('what_to_change')} | "
                        f"RESEARCH: {research.get('repair_insight')}"
                    )[:1000]
            self.ledger.log(
                task_id,
                "RESEARCH",
                f"adaptive repair research similar={similar_count}",
                {
                    "queries": queries[:6],
                    "similar_failure_count": similar_count,
                    "approach": diagnosis.get("approach"),
                    "libraries": research.get("libraries"),
                },
            )

        if diagnosis.get("needs_new_deps"):
            dep_r = self.deps.ensure(list(diagnosis["needs_new_deps"]))
            self.ledger.log(task_id, "INSTALL_DEPS", "from diagnosis", dep_r)
            research.setdefault("libraries", [])
            research["libraries"] = list(
                dict.fromkeys(
                    list(research.get("libraries") or [])
                    + list(diagnosis["needs_new_deps"])
                )
            )

        research["fix_plan"] = diagnosis.get("test_plan")
        research["approach"] = diagnosis.get("approach")
        research["diagnosis"] = diagnosis
        research["adaptive_research"] = diagnosis.get("adaptive_research")

        last_error = (
            f"ROOT CAUSE: {diagnosis.get('root_cause')}\n"
            f"FAULT_LAYER: {diagnosis.get('fault_layer')}\n"
            f"CHANGE: {diagnosis.get('what_to_change')}\n"
            f"APPROACH: {diagnosis.get('approach')}\n"
            f"RESEARCH_INSIGHT: {research.get('repair_insight') or ''}\n"
            f"ARGS: {json.dumps(new_args, default=str)[:500]}\n"
            f"EXCEPTION: {observation.get('exception')}\n"
            f"STDERR: {(observation.get('stderr') or '')[:800]}\n"
            f"STDOUT: {(observation.get('stdout') or '')[:400]}\n"
            f"TRACEBACK: {(observation.get('traceback') or '')[:800]}\n"
            f"VERIFIER: {json.dumps(observation.get('verifier_result'), default=str)[:600]}"
        )
        self._status("REPAIR")
        self.ledger.log(
            task_id,
            "REPAIR",
            f"prepare repair layer={diagnosis.get('fault_layer')} "
            f"rewrite_skill={diagnosis.get('rewrite_skill')} "
            f"approach={diagnosis.get('approach')} "
            f"adaptive_research={diagnosis.get('adaptive_research')}",
        )
        return (
            diagnosis,
            research,
            str(diagnosis.get("approach") or approach_label),
            last_error,
            code,
            new_args,
        )

    @staticmethod
    def _merge_research(prior: dict, new: dict) -> dict:
        """Merge adaptive research into prior notes — prefer new approach/insights."""
        out = dict(prior or {})
        new = dict(new or {})
        for key in ("approach", "test_idea", "repair_insight", "raw"):
            if new.get(key):
                out[key] = new[key]
        for key in ("libraries", "key_apis", "pitfalls"):
            merged = list(out.get(key) or [])
            for item in new.get(key) or []:
                if item not in merged:
                    merged.append(item)
            out[key] = merged
        for key in ("results", "sources"):
            merged = list(new.get(key) or []) + list(out.get(key) or [])
            # de-dupe by url+title
            seen: set[str] = set()
            uniq = []
            for r in merged:
                if not isinstance(r, dict):
                    continue
                sig = f"{r.get('url')}|{r.get('title')}"
                if sig in seen:
                    continue
                seen.add(sig)
                uniq.append(r)
            out[key] = uniq[:40]
        out["adaptive"] = True
        return out

    def _register_candidate_safe(
        self,
        skill_name: str,
        description: str,
        built_path: str,
        capabilities: list,
        deps_list: list,
        attempt_version: int,
        protect_active_path: Optional[str],
        existing: Optional[dict],
    ) -> None:
        protect = bool(protect_active_path) and Path(str(protect_active_path)).exists()
        keep_active_row = bool(
            existing and existing.get("status") == "ACTIVE" and protect
        )
        pending_meta = {
            "description": description,
            "capabilities": capabilities,
            "dependencies": list(deps_list),
            "version": attempt_version,
            "file_path": built_path,
        }
        if keep_active_row:
            self.registry.register_candidate(
                name=skill_name,
                description=description,
                file_path=built_path,
                capabilities=capabilities,
                dependencies=list(deps_list),
                version=attempt_version,
                protect_active=True,
            )
        elif protect:
            self.registry.register_candidate(
                name=skill_name,
                description=description,
                file_path=str(protect_active_path),
                capabilities=capabilities,
                dependencies=list(deps_list),
                version=existing.get("version") if existing else attempt_version,
                protect_active=False,
            )
            self.registry.set_pending(
                skill_name,
                built_path,
                attempt_version,
                pending_meta,
                keep_file_path=str(protect_active_path),
            )
        else:
            self.registry.register_candidate(
                name=skill_name,
                description=description,
                file_path=built_path,
                capabilities=capabilities,
                dependencies=list(deps_list),
                version=attempt_version,
                protect_active=False,
            )
        self.memory.log_skill_event(
            skill_name, "candidate", "CANDIDATE",
            {"path": built_path, "version": attempt_version, "protected": protect},
        )

    def _activate_and_remember(
        self,
        task_id: str,
        skill_name: str,
        goal: str,
        built: dict,
        meta: dict,
        caps: list,
        deps_list: list,
        attempt_version: int,
        attempt: int,
        approach: str,
        diagnosis: Optional[dict],
        description: str,
    ) -> Optional[dict]:
        self._status("SAVE_SKILL")
        promoted = self.registry.promote_candidate(
            name=skill_name,
            candidate_path=built["path"],
            version=attempt_version,
            description=meta.get("description") or description,
            capabilities=caps,
            dependencies=list(deps_list),
            archive_dir=self.skills_dir / "archive",
        )
        approach_fp = Observer.fingerprint_approach(approach)
        code_fp = Observer.fingerprint_code(built.get("code") or "")
        self.memory.save_solution(
            skill_name=skill_name,
            goal=goal,
            version=attempt_version,
            approach=approach,
            approach_fingerprint=approach_fp,
            diagnosis_summary=str(
                (diagnosis or {}).get("diagnosis")
                or (diagnosis or {}).get("root_cause")
                or "success"
            )[:1000],
            code_fingerprint=code_fp,
            details={
                "attempt": attempt,
                "path": built.get("path"),
                "test_plan": (diagnosis or {}).get("test_plan"),
            },
        )
        self.memory.log_skill_event(skill_name, "activated", "ACTIVE", {
            "version": attempt_version,
            "attempt": attempt,
            "approach": approach,
            "archived_from": promoted.get("archived_from"),
        })
        self.ledger.log(
            task_id,
            "SAVE_SKILL",
            f"{skill_name} → ACTIVE v{attempt_version}",
            {"archived_from": promoted.get("archived_from"), "approach": approach},
        )
        self._log(
            f"[{task_id}] Skill ACTIVE: {skill_name} v{attempt_version} "
            f"approach={approach!r}"
        )
        return self.registry.get_skill(skill_name)

    def _execute_trusted(
        self,
        task_id: str,
        skill: dict,
        goal: str,
        args: Optional[dict] = None,
    ) -> dict:
        self._status("EXECUTE")
        exec_args = dict(args or {})
        # Refresh from goal + skill meta so EXECUTE gets the same structured args as TEST
        exec_args = self.contexts.build(
            goal,
            str(self.workspace),
            mode="execute",
            skill_meta=skill,
            prior_args=exec_args,
        ).get("args") or exec_args
        self._log(
            f"[{task_id}] EXECUTE: {skill['name']} args_keys={list(exec_args.keys())}"
        )
        if skill.get("status") != "ACTIVE":
            return {
                "ok": False,
                "error": f"Refusing non-ACTIVE skill ({skill.get('status')})",
                "evidence": "",
                "result": None,
            }
        result = self.loader.execute(
            skill["name"],
            goal=goal,
            args=exec_args,
            file_path=skill.get("file_path"),
        )
        self.ledger.log(
            task_id,
            "EXECUTE",
            f"ok={result.get('ok')} args_keys={list(exec_args.keys())}",
            {
                "error": result.get("error"),
                "evidence": (result.get("evidence") or "")[:500],
                "args": exec_args,
            },
        )
        return result

    # ── Offline / helpers ───────────────────────────────────────────────

    def _offline_classify(self, text: str) -> dict:
        action_words = (
            "izveido", "uzraksti", "izpildi", "lejupielādē", "saglabā", "pārveido",
            "create", "write", "make", "download", "fetch", "convert", "run",
            "generate", "build", "install", "scrape", "parse", "compute", "calculate",
        )
        low = text.lower()
        if any(w in low for w in action_words):
            return {
                "intent": "task",
                "goal": text,
                "needs_capability": True,
                "keywords": text.split()[:8],
            }
        return {
            "intent": "conversation",
            "goal": text,
            "needs_capability": False,
            "keywords": [],
        }

    def _offline_plan(self, goal: str, caps: list[str]) -> dict:
        matched = self.registry.match_skills(goal.split(), only_active=True)
        # Universal offline arg draft — no task-specific key hardcoding
        draft_args = ContextBuilder._offline_extract(goal)
        return {
            "steps": [f"Handle: {goal}"],
            "can_reuse": [m["name"] for m in matched[:3]],
            "missing": [] if matched else [goal],
            "needs_research": not bool(matched),
            "needs_new_skill": not bool(matched),
            "skill_name": Brain._slug(goal)[:40],
            "skill_description": goal,
            "research_queries": [goal],
            "args": draft_args,
            "required_args": list(draft_args.keys()),
        }

    def _format_status(self) -> str:
        mem = self.memory.stats()
        reg = self.registry.stats()
        led = self.ledger.stats()
        brain = self.brain.model_status()
        return (
            f"Brain (Ollama/{self.brain.model}): {brain}\n"
            f"Memory — conv:{mem['conversations']} exp:{mem['experiences']} "
            f"ok:{mem['successes']} facts:{mem['facts']} "
            f"fail:{mem.get('learning_failures', 0)} "
            f"diag:{mem.get('learning_diagnoses', 0)} "
            f"sol:{mem.get('learning_solutions', 0)}\n"
            f"Skills — total:{reg['total']} ACTIVE:{reg.get('ACTIVE', 0)} "
            f"TESTING:{reg.get('TESTING', 0)} CANDIDATE:{reg.get('CANDIDATE', 0)} "
            f"BROKEN:{reg.get('BROKEN', 0)} REPAIRING:{reg.get('REPAIRING', 0)} "
            f"ARCHIVED:{reg.get('ARCHIVED', 0)} archive_files:{reg.get('archived_versions', 0)}\n"
            f"Ledger — tasks:{led['tasks']} done:{led['done']} fail:{led['failed']} "
            f"actions:{led['actions']}"
        )

    def _format_skills(self) -> str:
        skills = self.registry.list_skills()
        if not skills:
            return "Nav reģistrētu skills. Dod uzdevumu — JARVIS iemācīsies."
        lines = []
        for s in skills:
            lines.append(
                f"[{s['status']}] {s['name']} v{s['version']} "
                f"(ok:{s['success_count']} fail:{s['fail_count']}) — {s['description'][:80]}"
            )
        return "\n".join(lines)

    def get_dashboard_stats(self) -> dict[str, Any]:
        status = self.brain.model_status()
        return {
            "memory": self.memory.stats(),
            "skills": self.registry.stats(),
            "ledger": self.ledger.stats(),
            "brain_online": status == "ONLINE",
            "brain_status": status,
            "model": self.brain.model,
            "skill_list": self.registry.list_skills(),
        }

    def _log(self, msg: str) -> None:
        logger.info(msg)
        try:
            self.on_log(msg)
        except Exception:
            pass

    def _status(self, status: str) -> None:
        try:
            self.on_status(status)
        except Exception:
            pass

    def close(self) -> None:
        self.memory.close()
        self.ledger.close()
        self.registry.close()
