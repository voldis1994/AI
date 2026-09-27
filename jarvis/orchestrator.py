"""
JARVIS orchestrator — the autonomous learning cycle.

REQUEST → PLAN → check capabilities → research → learn → build skill →
install deps → test → repair loop → verify → save ACTIVE skill →
execute original task → save experience → DONE

DONE is only allowed after factual verification.
"""

from __future__ import annotations

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
                exec_result = self._execute_trusted(task_id, skill_record, goal)
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
                    skill_record = self._learn_or_repair(
                        task_id, goal, plan, repair_of=skill_record
                    )
                    if skill_record and skill_record["status"] == "ACTIVE":
                        exec_result = self._execute_trusted(task_id, skill_record, goal)
            else:
                skill_record = self._learn_or_repair(task_id, goal, plan, repair_of=None)
                if skill_record and skill_record["status"] == "ACTIVE":
                    exec_result = self._execute_trusted(task_id, skill_record, goal)

            # VERIFY final execution — SKILL RESULT ≠ VERIFIER RESULT
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
            verification = self.verifier.verify(goal, exec_result, require_brain_confirm=False)
            self.ledger.log(
                task_id,
                "VERIFY",
                verification.get("reason"),
                {
                    "skill_result": verification.get("skill_result"),
                    "verifier_result": verification.get("verifier_result"),
                },
            )
            self._log(
                f"[{task_id}] VERIFIER RESULT: "
                f"{'PASS' if verification.get('verified') else 'FAIL'} — "
                f"{verification.get('reason')}"
            )

            if not verification.get("verified"):
                if skill_record:
                    self.registry.mark_failure(
                        skill_record["name"],
                        verification.get("reason") or "verify failed",
                    )
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
    ) -> Optional[dict]:
        """
        Universal learning loop (no task-specific hardcoding):

        BUILD → TEST → OBSERVE → DIAGNOSE → RESEARCH? → REPAIR → RETEST → VERIFY → ACTIVE
        """
        skill_name = (
            (repair_of or {}).get("name")
            or plan.get("skill_name")
            or self.brain._slug(goal)[:40]
        )
        description = plan.get("skill_description") or goal
        version = self.registry.next_version(skill_name)

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

        for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
            attempt_version = version + (attempt - 1)
            is_retest = attempt > 1
            phase_build = "REPAIR" if is_retest else "BUILD_SKILL"
            phase_test = "RETEST" if is_retest else "TEST"

            # ── BUILD / REPAIR ──────────────────────────────────────────
            self._status(phase_build)
            built = self.builder.build(
                skill_name=skill_name,
                description=description,
                research=research,
                version=attempt_version,
                previous_code=last_code,
                error_log=last_error,
                protect_active_path=protect_active_path,
                diagnosis=diagnosis,
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
                },
            )
            if not built.get("ok"):
                obs = self.observer.observe_failure(
                    goal=goal,
                    skill_name=skill_name,
                    version=attempt_version,
                    phase="BUILD",
                    skill_code=built.get("code"),
                    context={"goal": goal, "workspace": str(self.workspace)},
                    dependencies=list(research.get("libraries") or []),
                    build_error=built.get("error"),
                    prior_approaches=failed_approaches,
                )
                diagnosis, research, current_approach, last_error, last_code = (
                    self._observe_diagnose_enrich(
                        task_id, skill_name, goal, obs, research,
                        approach_label=current_approach,
                        code=built.get("code"),
                    )
                )
                failed_approaches = self.memory.get_failed_approaches(skill_name)
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
                    skill_code=built.get("code"),
                    context={"goal": goal, "workspace": str(self.workspace)},
                    dependencies=deps_list,
                    deps_result=dep_result,
                    build_error=f"Dependency install failed: {dep_result.get('failed')}",
                    prior_approaches=failed_approaches,
                )
                diagnosis, research, current_approach, last_error, last_code = (
                    self._observe_diagnose_enrich(
                        task_id, skill_name, goal, obs, research,
                        approach_label=current_approach,
                        code=built.get("code"),
                    )
                )
                failed_approaches = self.memory.get_failed_approaches(skill_name)
                continue

            # ── TEST / RETEST (subprocess) ──────────────────────────────
            self._status(phase_test)
            protect = bool(protect_active_path) and Path(str(protect_active_path)).exists()
            if not protect:
                self.registry.set_status(skill_name, "TESTING")
            self.memory.log_skill_event(skill_name, "testing", "TESTING")
            test_context = {
                "goal": goal,
                "args": {},
                "workspace": str(self.workspace),
                "mode": "test",
            }
            test_result = self.tester.test(built["path"], goal=goal)
            self.ledger.log(
                task_id,
                phase_test,
                f"skill_ok={test_result.get('ok')} rc={test_result.get('returncode')} "
                f"timed_out={test_result.get('timed_out')}",
                {
                    "error": test_result.get("error"),
                    "stdout": (test_result.get("stdout") or "")[:400],
                    "stderr": (test_result.get("stderr") or "")[:400],
                    "returncode": test_result.get("returncode"),
                    "timed_out": test_result.get("timed_out"),
                    "killed": test_result.get("killed"),
                },
            )
            self._log(
                f"[{task_id}] SKILL RESULT ({phase_test}): ok={test_result.get('ok')} "
                f"rc={test_result.get('returncode')}"
            )

            test_failed = (
                test_result.get("timed_out")
                or test_result.get("crash")
                or not test_result.get("ok")
            )
            verification = None
            if not test_failed:
                # ── VERIFY ──────────────────────────────────────────────
                self._status("VERIFY")
                verification = self.verifier.verify(goal, test_result)
                self.ledger.log(
                    task_id,
                    "VERIFY",
                    verification.get("reason"),
                    {
                        "skill_result": verification.get("skill_result"),
                        "verifier_result": verification.get("verifier_result"),
                    },
                )
                if verification.get("verified"):
                    # ── ACTIVE + remember solution ──────────────────────
                    return self._activate_and_remember(
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

            # ── OBSERVE → DIAGNOSE → (RESEARCH?) → prepare REPAIR ─────
            obs = self.observer.observe_failure(
                goal=goal,
                skill_name=skill_name,
                version=attempt_version,
                phase="VERIFY" if (not test_failed and verification) else phase_test,
                skill_code=built.get("code"),
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
                {"phase": obs["phase"], "exception": obs.get("exception", "")[:500]},
            )
            diagnosis, research, current_approach, last_error, last_code = (
                self._observe_diagnose_enrich(
                    task_id, skill_name, goal, obs, research,
                    approach_label=current_approach,
                    code=built.get("code"),
                )
            )
            failed_approaches = self.memory.get_failed_approaches(skill_name)
            # loop → REPAIR build on next iteration

        self._log(
            f"[{task_id}] Failed to produce ACTIVE skill after "
            f"{MAX_REPAIR_ATTEMPTS} repair attempts"
        )
        return self.registry.get_skill(skill_name)

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
    ) -> tuple[dict, dict, str, str, Optional[str]]:
        """OBSERVE (already done) → DIAGNOSE → optional RESEARCH → return repair state."""
        self._status("OBSERVE")
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
            "artifact_count": len(observation.get("artifacts") or []),
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
            # Offline fallback still forces approach change (universal, not task-specific)
            diagnosis = {
                "root_cause": str(observation.get("exception") or "unknown")[:500],
                "what_to_change": "Rebuild with a different strategy using observation data",
                "approach": f"offline_alt_v{int(observation.get('version') or 0) + 1}",
                "approach_changed": True,
                "needs_research": True,
                "research_queries": [
                    goal,
                    str(observation.get("exception") or "")[:160],
                ],
                "needs_new_deps": [],
                "test_plan": "subprocess retest + independent artifact verification",
                "expected_artifacts": [],
                "is_unfixable": False,
                "diagnosis": str(observation.get("exception") or "failure")[:500],
            }
            diagnosis["approach_fingerprint"] = Observer.fingerprint_approach(
                diagnosis["approach"]
            )

        # If diagnosis repeats a failed approach fingerprint — force divergence
        failed_fps = {a.get("approach_fingerprint") for a in failed}
        if diagnosis.get("approach_fingerprint") in failed_fps:
            diagnosis["approach"] = (
                f"{diagnosis.get('approach')}::changed::{observation.get('version')}"
            )
            diagnosis["approach_fingerprint"] = Observer.fingerprint_approach(
                diagnosis["approach"]
            )
            diagnosis["approach_changed"] = True

        diag_id = self.memory.save_diagnosis(
            skill_name, goal, diagnosis, failure_id=failure_id
        )
        self.ledger.log(
            task_id,
            "DIAGNOSE",
            diagnosis.get("root_cause") or diagnosis.get("diagnosis"),
            {
                "diagnosis_id": diag_id,
                "approach": diagnosis.get("approach"),
                "approach_changed": diagnosis.get("approach_changed"),
                "needs_research": diagnosis.get("needs_research"),
                "needs_new_deps": diagnosis.get("needs_new_deps"),
                "test_plan": diagnosis.get("test_plan"),
            },
        )
        self._log(
            f"[{task_id}] DIAGNOSE: {diagnosis.get('root_cause', '')[:160]} "
            f"| approach={diagnosis.get('approach')!r} "
            f"changed={diagnosis.get('approach_changed')}"
        )

        # RESEARCH if brain says so
        if diagnosis.get("needs_research"):
            queries = diagnosis.get("research_queries") or [goal]
            research = self._do_research(task_id, skill_name, goal, list(queries))

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

        last_error = (
            f"ROOT CAUSE: {diagnosis.get('root_cause')}\n"
            f"CHANGE: {diagnosis.get('what_to_change')}\n"
            f"APPROACH: {diagnosis.get('approach')}\n"
            f"EXCEPTION: {observation.get('exception')}\n"
            f"STDERR: {(observation.get('stderr') or '')[:800]}\n"
            f"STDOUT: {(observation.get('stdout') or '')[:400]}\n"
            f"TRACEBACK: {(observation.get('traceback') or '')[:800]}"
        )
        self._status("REPAIR")
        self.ledger.log(task_id, "REPAIR", f"prepare repair via approach={diagnosis.get('approach')}")
        return (
            diagnosis,
            research,
            str(diagnosis.get("approach") or approach_label),
            last_error,
            code,
        )

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

    def _execute_trusted(self, task_id: str, skill: dict, goal: str) -> dict:
        self._status("EXECUTE")
        self._log(f"[{task_id}] EXECUTE: {skill['name']}")
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
            file_path=skill.get("file_path"),
        )
        self.ledger.log(
            task_id,
            "EXECUTE",
            f"ok={result.get('ok')}",
            {
                "error": result.get("error"),
                "evidence": (result.get("evidence") or "")[:500],
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
        return {
            "steps": [f"Handle: {goal}"],
            "can_reuse": [m["name"] for m in matched[:3]],
            "missing": [] if matched else [goal],
            "needs_research": not bool(matched),
            "needs_new_skill": not bool(matched),
            "skill_name": Brain._slug(goal)[:40],
            "skill_description": goal,
            "research_queries": [goal],
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
