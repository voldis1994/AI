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

logger = logging.getLogger("jarvis.orchestrator")

MAX_REPAIR_ATTEMPTS = 3


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

    # ── Public API ──────────────────────────────────────────────────────

    def handle_user_message(self, text: str) -> dict[str, Any]:
        """Route a user message to conversation or full task cycle."""
        text = (text or "").strip()
        if not text:
            return {"type": "error", "reply": "Empty message."}

        self.memory.add_message("user", text)
        self._status("THINKING")
        self._log(f"USER: {text}")

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
        brain_up = self.brain.is_available()
        if not brain_up:
            # Offline fallback: treat actionable verbs as tasks, else echo
            intent = self._offline_classify(text)
        else:
            intent = self.brain.classify_intent(text)

        if intent.get("intent") == "conversation" and not intent.get("needs_capability"):
            self._status("CONVERSING")
            history = self.memory.chat_history_for_llm(10)
            # history already includes current user msg — drop last duplicate
            history = [m for m in history if not (m["role"] == "user" and m["content"] == text)]
            if brain_up:
                reply = self.brain.converse(text, history=history)
            else:
                reply = (
                    "[Ollama offline] Esmu JARVIS. Palaiž Ollama ar modeli "
                    "qwen2.5-coder:7b, lai runātu un mācītos. "
                    f"Saņēmu: {text}"
                )
            self.memory.add_message("assistant", reply)
            self._status("IDLE")
            self._log(f"JARVIS: {reply[:500]}")
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

            # VERIFY final execution (not just test)
            self._status("VERIFY")
            if not exec_result or not exec_result.get("ok"):
                outcome = (
                    f"Neizdevās izpildīt uzdevumu. "
                    f"Kļūda: {(exec_result or {}).get('error') or 'nav skill'}"
                )
                self.memory.save_experience(
                    goal, outcome, False,
                    skill_name=(skill_record or {}).get("name"),
                    details={"exec": exec_result, "task_id": task_id},
                )
                self.ledger.log(task_id, "SAVE_EXPERIENCE", outcome)
                self.ledger.finish(task_id, False, {"outcome": outcome})
                self._log(f"[{task_id}] FAIL: {outcome}")
                return {
                    "type": "task",
                    "success": False,
                    "task_id": task_id,
                    "reply": outcome,
                    "outcome": outcome,
                }

            verification = self.verifier.verify(
                goal,
                {
                    "ok": exec_result.get("ok"),
                    "passed": exec_result.get("ok") and bool(exec_result.get("evidence")),
                    "result": exec_result.get("result"),
                    "error": exec_result.get("error"),
                    "evidence": exec_result.get("evidence"),
                },
                require_brain_confirm=False,
            )
            self.ledger.log(task_id, "VERIFY", verification.get("reason"), verification)

            if not verification.get("verified"):
                if skill_record:
                    self.registry.mark_failure(
                        skill_record["name"],
                        verification.get("reason") or "verify failed",
                    )
                outcome = (
                    f"Izpilde neattaisnojās verificēšanā: {verification.get('reason')}. "
                    "DONE nav atļauts bez pierādījumiem."
                )
                self.memory.save_experience(
                    goal, outcome, False,
                    skill_name=(skill_record or {}).get("name"),
                    details={"verification": verification, "task_id": task_id},
                )
                self.ledger.log(task_id, "SAVE_EXPERIENCE", outcome)
                self.ledger.finish(task_id, False, {"outcome": outcome})
                return {
                    "type": "task",
                    "success": False,
                    "task_id": task_id,
                    "reply": outcome,
                    "outcome": outcome,
                }

            # SUCCESS path
            if skill_record:
                self.registry.mark_success(skill_record["name"])
            outcome = (
                f"DONE. Uzdevums izpildīts un verificēts.\n"
                f"Skill: {(skill_record or {}).get('name')}\n"
                f"Rezultāts: {exec_result.get('result')!r}\n"
                f"Pierādījums: {exec_result.get('evidence')}"
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
        skill_name = (
            (repair_of or {}).get("name")
            or plan.get("skill_name")
            or self.brain._slug(goal)[:40]
        )
        description = plan.get("skill_description") or goal
        version = self.registry.next_version(skill_name)

        if repair_of:
            self.registry.set_status(skill_name, "REPAIRING")
            self.memory.log_skill_event(skill_name, "repair_start", "REPAIRING")
            self.ledger.log(task_id, "REPAIR", f"Repairing {skill_name}")
            previous_code = self.builder.read_skill_code(skill_name)
            error_log = repair_of.get("last_error") or "Skill marked BROKEN"
        else:
            previous_code = None
            error_log = None

        # RESEARCH
        self._status("RESEARCH")
        queries = plan.get("research_queries") or [goal]
        if not isinstance(queries, list):
            queries = [str(queries)]
        research = self.research.research(queries, goal=goal)
        self.ledger.log(task_id, "RESEARCH", "Research complete", {
            "libraries": research.get("libraries"),
            "sources": len(research.get("sources") or []),
        })

        # LEARN (notes already in research)
        self._status("LEARN")
        self.ledger.log(task_id, "LEARN", research.get("approach", "")[:500])
        self.memory.set_fact(f"research:{skill_name}", research)

        last_error = error_log
        last_code = previous_code
        built = None

        for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
            # BUILD
            self._status("BUILD_SKILL")
            built = self.builder.build(
                skill_name=skill_name,
                description=description,
                research=research,
                version=version,
                previous_code=last_code,
                error_log=last_error,
            )
            self.ledger.log(
                task_id,
                "BUILD_SKILL",
                f"Attempt {attempt}: {'ok' if built['ok'] else built.get('error')}",
                {"path": built.get("path"), "attempt": attempt},
            )
            if not built.get("ok"):
                last_error = built.get("error")
                last_code = built.get("code")
                continue

            meta = built["meta"]
            self.registry.register_candidate(
                name=skill_name,
                description=meta.get("description") or description,
                file_path=built["path"],
                capabilities=meta.get("capabilities") or [description],
                dependencies=meta.get("dependencies") or research.get("libraries") or [],
                version=version,
            )
            self.memory.log_skill_event(skill_name, "candidate", "CANDIDATE", meta)

            # INSTALL DEPS
            self._status("INSTALL_DEPS")
            deps_list = meta.get("dependencies") or research.get("libraries") or []
            dep_result = self.deps.ensure(list(deps_list))
            self.ledger.log(task_id, "INSTALL_DEPS", dep_result.get("details"), dep_result)
            if not dep_result.get("ok"):
                last_error = f"Dependency install failed: {dep_result.get('failed')}"
                last_code = built.get("code")
                self.registry.set_status(skill_name, "BROKEN", last_error)
                continue

            # TEST
            self._status("TEST")
            self.registry.set_status(skill_name, "TESTING")
            self.memory.log_skill_event(skill_name, "testing", "TESTING")
            test_result = self.tester.test(built["path"], goal=goal)
            self.ledger.log(
                task_id,
                "TEST",
                f"passed={test_result.get('passed')}",
                {
                    "error": test_result.get("error"),
                    "evidence": (test_result.get("evidence") or "")[:500],
                },
            )

            if not test_result.get("passed"):
                last_error = (
                    f"{test_result.get('error')}\n{test_result.get('traceback') or ''}"
                )
                last_code = built.get("code")
                self.registry.set_status(skill_name, "BROKEN", last_error[:2000])
                self.memory.log_skill_event(
                    skill_name, "test_failed", "BROKEN", {"error": last_error[:1000]}
                )
                # Analyze for next repair
                if self.brain.is_available() and last_code:
                    analysis = self.brain.analyze_failure(
                        last_code, last_error, test_result.get("evidence") or ""
                    )
                    self.ledger.log(task_id, "REPAIR", analysis.get("diagnosis"), analysis)
                    if analysis.get("needs_new_deps"):
                        self.deps.ensure(list(analysis["needs_new_deps"]))
                        research.setdefault("libraries", [])
                        research["libraries"] = list(
                            set(research["libraries"] + list(analysis["needs_new_deps"]))
                        )
                continue

            # VERIFY test evidence
            self._status("VERIFY")
            verification = self.verifier.verify(goal, test_result)
            self.ledger.log(task_id, "VERIFY", verification.get("reason"), verification)
            if not verification.get("verified"):
                last_error = verification.get("reason")
                last_code = built.get("code")
                self.registry.set_status(skill_name, "BROKEN", str(last_error)[:2000])
                continue

            # SAVE as ACTIVE
            self._status("SAVE_SKILL")
            self.registry.set_status(skill_name, "ACTIVE")
            self.memory.log_skill_event(skill_name, "activated", "ACTIVE", {
                "version": version,
                "attempt": attempt,
            })
            self.ledger.log(task_id, "SAVE_SKILL", f"{skill_name} → ACTIVE v{version}")
            self._log(f"[{task_id}] Skill ACTIVE: {skill_name} v{version}")
            return self.registry.get_skill(skill_name)

        self._log(f"[{task_id}] Failed to produce ACTIVE skill after repairs")
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
        brain = "ONLINE" if self.brain.is_available() else "OFFLINE"
        return (
            f"Brain (Ollama/{self.brain.model}): {brain}\n"
            f"Memory — conv:{mem['conversations']} exp:{mem['experiences']} "
            f"ok:{mem['successes']} facts:{mem['facts']}\n"
            f"Skills — total:{reg['total']} ACTIVE:{reg.get('ACTIVE', 0)} "
            f"TESTING:{reg.get('TESTING', 0)} CANDIDATE:{reg.get('CANDIDATE', 0)} "
            f"BROKEN:{reg.get('BROKEN', 0)} REPAIRING:{reg.get('REPAIRING', 0)}\n"
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
        return {
            "memory": self.memory.stats(),
            "skills": self.registry.stats(),
            "ledger": self.ledger.stats(),
            "brain_online": self.brain.is_available(),
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
