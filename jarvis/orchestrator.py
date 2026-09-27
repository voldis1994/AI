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
import re
import traceback
import uuid
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
from jarvis.task_goal import TaskGoal
from jarvis.intent import IntentClassifier
from jarvis import learning_verify as learn_v

logger = logging.getLogger("jarvis.orchestrator")

MAX_REPAIR_ATTEMPTS = 5
MAX_LEARNING_ATTEMPTS = 5

# Layers that must NEVER trigger a skill rewrite (rewrite only for skill_code).
_NON_REWRITE_LAYERS = frozenset({
    "goal_parsing",
    "context_mapping",
    "execution",
    "environment",
    "verifier",
    # legacy aliases (normalized elsewhere, kept for safety)
    "context_args",
    "test_harness",
    "dependency",
})

# Default skill-oriented test_idea from ResearchSystem — not valid learning notes.
_SKILL_TEST_IDEA_RE = re.compile(
    r"(?i)execute\s+skill\.run|independently verify artifacts"
)


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
        # Isolates the in-flight USER REQUEST (never share final_result across turns)
        self._active_request_id: Optional[str] = None

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
        """
        Route ONE user message in isolation:

        INTENT → correct handler → EXECUTE/ANSWER → own reply → DONE

        Never reuses a previous request's final_result / DONE outcome as the
        answer for a new request (conversation must not echo prior learning).
        """
        text = (text or "").strip()
        request_id = uuid.uuid4().hex[:12]
        if not text:
            return {
                "type": "error",
                "reply": "Empty message.",
                "request_id": request_id,
            }

        # Per-request state — do not read/write cross-request result caches
        self._active_request_id = request_id
        self.memory.add_message(
            "user", text, meta={"request_id": request_id, "phase": "received"}
        )
        self._status("THINKING")
        self._log(f"REQUEST_ID: {request_id} text={text[:120]!r}")
        # Do not _log USER/JARVIS reply here — GUI/CLI already display them once.

        # Fast-path commands
        low = text.lower().strip()
        if low in ("/status", "status"):
            reply = self._format_status()
            return self._finish_request(
                request_id, text, "status", reply, intent_name="status"
            )
        if low in ("/skills", "skills"):
            reply = self._format_skills()
            return self._finish_request(
                request_id, text, "skills", reply, intent_name="skills"
            )
        if low.startswith("/help"):
            reply = (
                "JARVIS commands:\n"
                "  /status  — memory & skill stats\n"
                "  /skills  — list skills & statuses\n"
                "  /help    — this help\n"
                "Or ask anything / give a task to learn & execute."
            )
            return self._finish_request(
                request_id, text, "help", reply, intent_name="help"
            )

        # Classify THIS request independently (never inherit prior skill/result)
        model_status = self.brain.model_status()
        brain_up = model_status == "ONLINE"
        if not brain_up:
            intent = self._offline_classify(text)
        else:
            intent = self.brain.classify_intent(text)
        intent = IntentClassifier.normalize(intent, text)
        intent_name = str(intent.get("intent") or "conversation")
        self._log(
            f"INTENT: {intent_name} request_id={request_id} "
            f"needs_capability={intent.get('needs_capability')} "
            f"goal={intent.get('goal')!r}"
        )

        if intent_name == "conversation":
            self._status("CONVERSING")
            reply = self._answer_conversation(
                text,
                request_id=request_id,
                brain_up=brain_up,
                model_status=model_status,
            )
            out = self._finish_request(
                request_id,
                text,
                "conversation",
                reply,
                intent_name="conversation",
                intent=intent,
                success=True,
            )
            self._status("DONE")
            self._status("IDLE")
            return out

        # Learning / knowledge — isolated cycle; no prior final_result as reply
        if intent_name == "learning":
            goal = intent.get("goal") or text
            result = self.run_learning_cycle(goal, original_request=text)
            reply = result.get("reply") or result.get("outcome") or str(result)
            reply = self._ensure_fresh_reply(
                reply, text, request_id=request_id, intent_name="learning"
            )
            out = self._finish_request(
                request_id,
                text,
                "learning",
                reply,
                intent_name="learning",
                intent=intent,
                success=bool(result.get("success")),
                extra={
                    k: v
                    for k, v in result.items()
                    if k
                    not in (
                        "reply", "outcome", "type", "intent", "request_id",
                        "success",
                    )
                },
            )
            return out

        # Action task — isolated cycle for THIS request only
        goal = intent.get("goal") or text
        result = self.run_cycle(goal, original_request=text)
        reply = result.get("reply") or result.get("outcome") or str(result)
        reply = self._ensure_fresh_reply(
            reply, text, request_id=request_id, intent_name="task"
        )
        out = self._finish_request(
            request_id,
            text,
            "task",
            reply,
            intent_name="task",
            intent=intent,
            success=bool(result.get("success")),
            extra={
                k: v
                for k, v in result.items()
                if k
                not in (
                    "reply", "outcome", "type", "intent", "request_id",
                    "success",
                )
            },
        )
        return out

    def _answer_conversation(
        self,
        text: str,
        *,
        request_id: str,
        brain_up: bool,
        model_status: str,
    ) -> str:
        """Produce a fresh conversational answer for THIS request only."""
        # Exclude learning/task DONE turns so prior final_result cannot leak in
        history = self.memory.conversation_history_for_llm(8)
        history = [
            m for m in history
            if not (m.get("role") == "user" and m.get("content") == text)
        ]
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
        return self._ensure_fresh_reply(
            reply, text, request_id=request_id, intent_name="conversation"
        )

    def _ensure_fresh_reply(
        self,
        reply: Any,
        user_text: str,
        *,
        request_id: str,
        intent_name: str,
    ) -> str:
        """
        Reject stale reuse of a previous learning/task final_result as this
        request's answer (especially for conversation).
        """
        text = str(reply or "").strip()
        prior = self.memory.last_cycle_assistant_reply()
        stale = False
        if prior and text:
            if text == prior.strip():
                stale = True
            elif (
                intent_name == "conversation"
                and Memory._looks_like_cycle_outcome(text)
                and (
                    prior.strip()[:120] in text
                    or text[:120] in prior.strip()
                )
            ):
                stale = True

        if stale and intent_name == "conversation":
            self._log(
                f"STALE_REPLY_REJECTED request_id={request_id} "
                f"— refusing prior cycle final_result as conversation answer"
            )
            # Retry once with zero history (no prior DONE in context)
            if self.brain.is_available():
                try:
                    text = self.brain.converse(user_text, history=[])
                except Exception:
                    text = ""
            if (
                not text
                or text.strip() == (prior or "").strip()
                or Memory._looks_like_cycle_outcome(text)
            ):
                text = (
                    f"Saņēmu jūsu jautājumu: {user_text}\n"
                    "(Iepriekšējā mācīšanās/uzdevuma rezultāts netiek atkārtots — "
                    "atbildi uz šo pieprasījumu.)"
                )
        elif stale:
            self._log(
                f"STALE_REPLY_WARN request_id={request_id} intent={intent_name} "
                f"— reply matched a prior cycle outcome"
            )
        return str(text or "").strip()

    def _finish_request(
        self,
        request_id: str,
        user_text: str,
        result_type: str,
        reply: str,
        *,
        intent_name: str,
        intent: Optional[dict] = None,
        success: Optional[bool] = None,
        extra: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Persist assistant reply tagged to THIS request_id and return payload."""
        meta = {
            "request_id": request_id,
            "intent": intent_name,
            "user_text": user_text[:500],
        }
        if extra and extra.get("task_id") is not None:
            meta["task_id"] = extra.get("task_id")
        self.memory.add_message("assistant", reply, meta=meta)
        if success is False:
            self._status("ERROR")
        elif result_type in ("learning", "task") and success:
            self._status("DONE")
            self._status("IDLE")
        elif result_type not in ("conversation",):
            self._status("IDLE")
        out: dict[str, Any] = {
            "type": result_type,
            "reply": reply,
            "request_id": request_id,
            "intent": intent or {"intent": intent_name, "goal": user_text},
        }
        if success is not None:
            out["success"] = success
        if extra:
            out.update(extra)
        # Clear active id only if we still own the turn
        if getattr(self, "_active_request_id", None) == request_id:
            self._active_request_id = None
        return out

    def run_learning_cycle(
        self, goal: str, original_request: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Self-correcting learning / knowledge path:

        RESEARCH → SAVE → VERIFY
        on FAIL: OBSERVE → diagnose missing knowledge → new research questions
                 → RESEARCH → update knowledge → VERIFY again

        Repeats with a changed approach until PASS or MAX_LEARNING_ATTEMPTS.
        Reuses prior failures + verified topic knowledge. Never repeats a failed
        approach fingerprint. Does not build/repair skills.
        """
        user_request = (original_request or goal or "").strip()
        task_goal = TaskGoal.from_request(user_request, goal=goal)
        goal = task_goal.goal
        topic = IntentClassifier.topic_slug(goal)
        learn_key = f"learning:{topic}"
        task_id = self.ledger.start_task(goal)
        self._status("REQUEST")
        self._log(f"[{task_id}] REQUEST (learning): {task_goal.user_request}")
        self.ledger.log(
            task_id,
            "REQUEST",
            "Learning TaskGoal frozen (self-correcting, no skill inherit)",
            {**task_goal.to_dict(), "topic": topic, "intent": "learning"},
        )

        try:
            # Reuse previously verified knowledge + failed approaches for this topic
            prior_history = self.memory.get_topic_knowledge(topic, limit=8)
            verified_prior = [
                e for e in prior_history
                if isinstance(e, dict) and e.get("verified")
            ]
            failed_approaches = self.memory.get_failed_approaches(learn_key)
            research: dict[str, Any] = {}
            if verified_prior:
                latest = verified_prior[-1]
                research = {
                    "approach": latest.get("approach") or latest.get("summary") or "",
                    "libraries": list(latest.get("libraries") or []),
                    "key_apis": list(latest.get("key_apis") or []),
                    "pitfalls": list(latest.get("pitfalls") or []),
                    "test_idea": latest.get("test_idea") or "",
                    "practical_result": latest.get("practical_result"),
                    "sources": list(latest.get("sources") or []),
                    "results": [],
                    "raw": str(latest.get("summary") or ""),
                    "from_prior_verified": True,
                }
                self._log(
                    f"[{task_id}] Reusing {len(verified_prior)} verified "
                    f"knowledge entries for topic:{topic}"
                )

            queries = self._initial_learning_queries(goal)
            current_approach = "initial_topic_research"
            verification: dict[str, Any] = {}
            saved: dict[str, Any] = {}
            diagnosis: Optional[dict] = None

            for attempt in range(1, MAX_LEARNING_ATTEMPTS + 1):
                # Never reuse a failed approach fingerprint
                ap_fp = Observer.fingerprint_approach(current_approach)
                failed_fps = {
                    str(a.get("approach_fingerprint") or "")
                    for a in failed_approaches
                }
                if ap_fp in failed_fps:
                    current_approach = (
                        f"{current_approach}::alt::{attempt}::{len(failed_fps)}"
                    )
                    ap_fp = Observer.fingerprint_approach(current_approach)

                self._status("RESEARCH")
                self._log(
                    f"[{task_id}] LEARNING RESEARCH attempt={attempt}/"
                    f"{MAX_LEARNING_ATTEMPTS} topic={topic!r} "
                    f"approach={current_approach!r} queries={len(queries)}"
                )
                fresh = self.research.research(queries, goal=goal)
                # Strip skill-oriented defaults; enrich with learning notes
                fresh = self._learning_strip_skill_defaults(fresh)
                fresh = self._enrich_learning_research(
                    fresh,
                    goal=goal,
                    user_request=task_goal.user_request,
                    missing=(diagnosis or {}).get("missing_knowledge") if diagnosis else None,
                    approach_label=current_approach,
                )
                research = (
                    self._merge_research(research, fresh) if research else fresh
                )
                research["approach_label"] = current_approach
                research["attempt"] = attempt
                # If USER REQUEST asks for a concrete answer/result, produce it
                research["practical_result"] = self._produce_learning_practical(
                    task_goal.user_request, research
                )

                # SAVE draft knowledge after every research pass
                saved = self.memory.save_topic_knowledge(
                    topic,
                    research,
                    goal=goal,
                    queries=queries,
                    summary=str(research.get("approach") or ""),
                    verified=False,
                )
                self.ledger.log(task_id, "RESEARCH", "Topic research saved", {
                    "topic": topic,
                    "attempt": attempt,
                    "approach": current_approach,
                    "libraries": research.get("libraries"),
                    "key_apis": list(research.get("key_apis") or [])[:8],
                    "results": len(research.get("results") or []),
                    "knowledge_entries": len(saved.get("history") or []),
                    "queries": queries[:6],
                    "skill_inherit": False,
                })
                self._log(
                    f"[{task_id}] KNOWLEDGE SAVED: topic:{topic} "
                    f"attempt={attempt} entries={len(saved.get('history') or [])} "
                    f"(draft, not bound to any skill)"
                )

                # VERIFY against original TaskGoal
                self._status("VERIFY")
                verification = self._verify_learning_knowledge(
                    task_goal, research, saved.get("entry") or {}
                )
                self.ledger.log(
                    task_id,
                    "VERIFY",
                    verification.get("reason"),
                    {
                        "verifier_result": verification,
                        "topic": topic,
                        "attempt": attempt,
                        "approach": current_approach,
                        "task_goal": task_goal.to_dict(),
                    },
                )
                self._log(
                    f"[{task_id}] LEARNING VERIFY: "
                    f"{'PASS' if verification.get('verified') else 'FAIL'} "
                    f"attempt={attempt} — {verification.get('reason')}"
                )

                if verification.get("verified"):
                    saved = self.memory.save_topic_knowledge(
                        topic,
                        research,
                        goal=goal,
                        queries=queries,
                        summary=str(
                            verification.get("summary")
                            or research.get("approach")
                            or goal
                        ),
                        verified=True,
                    )
                    practical = research.get("practical_result") or {}
                    practical_line = ""
                    if isinstance(practical, dict) and (
                        practical.get("answer") or practical.get("result") is not None
                    ):
                        practical_line = (
                            f"\nPractical result: "
                            f"{practical.get('result', practical.get('answer'))!r}"
                        )
                    outcome = (
                        f"DONE. Learning complete — knowledge verified & saved.\n"
                        f"Topic: {topic}\n"
                        f"Attempts: {attempt}\n"
                        f"Approach: {current_approach}\n"
                        f"Summary: {(saved.get('entry') or {}).get('summary', '')[:400]}\n"
                        f"Sources: {len((research.get('sources') or []))}"
                        f"{practical_line}"
                    )
                    self._status("SAVE_EXPERIENCE")
                    self.memory.save_experience(
                        goal, outcome, True,
                        details={
                            "topic": topic,
                            "verification": verification,
                            "knowledge": saved.get("entry"),
                            "task_id": task_id,
                            "intent": "learning",
                            "attempts": attempt,
                            "approach": current_approach,
                        },
                    )
                    self.ledger.finish(
                        task_id,
                        True,
                        {
                            "topic": topic,
                            "verified": True,
                            "intent": "learning",
                            "attempts": attempt,
                        },
                    )
                    self._status("DONE")
                    self._log(
                        f"[{task_id}] DONE (learning topic={topic} "
                        f"attempts={attempt})"
                    )
                    return {
                        "type": "learning",
                        "success": True,
                        "task_id": task_id,
                        "reply": outcome,
                        "outcome": outcome,
                        "topic": topic,
                        "knowledge": saved.get("entry"),
                        "verification": verification,
                        "attempts": attempt,
                    }

                # ── VERIFY FAIL → OBSERVE → diagnose → new questions ──
                self._status("OBSERVE")
                observation = self.observer.observe_failure(
                    goal=goal,
                    skill_name=learn_key,
                    version=attempt,
                    phase="VERIFY",
                    skill_code="",
                    context={
                        "goal": goal,
                        "user_request": task_goal.user_request,
                        "topic": topic,
                        "approach": current_approach,
                        "queries": queries,
                        "key_apis": list(research.get("key_apis") or []),
                        "mode": "learning",
                    },
                    test_result={
                        "ok": False,
                        "error": verification.get("reason"),
                        "result": {
                            "topic": topic,
                            "key_apis": list(research.get("key_apis") or []),
                        },
                        "evidence": str(research.get("approach") or "")[:1000],
                        "returncode": 0,
                    },
                    verification=verification,
                    prior_approaches=failed_approaches,
                )
                observation["error_fingerprint"] = Observer.fingerprint_error(
                    observation
                )
                failure_id = self.memory.save_failure(
                    learn_key, goal, observation,
                    version=attempt,
                    phase="VERIFY",
                )
                self.memory.record_failed_approach(
                    learn_key,
                    current_approach,
                    ap_fp,
                    last_error=str(verification.get("reason") or "")[:1000],
                )
                self.ledger.log(task_id, "OBSERVE", f"failure_id={failure_id}", {
                    "attempt": attempt,
                    "approach": current_approach,
                    "missing": verification.get("missing_knowledge"),
                    "error_fingerprint": observation.get("error_fingerprint"),
                })

                self._status("DIAGNOSE")
                failed_approaches = self.memory.get_failed_approaches(learn_key)
                diagnosis = self._diagnose_learning_gap(
                    task_goal=task_goal,
                    observation=observation,
                    verification=verification,
                    research=research,
                    failed_approaches=failed_approaches,
                    prior_knowledge=prior_history + list(
                        (saved.get("history") or [])
                    ),
                    attempt=attempt,
                )
                # Enforce approach change — never repeat a failed fingerprint
                new_approach = str(
                    diagnosis.get("approach") or f"learning_repair_v{attempt + 1}"
                )
                new_fp = Observer.fingerprint_approach(new_approach)
                used_fps = {
                    str(a.get("approach_fingerprint") or "")
                    for a in failed_approaches
                }
                if new_fp in used_fps or new_fp == ap_fp:
                    new_approach = (
                        f"{new_approach}::divergent::{attempt + 1}::"
                        f"{observation.get('error_fingerprint', '')[:8]}"
                    )
                    diagnosis["approach"] = new_approach
                    diagnosis["approach_changed"] = True
                diagnosis["approach_fingerprint"] = Observer.fingerprint_approach(
                    diagnosis["approach"]
                )
                self.memory.save_diagnosis(
                    learn_key, goal, diagnosis, failure_id=failure_id
                )
                self.ledger.log(
                    task_id,
                    "DIAGNOSE",
                    diagnosis.get("root_cause") or diagnosis.get("diagnosis"),
                    {
                        "missing_knowledge": diagnosis.get("missing_knowledge"),
                        "approach": diagnosis.get("approach"),
                        "approach_changed": diagnosis.get("approach_changed"),
                        "research_queries": diagnosis.get("research_queries"),
                        "attempt": attempt,
                    },
                )
                self._log(
                    f"[{task_id}] LEARNING DIAGNOSE: "
                    f"{diagnosis.get('root_cause', '')[:160]} "
                    f"| missing={diagnosis.get('missing_knowledge')!r} "
                    f"| next_approach={diagnosis.get('approach')!r}"
                )

                queries = list(diagnosis.get("research_queries") or []) or [
                    f"{goal} — fill gap: {g}"
                    for g in (diagnosis.get("missing_knowledge") or ["coverage"])[:4]
                ]
                if goal not in queries:
                    queries = [goal] + queries
                queries = queries[:6]
                current_approach = str(diagnosis.get("approach") or current_approach)
                # Refresh prior history for next merge
                prior_history = self.memory.get_topic_knowledge(topic, limit=8)

            outcome = (
                f"LEARNING VERIFY FAIL after {MAX_LEARNING_ATTEMPTS} attempts: "
                f"{(verification or {}).get('reason')}. DONE nav atļauts. "
                f"Knowledge draft under topic:{topic}."
            )
            self.memory.save_experience(
                goal, outcome, False,
                details={
                    "topic": topic,
                    "verification": verification,
                    "task_id": task_id,
                    "intent": "learning",
                    "attempts": MAX_LEARNING_ATTEMPTS,
                    "failed_approaches": [
                        a.get("approach") for a in failed_approaches[:10]
                    ],
                },
            )
            self.ledger.finish(
                task_id, False,
                {"outcome": outcome, "topic": topic, "attempts": MAX_LEARNING_ATTEMPTS},
            )
            return {
                "type": "learning",
                "success": False,
                "task_id": task_id,
                "reply": outcome,
                "outcome": outcome,
                "topic": topic,
                "verification": verification,
                "attempts": MAX_LEARNING_ATTEMPTS,
            }
        except Exception as exc:
            tb = traceback.format_exc()
            outcome = f"Learning error: {exc}"
            self._log(f"[{task_id}] EXCEPTION: {tb}")
            self.ledger.finish(task_id, False, {"error": str(exc), "traceback": tb})
            self.memory.save_experience(goal, outcome, False, details={"traceback": tb})
            return {
                "type": "learning",
                "success": False,
                "task_id": task_id,
                "reply": outcome,
                "outcome": outcome,
            }

    def _initial_learning_queries(self, goal: str) -> list[str]:
        queries = [goal]
        if self.brain.is_available():
            try:
                extra = self.brain.generate(
                    f"Learning request:\n{goal}\n\n"
                    "Reply ONLY with JSON: {\"queries\":[\"...\"]}\n"
                    "3-6 focused research queries for this topic. "
                    "Do not mention skills, file creation, or prior capabilities.",
                    system="You plan research queries for knowledge learning.",
                    temperature=0.2,
                )
                parsed = self.brain._parse_json(extra, {"queries": []})
                if isinstance(parsed.get("queries"), list) and parsed["queries"]:
                    queries = [str(q) for q in parsed["queries"][:6]]
            except Exception:
                pass
        if goal not in queries:
            queries = [goal] + [q for q in queries if q != goal]
        return queries[:6]

    @classmethod
    def _learning_strip_skill_defaults(cls, research: dict[str, Any]) -> dict[str, Any]:
        """Remove skill-repair defaults that pollute learning knowledge."""
        out = dict(research or {})
        test_idea = str(out.get("test_idea") or "")
        if _SKILL_TEST_IDEA_RE.search(test_idea):
            out["test_idea"] = ""
        approach = str(out.get("approach") or "")
        if any(
            tok in approach.lower()
            for tok in (
                "skill must define skill_meta",
                "return concrete result.path",
                "implement python solution for",
            )
        ):
            # Keep raw; clear skill-centric approach so enrich can rebuild
            out["approach"] = ""
        return out

    def _enrich_learning_research(
        self,
        research: dict[str, Any],
        *,
        goal: str,
        user_request: Optional[str] = None,
        missing: Optional[list] = None,
        approach_label: str = "",
    ) -> dict[str, Any]:
        """Build structured learning notes that answer the USER REQUEST (no keyword stuffing)."""
        out = dict(research or {})
        missing = [str(m) for m in (missing or []) if m]
        request = (user_request or goal or "").strip()

        if self.brain.is_available():
            try:
                notes = self.brain.research_notes(
                    f"LEARNING (not a skill): {request}\n"
                    f"Missing aspects to fill: "
                    f"{missing or ['core concepts', 'worked examples', 'practice checks']}\n"
                    f"Approach label: {approach_label}\n"
                    "Produce conceptual notes that answer the user goal; "
                    "do not optimize for keyword overlap.",
                    str(out.get("raw") or "")[:8000],
                )
                if isinstance(notes, dict):
                    if notes.get("approach"):
                        out["approach"] = notes["approach"]
                    for key in ("libraries", "key_apis", "pitfalls"):
                        merged = list(out.get(key) or [])
                        for item in notes.get(key) or []:
                            if item not in merged:
                                merged.append(item)
                        out[key] = merged
                    tip = str(notes.get("test_idea") or "")
                    if tip and not _SKILL_TEST_IDEA_RE.search(tip):
                        out["test_idea"] = tip
            except Exception as exc:
                self._log(f"LEARNING enrich brain notes failed ({exc}); offline")

        key_apis = list(out.get("key_apis") or [])
        pitfalls = list(out.get("pitfalls") or [])
        test_idea = str(out.get("test_idea") or "")
        approach = str(out.get("approach") or "").strip()
        raw = str(out.get("raw") or "")

        if not key_apis:
            # Prefer research Hint lines — never stuff raw request tokens as "APIs"
            for line in raw.splitlines():
                s = line.strip().lstrip("-").strip()
                if s.lower().startswith("hint:"):
                    tip = s.split(":", 1)[-1].strip()
                    if tip and tip not in key_apis:
                        key_apis.append(tip)
            if not key_apis and approach:
                # Split approach into short conceptual bullets
                for part in re.split(r"[.;\n]", approach):
                    part = part.strip()
                    if 12 <= len(part) <= 120 and part not in key_apis:
                        key_apis.append(part)
                    if len(key_apis) >= 6:
                        break
            out["key_apis"] = key_apis[:12]

        if not pitfalls and missing:
            out["pitfalls"] = [
                f"Previously missing aspect: {m}" for m in missing[:6]
            ]

        if not test_idea or _SKILL_TEST_IDEA_RE.search(test_idea):
            out["test_idea"] = (
                "Self-check: explain the topic in your own words, "
                "give one worked example, and answer practice questions "
                "that match the USER REQUEST goal."
            )

        if not approach or len(approach) < 40:
            focus = "; ".join(str(x) for x in (out.get("key_apis") or [])[:4])
            out["approach"] = (
                f"{approach_label or 'learning'}: build understanding for the "
                f"user goal — {request}. "
                f"Focus: {focus or 'core concepts and worked examples'}. "
                f"Verify with: {out.get('test_idea')}"
            )[:1200]

        return out

    def _produce_learning_practical(
        self, user_request: str, research: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Produce a concrete answer/result when the USER REQUEST asks for one."""
        if not learn_v.requests_practical_result(user_request):
            # Still useful to keep a short derived answer for semantic verify
            # only when knowledge already has substance — optional.
            return research.get("practical_result")

        knowledge = {
            "approach": research.get("approach"),
            "key_apis": list(research.get("key_apis") or [])[:12],
            "pitfalls": list(research.get("pitfalls") or [])[:8],
            "test_idea": research.get("test_idea"),
            "summary": str(research.get("approach") or "")[:2000],
        }
        if self.brain.is_available():
            try:
                produced = self.brain.produce_learning_answer(user_request, knowledge)
                if isinstance(produced, dict) and (
                    produced.get("answer") or produced.get("result") is not None
                ):
                    # Prefer offline arithmetic when expression present (deterministic)
                    offline = learn_v.produce_practical_offline(user_request, research)
                    if offline.get("source") == "offline_expression":
                        return offline
                    return produced
            except Exception as exc:
                self._log(f"LEARNING practical brain produce failed ({exc}); offline")
        return learn_v.produce_practical_offline(user_request, research)

    def _diagnose_learning_gap(
        self,
        *,
        task_goal: TaskGoal,
        observation: dict[str, Any],
        verification: dict[str, Any],
        research: dict[str, Any],
        failed_approaches: list,
        prior_knowledge: list,
        attempt: int,
    ) -> dict[str, Any]:
        """Identify missing knowledge and propose a NEW research approach."""
        missing = list(verification.get("missing_knowledge") or [])
        if not missing:
            missing = [
                c.get("name")
                for c in (verification.get("checks") or [])
                if not c.get("ok") and c.get("name") != "brain_advisory"
            ]
        failed_labels = [
            str(a.get("approach") or "")
            for a in (failed_approaches or [])
            if a.get("approach")
        ]
        root = str(verification.get("reason") or observation.get("exception") or "")[:500]

        if self.brain.is_available():
            try:
                system = (
                    "You diagnose a failed LEARNING/knowledge verification. "
                    "Reply ONLY with JSON:\n"
                    "{\n"
                    '  "root_cause": "...",\n'
                    '  "missing_knowledge": ["gap1", "gap2"],\n'
                    '  "approach": "NEW strategy label not in failed_approaches",\n'
                    '  "approach_changed": true,\n'
                    '  "research_queries": ["..."],\n'
                    '  "diagnosis": "summary"\n'
                    "}\n"
                    "Rules:\n"
                    "- Identify what knowledge is missing vs the USER REQUEST.\n"
                    "- research_queries must target those gaps (2-6 queries).\n"
                    "- approach MUST differ from every failed_approaches label.\n"
                    "- This is learning — do NOT propose skill code or file creation.\n"
                    "- No topic hardcoding; stay grounded in the request + failure."
                )
                payload = {
                    "user_request": task_goal.user_request,
                    "goal": task_goal.goal,
                    "verification": {
                        "reason": verification.get("reason"),
                        "missing_knowledge": missing,
                        "checks": verification.get("checks"),
                    },
                    "research_summary": {
                        "approach": research.get("approach"),
                        "key_apis": list(research.get("key_apis") or [])[:10],
                        "pitfalls": list(research.get("pitfalls") or [])[:8],
                        "test_idea": research.get("test_idea"),
                    },
                    "failed_approaches": failed_labels[:10],
                    "prior_knowledge_count": len(prior_knowledge or []),
                    "attempt": attempt,
                }
                raw = self.brain.generate(
                    json.dumps(payload, ensure_ascii=False, default=str)[:10000],
                    system=system,
                    temperature=0.3,
                )
                parsed = self.brain._parse_json(raw, {})
                if isinstance(parsed, dict) and parsed.get("research_queries"):
                    parsed.setdefault("missing_knowledge", missing)
                    parsed.setdefault("root_cause", root)
                    parsed.setdefault(
                        "approach",
                        f"fill_gaps_v{attempt + 1}",
                    )
                    parsed["approach_changed"] = True
                    return parsed
            except Exception as exc:
                self._log(f"LEARNING diagnose brain failed ({exc}); offline")

        # Offline gap → query synthesis (universal)
        queries = [
            f"{task_goal.user_request} — deepen: {m}"
            for m in (missing or ["core concepts"])[:4]
        ]
        queries.append(f"fundamentals and practice checks for: {task_goal.goal}")
        for tok in IntentClassifier._keywords(task_goal.user_request)[:4]:
            queries.append(f"{tok} explained with examples and self-test questions")
        # Dedupe
        out_q: list[str] = []
        for q in queries:
            q = " ".join(str(q).split())
            if q and q not in out_q:
                out_q.append(q)
        approach = f"fill_missing_{'+'.join(str(m)[:24] for m in missing[:3]) or 'coverage'}_v{attempt + 1}"
        return {
            "root_cause": root,
            "missing_knowledge": missing or ["goal_coverage", "knowledge_substance"],
            "approach": approach,
            "approach_changed": True,
            "research_queries": out_q[:6],
            "diagnosis": f"Learning verify failed; missing={missing}",
        }

    def _verify_learning_knowledge(
        self,
        task_goal: TaskGoal,
        research: dict[str, Any],
        entry: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Semantically verify learning: USER REQUEST → knowledge → goal answered.

        Does NOT require exact keyword/token matches. If the request also asks
        for a concrete/practical result, that result must be present and correct.
        """
        checks: list[dict[str, Any]] = []
        missing: list[str] = []
        request = task_goal.user_request
        approach = str(research.get("approach") or entry.get("summary") or "")
        practical = research.get("practical_result") or entry.get("practical_result")
        blob = learn_v.knowledge_blob(research, entry)

        # Baseline: real knowledge present (not skill-repair template)
        substance = learn_v.has_substance(research, entry)
        checks.append({
            "name": "knowledge_substance",
            "ok": substance,
            "detail": (
                f"substance={substance} approach_len={len(approach)} "
                f"blob_len={len(blob)}"
            ),
        })
        if not substance:
            missing.append("knowledge_substance")

        sources = list(research.get("sources") or entry.get("sources") or [])
        results = list(research.get("results") or [])
        has_sources = bool(sources) or bool(results) or bool(blob)
        checks.append({
            "name": "knowledge_sources",
            "ok": has_sources,
            "detail": f"sources={len(sources)} results={len(results)}",
        })
        if not has_sources:
            missing.append("knowledge_sources")

        contaminated = bool(learn_v._SKILL_JARGON_RE.search(approach))
        checks.append({
            "name": "no_skill_inherit",
            "ok": not contaminated,
            "detail": (
                "knowledge not bound to unrelated skill repair"
                if not contaminated
                else "knowledge contaminated by prior skill repair jargon"
            ),
        })
        if contaminated:
            missing.append("no_skill_inherit")

        # Semantic goal coverage (brain when available; fuzzy offline otherwise)
        judgment: dict[str, Any]
        if self.brain.is_available() and substance:
            try:
                knowledge_pkg = {
                    "approach": approach,
                    "key_apis": list(research.get("key_apis") or [])[:12],
                    "pitfalls": list(research.get("pitfalls") or [])[:8],
                    "test_idea": research.get("test_idea"),
                    "practical_result": practical,
                    "summary": blob[:3000],
                }
                judgment = self.brain.judge_learning_coverage(request, knowledge_pkg)
            except Exception as exc:
                self._log(f"LEARNING semantic judge failed ({exc}); offline")
                judgment = learn_v.offline_semantic_judgment(request, research, entry)
        else:
            judgment = learn_v.offline_semantic_judgment(request, research, entry)

        covers = bool(judgment.get("covers_goal"))
        checks.append({
            "name": "semantic_goal_coverage",
            "ok": covers,
            "detail": str(judgment.get("reason") or "")[:300],
        })
        if not covers:
            missing.append("goal_coverage")
            for aspect in judgment.get("missing_aspects") or []:
                if aspect and aspect not in missing:
                    missing.append(str(aspect))

        # Practical result when the request asks for a concrete answer
        needs_practical = bool(
            judgment.get("requires_practical_result")
            if judgment.get("requires_practical_result") is not None
            else learn_v.requests_practical_result(request)
        )
        if needs_practical:
            # Prefer model practical_ok when present; always cross-check offline
            offline_p = learn_v.verify_practical_offline(request, practical)
            model_p = judgment.get("practical_ok")
            if model_p is None:
                practical_ok = offline_p["ok"]
                detail = offline_p["detail"]
            else:
                practical_ok = bool(model_p) and offline_p["ok"]
                detail = (
                    f"model_practical_ok={model_p}; offline={offline_p['detail']}; "
                    f"{judgment.get('practical_feedback') or ''}"
                )[:300]
            checks.append({
                "name": "practical_result",
                "ok": practical_ok,
                "detail": detail,
            })
            if not practical_ok:
                for m in offline_p.get("missing") or ["practical_result"]:
                    if m not in missing:
                        missing.append(m)
        else:
            checks.append({
                "name": "practical_result",
                "ok": True,
                "detail": "not required by USER REQUEST",
            })

        verified = all(c["ok"] for c in checks) and bool(checks)
        reason = (
            "LEARNING VERIFY PASS — knowledge semantically covers USER REQUEST"
            if verified
            else "LEARNING VERIFY FAIL — " + "; ".join(
                f"{c['name']}:{c.get('detail')}" for c in checks if not c["ok"]
            )
        )
        return {
            "verified": verified,
            "reason": reason,
            "checks": checks,
            "missing_knowledge": missing,
            "summary": (approach or blob)[:2000],
            "judgment": judgment,
            "practical_result": practical,
        }

    def run_cycle(self, goal: str, original_request: Optional[str] = None) -> dict[str, Any]:
        """Execute the full REQUEST→…→DONE action/skill cycle."""
        # Immutable TaskGoal for the whole cycle — never overwrite user_request.
        user_request = (original_request or goal or "").strip()
        task_goal = TaskGoal.from_request(user_request, goal=goal)
        goal = task_goal.goal  # planner summary; VERIFY always uses task_goal

        task_id = self.ledger.start_task(goal)
        self._status("REQUEST")
        self._log(f"[{task_id}] REQUEST: {task_goal.user_request}")
        self.ledger.log(
            task_id,
            "REQUEST",
            "TaskGoal frozen",
            task_goal.to_dict(),
        )

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

            # CONTEXT: map TaskGoal → skill input schema (no invented defaults)
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
                user_request=task_goal.user_request,
                task_goal=task_goal,
            )
            task_args = dict(ctx0.get("args") or {})
            task_goal = task_goal.with_args(task_args)
            self.ledger.log(
                task_id,
                "CONTEXT",
                f"Mapped TaskGoal → args keys={list(task_args.keys())}",
                {"args": task_args, "constraints": task_goal.constraints},
            )

            # CHECK CAPABILITIES — only skills that fit THIS request
            self._status("CHECK_CAPABILITIES")
            plan = self._sanitize_action_plan(plan, goal)
            reuse = plan.get("can_reuse") or []
            matched = []
            for name in reuse:
                sk = self.registry.get_skill(name)
                if sk and sk["status"] == "ACTIVE" and self._skill_fits_goal(sk, goal):
                    matched.append(sk)
            # Loose keyword fallback only when planner did not demand a new skill
            if not matched and not plan.get("needs_new_skill"):
                keywords = plan.get("research_queries") or [goal]
                tokens = []
                for k in keywords:
                    tokens.extend(str(k).split())
                tokens.extend(str(goal).split())
                for sk in self.registry.match_skills(tokens, only_active=True):
                    if self._skill_fits_goal(sk, goal):
                        matched.append(sk)
            self.ledger.log(
                task_id,
                "CHECK_CAPABILITIES",
                f"Matched {len(matched)} active skills (goal-filtered)",
                {"names": [m["name"] for m in matched], "plan_skill": plan.get("skill_name")},
            )

            skill_record = None
            exec_result = None

            if matched and not plan.get("needs_new_skill"):
                skill_record = matched[0]
                self._log(f"[{task_id}] Reusing ACTIVE skill: {skill_record['name']}")
                # Refresh args with skill meta hints — still grounded to TaskGoal
                task_args = self.contexts.build(
                    goal,
                    str(self.workspace),
                    mode="execute",
                    skill_meta=skill_record,
                    prior_args=task_args,
                    user_request=task_goal.user_request,
                    task_goal=task_goal,
                ).get("args") or task_args
                task_goal = task_goal.with_args(task_args)
                exec_result = self._execute_trusted(
                    task_id, skill_record, goal, args=task_args, task_goal=task_goal
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
                    skill_record, task_args, task_goal = self._learn_or_repair(
                        task_id, goal, plan, repair_of=skill_record,
                        task_args=task_args, task_goal=task_goal,
                    )
                    if skill_record and skill_record["status"] == "ACTIVE":
                        exec_result = self._execute_trusted(
                            task_id, skill_record, goal,
                            args=task_args, task_goal=task_goal,
                        )
            else:
                skill_record, task_args, task_goal = self._learn_or_repair(
                    task_id, goal, plan, repair_of=None,
                    task_args=task_args, task_goal=task_goal,
                )
                if skill_record and skill_record["status"] == "ACTIVE":
                    exec_result = self._execute_trusted(
                        task_id, skill_record, goal,
                        args=task_args, task_goal=task_goal,
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
                user_request=task_goal.user_request,
                constraints=task_goal.with_args(task_args).constraints,
                task_goal=task_goal,
            )
            self.ledger.log(
                task_id,
                "VERIFY",
                verification.get("reason"),
                {
                    "skill_result": verification.get("skill_result"),
                    "verifier_result": verification.get("verifier_result"),
                    "args": task_args,
                    "task_goal": task_goal.to_dict(),
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
                    f"[{task_id}] VERIFY FAIL vs TaskGoal — "
                    f"entering OBSERVE→DIAGNOSE→REPAIR→RETEST"
                )
                skill_record, task_args, task_goal = self._learn_or_repair(
                    task_id,
                    goal,
                    plan,
                    repair_of=skill_record,
                    task_args=task_args,
                    verify_failure=verification,
                    task_goal=task_goal,
                )
                if skill_record and skill_record.get("status") == "ACTIVE":
                    exec_result = self._execute_trusted(
                        task_id, skill_record, goal,
                        args=task_args, task_goal=task_goal,
                    )
                    self._status("VERIFY")
                    verification = self.verifier.verify(
                        goal,
                        exec_result,
                        require_brain_confirm=False,
                        args=task_args,
                        user_request=task_goal.user_request,
                        constraints=task_goal.with_args(task_args).constraints,
                        task_goal=task_goal,
                    )
                    self.ledger.log(
                        task_id,
                        "VERIFY",
                        verification.get("reason"),
                        {
                            "skill_result": verification.get("skill_result"),
                            "verifier_result": verification.get("verifier_result"),
                            "args": task_args,
                            "task_goal": task_goal.to_dict(),
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
        task_goal: Optional[TaskGoal] = None,
    ) -> tuple[Optional[dict], dict, TaskGoal]:
        """
        Universal learning loop (no task-specific hardcoding):

        BUILD → TEST → OBSERVE → DIAGNOSE → RESEARCH? → REPAIR → RETEST → VERIFY → ACTIVE

        TaskGoal (original USER REQUEST) is immutable across the whole loop.
        Returns (skill_record, task_args, task_goal).
        """
        if task_goal is None:
            task_goal = TaskGoal.from_request(goal, goal=goal)
        # Skill name from THIS goal — never inherit an unrelated EXISTING skill
        planned_name = str(plan.get("skill_name") or "").strip()
        if planned_name and not repair_of:
            existing_planned = self.registry.get_skill(planned_name)
            if existing_planned and not self._skill_fits_goal(existing_planned, goal):
                self._log(
                    f"[{task_id}] Ignoring unrelated existing skill_name="
                    f"{planned_name!r} for goal={goal!r}"
                )
                planned_name = ""
        skill_name = (
            (repair_of or {}).get("name")
            or planned_name
            or self.brain._slug(goal)[:40]
            or "new_skill"
        )
        description = plan.get("skill_description") or goal
        version = self.registry.next_version(skill_name)
        args: dict = TaskGoal.ground_args(dict(task_args or {}), task_goal.user_request)
        if isinstance(plan.get("args"), dict):
            args = TaskGoal.ground_args(
                ContextBuilder.merge_args(plan.get("args"), args),
                task_goal.user_request,
            )
        task_goal = task_goal.with_args(args)

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

        # Load previously saved research knowledge, then gather new research
        prior_knowledge = self.memory.get_research_knowledge(skill_name, limit=8)
        prior_research = self.memory.get_latest_research(skill_name)
        research = self._do_research(
            task_id, skill_name, goal, plan.get("research_queries") or [goal]
        )
        if prior_research:
            research = self._merge_research(prior_research, research)
        if prior_knowledge:
            research["knowledge_history"] = prior_knowledge
            self._log(
                f"[{task_id}] Loaded {len(prior_knowledge)} saved knowledge "
                f"entries for {skill_name}"
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
                    "user_request": task_goal.user_request,
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
                    task_goal=task_goal,
                )
            )
            task_goal = task_goal.with_args(args)
            failed_approaches = self.memory.get_failed_approaches(skill_name)
            layer = TaskGoal.normalize_fault_layer(
                diagnosis.get("fault_layer") or "skill_code"
            )
            diagnosis["fault_layer"] = layer
            diagnosis["rewrite_skill"] = TaskGoal.rewrite_skill_for_layer(layer)
            # Rewrite skill ONLY when fault_layer is skill_code
            skip_rebuild = not diagnosis.get("rewrite_skill", False)
            if existing and existing.get("status") == "ACTIVE":
                protect_active_path = existing.get("file_path")
                self.registry.set_status(skill_name, "REPAIRING")

        for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
            attempt_version = version + (attempt - 1)
            is_retest = attempt > 1
            phase_build = "REPAIR" if is_retest else "BUILD_SKILL"
            phase_test = "RETEST" if is_retest else "TEST"

            # Refresh args each attempt — always grounded to original TaskGoal
            args = self.contexts.build(
                goal,
                str(self.workspace),
                mode="test",
                skill_meta={"name": skill_name, "description": description, **meta},
                prior_args=args,
                diagnosis=diagnosis,
                user_request=task_goal.user_request,
                task_goal=task_goal,
            ).get("args") or args
            task_goal = task_goal.with_args(args)

            # ── BUILD / REPAIR (skip unless fault_layer == skill_code) ───
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
                            "user_request": task_goal.user_request,
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
                            task_goal=task_goal,
                        )
                    )
                    task_goal = task_goal.with_args(args)
                    failed_approaches = self.memory.get_failed_approaches(skill_name)
                    skip_rebuild = not diagnosis.get("rewrite_skill", False)
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
                        "user_request": task_goal.user_request,
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
                        task_goal=task_goal,
                    )
                )
                task_goal = task_goal.with_args(args)
                failed_approaches = self.memory.get_failed_approaches(skill_name)
                skip_rebuild = not TaskGoal.rewrite_skill_for_layer(
                    diagnosis.get("fault_layer")
                )
                continue

            # ── TEST / RETEST (subprocess) with prepared args ───────────
            self._status(phase_test)
            protect = bool(protect_active_path) and Path(str(protect_active_path)).exists()
            if not protect:
                self.registry.set_status(skill_name, "TESTING")
            self.memory.log_skill_event(skill_name, "testing", "TESTING")
            test_context = {
                "goal": goal,
                "user_request": task_goal.user_request,
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
                # ── VERIFY against original TaskGoal (not skill claims) ─
                self._status("VERIFY")
                verification = self.verifier.verify(
                    goal,
                    test_result,
                    args=args,
                    user_request=task_goal.user_request,
                    constraints=task_goal.with_args(args).constraints,
                    task_goal=task_goal,
                )
                self.ledger.log(
                    task_id,
                    "VERIFY",
                    verification.get("reason"),
                    {
                        "skill_result": verification.get("skill_result"),
                        "verifier_result": verification.get("verifier_result"),
                        "args": args,
                        "task_goal": task_goal.to_dict(),
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
                    return skill, args, task_goal

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
                    task_goal=task_goal,
                )
            )
            task_goal = task_goal.with_args(args)
            failed_approaches = self.memory.get_failed_approaches(skill_name)
            layer = TaskGoal.normalize_fault_layer(
                diagnosis.get("fault_layer") or "skill_code"
            )
            diagnosis["fault_layer"] = layer
            diagnosis["rewrite_skill"] = TaskGoal.rewrite_skill_for_layer(layer)
            # Only rewrite skill when the fault is skill_code
            skip_rebuild = not diagnosis.get("rewrite_skill", False)
            if layer == "environment" and diagnosis.get("needs_new_deps"):
                self.deps.ensure(list(diagnosis["needs_new_deps"]))
            # legacy alias
            if layer in ("dependency",) and diagnosis.get("needs_new_deps"):
                self.deps.ensure(list(diagnosis["needs_new_deps"]))
            # loop → REPAIR correct layer on next iteration

        self._log(
            f"[{task_id}] Failed to produce ACTIVE skill after "
            f"{MAX_REPAIR_ATTEMPTS} repair attempts"
        )
        return self.registry.get_skill(skill_name), args, task_goal

    def _do_research(
        self, task_id: str, skill_name: str, goal: str, queries: list
    ) -> dict:
        self._status("RESEARCH")
        if not isinstance(queries, list):
            queries = [str(queries)]
        research = self.research.research(queries, goal=goal)
        # Persist knowledge (history + latest snapshot) so repairs can reuse it
        saved = self.memory.save_research_knowledge(
            skill_name, research, goal=goal, queries=queries
        )
        research["knowledge_history"] = saved.get("history") or []
        self.ledger.log(task_id, "RESEARCH", "Research complete", {
            "libraries": research.get("libraries"),
            "results": len(research.get("results") or []),
            "knowledge_saved": True,
            "knowledge_entries": len(research["knowledge_history"]),
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
            "knowledge_entries": len(research["knowledge_history"]),
            "repair_insight": (research.get("repair_insight") or "")[:300],
        })
        self._log(
            f"[{task_id}] KNOWLEDGE SAVED: {skill_name} "
            f"entries={len(research['knowledge_history'])} "
            f"approach={(research.get('approach') or '')[:80]!r}"
        )
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
        task_goal: Optional[TaskGoal] = None,
    ) -> tuple[dict, dict, str, str, Optional[str], dict]:
        """OBSERVE → DIAGNOSE → optional RESEARCH → return repair state + updated args."""
        self._status("OBSERVE")
        if task_goal is None:
            task_goal = TaskGoal.from_request(goal, goal=goal)
        # Ensure observation carries the args + immutable USER REQUEST
        if isinstance(observation.get("context"), dict) and task_args is not None:
            observation["context"] = dict(observation["context"])
            observation["context"]["args"] = dict(task_args)
            observation["context"]["user_request"] = task_goal.user_request
        observation["user_request"] = task_goal.user_request

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
            if not isinstance(ctx_args, dict):
                ctx_args = {}
            empty_args = not ctx_args
            invented = any(
                TaskGoal.is_invented_default(v, task_goal.user_request)
                for v in ctx_args.values()
            )
            err = str(observation.get("exception") or "")
            err_l = err.lower()
            phase = str(observation.get("phase") or "").upper()
            parsed_missing = ContextBuilder.parse_missing_arg_names(err)
            args_fault = (
                empty_args or bool(parsed_missing) or invented
            ) and any(
                t in err_l
                for t in (
                    "argument", "args", "missing", "required", "keyerror",
                    "default", "untrusted", "claim_aligns", "user_provided",
                )
            )
            goal_mismatch = phase == "VERIFY" and any(
                t in err_l
                for t in (
                    "default", "placeholder", "untrusted", "claim_aligns",
                    "not in user", "user constraints", "user request",
                    "reject_defaults", "missing from expected",
                )
            ) and not invented and not empty_args
            no_constraints = phase == "VERIFY" and "no user-derived constraints" in err_l
            env_fault = any(
                t in err_l for t in ("modulenotfound", "no module named", "importerror")
            )
            if no_constraints and empty_args:
                layer = "goal_parsing"
                rewrite = False
                approach = "goal_parsing_repair"
                change = "Re-parse immutable TaskGoal / USER REQUEST into constraints"
            elif args_fault and not goal_mismatch:
                layer = "context_mapping"
                rewrite = False
                approach = "context_mapping"
                change = (
                    "Map TaskGoal → skill args from USER REQUEST "
                    "(no invented defaults)"
                )
            elif env_fault:
                layer = "environment"
                rewrite = False
                approach = "fix_environment"
                change = "Install / fix missing dependencies"
            elif goal_mismatch:
                layer = "skill_code"
                rewrite = True
                approach = "honor_user_request"
                change = (
                    "Rewrite skill to satisfy original TaskGoal "
                    "(no default/placeholder artifacts)"
                )
            else:
                layer = "skill_code"
                rewrite = True
                approach = f"offline_alt_v{int(observation.get('version') or 0) + 1}"
                change = "Rebuild with a different strategy using observation data"
            layer = TaskGoal.normalize_fault_layer(layer)
            rewrite = TaskGoal.rewrite_skill_for_layer(layer)
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
                    "subprocess retest with grounded args; "
                    "VERIFY against original TaskGoal (reject defaults)"
                ),
                "expected_artifacts": [],
                "is_unfixable": False,
                "diagnosis": str(observation.get("exception") or "failure")[:500],
            }
            diagnosis["approach_fingerprint"] = Observer.fingerprint_approach(
                diagnosis["approach"]
            )

        # Universal enrichment: parse missing arg names + suggest values from TaskGoal
        diagnosis = ContextBuilder.enrich_diagnosis_args(
            diagnosis, observation, goal, user_request=task_goal.user_request
        )
        diagnosis["fault_layer"] = TaskGoal.normalize_fault_layer(
            diagnosis.get("fault_layer")
        )
        diagnosis["rewrite_skill"] = TaskGoal.rewrite_skill_for_layer(
            diagnosis["fault_layer"]
        )

        # If diagnosis repeats a failed approach fingerprint — force divergence
        # (but never force-rewrite context_mapping / non-skill layers)
        failed_fps = {a.get("approach_fingerprint") for a in failed}
        if (
            diagnosis.get("approach_fingerprint") in failed_fps
            and diagnosis.get("fault_layer") == "skill_code"
        ):
            diagnosis["approach"] = (
                f"{diagnosis.get('approach')}::changed::{observation.get('version')}"
            )
            diagnosis["approach_fingerprint"] = Observer.fingerprint_approach(
                diagnosis["approach"]
            )
            diagnosis["approach_changed"] = True

        # Rebuild args when fault is context_mapping / goal_parsing (or suggestions)
        new_args = TaskGoal.ground_args(dict(task_args or {}), task_goal.user_request)
        if diagnosis.get("fault_layer") in (
            "context_mapping", "goal_parsing", "context_args"
        ) or diagnosis.get("suggested_args"):
            new_args = self.contexts.build(
                goal,
                str(self.workspace),
                mode="test",
                prior_args=new_args,
                diagnosis=diagnosis,
                user_request=task_goal.user_request,
                task_goal=task_goal,
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
            diagnosis.get("needs_research")
            and layer_now not in ("context_mapping", "context_args", "goal_parsing")
        ) or (
            many_failed_approaches and layer_now == "skill_code"
        )
        # Pure first-time context mapping does not need web research
        if layer_now in ("context_mapping", "context_args", "goal_parsing") and not repeated_error:
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
            research["knowledge_history"] = self.memory.get_research_knowledge(
                skill_name, limit=8
            )
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
        task_goal: Optional[TaskGoal] = None,
    ) -> dict:
        self._status("EXECUTE")
        if task_goal is None:
            task_goal = TaskGoal.from_request(goal, goal=goal)
        exec_args = TaskGoal.ground_args(dict(args or {}), task_goal.user_request)
        # Refresh from TaskGoal + skill meta so EXECUTE matches TEST args
        exec_args = self.contexts.build(
            goal,
            str(self.workspace),
            mode="execute",
            skill_meta=skill,
            prior_args=exec_args,
            user_request=task_goal.user_request,
            task_goal=task_goal,
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
        return IntentClassifier.classify_offline(text)

    def _sanitize_action_plan(self, plan: dict, goal: str) -> dict:
        """Drop unrelated can_reuse / skill_name inherited from prior capabilities."""
        plan = dict(plan or {})
        reuse = []
        for name in plan.get("can_reuse") or []:
            sk = self.registry.get_skill(str(name))
            if sk and self._skill_fits_goal(sk, goal):
                reuse.append(sk["name"])
        plan["can_reuse"] = reuse
        planned = str(plan.get("skill_name") or "").strip()
        if planned:
            existing = self.registry.get_skill(planned)
            # Only reject when the name refers to an EXISTING unrelated skill.
            # Fresh planner names (not yet registered) are kept.
            if existing and not self._skill_fits_goal(existing, goal):
                plan["skill_name"] = Brain._slug(goal)[:40] or "new_skill"
                plan["needs_new_skill"] = True
        else:
            plan["skill_name"] = Brain._slug(goal)[:40] or "new_skill"
        if not reuse and not plan.get("needs_new_skill"):
            # No fitting skill → must build for THIS goal
            plan["needs_new_skill"] = True
            plan["needs_research"] = True
        return plan

    @staticmethod
    def _name_fits_goal(name: str, goal: str) -> bool:
        """True if skill/plan name shares meaningful tokens with the goal."""
        if not name or not goal:
            return False
        name_l = str(name).lower().replace("-", "_")
        goal_l = str(goal).lower()
        if name_l in goal_l:
            return True
        parts = [p for p in re.split(r"[_\s]+", name_l) if len(p) >= 4]
        # Generic verbs alone are not enough to claim a fit
        generic = {
            "create", "write", "make", "build", "file", "files", "skill",
            "data", "test", "tests", "run", "exec", "handle", "new",
        }
        meaningful = [p for p in parts if p not in generic]
        if not meaningful:
            # Name is only generic tokens — require full slug fragment in goal
            return any(p in goal_l for p in parts) and len(parts) >= 2
        hits = sum(1 for p in meaningful if p in goal_l)
        return hits >= 1

    def _skill_fits_goal(self, skill: dict, goal: str) -> bool:
        """Whether an ACTIVE skill is relevant to THIS user goal (not prior context)."""
        if not skill or not goal:
            return False
        if self._name_fits_goal(str(skill.get("name") or ""), goal):
            return True
        hay_parts = [
            str(skill.get("description") or ""),
            *list(skill.get("capabilities") or []),
        ]
        hay = " ".join(hay_parts).lower()
        goal_tokens = [
            t.lower() for t in re.findall(r"[A-Za-z0-9_]{4,}", goal)
            if t.lower() not in {
                "create", "write", "make", "build", "file", "files",
                "with", "that", "this", "from", "into", "please",
                "izveido", "uzraksti", "failu", "saturu", "learn",
                "study", "basics", "knowledge", "using", "about",
            }
        ]
        if not goal_tokens:
            return False
        hits = sum(1 for t in goal_tokens if t in hay or t in str(skill.get("name") or "").lower())
        # Need solid overlap — one shared generic word is not enough
        return hits >= 2 or (
            hits >= 1 and self._name_fits_goal(str(skill.get("name") or ""), goal)
        )

    def _offline_plan(self, goal: str, caps: list[str]) -> dict:
        matched = [
            m for m in self.registry.match_skills(goal.split(), only_active=True)
            if self._skill_fits_goal(m, goal)
        ]
        # Universal offline arg draft — no task-specific key hardcoding
        draft_args = ContextBuilder._offline_extract(goal)
        return {
            "steps": [f"Handle: {goal}"],
            "can_reuse": [m["name"] for m in matched[:3]],
            "missing": [] if matched else [goal],
            "needs_research": not bool(matched),
            "needs_new_skill": not bool(matched),
            "skill_name": Brain._slug(goal)[:40] or "new_skill",
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
