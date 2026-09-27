"""
JARVIS orchestrator — the autonomous learning cycle.

REQUEST → PLAN → check capabilities → research → learn → build/reuse skill →
install deps → TEST skill (process) → EXECUTE original task →
collect artifacts → VERIFY against original TaskGoal → DONE.

TEST PASS is never task success. DONE requires EXECUTE then VERIFY PASS.
On VERIFY FAIL: repair with full TaskGoal context (never unchanged approach).
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
from jarvis.knowledge_artifact import (
    KnowledgeArtifact,
    gap_fill_queries,
    synthesize_offline,
)
from jarvis.recovery import ProgressAwareRecovery
from jarvis.calibration import AutoCalibration
from jarvis.perf import PerfTracker, set_active_tracker
from jarvis.capability_match import (
    evaluate_capability,
    format_decision_log,
    format_match_log,
    verify_implies_capability_mismatch,
)

logger = logging.getLogger("jarvis.orchestrator")

# Defaults — runtime values come from AutoCalibration (persisted, measurable).
MAX_REPAIR_ATTEMPTS = 5
MAX_LEARNING_ATTEMPTS = 5
# Learning: at most one networked research gather; later attempts are local gap-fill.
MAX_LEARNING_NETWORK_RESEARCH = 1

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
        on_event: Optional[Callable[[str, dict], None]] = None,
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
        # Optional structured workspace events for GUI (fire-and-forget)
        self.on_event = on_event or (lambda _k, _p: None)
        # Isolates the in-flight USER REQUEST (never share final_result across turns)
        self._active_request_id: Optional[str] = None

        db = self.data_dir / "jarvis.db"
        self.brain = brain or Brain(on_log=self._log)
        # Multi-model pool logs (ROUTE / WARM / FALLBACK / ESCALATION) → cycle log
        if hasattr(self.brain, "set_logger"):
            self.brain.set_logger(self._log)
        self.memory = Memory(db)
        self.ledger = Ledger(db)
        self.registry = CapabilityRegistry(db)
        # Self-improvement from VERIFY outcomes (survives restart via SQLite facts)
        self.calibration = AutoCalibration(self.memory, on_log=self._log)
        # Apply calibrated timeouts to brain / research (defaults if untouched)
        try:
            self.brain.timeout = float(self.calibration.params.chat_timeout_sec)
        except Exception:
            pass
        # Concurrent persistent model pool — warm workers that fit GPU/RAM.
        # Test doubles (Brain subclasses) skip warm unless they opt in via
        # pool_warm_on_boot=True (avoids multi-minute Ollama preloads in e2e).
        self._pool_report: dict[str, Any] = {}
        if hasattr(self.brain, "ensure_pool_ready"):
            try:
                warm_boot = bool(
                    getattr(self.brain, "pool_warm_on_boot", type(self.brain) is Brain)
                )
                self._pool_report = (
                    self.brain.ensure_pool_ready(warm=warm_boot) or {}
                )
                warmed = self._pool_report.get("warmed") or []
                self._log(
                    f"MODEL POOL: boot online={self._pool_report.get('online')} "
                    f"warm={warm_boot} warmed={len(warmed)} "
                    f"budget_GiB="
                    f"{(self._pool_report.get('budget_bytes') or 0) // (1024**3)}"
                )
            except Exception as exc:
                self._log(f"MODEL POOL: boot warm skipped ({exc})")
        self.research = ResearchSystem(
            brain=self.brain,
            on_log=self._log,
            on_event=self._event,
            timeout=float(self.calibration.research_timeout()),
        )
        self.research.max_open_sources = int(self.calibration.max_open_sources())
        self.deps = DependencyManager(on_log=self._log)
        self.builder = SkillBuilder(
            self.skills_dir,
            brain=self.brain,
            on_log=self._log,
            on_event=self._event,
        )
        self.tester = SkillTester(
            self.workspace, on_log=self._log, on_event=self._event
        )
        self.verifier = Verifier(self.workspace, brain=self.brain, on_log=self._log)
        self.loader = SkillLoader(
            self.skills_dir, self.workspace, on_event=self._event
        )
        self.observer = Observer(self.workspace, on_log=self._log)
        self.contexts = ContextBuilder(brain=self.brain, on_log=self._log)
        # Last EXECUTE+VERIFY bundle for this task (never from TEST alone)
        self._verified_exec_bundle: Optional[dict[str, Any]] = None
        # Per-request reuse cache + performance telemetry
        self._request_cache: dict[str, Any] = {}
        self.perf = PerfTracker()
        self._log(
            f"CALIBRATE: loaded params success_rate="
            f"{self.calibration.window_success_rate():.2f} "
            f"approaches={len(self.calibration.approaches)}"
        )

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
        self._request_cache = {}
        self.perf = PerfTracker(request_id)
        set_active_tracker(self.perf)
        # Pool request context: collect results + dedupe same work across models
        if hasattr(self.brain, "begin_request_pool"):
            try:
                self.brain.begin_request_pool(request_id)
            except Exception:
                pass
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
            from jarvis.model_config import TIER_MODELS, TIER_FAST

            primary = TIER_MODELS[TIER_FAST]["primary"]
            reply = (
                f"[MODEL MISSING] Ollama darbojas, bet katalogā nav neviena "
                f"konfigurēta modeļa. Palaid piem.: ollama pull {primary}. "
                f"Saņēmu: {text}"
            )
        else:
            reply = (
                "[Ollama OFFLINE] Esmu JARVIS. Palaiž Ollama (multi-model: "
                "FAST/REASONING/CODING), lai runātu un mācītos. "
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
        # Emit performance telemetry for this request
        try:
            for line in self.perf.summary_lines():
                self._log(line)
        except Exception:
            pass
        perf_data = {}
        try:
            perf_data = self.perf.as_dict()
        except Exception:
            perf_data = {}
        meta["perf"] = {
            "total_ms": perf_data.get("total_ms"),
            "model_call_count": perf_data.get("model_call_count"),
        }
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
            "perf": perf_data,
        }
        if success is not None:
            out["success"] = success
        if extra:
            out.update(extra)
        # Clear active id only if we still own the turn
        if getattr(self, "_active_request_id", None) == request_id:
            self._active_request_id = None
            set_active_tracker(None)
            self._request_cache = {}
            if hasattr(self.brain, "end_request_pool"):
                try:
                    pool_ctx = self.brain.end_request_pool()
                    if pool_ctx and pool_ctx.get("result_keys"):
                        out["pool_results"] = pool_ctx
                except Exception:
                    pass
        return out

    def run_learning_cycle(
        self, goal: str, original_request: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Self-correcting learning / knowledge path:

        USER REQUEST → research → KnowledgeArtifact synthesis → VERIFY
        on FAIL: DIAGNOSE missing artifact fields → gap-fill only → VERIFY

        VERIFY compares USER REQUEST ↔ clean KnowledgeArtifact only
        (never research/debug/repair logs). Reuses verified topic knowledge
        before new research. Does not build/repair skills.
        """
        user_request = (original_request or goal or "").strip()
        task_goal = TaskGoal.from_request(user_request, goal=goal)
        goal = task_goal.goal
        topic = IntentClassifier.topic_slug(goal)
        learn_key = f"learning:{topic}"
        request_id = uuid.uuid4().hex[:12]
        task_id = self.ledger.start_task(goal)
        self._status("REQUEST")
        self._log(
            f"[{task_id}] REQUEST (learning): {task_goal.user_request} "
            f"request_id={request_id}"
        )
        self.ledger.log(
            task_id,
            "REQUEST",
            "Learning TaskGoal frozen (KnowledgeArtifact path)",
            {
                **task_goal.to_dict(),
                "topic": topic,
                "intent": "learning",
                "request_id": request_id,
            },
        )

        try:
            # ── MEMORY RETRIEVAL (before any SEARCH) ────────────────────
            self._status("MEMORY")
            retrieved = self.memory.retrieve_relevant_knowledge(
                task_goal.user_request,
                topic=topic,
                limit=8,
                verified_only=False,
            )
            self._log(
                f"MEMORY RETRIEVAL: topic={topic} "
                f"entries={retrieved.get('count', 0)} "
                f"verified={retrieved.get('verified_count', 0)} "
                f"request_id={request_id}"
            )
            prior_history = list(retrieved.get("entries") or [])
            # Prefer exact-topic history order for gap diagnostics
            exact_hist = self.memory.get_topic_knowledge(topic, limit=8)
            if exact_hist:
                prior_history = exact_hist
            verified_prior = [
                e for e in (retrieved.get("verified") or prior_history)
                if isinstance(e, dict) and e.get("verified")
            ]
            failed_approaches = self.memory.get_failed_approaches(learn_key)
            artifact = KnowledgeArtifact(
                request_id=request_id,
                topic=topic,
                user_request=task_goal.user_request,
            )
            research: dict[str, Any] = {}
            # Prefer verified knowledge BEFORE any new research
            if verified_prior:
                artifact = KnowledgeArtifact.from_dict(verified_prior[-1])
                artifact.request_id = request_id
                artifact.topic = topic
                artifact.user_request = task_goal.user_request
                artifact.practical_result = self._produce_learning_practical(
                    task_goal.user_request, artifact.research_compat()
                )
                research = self._research_from_artifact(artifact)
                self._log(
                    f"MEMORY USED: verified knowledge for topic:{topic} "
                    f"(skip full research) request_id={request_id}"
                )
                verification = self._verify_learning_knowledge(
                    task_goal, research, verified_prior[-1]
                )
                if verification.get("verified"):
                    self.calibration.observe_verify(
                        verified=True,
                        domain="learning",
                        approach="reuse_verified_artifact",
                        skill_or_topic=topic,
                        extra={"request_id": request_id, "memory_used": True},
                    )
                    return self._finish_learning_success(
                        task_id=task_id,
                        topic=topic,
                        goal=goal,
                        request_id=request_id,
                        artifact=artifact,
                        research=research,
                        verification=verification,
                        attempt=1,
                        approach="reuse_verified_artifact",
                        queries=[],
                        memory_used=True,
                    )
                # Verified blob exists but gaps vs THIS request → search only missing
                missing_from_mem = list(
                    verification.get("missing_knowledge")
                    or artifact.missing_fields(task_goal.user_request)
                )
                self._log(
                    f"MEMORY USED: partial — verified base kept; "
                    f"gap-only SEARCH for {missing_from_mem!r}"
                )
                gap_fill_seed = True
            else:
                gap_fill_seed = False
                missing_from_mem = []

            queries = self._initial_learning_queries(goal)
            if gap_fill_seed and missing_from_mem:
                queries = gap_fill_queries(
                    task_goal.user_request, missing_from_mem
                )[:6]
            current_approach = "knowledge_artifact_synthesis"
            # Prefer historically successful learning approaches when known
            calib_learning = [
                a.approach
                for a in self.calibration.approaches.values()
                if a.domain == "learning" and a.successes > 0 and a.approach
            ]
            if calib_learning:
                current_approach = self.calibration.pick_best_approach(
                    [current_approach] + [str(x) for x in calib_learning[:5]],
                    domain="learning",
                )
            verification: dict[str, Any] = {}
            saved: dict[str, Any] = {}
            diagnosis: Optional[dict] = None
            gap_fill_only = bool(gap_fill_seed)
            network_research_used = 0
            prior_gap_sig = ""
            stagnant_gap_rounds = 0
            learn_max = int(self.calibration.learning_attempts() or MAX_LEARNING_ATTEMPTS)
            net_budget = int(
                self.calibration.learning_network_budget()
                if self.calibration.learning_network_budget() is not None
                else MAX_LEARNING_NETWORK_RESEARCH
            )
            gap_stag_limit = int(self.calibration.gap_stagnation_rounds())
            if gap_fill_seed:
                diagnosis = {
                    "missing_knowledge": list(missing_from_mem),
                    "research_queries": list(queries),
                    "gap_fill_only": True,
                    "approach": "memory_gap_fill",
                }

            for attempt in range(1, learn_max + 1):
                ap_fp = Observer.fingerprint_approach(current_approach)
                failed_fps = {
                    str(a.get("approach_fingerprint") or "")
                    for a in failed_approaches
                }
                # Also avoid approaches calibration marked as repeatedly failing
                if (
                    ap_fp in failed_fps
                    or self.calibration.should_avoid_approach(
                        self.calibration.fingerprint(current_approach, "learning")
                    )
                ):
                    current_approach = (
                        f"{current_approach}::alt::{attempt}::{len(failed_fps)}"
                    )
                    ap_fp = Observer.fingerprint_approach(current_approach)

                missing_gaps = list(
                    (diagnosis or {}).get("missing_knowledge") or []
                )

                # RESEARCH — at most calibrated networked SEARCH→OPEN→EXTRACT;
                # further attempts are local gap-fill only
                use_network = network_research_used < net_budget
                self._status("RESEARCH")
                self._log(
                    f"[{task_id}] LEARNING RESEARCH attempt={attempt}/"
                    f"{learn_max} topic={topic!r} "
                    f"approach={current_approach!r} "
                    f"gap_fill={gap_fill_only} network={use_network} "
                    f"gaps={missing_gaps!r} queries={len(queries)}"
                )
                fresh = self.research.research(
                    queries,
                    goal=goal,
                    mode="learning",
                    network=use_network,
                    use_brain=False,
                    max_queries=3 if use_network else max(1, min(3, len(queries))),
                    append_python=False,
                )
                if use_network:
                    network_research_used += 1
                fresh = self._learning_strip_skill_defaults(fresh)

                # Deterministic KnowledgeArtifact synthesis from EXTRACTED pages
                # (SEARCH snippets alone must not become knowledge)
                artifact = self._synthesize_learning_artifact(
                    user_request=task_goal.user_request,
                    topic=topic,
                    request_id=request_id,
                    research=fresh,
                    prior=artifact if (gap_fill_only or attempt > 1) else None,
                    missing=missing_gaps,
                )
                # Persist opened-source evidence (full-page extracts, not snippets)
                if fresh.get("knowledge_from_extracts"):
                    evidence = [
                        s for s in (fresh.get("sources") or [])
                        if isinstance(s, dict) and s.get("extracted")
                    ]
                    if evidence:
                        artifact.source_evidence = evidence[:15]
                    self._log(
                        f"[{task_id}] SYNTHESIZE: from "
                        f"{sum(1 for e in (fresh.get('extracts') or []) if e.get('ok'))} "
                        f"opened sources (snippets excluded)"
                    )
                # Practical result on the clean artifact (offline first)
                artifact.practical_result = self._produce_learning_practical(
                    task_goal.user_request, artifact.research_compat()
                )
                research = self._research_from_artifact(artifact)
                # Keep scrape metadata out of VERIFY path but retain for ledger
                research["_research_meta"] = {
                    "result_count": len(fresh.get("results") or []),
                    "query_count": len(queries),
                    "gap_fill": gap_fill_only,
                }
                research["approach_label"] = current_approach
                research["attempt"] = attempt
                research["request_id"] = request_id

                saved = self.memory.save_topic_knowledge(
                    topic,
                    research,
                    goal=goal,
                    queries=queries,
                    summary=artifact.summary,
                    verified=False,
                )
                self.ledger.log(task_id, "RESEARCH", "KnowledgeArtifact synthesized", {
                    "topic": topic,
                    "attempt": attempt,
                    "approach": current_approach,
                    "request_id": request_id,
                    "concepts": artifact.concepts[:8],
                    "explanations": len(artifact.explanations),
                    "examples": len(artifact.examples),
                    "gap_fill": gap_fill_only,
                    "knowledge_entries": len(saved.get("history") or []),
                    "queries": queries[:6],
                    "skill_inherit": False,
                })
                self._log(
                    f"[{task_id}] KNOWLEDGE ARTIFACT: topic:{topic} "
                    f"attempt={attempt} concepts={len(artifact.concepts)} "
                    f"explanations={len(artifact.explanations)} "
                    f"examples={len(artifact.examples)} "
                    f"(VERIFY-ready; raw/logs excluded)"
                )

                # VERIFY: USER REQUEST ↔ KnowledgeArtifact only
                self._status("VERIFY")
                verification = self._verify_learning_knowledge(
                    task_goal, research, saved.get("entry") or {}
                )
                self._emit_verify(verification, context="learning")
                self.ledger.log(
                    task_id,
                    "VERIFY",
                    verification.get("reason"),
                    {
                        "verifier_result": verification,
                        "topic": topic,
                        "attempt": attempt,
                        "approach": current_approach,
                        "request_id": request_id,
                        "artifact_fields": {
                            "concepts": len(artifact.concepts),
                            "explanations": len(artifact.explanations),
                            "examples": len(artifact.examples),
                            "has_practical": bool(artifact.practical_result),
                        },
                        "task_goal": task_goal.to_dict(),
                    },
                )
                self._log(
                    f"[{task_id}] LEARNING VERIFY: "
                    f"{'PASS' if verification.get('verified') else 'FAIL'} "
                    f"attempt={attempt} — {verification.get('reason')}"
                )
                # Auto-calibration feedback (measurable VERIFY outcome)
                self.calibration.observe_verify(
                    verified=bool(verification.get("verified")),
                    domain="learning",
                    approach=current_approach,
                    approach_fingerprint=ap_fp,
                    skill_or_topic=topic,
                    extra={"attempt": attempt, "request_id": request_id},
                )

                if verification.get("verified"):
                    artifact.verified = True
                    research = self._research_from_artifact(artifact)
                    return self._finish_learning_success(
                        task_id=task_id,
                        topic=topic,
                        goal=goal,
                        request_id=request_id,
                        artifact=artifact,
                        research=research,
                        verification=verification,
                        attempt=attempt,
                        approach=current_approach,
                        queries=queries,
                    )

                # ── FAIL → OBSERVE → DIAGNOSE concrete artifact gaps ──
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
                        "request_id": request_id,
                        "mode": "learning",
                        "artifact_gaps_hint": artifact.missing_fields(
                            task_goal.user_request
                        ),
                    },
                    test_result={
                        "ok": False,
                        "error": verification.get("reason"),
                        "result": {
                            "topic": topic,
                            "concepts": list(artifact.concepts)[:8],
                        },
                        "evidence": artifact.narrative()[:1000],
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
                new_approach = str(
                    diagnosis.get("approach") or f"gap_fill_v{attempt + 1}"
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
                        "gap_fill_only": True,
                        "attempt": attempt,
                    },
                )
                self._log(
                    f"[{task_id}] LEARNING DIAGNOSE: "
                    f"{diagnosis.get('root_cause', '')[:160]} "
                    f"| missing={diagnosis.get('missing_knowledge')!r} "
                    f"| next_approach={diagnosis.get('approach')!r} "
                    f"(gap-fill only — not full restart)"
                )

                # Next attempt fills ONLY missing fields (local — no full restart)
                queries = list(diagnosis.get("research_queries") or []) or gap_fill_queries(
                    task_goal.user_request,
                    list(diagnosis.get("missing_knowledge") or ["explanations"]),
                )
                queries = queries[:6]
                next_ap = str(diagnosis.get("approach") or current_approach)
                # Prefer calibrated gap-fill strategies over repeatedly failing ones
                current_approach = self.calibration.pick_best_approach(
                    [next_ap, f"gap_fill_v{attempt + 1}", current_approach],
                    domain="learning",
                    failed_fingerprints=failed_fps,
                )
                gap_fill_only = True
                gap_sig = "|".join(
                    sorted(str(g) for g in (diagnosis.get("missing_knowledge") or []))
                )
                if gap_sig and gap_sig == prior_gap_sig:
                    stagnant_gap_rounds += 1
                else:
                    stagnant_gap_rounds = 0
                prior_gap_sig = gap_sig
                if stagnant_gap_rounds >= gap_stag_limit:
                    self._log(
                        f"[{task_id}] LEARNING stagnation — same gaps "
                        f"{gap_sig!r} x{stagnant_gap_rounds}; stopping early"
                    )
                    self.calibration.observe_recovery(
                        stagnated=True,
                        stopped=True,
                        allow_research=False,
                        failure_signature=gap_sig[:64],
                    )
                    break
                prior_history = self.memory.get_topic_knowledge(topic, limit=8)

            outcome = (
                f"LEARNING VERIFY FAIL after {learn_max} attempts: "
                f"{(verification or {}).get('reason')}. DONE nav atļauts. "
                f"Knowledge draft under topic:{topic}."
            )
            self.memory.save_experience(
                goal, outcome, False,
                details={
                    "topic": topic,
                    "verification": verification,
                    "task_id": task_id,
                    "request_id": request_id,
                    "intent": "learning",
                    "attempts": learn_max,
                    "failed_approaches": [
                        a.get("approach") for a in failed_approaches[:10]
                    ],
                    "calibration": self.calibration.status(),
                },
            )
            self.ledger.finish(
                task_id, False,
                {"outcome": outcome, "topic": topic, "attempts": learn_max},
            )
            return {
                "type": "learning",
                "success": False,
                "task_id": task_id,
                "request_id": request_id,
                "reply": outcome,
                "outcome": outcome,
                "topic": topic,
                "verification": verification,
                "attempts": learn_max,
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

    def _synthesize_learning_artifact(
        self,
        *,
        user_request: str,
        topic: str,
        request_id: str,
        research: dict[str, Any],
        prior: Optional[KnowledgeArtifact] = None,
        missing: Optional[list] = None,
    ) -> KnowledgeArtifact:
        """Hookable offline synthesis — tests may override; production stays deterministic."""
        return synthesize_offline(
            user_request=user_request,
            topic=topic,
            request_id=request_id,
            research=research,
            prior=prior,
            missing=missing,
        )

    def _research_from_artifact(self, artifact: KnowledgeArtifact) -> dict[str, Any]:
        """Build VERIFY/memory payload from a clean KnowledgeArtifact (no raw)."""
        data = artifact.research_compat()
        data["artifact"] = artifact.to_dict()
        data["request_id"] = artifact.request_id
        data["raw"] = ""  # hard guarantee — never verify logs
        data["results"] = []
        data["pitfalls"] = []
        return data

    def _finish_learning_success(
        self,
        *,
        task_id: str,
        topic: str,
        goal: str,
        request_id: str,
        artifact: KnowledgeArtifact,
        research: dict[str, Any],
        verification: dict[str, Any],
        attempt: int,
        approach: str,
        queries: list,
        memory_used: bool = False,
    ) -> dict[str, Any]:
        artifact.verified = True
        research = self._research_from_artifact(artifact)
        # Attach extract evidence into source_evidence when present
        if not artifact.source_evidence and research.get("sources"):
            artifact.source_evidence = list(research.get("sources") or [])[:15]
        saved = self.memory.save_topic_knowledge(
            topic,
            research,
            goal=goal,
            queries=queries,
            summary=artifact.summary,
            verified=True,
        )
        self._log(
            f"KNOWLEDGE SAVED: topic={topic} verified=True "
            f"concepts={len(artifact.concepts)} "
            f"sources={len(artifact.source_evidence)} "
            f"request_id={request_id}"
        )
        if memory_used:
            self._log(
                f"MEMORY USED: served from prior verified knowledge "
                f"topic={topic} request_id={request_id}"
            )
        practical = artifact.practical_result or {}
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
            f"Approach: {approach}\n"
            f"Summary: {(saved.get('entry') or {}).get('summary', '')[:400]}\n"
            f"Sources: {len(artifact.source_evidence)}"
            f"{practical_line}"
        )
        self._status("SAVE_EXPERIENCE")
        self.memory.save_experience(
            goal, outcome, True,
            details={
                "topic": topic,
                "verification": verification,
                "knowledge": saved.get("entry"),
                "artifact": artifact.to_dict(),
                "task_id": task_id,
                "request_id": request_id,
                "intent": "learning",
                "attempts": attempt,
                "approach": approach,
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
                "request_id": request_id,
            },
        )
        self._status("DONE")
        self._log(
            f"[{task_id}] DONE (learning topic={topic} attempts={attempt} "
            f"request_id={request_id})"
        )
        return {
            "type": "learning",
            "success": True,
            "task_id": task_id,
            "request_id": request_id,
            "reply": outcome,
            "outcome": outcome,
            "topic": topic,
            "knowledge": saved.get("entry"),
            "artifact": artifact.to_dict(),
            "verification": verification,
            "attempts": attempt,
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
                    work="query_generation",
                    allow_escalate=True,
                    escalate_error_only=True,
                    expect_json=True,
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
        """Remove skill-repair defaults before KnowledgeArtifact synthesis."""
        from jarvis.knowledge_artifact import scrub_text, is_polluted

        out = dict(research or {})
        test_idea = str(out.get("test_idea") or "")
        if _SKILL_TEST_IDEA_RE.search(test_idea) or is_polluted(test_idea):
            out["test_idea"] = ""
        approach = scrub_text(str(out.get("approach") or ""))
        if is_polluted(approach) or any(
            tok in approach.lower()
            for tok in (
                "skill must define skill_meta",
                "return concrete result.path",
                "implement python solution for",
            )
        ):
            approach = ""
        out["approach"] = approach
        # Scrub skill/debug lines from raw — synthesis must not ingest them
        out["raw"] = scrub_text(str(out.get("raw") or ""))
        out["pitfalls"] = [
            p for p in (out.get("pitfalls") or [])
            if p and not is_polluted(str(p)) and "previously missing" not in str(p).lower()
        ]
        out["key_apis"] = [
            k for k in (out.get("key_apis") or [])
            if k and not is_polluted(str(k))
        ]
        # Drop local skill-hint pseudo-results before synthesis
        out["results"] = [
            r for r in (out.get("results") or [])
            if isinstance(r, dict)
            and str(r.get("title") or "").lower() not in ("local capability hints",)
            and not is_polluted(str(r.get("snippet") or r.get("title") or ""))
        ]
        out["sources"] = [
            s for s in (out.get("sources") or [])
            if isinstance(s, dict)
            and str(s.get("title") or "").lower() not in ("local capability hints",)
            and not is_polluted(str(s.get("title") or ""))
        ]
        out["repair_insight"] = ""
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
        """
        Legacy enrich hook — learning path now uses synthesize_offline.

        Kept for tests that override/call it; synthesizes a clean artifact
        offline without skill-repair brain prompts (avoids 30B + pollution).
        """
        from jarvis.knowledge_artifact import scrub_text

        request = (user_request or goal or "").strip()
        topic = IntentClassifier.topic_slug(goal or request)
        cleaned = dict(research or {})
        cleaned["raw"] = scrub_text(str(cleaned.get("raw") or ""))
        art = synthesize_offline(
            user_request=request,
            topic=topic,
            research=cleaned,
            missing=[str(m) for m in (missing or []) if m],
        )
        out = dict(cleaned)
        out.update(art.research_compat())
        out["artifact"] = art.to_dict()
        out["raw"] = cleaned.get("raw") or ""
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
        # Offline Python first — never burn 30B when a local result exists
        offline = learn_v.produce_practical_offline(user_request, research)
        if offline.get("source") == "offline_expression":
            return offline
        ans = str(offline.get("answer") or "").strip()
        if ans and not ans.lower().startswith("insufficient"):
            return offline
        if self.brain.is_available():
            try:
                produced = self.brain.produce_learning_answer(user_request, knowledge)
                if isinstance(produced, dict) and (
                    produced.get("answer") or produced.get("result") is not None
                ):
                    return produced
            except Exception as exc:
                self._log(f"LEARNING practical brain produce failed ({exc}); offline")
        return offline

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
        """
        Identify concrete KnowledgeArtifact gaps (deterministic first).

        When relatedness is high but substance fails, name missing fields
        (concepts/explanations/examples/…) — do not restart full research.
        """
        # Prefer artifact-field gaps over coarse check names
        artifact_gaps = learn_v.diagnose_artifact_gaps(
            task_goal.user_request, research, None
        )
        missing = list(artifact_gaps)
        if not missing:
            missing = list(verification.get("missing_knowledge") or [])
        if not missing:
            missing = [
                c.get("name")
                for c in (verification.get("checks") or [])
                if not c.get("ok") and c.get("name") != "brain_advisory"
            ]
        # Map coarse check names → artifact fields
        mapped: list[str] = []
        for m in missing:
            if m in ("knowledge_substance", "goal_coverage"):
                mapped.extend(artifact_gaps or ["explanations", "concepts"])
            elif m == "practical_result" or str(m).startswith("practical_"):
                mapped.append("practical_result")
            elif m == "no_skill_inherit":
                mapped.append("contamination")
            else:
                mapped.append(str(m))
        # Dedupe
        missing = []
        for m in mapped:
            if m and m not in missing:
                missing.append(m)

        root = str(verification.get("reason") or observation.get("exception") or "")[:500]
        queries = gap_fill_queries(task_goal.user_request, missing)
        approach = (
            f"gap_fill_{'+'.join(str(m)[:20] for m in missing[:3]) or 'fields'}"
            f"_v{attempt + 1}"
        )
        # Deterministic diagnose — skip 30B unless gaps are empty (should not happen)
        return {
            "root_cause": root or f"KnowledgeArtifact missing: {missing}",
            "missing_knowledge": missing or ["explanations"],
            "approach": approach,
            "approach_changed": True,
            "research_queries": queries[:6],
            "diagnosis": (
                f"Gap-fill KnowledgeArtifact fields {missing}; "
                f"do not repeat full research. prior_entries={len(prior_knowledge or [])}"
            ),
            "gap_fill_only": True,
        }

    def _verify_learning_knowledge(
        self,
        task_goal: TaskGoal,
        research: dict[str, Any],
        entry: dict[str, Any],
    ) -> dict[str, Any]:
        """
        VERIFY: original USER REQUEST ↔ clean KnowledgeArtifact only.

        Never judges research raw dumps, skill-repair jargon, failed approaches,
        or diagnostic logs. Deterministic Python checks first; brain advisory only.
        """
        checks: list[dict[str, Any]] = []
        missing: list[str] = []
        request = task_goal.user_request

        # Normalize to artifact payload (strip raw/results if sneaked in)
        verify_research = dict(research or {})
        verify_research["raw"] = ""
        verify_research["results"] = []
        verify_research["pitfalls"] = []
        if isinstance(verify_research.get("artifact"), dict):
            art = KnowledgeArtifact.from_dict(verify_research["artifact"])
            verify_research = self._research_from_artifact(art)
        elif isinstance(entry, dict) and entry.get("artifact"):
            art = KnowledgeArtifact.from_dict(entry["artifact"])
            verify_research = self._research_from_artifact(art)

        approach = str(
            verify_research.get("approach")
            or verify_research.get("summary")
            or entry.get("summary")
            or ""
        )
        practical = (
            verify_research.get("practical_result") or entry.get("practical_result")
        )
        blob = learn_v.knowledge_blob(verify_research, entry)

        substance = learn_v.has_substance(verify_research, entry)
        checks.append({
            "name": "knowledge_substance",
            "ok": substance,
            "detail": (
                f"substance={substance} approach_len={len(approach)} "
                f"blob_len={len(blob)} (artifact-only; raw excluded)"
            ),
        })
        if not substance:
            missing.extend(
                learn_v.diagnose_artifact_gaps(request, verify_research, entry)
            )

        sources = list(
            verify_research.get("sources")
            or verify_research.get("source_evidence")
            or entry.get("sources")
            or []
        )
        # Narrative alone is enough evidence when clean (sources optional)
        has_sources = bool(sources) or bool(blob)
        checks.append({
            "name": "knowledge_sources",
            "ok": has_sources,
            "detail": f"sources={len(sources)} narrative_len={len(blob)}",
        })
        if not has_sources:
            missing.append("knowledge_sources")

        contaminated = bool(learn_v._SKILL_JARGON_RE.search(approach))
        if isinstance(verify_research.get("artifact"), dict):
            contaminated = contaminated or KnowledgeArtifact.from_dict(
                verify_research["artifact"]
            ).is_contaminated()
        checks.append({
            "name": "no_skill_inherit",
            "ok": not contaminated,
            "detail": (
                "KnowledgeArtifact free of skill/debug pollution"
                if not contaminated
                else "KnowledgeArtifact contaminated by skill/debug jargon"
            ),
        })
        if contaminated:
            missing.append("contamination")

        # Prefer deterministic offline semantic judgment (minimize 30B)
        judgment = learn_v.offline_semantic_judgment(request, verify_research, entry)
        # Optional brain advisory only when offline already covers + substance
        if (
            self.brain.is_available()
            and substance
            and judgment.get("covers_goal")
            and False  # keep brain off by default for learning VERIFY
        ):
            try:
                knowledge_pkg = {
                    "approach": approach,
                    "concepts": list(
                        verify_research.get("concepts")
                        or verify_research.get("key_apis")
                        or []
                    )[:12],
                    "explanations": list(verify_research.get("explanations") or [])[:8],
                    "examples": list(verify_research.get("examples") or [])[:6],
                    "test_idea": verify_research.get("practice")
                    or verify_research.get("test_idea"),
                    "practical_result": practical,
                    "summary": blob[:3000],
                }
                judgment = self.brain.judge_learning_coverage(request, knowledge_pkg)
            except Exception as exc:
                self._log(f"LEARNING semantic judge failed ({exc}); offline")

        covers = bool(judgment.get("covers_goal"))
        checks.append({
            "name": "semantic_goal_coverage",
            "ok": covers,
            "detail": str(judgment.get("reason") or "")[:300],
        })
        if not covers:
            # High relatedness + substance fail → concrete artifact gaps
            related = float(judgment.get("relatedness") or 0.0)
            if related >= 0.45 or not substance:
                for g in learn_v.diagnose_artifact_gaps(
                    request, verify_research, entry
                ):
                    if g not in missing:
                        missing.append(g)
            else:
                missing.append("goal_coverage")
            for aspect in judgment.get("missing_aspects") or []:
                if aspect and aspect not in missing and aspect != "knowledge_substance":
                    # Replace coarse substance with field gaps already added
                    if aspect == "knowledge_substance":
                        continue
                    missing.append(str(aspect))

        needs_practical = bool(
            judgment.get("requires_practical_result")
            if judgment.get("requires_practical_result") is not None
            else learn_v.requests_practical_result(request)
        )
        if needs_practical:
            offline_p = learn_v.verify_practical_offline(request, practical)
            practical_ok = offline_p["ok"]
            detail = offline_p["detail"]
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

        # Dedupe missing
        miss_out: list[str] = []
        for m in missing:
            if m and m not in miss_out:
                miss_out.append(m)

        verified = all(c["ok"] for c in checks) and bool(checks)
        reason = (
            "LEARNING VERIFY PASS — KnowledgeArtifact covers USER REQUEST"
            if verified
            else "LEARNING VERIFY FAIL — " + "; ".join(
                f"{c['name']}:{c.get('detail')}" for c in checks if not c["ok"]
            )
        )
        return {
            "verified": verified,
            "reason": reason,
            "checks": checks,
            "missing_knowledge": miss_out,
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
        # Fresh cycle — never carry TEST/prior EXECUTE proof into DONE
        self._verified_exec_bundle = None
        self._status("REQUEST")
        self._log(f"[{task_id}] REQUEST: {task_goal.user_request}")
        self.ledger.log(
            task_id,
            "REQUEST",
            "TaskGoal frozen",
            task_goal.to_dict(),
        )
        # Classify PATH/FILE/LIBRARY/… before any research or install
        self._log_request_items(task_id, task_goal)

        try:
            # MEMORY first — reuse verified knowledge before PLAN/RESEARCH
            self.perf.begin("MEMORY")
            self._status("MEMORY")
            mem_pack = (
                self.memory.retrieve_relevant_knowledge(
                    task_goal.user_request, limit=6
                )
                if hasattr(self.memory, "retrieve_relevant_knowledge")
                else {"entries": [], "verified": []}
            )
            mem_hits = list(
                (mem_pack or {}).get("entries")
                or (mem_pack or {}).get("verified")
                or []
            )
            self._request_cache["memory_hits"] = mem_pack
            if mem_hits:
                self._log(
                    f"[{task_id}] MEMORY: {len(mem_hits)} relevant knowledge hit(s) "
                    f"— prefer reuse before research"
                )
            self.perf.end("MEMORY", hits=len(mem_hits))

            # PLAN (FAST / offline — never burn REASONING for skeleton)
            self.perf.begin("PLAN")
            self._status("PLAN")
            caps = self.registry.active_capabilities()
            cache_key = f"plan:{task_goal.user_request}"
            if cache_key in self._request_cache:
                plan = dict(self._request_cache[cache_key])
                self._log(f"[{task_id}] PLAN: cache hit (reuse within request)")
            else:
                # Deterministic baseline first; optionally refine with FAST
                plan = self._offline_plan(goal, caps)
                if self.brain.is_available() and not plan.get("can_reuse"):
                    try:
                        llm_plan = self.brain.plan(goal, caps)
                        if isinstance(llm_plan, dict) and llm_plan.get("steps"):
                            # Merge — keep offline grounded args if LLM invented word-keys
                            offline_args = TaskGoal.sanitize_args(
                                dict(plan.get("args") or {}),
                                task_goal.user_request,
                            )
                            llm_args = TaskGoal.sanitize_args(
                                dict(llm_plan.get("args") or {}),
                                task_goal.user_request,
                            )
                            merged = dict(plan)
                            merged.update({
                                k: v for k, v in llm_plan.items()
                                if k not in ("args", "required_args")
                            })
                            merged["args"] = {**offline_args, **llm_args}
                            merged["required_args"] = list(
                                dict.fromkeys(
                                    list(merged["args"].keys())
                                    + list(llm_plan.get("required_args") or [])
                                    + list(plan.get("required_args") or [])
                                )
                            )
                            # Drop polluted required_args that are request tokens
                            merged["required_args"] = [
                                k for k in merged["required_args"]
                                if not TaskGoal.is_polluted_arg_key(
                                    k, task_goal.user_request
                                )
                            ]
                            plan = merged
                    except Exception as exc:
                        self._log(f"[{task_id}] PLAN: FAST refine failed ({exc})")
                self._request_cache[cache_key] = dict(plan)
            self.ledger.log(task_id, "PLAN", "Plan created", plan)
            self._log(f"[{task_id}] PLAN: {plan.get('steps')}")
            self.perf.end("PLAN")

            # CONTEXT: map TaskGoal → skill input schema (no invented defaults)
            self.perf.begin("CONTEXT")
            task_args = {}
            if isinstance(plan.get("args"), dict):
                task_args.update(plan["args"])
            # Filter polluted required_args before diagnosis fill
            req_args = [
                k for k in (plan.get("required_args") or [])
                if not TaskGoal.is_polluted_arg_key(str(k), task_goal.user_request)
            ]
            ctx0 = self.contexts.build(
                goal,
                str(self.workspace),
                mode="plan",
                prior_args=task_args,
                diagnosis={"required_args": req_args},
                user_request=task_goal.user_request,
                task_goal=task_goal,
            )
            task_args = dict(ctx0.get("args") or {})
            task_goal = task_goal.with_args(task_args)
            self.ledger.log(
                task_id,
                "CONTEXT",
                f"Mapped TaskGoal → args keys={list(task_args.keys())}",
                {
                    "args": task_args,
                    "constraints": task_goal.constraints,
                    "actions": list(task_goal.actions),
                    "artifacts": list(task_goal.artifacts),
                    "content_requirements": list(task_goal.content_requirements),
                    "success_criteria": list(task_goal.success_criteria),
                },
            )
            self.perf.end("CONTEXT", arg_count=len(task_args))

            # CHECK CAPABILITIES — TaskGoal-compatible skills only (not name similarity)
            self._status("CHECK_CAPABILITIES")
            plan = self._sanitize_action_plan(plan, goal, task_goal=task_goal)
            matched, match_reports = self._select_compatible_skills(
                plan, task_goal, task_id=task_id
            )
            decision = "REUSE" if (matched and not plan.get("needs_new_skill")) else "BUILD_NEW"
            self._log(
                format_decision_log(
                    decision,
                    skill=matched[0]["name"] if matched else "",
                    detail=f"candidates={len(matched)}",
                )
            )
            self.ledger.log(
                task_id,
                "CHECK_CAPABILITIES",
                f"decision={decision} matched={len(matched)}",
                {
                    "decision": decision,
                    "names": [m["name"] for m in matched],
                    "plan_skill": plan.get("skill_name"),
                    "matches": match_reports[:8],
                },
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
                    # Compatible skill failed execution → REPAIR (not BUILD_NEW)
                    self.registry.mark_failure(
                        skill_record["name"],
                        str(exec_result.get("error") or "execution failed"),
                    )
                    self.memory.log_skill_event(
                        skill_record["name"], "broken_detected", "BROKEN", exec_result
                    )
                    self._log(
                        format_decision_log(
                            "REPAIR",
                            skill=skill_record["name"],
                            detail="execution failed on compatible skill",
                        )
                    )
                    self._log(
                        f"[{task_id}] Skill broken — entering repair path"
                    )
                    skill_record, task_args, task_goal = self._learn_or_repair(
                        task_id, goal, plan, repair_of=skill_record,
                        task_args=task_args, task_goal=task_goal,
                    )
                    if skill_record and skill_record["status"] == "ACTIVE":
                        exec_result = self._exec_after_repair(
                            task_id, skill_record, goal, task_args, task_goal
                        )
            else:
                self._log(
                    format_decision_log(
                        "BUILD_NEW",
                        detail="no TaskGoal-compatible ACTIVE skill",
                    )
                )
                skill_record, task_args, task_goal = self._learn_or_repair(
                    task_id, goal, plan, repair_of=None,
                    task_args=task_args, task_goal=task_goal,
                )
                if skill_record and skill_record["status"] == "ACTIVE":
                    exec_result = self._exec_after_repair(
                        task_id, skill_record, goal, task_args, task_goal
                    )

            # VERIFY only after EXECUTE of the original task (TEST ≠ task success)
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
                f"[{task_id}] SKILL RESULT (EXECUTE): ok={exec_result.get('ok')} "
                f"rc={exec_result.get('returncode')} error={exec_result.get('error')}"
            )
            # Reuse VERIFY from EXECUTE+VERIFY bundle when learn loop already proved it
            proof_v = None
            if (
                isinstance(exec_result, dict)
                and exec_result.get("_from_verified_exec_bundle")
                and isinstance(exec_result.get("_verification"), dict)
                and exec_result["_verification"].get("verified")
            ):
                proof_v = exec_result["_verification"]
            # Collect workspace artifacts for VERIFY / repair context
            actual_artifacts = (
                list(exec_result.get("_actual_artifacts") or [])
                if proof_v is not None
                else self._list_workspace_artifacts()
            )
            if not actual_artifacts:
                actual_artifacts = self._list_workspace_artifacts()
            self._log(
                f"[{task_id}] ARTIFACTS after EXECUTE: "
                f"{len(actual_artifacts)} file(s)"
            )
            if proof_v is not None:
                verification = proof_v
                self.ledger.log(
                    task_id,
                    "VERIFY",
                    verification.get("reason"),
                    {
                        "skill_result": verification.get("skill_result"),
                        "verifier_result": verification.get("verifier_result"),
                        "args": task_args,
                        "task_goal": task_goal.to_dict(),
                        "actual_artifacts": actual_artifacts[:20],
                        "after_execute": True,
                        "reused_verified_exec_bundle": True,
                    },
                )
            else:
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
                        "actual_artifacts": actual_artifacts[:20],
                        "after_execute": True,
                    },
                )
            self._emit_verify(verification, context="execute")
            self._log(
                f"[{task_id}] VERIFIER RESULT: "
                f"{'PASS' if verification.get('verified') else 'FAIL'} — "
                f"{verification.get('reason')}"
            )
            self.calibration.observe_verify(
                verified=bool(verification.get("verified")),
                domain="repair",
                approach=str((skill_record or {}).get("name") or "execute"),
                skill_or_topic=str((skill_record or {}).get("name") or ""),
                extra={"phase": "outer_verify", "task_id": task_id},
            )

            # VERIFY FAIL → capability mismatch? BUILD_NEW : REPAIR compatible skill
            if not verification.get("verified"):
                repair_target = skill_record
                if skill_record and task_goal is not None:
                    compat = evaluate_capability(skill_record, task_goal)
                    self._log(format_match_log(compat))
                    mismatch = verify_implies_capability_mismatch(
                        verification, skill_record, task_goal
                    )
                    if mismatch or not compat.get("compatible"):
                        self._log(
                            format_decision_log(
                                "BUILD_NEW",
                                skill=skill_record["name"],
                                detail=(
                                    "VERIFY fail + capability mismatch — "
                                    "will not repair unrelated skill"
                                ),
                            )
                        )
                        # Do not mutate the unrelated specialized skill
                        plan = dict(plan or {})
                        plan["needs_new_skill"] = True
                        plan["needs_research"] = True
                        plan["can_reuse"] = []
                        plan["skill_name"] = (
                            Brain._slug(task_goal.user_request)[:40] or "new_skill"
                        )
                        repair_target = None
                    else:
                        self._log(
                            format_decision_log(
                                "REPAIR",
                                skill=skill_record["name"],
                                detail="VERIFY fail on TaskGoal-compatible skill",
                            )
                        )
                if skill_record and repair_target is not None:
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
                    + (
                        "BUILD_NEW (capability mismatch)"
                        if repair_target is None
                        else "entering OBSERVE→DIAGNOSE→REPAIR→RETEST"
                    )
                )
                skill_record, task_args, task_goal = self._learn_or_repair(
                    task_id,
                    goal,
                    plan,
                    repair_of=repair_target,
                    task_args=task_args,
                    verify_failure=verification if repair_target is not None else None,
                    task_goal=task_goal,
                )
                if skill_record and skill_record.get("status") == "ACTIVE":
                    # EXECUTE+VERIFY after repair — reuse bundle only if post-EXECUTE
                    exec_result = self._exec_after_repair(
                        task_id, skill_record, goal, task_args, task_goal
                    )
                    self._status("VERIFY")
                    if (
                        isinstance(exec_result, dict)
                        and exec_result.get("_from_verified_exec_bundle")
                        and isinstance(exec_result.get("_verification"), dict)
                    ):
                        verification = exec_result["_verification"]
                        actual_artifacts = list(
                            exec_result.get("_actual_artifacts")
                            or self._list_workspace_artifacts()
                        )
                        self._log(
                            f"[{task_id}] Reusing EXECUTE+VERIFY PASS "
                            f"(after outer VERIFY FAIL repair)"
                        )
                    else:
                        actual_artifacts = self._list_workspace_artifacts()
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
                            "actual_artifacts": actual_artifacts[:20],
                            "after_repair": True,
                            "after_execute": True,
                        },
                    )
                    self._log(
                        f"[{task_id}] VERIFIER RESULT (after repair): "
                        f"{'PASS' if verification.get('verified') else 'FAIL'} — "
                        f"{verification.get('reason')}"
                    )
                    self.calibration.observe_verify(
                        verified=bool(verification.get("verified")),
                        domain="repair",
                        approach=str((skill_record or {}).get("name") or "repair"),
                        skill_or_topic=str((skill_record or {}).get("name") or ""),
                        extra={"phase": "after_repair_verify", "task_id": task_id},
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

        BUILD/REPAIR → TEST (process only) → EXECUTE original task →
        collect artifacts → VERIFY vs TaskGoal → ACTIVE + verified bundle.

        TEST PASS never activates or marks task success. On VERIFY FAIL:
        OBSERVE → DIAGNOSE → REPAIR with full TaskGoal context (never an
        unchanged repair approach). Outer run_cycle may reuse the verified
        EXECUTE bundle for DONE, or re-enter here with verify_failure.

        TaskGoal (original USER REQUEST) is immutable across the whole loop.
        Returns (skill_record, task_args, task_goal).
        """
        if task_goal is None:
            task_goal = TaskGoal.from_request(goal, goal=goal)
        # Skill name from THIS goal — never inherit an unrelated EXISTING skill
        planned_name = str(plan.get("skill_name") or "").strip()
        if planned_name and not repair_of:
            existing_planned = self.registry.get_skill(planned_name)
            if existing_planned and not self._skill_fits_goal(
                existing_planned, goal, task_goal=task_goal
            ):
                self._log(
                    f"[{task_id}] Ignoring unrelated existing skill_name="
                    f"{planned_name!r} for goal={goal!r}"
                )
                self._log(
                    format_decision_log(
                        "BUILD_NEW",
                        skill=planned_name,
                        detail="planned skill incompatible with TaskGoal",
                    )
                )
                planned_name = ""
        skill_name = (
            (repair_of or {}).get("name")
            or planned_name
            or self.brain._slug(task_goal.user_request or goal)[:40]
            or "new_skill"
        )
        description = plan.get("skill_description") or goal
        version = self.registry.next_version(skill_name)
        args: dict = TaskGoal.sanitize_args(
            dict(task_args or {}), task_goal.user_request
        )
        if isinstance(plan.get("args"), dict):
            args = TaskGoal.sanitize_args(
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

        # Progress-aware recovery — params from AutoCalibration (measurable history)
        recovery = ProgressAwareRecovery(**self.calibration.recovery_kwargs())

        # Load prior knowledge. Research is NOT a universal entry fallback —
        # only gather when we lack knowledge for a new skill (not mid-repair).
        prior_knowledge = self.memory.get_research_knowledge(skill_name, limit=8)
        prior_research = self.memory.get_latest_research(skill_name)
        research: dict[str, Any] = {
            "approach": "coding_first",
            "libraries": [],
            "key_apis": [],
            "pitfalls": [],
            "test_idea": "",
            "raw": "",
            "sources": [],
            "results": [],
        }
        need_entry_research = (
            not repair_of
            and not verify_failure
            and not prior_knowledge
            and not prior_research
            and bool(plan.get("needs_research", True))
        )
        if need_entry_research:
            research = self._do_research(
                task_id, skill_name, goal, plan.get("research_queries") or [goal],
                task_goal=task_goal,
            )
            recovery.research_count += 1
        elif prior_research:
            research = dict(prior_research)
            self._log(
                f"[{task_id}] RECOVERY: reuse prior research "
                f"(skip universal entry RESEARCH)"
            )
        else:
            self._log(
                f"[{task_id}] RECOVERY: CODING-first — no entry RESEARCH "
                f"(research only on diagnosed knowledge gap)"
            )
        if prior_research and need_entry_research:
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
            expected_arts = list(task_goal.artifacts)
            actual_arts = self._list_workspace_artifacts()
            seed_obs = self.observer.observe_failure(
                goal=goal,
                skill_name=skill_name,
                version=int(repair_of.get("version") or version),
                phase="VERIFY",
                skill_code=previous_code,
                context={
                    "goal": goal,
                    "user_request": task_goal.user_request,
                    "task_goal": task_goal.to_dict(),
                    "args": args,
                    "workspace": str(self.workspace),
                    "expected_artifacts": expected_arts,
                    "actual_artifacts": actual_arts[:40],
                    "content_requirements": list(task_goal.content_requirements),
                    "success_criteria": list(task_goal.success_criteria),
                    "verifier_error": str(verify_failure.get("reason") or "")[:2000],
                    "previous_failures": failed_approaches[:8],
                },
                dependencies=list(repair_of.get("dependencies") or []),
                test_result={
                    "ok": bool((verify_failure.get("skill_result") or {}).get("ok", True)),
                    "error": verify_failure.get("reason"),
                    "result": (verify_failure.get("skill_result") or {}).get("result"),
                    "evidence": (verify_failure.get("skill_result") or {}).get("evidence"),
                    "stdout": (verify_failure.get("skill_result") or {}).get("stdout"),
                    "stderr": (verify_failure.get("skill_result") or {}).get("stderr"),
                    "returncode": (verify_failure.get("skill_result") or {}).get(
                        "returncode", 0
                    ),
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
                    recovery=recovery,
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
            if diagnosis.get("stop_recovery"):
                self._log(
                    f"[{task_id}] RECOVERY STOP (seed): "
                    f"{(diagnosis.get('recovery_report') or {}).get('focus_signature')}"
                )
                return self.registry.get_skill(skill_name), args, task_goal

        repair_max = int(self.calibration.repair_attempts() or MAX_REPAIR_ATTEMPTS)
        for attempt in range(1, repair_max + 1):
            if recovery.stopped:
                break
            attempt_version = version + (attempt - 1)
            is_retest = attempt > 1
            phase_build = "REPAIR" if is_retest else "BUILD_SKILL"
            phase_test = "RETEST" if is_retest else "TEST"

            # Refresh args each attempt — always grounded to original TaskGoal
            skill_meta_ctx = {
                "name": skill_name,
                "description": description,
                **(meta or {}),
            }
            args = self.contexts.build(
                goal,
                str(self.workspace),
                mode="test",
                skill_meta=skill_meta_ctx,
                prior_args=args,
                diagnosis=diagnosis,
                user_request=task_goal.user_request,
                task_goal=task_goal,
            ).get("args") or args
            task_goal = task_goal.with_args(args, skill_meta=skill_meta_ctx)

            # ── BUILD / REPAIR (skip unless fault_layer == skill_code) ───
            if not skip_rebuild or built is None or not built.get("ok"):
                self._status(phase_build)
                self.perf.begin(phase_build)
                missing_reqs = self._missing_requirements_from_diagnosis(
                    diagnosis, task_goal
                )
                artifacts_now = self._list_workspace_artifacts()
                # Avoid identical CODING approach when calibration says avoid
                if diagnosis and self.calibration.should_avoid_approach(
                    Observer.fingerprint_approach(
                        str(diagnosis.get("approach") or current_approach)
                    )
                ):
                    alt = self.calibration.pick_best_approach(
                        [
                            f"honor_user_request::v{attempt_version}",
                            f"offline_alt_v{attempt_version}",
                            f"constraint_focus::v{attempt_version}",
                        ],
                        domain="repair",
                    )
                    diagnosis = dict(diagnosis)
                    diagnosis["approach"] = alt
                    diagnosis["approach_changed"] = True
                    current_approach = alt
                    self._log(
                        f"[{task_id}] CALIBRATE: avoid prior approach — "
                        f"switch to {alt!r}"
                    )
                built = self.builder.build(
                    skill_name=skill_name,
                    description=description or task_goal.user_request,
                    research=research,
                    version=attempt_version,
                    previous_code=last_code,
                    error_log=last_error,
                    protect_active_path=protect_active_path,
                    diagnosis=diagnosis if (diagnosis or {}).get("rewrite_skill", True) else None,
                    failed_approaches=failed_approaches,
                    test_plan=(diagnosis or {}).get("test_plan") if diagnosis else None,
                    user_request=task_goal.user_request,
                    task_goal=task_goal.to_dict(),
                    grounded_args=args,
                    artifacts=artifacts_now,
                    missing_requirements=missing_reqs,
                    constraints=task_goal.with_args(args).constraints,
                )
                self.perf.end(phase_build)
                if built.get("path"):
                    self._event(
                        "skill_file",
                        path=built.get("path"),
                        name=built.get("name") or skill_name,
                        version=attempt_version,
                        ok=bool(built.get("ok")),
                        source=built.get("source") or phase_build,
                        code=str(built.get("code") or "")[:12000],
                    )
                # Never accept an unchanged skill_code repair as progress
                if (
                    built.get("ok")
                    and last_code
                    and (diagnosis or {}).get("rewrite_skill", True)
                ):
                    new_fp = Observer.fingerprint_code(str(built.get("code") or ""))
                    old_fp = Observer.fingerprint_code(str(last_code or ""))
                    if new_fp and new_fp == old_fp:
                        self._log(
                            f"[{task_id}] REPAIR REJECTED: unchanged skill code "
                            f"(fp={new_fp[:8]}) approach={current_approach!r} — "
                            f"forcing divergent approach"
                        )
                        self.memory.record_failed_approach(
                            skill_name,
                            current_approach,
                            Observer.fingerprint_approach(current_approach),
                            last_error="unchanged_repair_code",
                        )
                        current_approach = (
                            f"{current_approach}::changed_code::{attempt_version}"
                        )
                        if isinstance(diagnosis, dict):
                            diagnosis = dict(diagnosis)
                            diagnosis["approach"] = current_approach
                            diagnosis["approach_changed"] = True
                            diagnosis["approach_fingerprint"] = (
                                Observer.fingerprint_approach(current_approach)
                            )
                        # Treat as failed build iteration — re-diagnose / rebuild
                        built = {
                            **built,
                            "ok": False,
                            "error": "unchanged repair code rejected",
                        }
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
                            recovery=recovery,
                        )
                    )
                    task_goal = task_goal.with_args(args)
                    failed_approaches = self.memory.get_failed_approaches(skill_name)
                    skip_rebuild = not diagnosis.get("rewrite_skill", False)
                    if diagnosis.get("stop_recovery"):
                        break
                    continue

                meta = built["meta"]
                caps = meta.get("capabilities") or [description]
                # Only install libraries the skill code actually imports
                deps_list = self._safe_deps(
                    meta.get("dependencies")
                    or research.get("libraries")
                    or (diagnosis or {}).get("needs_new_deps")
                    or [],
                    task_goal=task_goal,
                    skill_code=str(built.get("code") or ""),
                )
                # Persist sanitized deps back onto meta / research
                if isinstance(meta, dict):
                    meta["dependencies"] = list(deps_list)
                research["libraries"] = list(deps_list)
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

            # ── INSTALL DEPS (LIBRARY only — never path/file tokens) ───
            self._status("INSTALL_DEPS")
            deps_list = self._safe_deps(
                deps_list, task_goal=task_goal,
                skill_code=str((built or {}).get("code") or ""),
            )
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
                        recovery=recovery,
                    )
                )
                task_goal = task_goal.with_args(args)
                failed_approaches = self.memory.get_failed_approaches(skill_name)
                skip_rebuild = not TaskGoal.rewrite_skill_for_layer(
                    diagnosis.get("fault_layer")
                )
                if diagnosis.get("stop_recovery"):
                    break
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
            self._emit_subprocess(test_result, mode=str(phase_test).lower())
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
            exec_result: Optional[dict[str, Any]] = None
            actual_artifacts: list[dict[str, Any]] = []

            if not test_failed:
                # TEST PASS ≠ task success — EXECUTE original task, then VERIFY TaskGoal
                self._log(
                    f"[{task_id}] SKILL TEST PASS — running EXECUTE of original task "
                    f"before TaskGoal VERIFY"
                )
                self._status("EXECUTE")
                from jarvis.skill_runner import run_skill_subprocess

                exec_result = run_skill_subprocess(
                    skill_path=built["path"],
                    goal=goal,
                    workspace=self.workspace,
                    args=args,
                    mode="execute",
                    timeout=float(getattr(self.tester, "timeout", 60.0) or 60.0),
                )
                self._emit_subprocess(exec_result, mode="execute")
                self.ledger.log(
                    task_id,
                    "EXECUTE",
                    f"ok={exec_result.get('ok')} rc={exec_result.get('returncode')}",
                    {
                        "error": exec_result.get("error"),
                        "args": args,
                        "after_test": True,
                    },
                )
                actual_artifacts = self._list_workspace_artifacts()
                self._log(
                    f"[{task_id}] ARTIFACTS after EXECUTE: "
                    f"{len(actual_artifacts)} file(s)"
                )
                self._status("VERIFY")
                verification = self.verifier.verify(
                    goal,
                    exec_result,
                    args=args,
                    user_request=task_goal.user_request,
                    constraints=task_goal.with_args(args).constraints,
                    task_goal=task_goal,
                )
                self._emit_verify(verification, context="after_execute")
                self.ledger.log(
                    task_id,
                    "VERIFY",
                    verification.get("reason"),
                    {
                        "skill_result": verification.get("skill_result"),
                        "verifier_result": verification.get("verifier_result"),
                        "args": args,
                        "task_goal": task_goal.to_dict(),
                        "actual_artifacts": actual_artifacts[:20],
                        "after_execute": True,
                    },
                )
                self.calibration.observe_verify(
                    verified=bool(verification.get("verified")),
                    domain="repair",
                    approach=current_approach,
                    approach_fingerprint=Observer.fingerprint_approach(
                        current_approach
                    ),
                    skill_or_topic=skill_name,
                    extra={"attempt": attempt, "phase": "execute_verify"},
                )
                if verification.get("verified"):
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
                    # Bundle is from EXECUTE+VERIFY — safe for DONE (not TEST)
                    self._verified_exec_bundle = {
                        "task_id": task_id,
                        "skill_name": skill_name,
                        "exec_result": exec_result,
                        "verification": verification,
                        "args": dict(args or {}),
                        "actual_artifacts": actual_artifacts[:20],
                    }
                    return skill, args, task_goal

            # ── OBSERVE → DIAGNOSE (TEST fail OR EXECUTE/VERIFY fail) ───
            fail_phase = (
                "VERIFY"
                if (not test_failed and verification and not verification.get("verified"))
                else phase_test
            )
            fail_result = exec_result if exec_result is not None else test_result
            obs = self.observer.observe_failure(
                goal=goal,
                skill_name=skill_name,
                version=attempt_version,
                phase=fail_phase,
                skill_code=(built or {}).get("code"),
                context={
                    **test_context,
                    "user_request": task_goal.user_request,
                    "task_goal": task_goal.to_dict(),
                    "expected_artifacts": list(task_goal.artifacts),
                    "actual_artifacts": actual_artifacts[:40]
                    or self._list_workspace_artifacts()[:40],
                    "content_requirements": list(task_goal.content_requirements),
                    "success_criteria": list(task_goal.success_criteria),
                    "verifier_error": str(
                        (verification or {}).get("reason")
                        or (fail_result or {}).get("error")
                        or ""
                    )[:2000],
                    "previous_failures": failed_approaches[:8],
                },
                dependencies=deps_list,
                test_result=fail_result,
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
                    recovery=recovery,
                )
            )
            task_goal = task_goal.with_args(args)
            failed_approaches = self.memory.get_failed_approaches(skill_name)
            layer = TaskGoal.normalize_fault_layer(
                diagnosis.get("fault_layer") or "skill_code"
            )
            diagnosis["fault_layer"] = layer
            diagnosis["rewrite_skill"] = TaskGoal.rewrite_skill_for_layer(layer)
            # Only rewrite skill when the fault is skill_code → CODING next
            skip_rebuild = not diagnosis.get("rewrite_skill", False)
            if diagnosis.get("prefer_coding") and layer == "skill_code":
                self._log(
                    f"[{task_id}] RECOVERY ROUTE: skill_code → CODING repair "
                    f"(no identical research cycle)"
                )
            if layer == "environment" and diagnosis.get("needs_new_deps"):
                safe = self._safe_deps(
                    diagnosis["needs_new_deps"],
                    task_goal=task_goal,
                    skill_code=str((built or {}).get("code") or last_code or ""),
                )
                diagnosis["needs_new_deps"] = safe
                if safe:
                    self.deps.ensure(safe)
            # legacy alias
            if layer in ("dependency",) and diagnosis.get("needs_new_deps"):
                safe = self._safe_deps(
                    diagnosis["needs_new_deps"],
                    task_goal=task_goal,
                    skill_code=str((built or {}).get("code") or last_code or ""),
                )
                diagnosis["needs_new_deps"] = safe
                if safe:
                    self.deps.ensure(safe)
            if diagnosis.get("stop_recovery"):
                break
            # loop → REPAIR correct layer on next iteration

        report = recovery.stop_report or recovery.build_report()
        self._log(
            f"[{task_id}] RECOVERY STOP: unresolved after "
            f"{len(recovery.attempts)} attempts — "
            f"signatures={report.get('signatures')}"
        )
        self.ledger.log(
            task_id,
            "RECOVERY_STOP",
            "unresolved failure — recovery budget / stagnation",
            report,
        )
        return self.registry.get_skill(skill_name), args, task_goal

    def _safe_deps(
        self,
        names: Any,
        *,
        task_goal: Optional[TaskGoal] = None,
        skill_code: str = "",
    ) -> list[str]:
        """Sanitize install targets — PATH/FILE/CONTENT never become pip packages."""
        req = (task_goal.user_request if task_goal else "") or ""
        arts = list(task_goal.artifacts) if task_goal else []
        # When skill code is present, only external imports are installable
        code = skill_code if skill_code else None
        return TaskGoal.sanitize_libraries(
            list(names or []),
            req,
            artifacts=arts,
            skill_code=code,
        )

    def _log_request_items(self, task_id: str, task_goal: TaskGoal) -> dict[str, list[str]]:
        """Classify request tokens before research/install (PATH ≠ LIBRARY)."""
        items = TaskGoal.classify_items(
            task_goal.user_request,
            artifacts=task_goal.artifacts,
        )
        by_kind: dict[str, list[str]] = {}
        for it in items:
            by_kind.setdefault(it.kind.value, []).append(it.text)
        self._log(
            f"[{task_id}] REQUEST ITEMS: "
            + ", ".join(
                f"{k}={v[:4]}" for k, v in sorted(by_kind.items()) if v
            )
        )
        self._event(
            "action",
            text="Classifying request items...",
            status="REQUEST",
            items={k: v[:8] for k, v in by_kind.items()},
        )
        self.ledger.log(
            task_id,
            "REQUEST_ITEMS",
            "classified",
            {k: v[:12] for k, v in by_kind.items()},
        )
        return by_kind

    def _do_research(
        self, task_id: str, skill_name: str, goal: str, queries: list,
        task_goal: Optional[TaskGoal] = None,
    ) -> dict:
        self._status("RESEARCH")
        if not isinstance(queries, list):
            queries = [str(queries)]
        if task_goal is None:
            task_goal = TaskGoal.from_request(goal, goal=goal)
        # Skill mode: brain notes only when Ollama is ONLINE (no offline hang)
        research = self.research.research(
            queries,
            goal=goal,
            mode="skill",
            network=True,
            use_brain=None,  # auto: available() only
            append_python=True,
        )
        # Drop path/file stems that research_notes or PyPI may have leaked
        research["libraries"] = self._safe_deps(
            research.get("libraries") or [],
            task_goal=task_goal,
            skill_code="",  # no skill yet — only explicit LIBRARY-safe names
        )
        self._event(
            "research",
            stage="done",
            queries=list(queries)[:8],
            sources=[
                {
                    "title": str(s.get("title") or "")[:100],
                    "url": str(s.get("url") or "")[:200],
                }
                for s in (research.get("sources") or research.get("results") or [])[:12]
                if isinstance(s, dict)
            ],
            extracts=[
                {
                    "title": str(e.get("title") or "")[:100],
                    "url": str(e.get("url") or "")[:200],
                    "chars": e.get("char_count"),
                    "preview": str(e.get("text_preview") or e.get("text") or "")[:400],
                }
                for e in (research.get("extracts") or [])[:8]
                if isinstance(e, dict)
            ],
            mode="skill",
            goal=goal,
        )
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
        recovery: Optional[ProgressAwareRecovery] = None,
    ) -> tuple[dict, dict, str, str, Optional[str], dict]:
        """
        OBSERVE → REASONING diagnose → optional RESEARCH (knowledge gap only)
        → CODING-oriented repair state.

        Progress-aware: identical failure_signature without progress must not
        repeat the same research/approach cycle.
        """
        self._status("OBSERVE")
        if task_goal is None:
            task_goal = TaskGoal.from_request(goal, goal=goal)
        if recovery is None:
            recovery = ProgressAwareRecovery(**self.calibration.recovery_kwargs())
        # Ensure observation carries the args + immutable USER REQUEST
        if isinstance(observation.get("context"), dict) and task_args is not None:
            observation["context"] = dict(observation["context"])
            observation["context"]["args"] = dict(task_args)
            observation["context"]["user_request"] = task_goal.user_request
        observation["user_request"] = task_goal.user_request

        # Fingerprints for stagnation / similarity (not task-specific)
        observation["error_fingerprint"] = Observer.fingerprint_error(observation)
        # Provisional signature (fault_layer filled after diagnose)
        observation["failure_signature"] = Observer.failure_signature(observation)

        failure_id = self.memory.save_failure(
            skill_name, goal, observation,
            version=observation.get("version"),
            phase=observation.get("phase"),
            failure_signature=observation.get("failure_signature"),
        )
        self.ledger.log(task_id, "OBSERVE", f"failure_id={failure_id}", {
            "phase": observation.get("phase"),
            "returncode": observation.get("returncode"),
            "exception": (observation.get("exception") or "")[:500],
            "code_fingerprint": observation.get("code_fingerprint"),
            "error_fingerprint": observation.get("error_fingerprint"),
            "failure_signature": observation.get("failure_signature"),
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
        # Deterministic heuristics first — skip 30B when fault layer is obvious
        diagnosis = self._deterministic_diagnose(observation, task_goal)
        clear_layers = {
            "context_mapping", "environment", "goal_parsing",
        }
        layer_clear = diagnosis.get("fault_layer") in clear_layers
        ambiguous_skill = (
            diagnosis.get("fault_layer") == "skill_code"
            and diagnosis.get("approach") not in ("honor_user_request",)
        )
        if (
            self.brain.is_available()
            and not layer_clear
            and ambiguous_skill
        ):
            try:
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
            except Exception as exc:
                self._log(f"DIAGNOSE brain failed ({exc}); using deterministic")
                diagnosis = self._deterministic_diagnose(observation, task_goal)
        elif layer_clear:
            self._log(
                f"DIAGNOSE: deterministic layer={diagnosis.get('fault_layer')} "
                f"(skipped 30B)"
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

        # Rebuild args ONLY for true context/goal mapping faults — never for
        # VERIFY content_constraint failures (those must go to CODING repair).
        new_args = TaskGoal.sanitize_args(
            dict(task_args or {}), task_goal.user_request
        )
        if diagnosis.get("fault_layer") in (
            "context_mapping", "goal_parsing", "context_args"
        ):
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
        else:
            # Still drop any polluted word-keys that leaked in
            new_args = TaskGoal.sanitize_args(new_args, task_goal.user_request)

        # Similarity vs prior failures (current observation already persisted)
        prior_obs = [
            f.get("observation")
            for f in self.memory.recent_failures(skill_name, limit=12)
            if isinstance(f.get("observation"), dict)
        ]
        similar_count = Observer.count_similar_errors(observation, prior_obs)
        layer_now = TaskGoal.normalize_fault_layer(
            diagnosis.get("fault_layer") or "skill_code"
        )
        diagnosis["fault_layer"] = layer_now
        diagnosis["error_fingerprint"] = observation.get("error_fingerprint")
        diagnosis["similar_failure_count"] = similar_count

        # Finalize failure_signature with diagnosed fault_layer
        fail_sig = ProgressAwareRecovery.failure_signature(
            phase=str(observation.get("phase") or ""),
            error_fingerprint=str(observation.get("error_fingerprint") or ""),
            fault_layer=layer_now,
            verifier_failure=observation.get("verifier_result"),
        )
        observation["failure_signature"] = fail_sig
        observation["fault_layer"] = layer_now
        diagnosis["failure_signature"] = fail_sig

        # Model used for this diagnose step (REASONING)
        model_used = ""
        if hasattr(self.brain, "router") and self.brain.router.last_route:
            model_used = str(self.brain.router.last_route.model or "")
        model_used = model_used or getattr(self.brain, "model", "") or "offline"

        # Persist recovery attempt fields on the observation (memory + policy)
        did_research = False
        missing_knowledge = list(diagnosis.get("missing_knowledge") or [])
        knowledge_gap = bool(missing_knowledge) or bool(
            diagnosis.get("knowledge_gap")
        )
        # Strip spurious needs_research when no knowledge gap is stated
        if diagnosis.get("needs_research") and not knowledge_gap:
            # Keep flag only if diagnosis explicitly listed research_queries
            # as a knowledge-gap fill — otherwise clear (no universal fallback)
            if not diagnosis.get("research_queries"):
                diagnosis["needs_research"] = False

        attempt_rec = recovery.record(
            fault_layer=layer_now,
            failure_signature=fail_sig,
            attempted_solution=str(
                diagnosis.get("approach") or approach_label or ""
            ),
            model_used=model_used,
            result=str(observation.get("exception") or "fail")[:500],
            artifacts=list(observation.get("artifacts") or []),
            verifier_failure=observation.get("verifier_result"),
            error_fingerprint=str(observation.get("error_fingerprint") or ""),
            code_fingerprint=str(observation.get("code_fingerprint") or ""),
            approach_fingerprint=str(
                diagnosis.get("approach_fingerprint")
                or Observer.fingerprint_approach(
                    str(diagnosis.get("approach") or approach_label)
                )
            ),
            researched=False,
        )
        observation["attempted_solution"] = attempt_rec.attempted_solution
        observation["model_used"] = model_used
        observation["recovery_progress"] = attempt_rec.progress

        decision = recovery.evaluate(
            failure_signature=fail_sig,
            fault_layer=layer_now,
            diagnosis_needs_research=bool(diagnosis.get("needs_research")),
            missing_knowledge=missing_knowledge,
            knowledge_gap=knowledge_gap,
        )
        diagnosis["stagnated"] = decision.stagnated
        diagnosis["prefer_coding"] = decision.prefer_coding
        diagnosis["recovery_reason"] = decision.reason
        if decision.stagnated or decision.stop:
            self.calibration.observe_recovery(
                stagnated=bool(decision.stagnated),
                stopped=bool(decision.stop),
                allow_research=bool(decision.allow_research),
                failure_signature=fail_sig,
            )

        if decision.force_new_approach and layer_now == "skill_code":
            # Must not repeat the same approach/research — force CODING divergence
            # Prefer historically better repair approaches when available
            old_ap = str(diagnosis.get("approach") or "repair")
            candidates = [
                f"{old_ap}::coding_diverge::v{observation.get('version')}::{fail_sig[:6]}",
                f"honor_user_request::div::{fail_sig[:6]}",
                f"offline_alt_v{int(observation.get('version') or 0) + 1}",
            ]
            diagnosis["approach"] = self.calibration.pick_best_approach(
                candidates,
                domain="repair",
            )
            diagnosis["approach_fingerprint"] = Observer.fingerprint_approach(
                diagnosis["approach"]
            )
            diagnosis["approach_changed"] = True
            diagnosis["rewrite_skill"] = True
            self._log(
                f"[{task_id}] STAGNATION: signature={fail_sig} — "
                f"force new CODING approach (no identical cycle)"
            )

        if decision.stop:
            report = recovery.mark_stopped(decision.report or recovery.build_report(fail_sig))
            diagnosis["stop_recovery"] = True
            diagnosis["recovery_report"] = report
            self._log(
                f"[{task_id}] RECOVERY BUDGET STOP: {decision.reason}"
            )

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
                "missing_knowledge": missing_knowledge,
                "needs_new_deps": diagnosis.get("needs_new_deps"),
                "missing_args": diagnosis.get("missing_args"),
                "suggested_args": diagnosis.get("suggested_args"),
                "args_keys": list(new_args.keys()),
                "test_plan": diagnosis.get("test_plan"),
                "error_fingerprint": diagnosis.get("error_fingerprint"),
                "failure_signature": fail_sig,
                "similar_failure_count": similar_count,
                "stagnated": decision.stagnated,
                "stop_recovery": diagnosis.get("stop_recovery"),
                "model_used": model_used,
            },
        )
        self._log(
            f"[{task_id}] DIAGNOSE: {diagnosis.get('root_cause', '')[:160]} "
            f"| layer={diagnosis.get('fault_layer')!r} "
            f"rewrite_skill={diagnosis.get('rewrite_skill')} "
            f"| approach={diagnosis.get('approach')!r} "
            f"changed={diagnosis.get('approach_changed')} "
            f"| sig={fail_sig} similar={similar_count} "
            f"stagnated={decision.stagnated}"
        )

        # ── RESEARCH only on diagnosed knowledge gap (never universal) ──
        if decision.allow_research and not diagnosis.get("stop_recovery"):
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
            # Ground queries in stated missing knowledge
            for mk in missing_knowledge[:4]:
                q = f"{goal} — knowledge gap: {mk}"
                if q not in queries:
                    queries.append(q)
            if not queries:
                queries = [
                    f"{goal} — fill knowledge: {m}"
                    for m in (missing_knowledge or ["fundamentals"])[:3]
                ]
            diagnosis["research_queries"] = queries[:6]
            diagnosis["adaptive_research"] = True
            self._log(
                f"[{task_id}] KNOWLEDGE-GAP RESEARCH: sig={fail_sig} "
                f"missing={missing_knowledge!r} queries={len(queries)}"
            )
            new_research = self._do_research(
                task_id, skill_name, goal, list(queries), task_goal=task_goal
            )
            research = self._merge_research(research, new_research)
            research["knowledge_history"] = self.memory.get_research_knowledge(
                skill_name, limit=8
            )
            did_research = True
            recovery.mark_researched(fail_sig)
            researched_approach = str(research.get("approach") or "").strip()
            if researched_approach:
                candidate = researched_approach.split("\n")[0][:100]
                cand_fp = Observer.fingerprint_approach(candidate)
                failed_fps = {a.get("approach_fingerprint") for a in failed}
                if cand_fp in failed_fps or not diagnosis.get("approach_changed"):
                    candidate = (
                        f"{candidate} | researched-v{observation.get('version')}"
                        f"|{fail_sig[:6]}"
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
                f"knowledge-gap research sig={fail_sig}",
                {
                    "queries": queries[:6],
                    "failure_signature": fail_sig,
                    "missing_knowledge": missing_knowledge,
                    "approach": diagnosis.get("approach"),
                    "libraries": research.get("libraries"),
                },
            )
        elif diagnosis.get("needs_research") and not decision.allow_research:
            self._log(
                f"[{task_id}] RESEARCH SKIPPED: {decision.reason} "
                f"— route to "
                f"{'CODING' if decision.prefer_coding else layer_now} repair"
            )
            diagnosis["adaptive_research"] = False

        if diagnosis.get("needs_new_deps"):
            safe_deps = self._safe_deps(
                diagnosis["needs_new_deps"],
                task_goal=task_goal,
                skill_code=str(code or ""),
            )
            diagnosis["needs_new_deps"] = safe_deps
            if safe_deps:
                dep_r = self.deps.ensure(safe_deps)
                self.ledger.log(task_id, "INSTALL_DEPS", "from diagnosis", dep_r)
                research.setdefault("libraries", [])
                research["libraries"] = list(
                    dict.fromkeys(
                        list(research.get("libraries") or []) + list(safe_deps)
                    )
                )

        research["fix_plan"] = diagnosis.get("test_plan")
        research["approach"] = diagnosis.get("approach")
        research["diagnosis"] = diagnosis
        research["adaptive_research"] = diagnosis.get("adaptive_research")
        research["failure_signature"] = fail_sig

        missing_reqs = self._missing_requirements_from_diagnosis(
            diagnosis, task_goal
        )
        last_error = (
            f"USER REQUEST: {task_goal.user_request}\n"
            f"ROOT CAUSE: {diagnosis.get('root_cause')}\n"
            f"FAULT_LAYER: {diagnosis.get('fault_layer')}\n"
            f"FAILURE_SIGNATURE: {fail_sig}\n"
            f"CHANGE: {diagnosis.get('what_to_change')}\n"
            f"APPROACH: {diagnosis.get('approach')}\n"
            f"MODEL_USED: {model_used}\n"
            f"RESEARCHED: {did_research}\n"
            f"RESEARCH_INSIGHT: {research.get('repair_insight') or ''}\n"
            f"TASK_GOAL: {json.dumps(task_goal.to_dict(), default=str)[:1200]}\n"
            f"GROUNDED_ARGS: {json.dumps(new_args, default=str)[:500]}\n"
            f"MISSING_REQUIREMENTS: {json.dumps(missing_reqs, default=str)[:500]}\n"
            f"ARTIFACTS: {json.dumps(observation.get('artifacts') or [], default=str)[:500]}\n"
            f"EXCEPTION: {observation.get('exception')}\n"
            f"STDERR: {(observation.get('stderr') or '')[:800]}\n"
            f"STDOUT: {(observation.get('stdout') or '')[:400]}\n"
            f"TRACEBACK: {(observation.get('traceback') or '')[:800]}\n"
            f"VERIFIER: {json.dumps(observation.get('verifier_result'), default=str)[:800]}"
        )
        self._status("REPAIR")
        self.ledger.log(
            task_id,
            "REPAIR",
            f"prepare repair layer={diagnosis.get('fault_layer')} "
            f"rewrite_skill={diagnosis.get('rewrite_skill')} "
            f"approach={diagnosis.get('approach')} "
            f"sig={fail_sig} research={did_research} "
            f"stagnated={decision.stagnated}",
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

    def _exec_after_repair(
        self,
        task_id: str,
        skill: dict,
        goal: str,
        task_args: dict,
        task_goal: TaskGoal,
    ) -> dict:
        """
        Prefer a just-completed EXECUTE+VERIFY PASS bundle from the learn loop.
        Never treat TEST-only proof as success — bundle is always post-EXECUTE.
        """
        proof = self._verified_exec_bundle
        self._verified_exec_bundle = None
        if (
            isinstance(proof, dict)
            and proof.get("task_id") == task_id
            and proof.get("skill_name") == skill.get("name")
            and isinstance(proof.get("verification"), dict)
            and proof["verification"].get("verified")
            and isinstance(proof.get("exec_result"), dict)
        ):
            out = dict(proof["exec_result"])
            out["_from_verified_exec_bundle"] = True
            out["_verification"] = proof["verification"]
            out["_actual_artifacts"] = list(proof.get("actual_artifacts") or [])
            self._log(
                f"[{task_id}] Reusing EXECUTE+VERIFY PASS bundle "
                f"(skip duplicate execute+VERIFY)"
            )
            return out
        return self._execute_trusted(
            task_id, skill, goal, args=task_args, task_goal=task_goal
        )

    @staticmethod
    def _deterministic_diagnose(
        observation: dict[str, Any],
        task_goal: TaskGoal,
    ) -> dict[str, Any]:
        """Python fault-layer diagnosis — no LLM. Prefer over 30B when clear."""
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
        content_fail = any(
            t in err_l
            for t in (
                "missing from expected",
                "content_constraint",
                "not found in workspace",
                "no user-derived constraints",
                "claim_aligns",
                "reject_defaults",
            )
        )
        parsed_missing = (
            [] if content_fail else ContextBuilder.parse_missing_arg_names(err)
        )
        # Never promote request tokens / content needles to missing_args
        parsed_missing = [
            n for n in parsed_missing
            if not TaskGoal.is_polluted_arg_key(n, task_goal.user_request)
        ]
        args_fault = (
            (empty_args or bool(parsed_missing) or invented)
            and not content_fail
            and any(
                t in err_l
                for t in (
                    "required argument", "missing required", "keyerror",
                    "when invoked", "argument",
                )
            )
        )
        goal_mismatch = phase == "VERIFY" and any(
            t in err_l
            for t in (
                "default", "placeholder", "untrusted", "claim_aligns",
                "not in user", "user constraints", "user request",
                "reject_defaults", "missing from expected",
                "content_constraint",
            )
        ) and not invented
        no_constraints = phase == "VERIFY" and "no user-derived constraints" in err_l
        env_fault = any(
            t in err_l for t in ("modulenotfound", "no module named", "importerror")
        )
        if no_constraints and empty_args:
            layer = "goal_parsing"
            approach = "goal_parsing_repair"
            change = "Re-parse immutable TaskGoal / USER REQUEST into constraints"
        elif args_fault and not goal_mismatch:
            layer = "context_mapping"
            approach = "context_mapping"
            change = (
                "Map TaskGoal → skill args from USER REQUEST "
                "(no invented defaults)"
            )
        elif env_fault:
            layer = "environment"
            approach = "fix_environment"
            change = "Install / fix missing dependencies"
        elif goal_mismatch or content_fail:
            layer = "skill_code"
            approach = "honor_user_request"
            change = (
                "Rewrite skill to satisfy original TaskGoal "
                "(no default/placeholder artifacts; honor content requirements)"
            )
        else:
            layer = "skill_code"
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
            "needs_research": False,
            "missing_knowledge": [],
            "research_queries": [],
            "needs_new_deps": [],
            "missing_args": list(parsed_missing),
            "required_args": list(parsed_missing),
            "suggested_args": {},
            "test_plan": (
                "subprocess retest with grounded args; "
                "VERIFY against original TaskGoal (reject defaults)"
            ),
            "expected_artifacts": list(task_goal.artifacts),
            "is_unfixable": False,
            "diagnosis": str(observation.get("exception") or "failure")[:500],
        }
        diagnosis["approach_fingerprint"] = Observer.fingerprint_approach(
            diagnosis["approach"]
        )
        return diagnosis

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
        self._emit_subprocess(
            {**result, "skill_path": skill.get("file_path") or skill.get("name")},
            mode="execute",
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

    def _sanitize_action_plan(
        self, plan: dict, goal: str, task_goal: Optional[TaskGoal] = None
    ) -> dict:
        """Drop unrelated can_reuse / skill_name inherited from prior capabilities."""
        plan = dict(plan or {})
        tg = task_goal or TaskGoal.from_request(goal, goal=goal)
        reuse = []
        for name in plan.get("can_reuse") or []:
            sk = self.registry.get_skill(str(name))
            if not sk:
                continue
            match = evaluate_capability(sk, tg)
            self._log(format_match_log(match))
            if match.get("compatible"):
                reuse.append(sk["name"])
        plan["can_reuse"] = reuse
        planned = str(plan.get("skill_name") or "").strip()
        if planned:
            existing = self.registry.get_skill(planned)
            # Only reject when the name refers to an EXISTING incompatible skill.
            # Fresh planner names (not yet registered) are kept.
            if existing:
                match = evaluate_capability(existing, tg)
                self._log(format_match_log(match))
                if not match.get("compatible"):
                    plan["skill_name"] = Brain._slug(tg.user_request or goal)[:40] or "new_skill"
                    plan["needs_new_skill"] = True
        else:
            plan["skill_name"] = Brain._slug(tg.user_request or goal)[:40] or "new_skill"
        if not reuse and not plan.get("needs_new_skill"):
            # No fitting skill → must build for THIS goal
            plan["needs_new_skill"] = True
            plan["needs_research"] = True
        return plan

    def _select_compatible_skills(
        self,
        plan: dict,
        task_goal: TaskGoal,
        *,
        task_id: str = "",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Rank ACTIVE skills that can fulfill this TaskGoal (not keyword bait)."""
        reports: list[dict[str, Any]] = []
        scored: list[tuple[float, dict[str, Any]]] = []
        seen: set[str] = set()

        def _consider(sk: dict[str, Any]) -> None:
            name = str(sk.get("name") or "")
            if not name or name in seen or sk.get("status") != "ACTIVE":
                return
            seen.add(name)
            match = evaluate_capability(sk, task_goal)
            self._log(format_match_log(match))
            reports.append(match)
            if match.get("compatible"):
                scored.append((float(match.get("score") or 0.0), sk))

        for name in plan.get("can_reuse") or []:
            sk = self.registry.get_skill(str(name))
            if sk:
                _consider(sk)

        # Fallback candidates from registry keywords — still require TaskGoal compatibility
        if not scored and not plan.get("needs_new_skill"):
            keywords = plan.get("research_queries") or [task_goal.user_request]
            tokens: list[str] = []
            for k in keywords:
                tokens.extend(str(k).split())
            tokens.extend(str(task_goal.user_request).split())
            tokens.extend(str(a) for a in task_goal.artifacts)
            for sk in self.registry.match_skills(tokens, only_active=True):
                _consider(sk)

        scored.sort(key=lambda x: -x[0])
        matched = [sk for _, sk in scored]
        if task_id:
            self._log(
                f"[{task_id}] capability candidates compatible="
                f"{[m['name'] for m in matched]}"
            )
        return matched, reports

    @staticmethod
    def _name_fits_goal(name: str, goal: str) -> bool:
        """Legacy name overlap helper (kept for offline fallbacks)."""
        if not name or not goal:
            return False
        tg = TaskGoal.from_request(goal, goal=goal)
        fake = {
            "name": name,
            "description": goal,
            "capabilities": [],
            "status": "ACTIVE",
            "file_path": "",
        }
        return bool(evaluate_capability(fake, tg).get("compatible"))

    def _skill_fits_goal(
        self, skill: dict, goal: str, task_goal: Optional[TaskGoal] = None
    ) -> bool:
        """Whether skill is TaskGoal-compatible (actions/artifacts/domain)."""
        if not skill or not goal:
            return False
        tg = task_goal or TaskGoal.from_request(goal, goal=goal)
        return bool(evaluate_capability(skill, tg).get("compatible"))

    def _offline_plan(self, goal: str, caps: list[str]) -> dict:
        tg = TaskGoal.from_request(goal, goal=goal)
        matched = []
        for m in self.registry.match_skills(goal.split(), only_active=True):
            match = evaluate_capability(m, tg)
            # Quiet offline plan — match logs happen at CHECK_CAPABILITIES
            if match.get("compatible"):
                matched.append(m)
        # Universal offline arg draft — no task-specific key hardcoding
        draft_args = TaskGoal.sanitize_args(
            ContextBuilder._offline_extract(goal), goal
        )
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

    def _list_workspace_artifacts(self, limit: int = 40) -> list[dict[str, Any]]:
        """Snapshot existing workspace files for CODING repair context."""
        out: list[dict[str, Any]] = []
        try:
            root = self.workspace
            if not root.exists():
                return out
            for p in sorted(root.rglob("*")):
                if not p.is_file():
                    continue
                try:
                    rel = str(p.relative_to(root))
                except Exception:
                    rel = str(p)
                try:
                    size = p.stat().st_size
                except Exception:
                    size = -1
                out.append({"path": rel, "bytes": size})
                if len(out) >= limit:
                    break
        except Exception:
            pass
        return out

    @staticmethod
    def _missing_requirements_from_diagnosis(
        diagnosis: Optional[dict],
        task_goal: TaskGoal,
    ) -> list[str]:
        """Human-readable missing requirements for CODING (not word-arg keys)."""
        out: list[str] = []
        for a in task_goal.artifacts:
            out.append(f"artifact:{a}")
        for c in task_goal.content_requirements:
            out.append(f"content:{c}")
        for s in task_goal.success_criteria:
            if s not in out:
                out.append(s)
        if diagnosis:
            for key in ("expected_artifacts", "missing_knowledge"):
                for item in diagnosis.get(key) or []:
                    s = str(item)
                    if s and s not in out:
                        out.append(s)
            root = str(diagnosis.get("root_cause") or "")
            if root and "missing from expected" in root.lower():
                out.append(root[:240])
        return out[:24]

    def _format_status(self) -> str:
        mem = self.memory.stats()
        reg = self.registry.stats()
        led = self.ledger.stats()
        brain = self.brain.model_status()
        catalog = ""
        if hasattr(self.brain, "model_catalog_summary"):
            catalog = f"\nModels — {self.brain.model_catalog_summary()}"
        cal = self.calibration.status()
        pool_line = ""
        if hasattr(self.brain, "router") and hasattr(self.brain.router, "status_detail"):
            try:
                detail = self.brain.router.status_detail()
                warm = [
                    f"{t}:{'♨' if inf.get('warm') else ('✓' if inf.get('ready') else '·')}"
                    for t, inf in (detail.get("tiers") or {}).items()
                ]
                pool_line = f"\nPool — {' '.join(warm)} budget_GiB=" + str(
                    (detail.get("budget_bytes") or 0) // (1024**3)
                )
            except Exception:
                pool_line = ""
        return (
            f"Brain (Ollama model pool): {brain}{catalog}{pool_line}\n"
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
            f"actions:{led['actions']}\n"
            f"Calibration — rate:{cal.get('success_rate', 0):.2f} "
            f"outcomes:{cal.get('outcomes', 0)} "
            f"approaches:{cal.get('approaches', 0)} "
            f"avoided:{len(cal.get('avoided') or [])} "
            f"adaptations:{(cal.get('meta') or {}).get('adaptations', 0)}"
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
        catalog = ""
        detail: dict[str, Any] = {}
        if hasattr(self.brain, "model_catalog_summary"):
            catalog = self.brain.model_catalog_summary()
        if hasattr(self.brain, "router"):
            try:
                detail = self.brain.router.status_detail()
            except Exception:
                detail = {}
        return {
            "memory": self.memory.stats(),
            "skills": self.registry.stats(),
            "ledger": self.ledger.stats(),
            "brain_online": status == "ONLINE",
            "brain_status": status,
            "model": self.brain.model,
            "model_catalog": catalog,
            "model_tiers": detail.get("tiers") or {},
            "model_pool": {
                "initialized": detail.get("initialized"),
                "budget_bytes": detail.get("budget_bytes"),
                "loaded": detail.get("loaded") or [],
                "tiers": detail.get("tiers") or {},
            },
            "skill_list": self.registry.list_skills(),
            "calibration": self.calibration.status(),
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
        # Mirror phase to workspace GUI (async consumer — never wait)
        self._event("phase", status=status)

    def _event(self, kind: str, payload: Any = None, **extra: Any) -> None:
        """Fire structured workspace event; must never block the cycle.

        Accepts either ``_event(kind, {..})`` or ``_event(kind, key=val)``.
        """
        data: dict[str, Any]
        if isinstance(payload, dict):
            data = dict(payload)
        elif payload is None:
            data = {}
        else:
            data = {"value": payload}
        if extra:
            data.update(extra)
        try:
            self.on_event(kind, data)
        except Exception:
            pass

    def _emit_verify(self, verification: dict[str, Any], *, context: str = "") -> None:
        self._event(
            "verify",
            verified=bool(verification.get("verified")),
            reason=str(verification.get("reason") or "")[:500],
            checks=verification.get("checks") or [],
            context=context,
        )

    def _emit_subprocess(self, result: dict[str, Any], *, mode: str) -> None:
        self._event(
            "subprocess",
            mode=mode,
            command=f"skill:{Path(str(result.get('skill_path') or '')).name or mode}",
            stdout=str(result.get("stdout") or "")[:8000],
            stderr=str(result.get("stderr") or "")[:8000],
            traceback=str(result.get("traceback") or "")[:8000],
            returncode=result.get("returncode"),
            timed_out=bool(result.get("timed_out")),
            ok=bool(result.get("ok")),
            error=str(result.get("error") or "")[:500],
            skill_path=str(result.get("skill_path") or ""),
        )

    def close(self) -> None:
        set_active_tracker(None)
        try:
            if hasattr(self.brain, "pool") and hasattr(self.brain.pool, "close"):
                self.brain.pool.close()
        except Exception:
            pass
        self.memory.close()
        self.ledger.close()
        self.registry.close()
