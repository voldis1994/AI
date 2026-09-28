"""
Progress-aware recovery for JARVIS skill repair loops.

Prevents VERIFY FAIL → RESEARCH → REASONING → RESEARCH… spinning without
progress. Tracks failure signatures, detects stagnation, budgets recovery,
and gates research to genuine knowledge gaps only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional


# After this many identical signatures with no material progress → must diverge.
STAGNATION_SAME_SIGNATURE = 2
# Distinct recovery attempts (observe/diagnose cycles) before hard STOP.
MAX_RECOVERY_ATTEMPTS = 5
# Research calls allowed per repair session (not a universal fallback).
MAX_RESEARCH_PER_REPAIR = 1


@dataclass
class RecoveryAttempt:
    """One persisted failure/repair attempt (universal — not task-specific)."""

    fault_layer: str
    failure_signature: str
    attempted_solution: str
    model_used: str
    result: str
    artifacts: list[Any]
    verifier_failure: Any
    error_fingerprint: str = ""
    code_fingerprint: str = ""
    approach_fingerprint: str = ""
    researched: bool = False
    progress: bool = False


@dataclass
class RecoveryDecision:
    stagnated: bool = False
    stop: bool = False
    allow_research: bool = False
    force_new_approach: bool = False
    prefer_coding: bool = False
    reason: str = ""
    report: dict[str, Any] = field(default_factory=dict)


class ProgressAwareRecovery:
    """
    Per-repair-session tracker (scoped to one skill/task_id).

    Does not replace Memory — enriches the loop with stagnation / budget policy.
    """

    def __init__(
        self,
        *,
        stagnation_limit: int = STAGNATION_SAME_SIGNATURE,
        max_attempts: int = MAX_RECOVERY_ATTEMPTS,
        max_research: int = MAX_RESEARCH_PER_REPAIR,
    ) -> None:
        self.stagnation_limit = max(2, int(stagnation_limit))
        self.max_attempts = max(2, int(max_attempts))
        self.max_research = max(0, int(max_research))
        self.attempts: list[RecoveryAttempt] = []
        self.research_count = 0
        self._researched_signatures: set[str] = set()
        self.stopped = False
        self.stop_report: Optional[dict[str, Any]] = None

    # ── signatures / progress ───────────────────────────────────────────

    @staticmethod
    def failure_signature(
        *,
        phase: str = "",
        error_fingerprint: str = "",
        fault_layer: str = "",
        verifier_failure: Any = None,
    ) -> str:
        """
        Stable signature for a failure class.

        Combines phase + error fingerprint + fault layer + verifier reason class.
        Paths/ids already normalized inside Observer.fingerprint_error.
        """
        vr_reason = ""
        if isinstance(verifier_failure, dict):
            vr_reason = str(
                verifier_failure.get("reason")
                or (verifier_failure.get("verifier_result") or {}).get("reason")
                or ""
            )[:240]
        elif verifier_failure:
            vr_reason = str(verifier_failure)[:240]
        blob = "|".join(
            [
                str(phase or "").upper(),
                str(error_fingerprint or ""),
                str(fault_layer or "").lower(),
                vr_reason.lower(),
            ]
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def material_progress(
        prev: Optional[RecoveryAttempt],
        *,
        code_fingerprint: str,
        artifacts: list[Any],
        verifier_failure: Any,
        fault_layer: str,
        approach_fingerprint: str,
    ) -> bool:
        """True when the new attempt differs in a way that can break the loop."""
        if prev is None:
            return True
        if code_fingerprint and code_fingerprint != (prev.code_fingerprint or ""):
            return True
        if approach_fingerprint and approach_fingerprint != (
            prev.approach_fingerprint or ""
        ):
            # Approach label alone is not enough if code/artifacts/verifier identical —
            # only counts if something else also moved OR we haven't compared code yet.
            pass
        prev_arts = ProgressAwareRecovery._artifact_key(prev.artifacts)
        cur_arts = ProgressAwareRecovery._artifact_key(artifacts)
        if cur_arts != prev_arts:
            return True
        prev_vr = ProgressAwareRecovery._verifier_key(prev.verifier_failure)
        cur_vr = ProgressAwareRecovery._verifier_key(verifier_failure)
        if cur_vr != prev_vr:
            return True
        if fault_layer and fault_layer != (prev.fault_layer or ""):
            return True
        return False

    @staticmethod
    def _artifact_key(artifacts: list[Any]) -> str:
        parts: list[str] = []
        for a in artifacts or []:
            if isinstance(a, dict):
                parts.append(
                    f"{a.get('path')}|{a.get('exists')}|{a.get('size')}|{a.get('preview')}"
                )
            else:
                parts.append(str(a)[:80])
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _verifier_key(verifier_failure: Any) -> str:
        if isinstance(verifier_failure, dict):
            text = str(
                verifier_failure.get("reason")
                or json.dumps(verifier_failure, default=str, sort_keys=True)[:400]
            )
        else:
            text = str(verifier_failure or "")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

    # ── record / decide ─────────────────────────────────────────────────

    def record(
        self,
        *,
        fault_layer: str,
        failure_signature: str,
        attempted_solution: str,
        model_used: str,
        result: str,
        artifacts: Optional[list[Any]] = None,
        verifier_failure: Any = None,
        error_fingerprint: str = "",
        code_fingerprint: str = "",
        approach_fingerprint: str = "",
        researched: bool = False,
    ) -> RecoveryAttempt:
        arts = list(artifacts or [])
        prev = self.attempts[-1] if self.attempts else None
        # Compare to last attempt with the SAME signature for progress
        same_sig_prev = None
        for a in reversed(self.attempts):
            if a.failure_signature == failure_signature:
                same_sig_prev = a
                break
        progress = self.material_progress(
            same_sig_prev or prev,
            code_fingerprint=code_fingerprint,
            artifacts=arts,
            verifier_failure=verifier_failure,
            fault_layer=fault_layer,
            approach_fingerprint=approach_fingerprint,
        )
        # Identical signature + identical code + identical verifier = no progress
        if same_sig_prev is not None:
            if (
                code_fingerprint
                and code_fingerprint == same_sig_prev.code_fingerprint
                and self._verifier_key(verifier_failure)
                == self._verifier_key(same_sig_prev.verifier_failure)
            ):
                progress = False

        attempt = RecoveryAttempt(
            fault_layer=str(fault_layer or ""),
            failure_signature=str(failure_signature or ""),
            attempted_solution=str(attempted_solution or "")[:500],
            model_used=str(model_used or "")[:120],
            result=str(result or "")[:500],
            artifacts=arts[:30],
            verifier_failure=verifier_failure,
            error_fingerprint=str(error_fingerprint or ""),
            code_fingerprint=str(code_fingerprint or ""),
            approach_fingerprint=str(approach_fingerprint or ""),
            researched=bool(researched),
            progress=progress,
        )
        self.attempts.append(attempt)
        if researched:
            self.research_count += 1
            self._researched_signatures.add(failure_signature)
        return attempt

    def same_signature_streak(self, signature: str) -> int:
        n = 0
        for a in reversed(self.attempts):
            if a.failure_signature != signature:
                break
            if a.progress:
                break
            n += 1
        return n

    def count_signature(self, signature: str) -> int:
        return sum(1 for a in self.attempts if a.failure_signature == signature)

    def evaluate(
        self,
        *,
        failure_signature: str,
        fault_layer: str,
        diagnosis_needs_research: bool = False,
        missing_knowledge: Optional[list] = None,
        knowledge_gap: bool = False,
    ) -> RecoveryDecision:
        """
        Decide next recovery action after a failure is observed/diagnosed.

        Research is allowed ONLY when diagnosis indicates a knowledge gap —
        never as a universal fallback for every VERIFY fail.
        """
        from jarvis.task_contract import TaskContract

        layer = TaskContract.normalize_fault_layer(fault_layer)
        # CODING rewrite only for evidence-backed skill_code — never unknown
        prefer_coding = layer == "skill_code"
        same = [
            a for a in self.attempts if a.failure_signature == failure_signature
        ]
        sig_count = len(same)
        # After 2 identical signatures where the latest shows no material progress
        stagnated = (
            sig_count >= self.stagnation_limit
            and bool(same)
            and not same[-1].progress
        )

        gap = bool(knowledge_gap) or bool(missing_knowledge)
        already_researched = failure_signature in self._researched_signatures
        research_budget_left = self.research_count < self.max_research

        allow_research = (
            bool(diagnosis_needs_research)
            and gap
            and research_budget_left
            and not already_researched
            and layer
            not in ("context_mapping", "context_args", "goal_parsing", "execution")
        )

        # Hard STOP: overall budget, or stagnated sig after we already forced
        # a diverge once more without progress (3rd identical no-progress hit)
        no_progress_same = sum(1 for a in same if not a.progress)
        exhausted_sig = stagnated and no_progress_same >= self.stagnation_limit + 1
        stop = (
            len(self.attempts) >= self.max_attempts
            or exhausted_sig
        )

        reason_parts = []
        if stagnated:
            reason_parts.append(
                f"stagnation: failure_signature={failure_signature} "
                f"count={sig_count} without progress"
            )
        if stop:
            reason_parts.append(
                f"recovery budget exhausted attempts={len(self.attempts)}/"
                f"{self.max_attempts}"
            )
        if allow_research:
            reason_parts.append("knowledge_gap→research allowed")
        elif diagnosis_needs_research and not gap:
            reason_parts.append("needs_research ignored (no knowledge gap)")
        elif already_researched:
            reason_parts.append("research already used for this signature")

        report = self.build_report(failure_signature) if stop or stagnated else {}

        return RecoveryDecision(
            stagnated=stagnated,
            stop=stop,
            allow_research=allow_research,
            force_new_approach=stagnated or stop,
            prefer_coding=prefer_coding,
            reason="; ".join(reason_parts) or "continue",
            report=report,
        )

    def build_report(self, focus_signature: str = "") -> dict[str, Any]:
        """Precise unresolved failure report for STOP."""
        return {
            "unresolved": True,
            "attempts": len(self.attempts),
            "research_count": self.research_count,
            "focus_signature": focus_signature,
            "signatures": sorted({a.failure_signature for a in self.attempts}),
            "history": [
                {
                    "fault_layer": a.fault_layer,
                    "failure_signature": a.failure_signature,
                    "attempted_solution": a.attempted_solution,
                    "model_used": a.model_used,
                    "result": a.result,
                    "code_fingerprint": a.code_fingerprint,
                    "approach_fingerprint": a.approach_fingerprint,
                    "researched": a.researched,
                    "progress": a.progress,
                    "verifier_failure": (
                        (a.verifier_failure or {}).get("reason")
                        if isinstance(a.verifier_failure, dict)
                        else a.verifier_failure
                    ),
                    "artifacts": [
                        (x.get("path") if isinstance(x, dict) else str(x))
                        for x in (a.artifacts or [])[:8]
                    ],
                }
                for a in self.attempts
            ],
        }

    def mark_researched(self, failure_signature: str) -> None:
        """Record that knowledge-gap research was used for this signature."""
        self.research_count += 1
        self._researched_signatures.add(str(failure_signature or ""))
        if self.attempts:
            self.attempts[-1].researched = True

    def mark_stopped(self, report: Optional[dict] = None) -> dict[str, Any]:
        self.stopped = True
        self.stop_report = report or self.build_report()
        return self.stop_report
