"""
Universal auto-calibration / self-improvement for JARVIS.

Learns HOW to solve tasks from measurable VERIFY outcomes — not LLM guesses.
Persists in SQLite via Memory.facts so improvements survive restarts.

Integrates with existing failed_approaches, ProgressAwareRecovery, ModelRouter
knobs — does not create a parallel architecture.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger("jarvis.calibration")

# Durable fact keys
FACT_PARAMS = "calib:params"
FACT_APPROACHES = "calib:approaches"
FACT_OUTCOMES = "calib:outcomes_window"
FACT_CONFIG_STACK = "calib:config_stack"
FACT_PENDING = "calib:pending_eval"
FACT_META = "calib:meta"

# Rolling window for measurable success rate
OUTCOME_WINDOW = 24
# Min outcomes before a pending config change is judged
EVAL_MIN_OUTCOMES = 6
# Relative drop that triggers rollback (absolute rate points)
ROLLBACK_MARGIN = 0.12
# Min fails before an approach is strongly avoided
AVOID_FAIL_THRESHOLD = 3
# Clamp ranges for calibrated knobs (universal — not task-specific)
PARAM_CLAMPS: dict[str, tuple[Any, Any]] = {
    "max_repair_attempts": (2, 8),
    "max_learning_attempts": (2, 8),
    "max_learning_network_research": (0, 3),
    "stagnation_same_signature": (2, 4),
    "max_recovery_attempts": (3, 8),
    "max_research_per_repair": (0, 3),
    "max_open_sources": (2, 6),
    "chat_timeout_sec": (45.0, 120.0),
    "research_timeout_sec": (6.0, 20.0),
    "gap_stagnation_rounds": (2, 4),
    "escalate_error_only": (0, 1),  # bool as 0/1
}


@dataclass
class CalibParams:
    """Work parameters JARVIS may auto-tune from outcomes."""

    max_repair_attempts: int = 5
    max_learning_attempts: int = 5
    max_learning_network_research: int = 1
    stagnation_same_signature: int = 2
    max_recovery_attempts: int = 5
    max_research_per_repair: int = 1
    max_open_sources: int = 4
    chat_timeout_sec: float = 90.0
    research_timeout_sec: float = 10.0
    gap_stagnation_rounds: int = 2
    escalate_error_only: int = 1  # prefer error-only escalate (safer/faster)

    def clamped(self) -> "CalibParams":
        d = asdict(self)
        for k, (lo, hi) in PARAM_CLAMPS.items():
            if k not in d:
                continue
            v = d[k]
            if isinstance(lo, float) or isinstance(v, float):
                d[k] = float(min(hi, max(lo, float(v))))
            else:
                d[k] = int(min(hi, max(lo, int(v))))
        return CalibParams(**d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.clamped())

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "CalibParams":
        from dataclasses import fields as dc_fields

        if not isinstance(data, dict):
            return cls()
        names = {f.name for f in dc_fields(cls)}
        kwargs = {k: data[k] for k in names if k in data}
        return cls(**kwargs).clamped()


@dataclass
class ApproachStats:
    approach: str = ""
    fingerprint: str = ""
    domain: str = ""  # learning | repair | plan | research
    successes: int = 0
    failures: int = 0
    last_result: str = ""
    updated_at: float = 0.0
    # Multi-attempt evidence (not single-error reactivity)
    recent: list[int] = field(default_factory=list)  # 1=pass 0=fail, newest last

    def score(self) -> float:
        """Laplace-smoothed success rate in [0, 1]."""
        s, f = self.successes, self.failures
        return (s + 1.0) / (s + f + 2.0)

    def recent_rate(self, n: int = 8) -> float:
        window = self.recent[-n:]
        if not window:
            return 0.5
        return sum(window) / len(window)

    def should_avoid(self) -> bool:
        if self.failures >= AVOID_FAIL_THRESHOLD and self.successes == 0:
            return True
        if len(self.recent) >= 4 and self.recent_rate(6) <= 0.1 and self.failures >= 2:
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "approach": self.approach,
            "fingerprint": self.fingerprint,
            "domain": self.domain,
            "successes": self.successes,
            "failures": self.failures,
            "last_result": self.last_result,
            "updated_at": self.updated_at,
            "recent": list(self.recent)[-16:],
            "score": round(self.score(), 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ApproachStats":
        return cls(
            approach=str(data.get("approach") or ""),
            fingerprint=str(data.get("fingerprint") or ""),
            domain=str(data.get("domain") or ""),
            successes=int(data.get("successes") or 0),
            failures=int(data.get("failures") or 0),
            last_result=str(data.get("last_result") or ""),
            updated_at=float(data.get("updated_at") or 0.0),
            recent=[int(x) for x in (data.get("recent") or [])][-16:],
        )


class AutoCalibration:
    """
    Measurable self-improvement controller.

    Feedback authority: verifier PASS/FAIL (and recovery stagnation reports).
    Persistence: Memory.set_fact — survives process restart.
    """

    def __init__(
        self,
        memory: Any,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.memory = memory
        self.on_log = on_log or (lambda _m: None)
        self.params = self._load_params()
        self.approaches: dict[str, ApproachStats] = self._load_approaches()
        self.outcomes: list[dict[str, Any]] = self._load_outcomes()
        self._ensure_meta()

    # ── persistence ─────────────────────────────────────────────────────

    def _load_params(self) -> CalibParams:
        return CalibParams.from_dict(self.memory.get_fact(FACT_PARAMS) or {})

    def _load_approaches(self) -> dict[str, ApproachStats]:
        raw = self.memory.get_fact(FACT_APPROACHES) or {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, ApproachStats] = {}
        for k, v in raw.items():
            if isinstance(v, dict):
                out[str(k)] = ApproachStats.from_dict(v)
        return out

    def _load_outcomes(self) -> list[dict[str, Any]]:
        raw = self.memory.get_fact(FACT_OUTCOMES) or []
        return list(raw)[-OUTCOME_WINDOW:] if isinstance(raw, list) else []

    def _ensure_meta(self) -> None:
        meta = self.memory.get_fact(FACT_META)
        if not isinstance(meta, dict):
            self.memory.set_fact(
                FACT_META,
                {"created_at": time.time(), "version": 1, "adaptations": 0},
                source="calibration",
            )

    def _save_params(self) -> None:
        self.memory.set_fact(FACT_PARAMS, self.params.to_dict(), source="calibration")

    def _save_approaches(self) -> None:
        payload = {k: v.to_dict() for k, v in self.approaches.items()}
        self.memory.set_fact(FACT_APPROACHES, payload, source="calibration")

    def _save_outcomes(self) -> None:
        self.memory.set_fact(
            FACT_OUTCOMES, list(self.outcomes)[-OUTCOME_WINDOW:], source="calibration"
        )

    # ── public knobs for orchestrator ───────────────────────────────────

    def repair_attempts(self) -> int:
        return int(self.params.max_repair_attempts)

    def learning_attempts(self) -> int:
        return int(self.params.max_learning_attempts)

    def learning_network_budget(self) -> int:
        return int(self.params.max_learning_network_research)

    def gap_stagnation_rounds(self) -> int:
        return int(self.params.gap_stagnation_rounds)

    def recovery_kwargs(self) -> dict[str, int]:
        return {
            "stagnation_limit": int(self.params.stagnation_same_signature),
            "max_attempts": int(self.params.max_recovery_attempts),
            "max_research": int(self.params.max_research_per_repair),
        }

    def research_timeout(self) -> float:
        return float(self.params.research_timeout_sec)

    def max_open_sources(self) -> int:
        return int(self.params.max_open_sources)

    # ── approach priority ───────────────────────────────────────────────

    @staticmethod
    def fingerprint(approach: str, domain: str = "") -> str:
        blob = f"{domain}|{approach or ''}".strip().lower()
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def approach_score(self, fingerprint: str) -> float:
        st = self.approaches.get(fingerprint)
        return st.score() if st else 0.5

    def should_avoid_approach(self, fingerprint: str) -> bool:
        st = self.approaches.get(fingerprint)
        return bool(st and st.should_avoid())

    def rank_approaches(self, candidates: list[str], *, domain: str = "") -> list[str]:
        """Order approach labels by calibrated score (best first)."""
        scored: list[tuple[float, str]] = []
        for a in candidates:
            fp = self.fingerprint(a, domain)
            st = self.approaches.get(fp)
            if st and st.should_avoid():
                scored.append((-1.0, a))
            else:
                scored.append((self.approach_score(fp), a))
        scored.sort(key=lambda x: -x[0])
        return [a for _, a in scored]

    def pick_best_approach(
        self,
        candidates: list[str],
        *,
        domain: str = "",
        failed_fingerprints: Optional[set[str]] = None,
    ) -> str:
        failed = failed_fingerprints or set()
        ordered = self.rank_approaches(candidates, domain=domain)
        for a in ordered:
            fp = self.fingerprint(a, domain)
            if fp in failed or self.should_avoid_approach(fp):
                continue
            return a
        return ordered[0] if ordered else (candidates[0] if candidates else "default")

    # ── observe VERIFY / recovery ───────────────────────────────────────

    def observe_verify(
        self,
        *,
        verified: bool,
        domain: str,
        approach: str = "",
        approach_fingerprint: str = "",
        skill_or_topic: str = "",
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Primary feedback: verifier PASS/FAIL.

        Updates approach stats (multi-attempt window) and global outcomes,
        then may adapt parameters or rollback a pending change.
        """
        fp = approach_fingerprint or self.fingerprint(approach, domain)
        now = time.time()
        st = self.approaches.get(fp) or ApproachStats(
            approach=approach or fp, fingerprint=fp, domain=domain
        )
        st.approach = approach or st.approach
        st.domain = domain or st.domain
        st.updated_at = now
        if verified:
            st.successes += 1
            st.last_result = "PASS"
            st.recent.append(1)
        else:
            st.failures += 1
            st.last_result = "FAIL"
            st.recent.append(0)
        st.recent = st.recent[-16:]
        self.approaches[fp] = st
        self._save_approaches()

        outcome = {
            "ts": now,
            "verified": bool(verified),
            "domain": domain,
            "approach": approach,
            "fingerprint": fp,
            "skill_or_topic": skill_or_topic,
            "extra": dict(extra or {}),
        }
        self.outcomes.append(outcome)
        self.outcomes = self.outcomes[-OUTCOME_WINDOW:]
        self._save_outcomes()

        self.on_log(
            f"CALIBRATE: observe domain={domain} "
            f"{'PASS' if verified else 'FAIL'} "
            f"approach={approach!r} score={st.score():.2f} "
            f"s={st.successes}/f={st.failures}"
        )

        adapt_report = self._maybe_adapt()
        rollback_report = self._maybe_rollback_pending()
        return {
            "fingerprint": fp,
            "score": st.score(),
            "avoid": st.should_avoid(),
            "adapt": adapt_report,
            "rollback": rollback_report,
            "success_rate": self.window_success_rate(),
        }

    def observe_recovery(
        self,
        *,
        stagnated: bool,
        stopped: bool,
        allow_research: bool,
        failure_signature: str = "",
    ) -> None:
        """Secondary signal: stagnation / budget exhaustion patterns."""
        if not stagnated and not stopped:
            return
        # Record as a synthetic outcome tag for adaptation heuristics
        self.outcomes.append({
            "ts": time.time(),
            "verified": False,
            "domain": "recovery",
            "approach": "stagnation" if stagnated else "budget_stop",
            "fingerprint": failure_signature or "recovery",
            "skill_or_topic": "",
            "extra": {
                "stagnated": stagnated,
                "stopped": stopped,
                "allow_research": allow_research,
            },
        })
        self.outcomes = self.outcomes[-OUTCOME_WINDOW:]
        self._save_outcomes()
        if stagnated:
            self.on_log(
                f"CALIBRATE: stagnation noted sig={failure_signature[:16]} "
                f"— will prefer strategy change"
            )
        self._maybe_adapt(force_stagnation=stagnated)

    def window_success_rate(self, n: Optional[int] = None) -> float:
        window = self.outcomes[-(n or OUTCOME_WINDOW):]
        usable = [o for o in window if o.get("domain") != "recovery"]
        if not usable:
            return 0.5
        return sum(1 for o in usable if o.get("verified")) / len(usable)

    # ── adapt + rollback ────────────────────────────────────────────────

    def _snapshot_params(self, reason: str) -> dict[str, Any]:
        stack = self.memory.get_fact(FACT_CONFIG_STACK) or []
        if not isinstance(stack, list):
            stack = []
        snap = {
            "ts": time.time(),
            "reason": reason,
            "params": self.params.to_dict(),
            "success_rate": self.window_success_rate(),
        }
        stack.append(snap)
        self.memory.set_fact(FACT_CONFIG_STACK, stack[-20:], source="calibration")
        return snap

    def _apply_params(self, new_params: CalibParams, *, reason: str) -> dict[str, Any]:
        snap = self._snapshot_params(reason)
        old = self.params.to_dict()
        self.params = new_params.clamped()
        self._save_params()
        baseline = self.window_success_rate()
        self.memory.set_fact(
            FACT_PENDING,
            {
                "ts": time.time(),
                "reason": reason,
                "baseline_rate": baseline,
                "outcomes_at_change": len(self.outcomes),
                "previous": snap,
                "new_params": self.params.to_dict(),
            },
            source="calibration",
        )
        meta = self.memory.get_fact(FACT_META) or {}
        if isinstance(meta, dict):
            meta["adaptations"] = int(meta.get("adaptations") or 0) + 1
            meta["last_adapt_reason"] = reason
            meta["last_adapt_at"] = time.time()
            self.memory.set_fact(FACT_META, meta, source="calibration")
        self.on_log(
            f"CALIBRATE: adapt reason={reason!r} "
            f"params {old} → {self.params.to_dict()}"
        )
        return {"adapted": True, "reason": reason, "params": self.params.to_dict()}

    def _maybe_adapt(self, *, force_stagnation: bool = False) -> dict[str, Any]:
        """
        Propose parameter tweaks from multi-outcome evidence.

        Conservative: only when window is large enough and no pending eval,
        or when stagnation is repeatedly observed.
        """
        pending = self.memory.get_fact(FACT_PENDING)
        if isinstance(pending, dict) and pending.get("ts"):
            return {"adapted": False, "reason": "pending_eval"}

        usable = [o for o in self.outcomes if o.get("domain") != "recovery"]
        if len(usable) < EVAL_MIN_OUTCOMES and not force_stagnation:
            return {"adapted": False, "reason": "insufficient_evidence"}

        rate = self.window_success_rate()
        stagnations = sum(
            1
            for o in self.outcomes[-12:]
            if o.get("domain") == "recovery"
            and (o.get("extra") or {}).get("stagnated")
        )
        repair_fails = sum(
            1
            for o in usable[-12:]
            if o.get("domain") == "repair" and not o.get("verified")
        )
        learning_fails = sum(
            1
            for o in usable[-12:]
            if o.get("domain") == "learning" and not o.get("verified")
        )

        new = CalibParams.from_dict(self.params.to_dict())
        reason = ""

        # Stagnation-heavy → detect earlier / allow one more research
        if force_stagnation or stagnations >= 2:
            if new.stagnation_same_signature > 2:
                new.stagnation_same_signature -= 1
                reason = "stagnation_tighten"
            elif new.max_research_per_repair < 2:
                new.max_research_per_repair += 1
                reason = "stagnation_research_budget"
            else:
                new.gap_stagnation_rounds = min(4, new.gap_stagnation_rounds + 0)
                # Force diverge sooner by keeping stagnation at 2
                reason = "stagnation_noted"

        # Many repair fails → slightly more repair budget (capped)
        elif rate < 0.35 and repair_fails >= 4:
            if new.max_repair_attempts < 7:
                new.max_repair_attempts += 1
                reason = "repair_budget_up"

        # Learning fails with network already used → keep network low, more local attempts
        elif rate < 0.35 and learning_fails >= 4:
            if new.max_learning_attempts < 7:
                new.max_learning_attempts += 1
                reason = "learning_budget_up"

        # High success → tighten budgets (faster)
        elif rate >= 0.75 and len(usable) >= EVAL_MIN_OUTCOMES:
            tightened = False
            if new.max_repair_attempts > 4:
                new.max_repair_attempts -= 1
                tightened = True
            if new.max_open_sources > 3:
                new.max_open_sources -= 1
                tightened = True
            if tightened:
                reason = "success_tighten"

        if not reason:
            return {"adapted": False, "reason": "no_change_needed", "rate": rate}

        if new.to_dict() == self.params.to_dict():
            return {"adapted": False, "reason": "clamped_noop", "rate": rate}

        return self._apply_params(new, reason=reason)

    def _maybe_rollback_pending(self) -> dict[str, Any]:
        pending = self.memory.get_fact(FACT_PENDING)
        if not isinstance(pending, dict) or not pending.get("ts"):
            return {"rollback": False}

        at = int(pending.get("outcomes_at_change") or 0)
        # Count outcomes since change
        since = [o for o in self.outcomes if o.get("ts", 0) >= float(pending.get("ts") or 0)]
        since_v = [o for o in since if o.get("domain") != "recovery"]
        if len(since_v) < EVAL_MIN_OUTCOMES:
            return {"rollback": False, "reason": "eval_pending", "n": len(since_v)}

        baseline = float(pending.get("baseline_rate") or 0.5)
        current = (
            sum(1 for o in since_v if o.get("verified")) / len(since_v)
            if since_v
            else 0.0
        )
        if current + ROLLBACK_MARGIN < baseline:
            prev = (pending.get("previous") or {}).get("params") or {}
            self.params = CalibParams.from_dict(prev)
            self._save_params()
            self.memory.set_fact(FACT_PENDING, {}, source="calibration")
            self.on_log(
                f"CALIBRATE: ROLLBACK — new config worse "
                f"(rate {current:.2f} < baseline {baseline:.2f} - margin) "
                f"restored previous params"
            )
            return {
                "rollback": True,
                "current_rate": current,
                "baseline_rate": baseline,
                "params": self.params.to_dict(),
            }

        # Accept change
        self.memory.set_fact(FACT_PENDING, {}, source="calibration")
        self.on_log(
            f"CALIBRATE: keep adaptation — rate {current:.2f} vs baseline {baseline:.2f}"
        )
        return {
            "rollback": False,
            "accepted": True,
            "current_rate": current,
            "baseline_rate": baseline,
        }

    def force_rollback(self) -> bool:
        """Manual/test helper: restore last snapshot."""
        stack = self.memory.get_fact(FACT_CONFIG_STACK) or []
        if not isinstance(stack, list) or not stack:
            return False
        last = stack[-1]
        self.params = CalibParams.from_dict(last.get("params") or {})
        self._save_params()
        self.memory.set_fact(FACT_PENDING, {}, source="calibration")
        self.on_log("CALIBRATE: force rollback to last snapshot")
        return True

    def status(self) -> dict[str, Any]:
        top = sorted(
            self.approaches.values(), key=lambda s: -s.score()
        )[:8]
        avoided = [s.to_dict() for s in self.approaches.values() if s.should_avoid()]
        return {
            "params": self.params.to_dict(),
            "success_rate": round(self.window_success_rate(), 4),
            "outcomes": len(self.outcomes),
            "approaches": len(self.approaches),
            "top_approaches": [s.to_dict() for s in top],
            "avoided": avoided[:8],
            "pending": self.memory.get_fact(FACT_PENDING) or {},
            "meta": self.memory.get_fact(FACT_META) or {},
        }
