"""
Universal Multi-Model Router for JARVIS.

REQUEST work kind → tier (FAST / REASONING / CODING) → best available model,
with automatic fallback when a model is missing and escalation when a FAST
result is insufficient.

Never hardcodes user tasks — only work type / complexity.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from jarvis import model_config as cfg

logger = logging.getLogger("jarvis.model_router")

# Avoid hammering Ollama /api/tags from GUI ticks and is_available().
TAGS_CACHE_TTL_SEC = 15.0


@dataclass(frozen=True)
class RouteDecision:
    """Outcome of a single routing decision."""

    tier: str
    model: str
    work: str
    primary: str
    used_fallback: bool
    reason: str = ""


class ModelRouter:
    """
    Selects an Ollama model by work kind / tier.

    Thread-safe for parallel independent LLM calls (shared tag cache).
    Does not store per-request results — callers keep request_id isolation.
    """

    def __init__(
        self,
        list_models: Callable[[], Optional[list[str]]],
        on_log: Optional[Callable[[str], None]] = None,
        *,
        prefer_installed: bool = True,
    ) -> None:
        self._list_models = list_models
        self.on_log = on_log or (lambda _m: None)
        self.prefer_installed = prefer_installed
        self._lock = threading.RLock()
        self._cached_tags: Optional[list[str]] = None
        self._cache_ok = False
        self._cache_ts: float = 0.0
        self._cache_ttl = TAGS_CACHE_TTL_SEC
        self._last_route: Optional[RouteDecision] = None

    def set_logger(self, on_log: Callable[[str], None]) -> None:
        self.on_log = on_log or (lambda _m: None)

    def invalidate_cache(self) -> None:
        with self._lock:
            self._cached_tags = None
            self._cache_ok = False
            self._cache_ts = 0.0

    def available_models(self, *, refresh: bool = False) -> Optional[list[str]]:
        """Installed model names, or None if Ollama unreachable."""
        with self._lock:
            now = time.time()
            fresh_enough = (
                self._cache_ok
                and (now - self._cache_ts) < self._cache_ttl
            )
            if not refresh and fresh_enough:
                return list(self._cached_tags) if self._cached_tags is not None else None
            tags = self._list_models()
            self._cached_tags = list(tags) if tags is not None else None
            self._cache_ok = True
            self._cache_ts = now
            return list(self._cached_tags) if self._cached_tags is not None else None

    def model_in_tags(self, model: str, tags: Optional[list[str]]) -> bool:
        if tags is None:
            return False
        target = (model or "").strip().lower()
        normalized = {n.strip().lower() for n in tags}
        return target in normalized

    def any_configured_online(self) -> bool:
        tags = self.available_models()
        if tags is None:
            return False
        return any(self.model_in_tags(m, tags) for m in cfg.all_configured_models())

    def status(self, *, refresh: bool = False) -> str:
        """
        Aggregate brain status across the multi-model catalog:
          ONLINE / MODEL MISSING / OFFLINE

        Uses cached tags by default (TTL) — GUI ticks must not force /api/tags.
        Pass refresh=True after connection failures or explicit /status.
        """
        tags = self.available_models(refresh=refresh)
        if tags is None:
            return "OFFLINE"
        if any(self.model_in_tags(m, tags) for m in cfg.all_configured_models()):
            return "ONLINE"
        return "MODEL MISSING"

    def status_detail(self) -> dict[str, Any]:
        """Per-tier primary + resolved model for dashboards."""
        tags = self.available_models()
        tiers: dict[str, Any] = {}
        for tier in (cfg.TIER_FAST, cfg.TIER_REASONING, cfg.TIER_CODING):
            primary = cfg.models_for_tier(tier)[0]
            resolved, fb = self._resolve_model(tier, tags)
            tiers[tier] = {
                "primary": primary,
                "resolved": resolved,
                "fallback": fb,
                "online": bool(resolved and tags is not None and self.model_in_tags(resolved, tags)),
            }
        if tags is None:
            agg = "OFFLINE"
        elif any(t.get("online") for t in tiers.values()):
            agg = "ONLINE"
        else:
            agg = "MODEL MISSING"
        return {
            "status": agg,
            "tiers": tiers,
            "installed": list(tags or []),
        }

    def tier_for_work(self, work: str) -> str:
        return cfg.tier_for_work(work)

    def route(
        self,
        work: str = "",
        *,
        tier: Optional[str] = None,
        force_model: Optional[str] = None,
        log: bool = True,
    ) -> RouteDecision:
        """
        Choose model for this work kind / tier.

        Logs: MODEL ROUTE: <TIER> → <model>
              MODEL FALLBACK: <primary> → <model> (<tier>)
        """
        work_key = (work or "").strip().lower() or "unknown"
        chosen_tier = (tier or cfg.tier_for_work(work_key)).strip().upper()
        if chosen_tier not in cfg.VALID_TIERS:
            chosen_tier = cfg.TIER_REASONING

        tags = self.available_models()
        candidates = cfg.models_for_tier(chosen_tier)
        primary = candidates[0]

        if force_model:
            decision = RouteDecision(
                tier=chosen_tier,
                model=force_model,
                work=work_key,
                primary=primary,
                used_fallback=force_model.strip() != primary,
                reason="forced",
            )
            self._commit(decision, log=log)
            return decision

        resolved, used_fb = self._resolve_model(chosen_tier, tags)
        if not resolved:
            # Last resort: primary name even if missing (call will error / offline path)
            resolved = primary
            used_fb = False
            reason = "no_installed_candidate"
        else:
            reason = "fallback" if used_fb else "primary"

        decision = RouteDecision(
            tier=chosen_tier,
            model=resolved,
            work=work_key,
            primary=primary,
            used_fallback=used_fb,
            reason=reason,
        )
        self._commit(decision, log=log)
        return decision

    def escalate(
        self,
        from_tier: str,
        work: str = "",
        *,
        log: bool = True,
    ) -> Optional[RouteDecision]:
        """
        Escalate to a stronger tier when the prior result is insufficient.

        FAST → REASONING. CODING → CODING (retry same tier; caller adds failure context).
        """
        src = (from_tier or "").strip().upper()
        dest = cfg.TIER_ESCALATION.get(src)
        if dest is None:
            return None
        if log:
            if dest == src:
                self.on_log(f"MODEL RETRY: {src} (failure context)")
                logger.info("MODEL RETRY: %s (failure context)", src)
            else:
                self.on_log(f"MODEL ESCALATION: {src} → {dest}")
                logger.info("MODEL ESCALATION: %s → %s", src, dest)
        return self.route(work=work or "escalated", tier=dest, log=log)

    def next_tier(self, tier: str) -> Optional[str]:
        return cfg.TIER_ESCALATION.get((tier or "").strip().upper())

    # ── quality / insufficiency (universal, not task-specific) ──────────

    @staticmethod
    def result_insufficient(
        text: str,
        *,
        expect_json: bool = False,
        expect_code: bool = False,
        error_only: bool = False,
    ) -> bool:
        """
        Deterministic checks that a model reply is unusable.

        Does NOT judge semantic task success — only empty/error/malformed output.
        Verifier / Python checks remain authoritative for goal achievement.

        error_only=True: escalate only on empty / [BRAIN ERROR] / unreachable
        (not on weak JSON) — use for cheap FAST micro-tasks with offline fallbacks.
        """
        raw = (text or "").strip()
        if not raw:
            return True
        if raw.startswith("[BRAIN ERROR]"):
            return True
        low = raw.lower()
        if "ollama unreachable" in low or "connection refused" in low:
            return True
        if error_only:
            return False
        if expect_json:
            if "{" not in raw and "[" not in raw:
                return True
        if expect_code:
            if "def run" not in raw and "SKILL_META" not in raw and "```" not in raw:
                # Clearly not code-shaped
                if len(raw) < 40:
                    return True
        return False

    # ── internals ───────────────────────────────────────────────────────

    def _resolve_model(
        self, tier: str, tags: Optional[list[str]]
    ) -> tuple[Optional[str], bool]:
        candidates = cfg.models_for_tier(tier)
        if not self.prefer_installed or tags is None:
            # Server unknown — still prefer primary; caller may fail soft
            return candidates[0], False
        for i, name in enumerate(candidates):
            if self.model_in_tags(name, tags):
                return name, i > 0
        return None, False

    def _commit(self, decision: RouteDecision, *, log: bool) -> None:
        with self._lock:
            self._last_route = decision
        if not log:
            return
        msg = f"MODEL ROUTE: {decision.tier} → {decision.model}"
        self.on_log(msg)
        logger.info(msg)
        if decision.used_fallback:
            fb = (
                f"MODEL FALLBACK: {decision.primary} → {decision.model} "
                f"({decision.tier})"
            )
            self.on_log(fb)
            logger.info(fb)

    @property
    def last_route(self) -> Optional[RouteDecision]:
        with self._lock:
            return self._last_route
