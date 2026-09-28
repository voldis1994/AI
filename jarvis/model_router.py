"""
JARVIS concurrent persistent model pool (ModelRouter).

FAST / REASONING / CODING are independent worker resources available to the
orchestrator immediately — not a sequential FAST→REASONING→CODING pipeline.

Startup: probe Ollama, warm models that fit GPU/RAM (LRU when they don't),
keep_alive residency, readiness status.

Request time: direct tier/work selection, optional parallel calls, escalation
only when a result is truly insufficient, and no duplicate work across models.
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from jarvis import model_config as cfg

logger = logging.getLogger("jarvis.model_router")

# Avoid hammering Ollama /api/tags from GUI ticks and is_available().
TAGS_CACHE_TTL_SEC = 15.0


@dataclass(frozen=True)
class RouteDecision:
    """Outcome of a single pool acquire / route decision."""

    tier: str
    model: str
    work: str
    primary: str
    used_fallback: bool
    reason: str = ""
    warm: bool = False
    ready: bool = True


@dataclass
class ModelWorker:
    """One independent pool worker (tier → resolved model)."""

    tier: str
    model: str
    primary: str
    ready: bool = False          # installed / resolvable
    warm: bool = False           # loaded (or successfully preloaded)
    size_bytes: int = 0
    last_used: float = 0.0
    in_flight: int = 0
    last_error: str = ""
    keep_alive: Any = cfg.KEEP_ALIVE_WARM


@dataclass
class RequestWorkContext:
    """
    Per-request registry: collect model results and prevent redoing the same
    work on a different model unless escalate=True.
    """

    request_id: str = ""
    results: dict[str, str] = field(default_factory=dict)
    decisions: dict[str, RouteDecision] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    @staticmethod
    def work_key(work: str, prompt_fingerprint: str = "", tier: str = "") -> str:
        blob = f"{(work or '').strip().lower()}|{tier}|{prompt_fingerprint}"
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            return self.results.get(key)

    def put(self, key: str, text: str, decision: Optional[RouteDecision] = None) -> None:
        with self._lock:
            self.results[key] = text
            if decision is not None:
                self.decisions[key] = decision

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "request_id": self.request_id,
                "result_keys": list(self.results.keys()),
                "decisions": {
                    k: {
                        "tier": d.tier,
                        "model": d.model,
                        "work": d.work,
                        "reason": d.reason,
                    }
                    for k, d in self.decisions.items()
                },
            }


class ModelPool:
    """
    Concurrent persistent pool of independent FAST / REASONING / CODING workers.

    Thread-safe. Does not invent task answers — only manages model residency
    and routing. Orchestrator / Brain choose work kinds directly.
    """

    def __init__(
        self,
        list_models: Callable[[], Optional[list[str]]],
        on_log: Optional[Callable[[str], None]] = None,
        *,
        prefer_installed: bool = True,
        host: str = "",
        preload_fn: Optional[Callable[[str, Any], bool]] = None,
        unload_fn: Optional[Callable[[str], bool]] = None,
        ps_fn: Optional[Callable[[], Optional[list[dict]]]] = None,
        size_fn: Optional[Callable[[str], Optional[int]]] = None,
    ) -> None:
        self._list_models = list_models
        self.on_log = on_log or (lambda _m: None)
        self.prefer_installed = prefer_installed
        self.host = (host or cfg.DEFAULT_HOST).rstrip("/")
        self._preload_fn = preload_fn
        self._unload_fn = unload_fn
        self._ps_fn = ps_fn
        self._size_fn = size_fn

        self._lock = threading.RLock()
        self._cached_tags: Optional[list[str]] = None
        self._cache_ok = False
        self._cache_ts: float = 0.0
        # Models that crashed the Ollama runner (llama-server) — skip until reset
        self._failed_models: dict[str, str] = {}
        self._cache_ttl = TAGS_CACHE_TTL_SEC
        self._last_route: Optional[RouteDecision] = None

        self.workers: dict[str, ModelWorker] = {}  # tier → worker
        self._model_semaphores: dict[str, threading.Semaphore] = {}
        self._initialized = False
        self._budget_bytes: int = 0
        self._request_ctx: Optional[RequestWorkContext] = None
        self._executor = ThreadPoolExecutor(
            max_workers=6, thread_name_prefix="jarvis-pool"
        )

    # ── logging / cache ─────────────────────────────────────────────────

    def set_logger(self, on_log: Callable[[str], None]) -> None:
        self.on_log = on_log or (lambda _m: None)

    def invalidate_cache(self) -> None:
        with self._lock:
            self._cached_tags = None
            self._cache_ok = False
            self._cache_ts = 0.0

    def available_models(self, *, refresh: bool = False) -> Optional[list[str]]:
        with self._lock:
            now = time.time()
            fresh_enough = (
                self._cache_ok and (now - self._cache_ts) < self._cache_ttl
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

    def mark_model_failed(self, model: str, reason: str = "") -> None:
        """
        Blacklist a model after runner crash / hard failure so the pool
        auto-selects the next installed candidate for that tier.
        """
        name = (model or "").strip()
        if not name:
            return
        key = name.lower()
        with self._lock:
            self._failed_models[key] = (reason or "runner_failed")[:200]
            # Drop warm flag so we don't keep trying the broken worker
            for w in self.workers.values():
                if (w.model or "").strip().lower() == key:
                    w.warm = False
                    w.ready = False
                    w.last_error = reason or "runner_failed"
        msg = f"MODEL SKIP: {name} marked failed — will auto-select next installed"
        self.on_log(msg)
        logger.warning("%s (%s)", msg, reason)
        # Re-bind workers that pointed at the failed model
        try:
            tags = self.available_models(refresh=True)
        except Exception:
            tags = None
        with self._lock:
            for tier, w in list(self.workers.items()):
                if (w.model or "").strip().lower() != key:
                    continue
                nxt, _, _ = self._resolve_model(tier, tags)
                if nxt and nxt.strip().lower() != key:
                    w.model = nxt
                    w.ready = True
                    w.warm = False
                    self.on_log(f"MODEL AUTO-SELECT: {tier} → {nxt} (after skip {name})")

    def clear_model_failures(self) -> None:
        with self._lock:
            self._failed_models.clear()

    def failed_models(self) -> dict[str, str]:
        with self._lock:
            return dict(self._failed_models)

    # ── request context (dedupe + collect) ──────────────────────────────

    def begin_request(self, request_id: str = "") -> RequestWorkContext:
        ctx = RequestWorkContext(request_id=request_id or "")
        with self._lock:
            self._request_ctx = ctx
        return ctx

    def end_request(self) -> Optional[dict[str, Any]]:
        with self._lock:
            ctx = self._request_ctx
            self._request_ctx = None
        return ctx.as_dict() if ctx else None

    def current_request(self) -> Optional[RequestWorkContext]:
        with self._lock:
            return self._request_ctx

    # ── init / warm ─────────────────────────────────────────────────────

    def initialize(self, *, warm: bool = True) -> dict[str, Any]:
        """
        Probe Ollama, resolve per-tier workers, optionally warm what fits.
        Safe to call multiple times (idempotent refresh + warm).
        """
        tags = self.available_models(refresh=True)
        online = tags is not None
        report: dict[str, Any] = {
            "online": online,
            "installed": list(tags or []),
            "workers": {},
            "warmed": [],
            "skipped_warm": [],
            "budget_bytes": 0,
        }
        if not online:
            self.on_log("MODEL POOL: Ollama OFFLINE — pool not warmed")
            logger.warning("MODEL POOL: Ollama OFFLINE")
            with self._lock:
                self._initialized = True
                self.workers = {}
            return report

        budget = self.detect_memory_budget_bytes()
        self._budget_bytes = budget
        report["budget_bytes"] = budget

        workers: dict[str, ModelWorker] = {}
        for tier in (cfg.TIER_FAST, cfg.TIER_REASONING, cfg.TIER_CODING):
            resolved, used_fb, resolve_reason = self._resolve_model(tier, tags)
            primary = cfg.models_for_tier(tier)[0]
            model = resolved or primary
            size = self._model_size_bytes(model, tags)
            w = ModelWorker(
                tier=tier,
                model=model,
                primary=primary,
                ready=bool(resolved),
                warm=False,
                size_bytes=size,
                last_used=0.0,
            )
            workers[tier] = w
            self._model_semaphores.setdefault(
                model, threading.Semaphore(cfg.MAX_INFLIGHT_PER_MODEL)
            )
            report["workers"][tier] = {
                "model": model,
                "ready": w.ready,
                "size_bytes": size,
                "fallback": used_fb,
                "reason": resolve_reason,
            }
            if resolved and used_fb:
                self.on_log(
                    f"MODEL AUTO-SELECT: {tier} → {resolved} "
                    f"(preferred {primary} not used; {resolve_reason})"
                )

        with self._lock:
            self.workers = workers
            self._initialized = True

        self.on_log(
            f"MODEL POOL: initialized workers="
            + ", ".join(
                f"{t}:{w.model}{'✓' if w.ready else '·'}"
                for t, w in workers.items()
            )
            + f" budget≈{budget // (1024**3)}GiB"
        )

        if warm:
            warm_report = self.warm_start()
            report["warmed"] = warm_report.get("warmed") or []
            report["skipped_warm"] = warm_report.get("skipped") or []
        return report

    def warm_start(self) -> dict[str, Any]:
        """
        Preload models that fit the memory budget (warm order).
        Excess tiers stay cold and load on demand via LRU.
        """
        with self._lock:
            workers = dict(self.workers)
            budget = self._budget_bytes or self.detect_memory_budget_bytes()
            self._budget_bytes = budget

        warmed: list[str] = []
        skipped: list[str] = []
        used = 0
        # Prefer already-resident models from /api/ps
        resident = {m.get("name", "").lower() for m in (self.loaded_models() or [])}

        for tier in cfg.WARM_ORDER:
            w = workers.get(tier)
            if not w or not w.ready:
                if w:
                    skipped.append(f"{tier}:not_ready")
                continue
            # Already resident counts as warm without re-preload cost
            if w.model.lower() in resident:
                w.warm = True
                w.last_used = time.time()
                used += w.size_bytes
                warmed.append(f"{tier}:{w.model}:resident")
                continue
            if used + w.size_bytes > budget and warmed:
                skipped.append(f"{tier}:{w.model}:budget")
                self.on_log(
                    f"MODEL POOL: skip warm {tier}→{w.model} "
                    f"(budget {budget // (1024**3)}GiB)"
                )
                continue
            ok = self.preload(w.model, keep_alive=cfg.KEEP_ALIVE_WARM)
            if ok:
                w.warm = True
                w.last_used = time.time()
                used += w.size_bytes
                warmed.append(f"{tier}:{w.model}")
                self.on_log(
                    f"MODEL POOL: WARM {tier} → {w.model} keep_alive={cfg.KEEP_ALIVE_WARM}"
                )
            else:
                skipped.append(f"{tier}:{w.model}:preload_fail")
                w.last_error = "preload_failed"

        with self._lock:
            self.workers.update(workers)

        return {"warmed": warmed, "skipped": skipped, "budget_bytes": budget}

    def preload(self, model: str, *, keep_alive: Any = None) -> bool:
        """Load model into Ollama memory (empty generate + keep_alive)."""
        ka = cfg.KEEP_ALIVE_WARM if keep_alive is None else keep_alive
        if self._preload_fn is not None:
            try:
                return bool(self._preload_fn(model, ka))
            except Exception as exc:
                logger.info("preload_fn failed (%s): %s", model, exc)
                return False
        return self._http_preload(model, ka)

    def unload(self, model: str) -> bool:
        """Evict model from Ollama memory (keep_alive=0)."""
        if self._unload_fn is not None:
            try:
                return bool(self._unload_fn(model))
            except Exception as exc:
                logger.info("unload_fn failed (%s): %s", model, exc)
                return False
        return self._http_preload(model, cfg.KEEP_ALIVE_UNLOAD)

    def ensure_warm(self, tier: str) -> bool:
        """Make sure the worker for tier is loaded; LRU-evict if needed."""
        with self._lock:
            w = self.workers.get((tier or "").strip().upper())
        if w is None:
            return False
        if w.warm:
            return True
        if not w.ready:
            return False
        # Evict LRU cold-space if budget tight
        self._maybe_evict_for(w.size_bytes, keep_model=w.model)
        ok = self.preload(w.model, keep_alive=cfg.KEEP_ALIVE_ACTIVE)
        with self._lock:
            ww = self.workers.get(w.tier)
            if ww and ok:
                ww.warm = True
                ww.last_used = time.time()
        if ok:
            self.on_log(f"MODEL POOL: on-demand WARM {w.tier} → {w.model}")
        return ok

    def _maybe_evict_for(self, need_bytes: int, *, keep_model: str) -> None:
        """LRU unload warm workers until need_bytes likely fits."""
        with self._lock:
            warm_list = [
                w for w in self.workers.values()
                if w.warm and w.model != keep_model and w.in_flight == 0
            ]
            warm_list.sort(key=lambda x: x.last_used)  # oldest first
            used = sum(w.size_bytes for w in self.workers.values() if w.warm)
            budget = self._budget_bytes or self.detect_memory_budget_bytes()

        for w in warm_list:
            if used + need_bytes <= budget:
                break
            if self.unload(w.model):
                with self._lock:
                    ww = self.workers.get(w.tier)
                    if ww:
                        ww.warm = False
                used = max(0, used - w.size_bytes)
                self.on_log(
                    f"MODEL POOL: LRU unload {w.tier}→{w.model} "
                    f"(free space for {keep_model})"
                )

    # ── status ──────────────────────────────────────────────────────────

    def any_configured_online(self) -> bool:
        tags = self.available_models()
        if tags is None:
            return False
        return any(self.model_in_tags(m, tags) for m in cfg.all_configured_models())

    def status(self, *, refresh: bool = False) -> str:
        tags = self.available_models(refresh=refresh)
        if tags is None:
            return "OFFLINE"
        if any(self.model_in_tags(m, tags) for m in cfg.all_configured_models()):
            return "ONLINE"
        # Auto-discovered installed models also count as ONLINE
        for tier in (cfg.TIER_FAST, cfg.TIER_REASONING, cfg.TIER_CODING):
            resolved, _, _ = self._resolve_model(tier, tags)
            if resolved:
                return "ONLINE"
        return "MODEL MISSING"

    def status_detail(self) -> dict[str, Any]:
        tags = self.available_models()
        resident = {
            str(m.get("name") or m.get("model") or "").lower()
            for m in (self.loaded_models() or [])
        }
        tiers: dict[str, Any] = {}
        with self._lock:
            workers = dict(self.workers)
        for tier in (cfg.TIER_FAST, cfg.TIER_REASONING, cfg.TIER_CODING):
            primary = cfg.models_for_tier(tier)[0]
            resolved, fb, reason = self._resolve_model(tier, tags)
            w = workers.get(tier)
            model = (w.model if w else resolved) or primary
            warm = bool(w.warm) if w else (model.lower() in resident)
            ready = bool(resolved)
            tiers[tier] = {
                "primary": primary,
                "resolved": model,
                "fallback": fb,
                "reason": reason,
                "online": bool(ready and tags is not None),
                "ready": ready,
                "warm": warm,
                "in_flight": int(w.in_flight) if w else 0,
                "size_bytes": int(w.size_bytes) if w else self._model_size_bytes(model, tags),
                "last_used": float(w.last_used) if w else 0.0,
            }
        if tags is None:
            agg = "OFFLINE"
        elif any(t.get("online") for t in tiers.values()):
            agg = "ONLINE"
        else:
            agg = "MODEL MISSING"
        return {
            "status": agg,
            "pool": True,
            "tiers": tiers,
            "installed": list(tags or []),
            "loaded": list(resident),
            "budget_bytes": self._budget_bytes,
            "initialized": self._initialized,
        }

    def tier_for_work(self, work: str) -> str:
        return cfg.tier_for_work(work)

    # ── acquire / route / escalate ──────────────────────────────────────

    def acquire(
        self,
        work: str = "",
        *,
        tier: Optional[str] = None,
        force_model: Optional[str] = None,
        ensure_warm: bool = True,
        log: bool = True,
    ) -> RouteDecision:
        """Direct worker selection — no pipeline gate through other tiers."""
        return self.route(
            work=work,
            tier=tier,
            force_model=force_model,
            ensure_warm=ensure_warm,
            log=log,
        )

    def route(
        self,
        work: str = "",
        *,
        tier: Optional[str] = None,
        force_model: Optional[str] = None,
        ensure_warm: bool = True,
        log: bool = True,
    ) -> RouteDecision:
        """
        Choose model for this work kind / tier (independent worker).

        Logs: MODEL ROUTE: <TIER> → <model>
              MODEL FALLBACK: <primary> → <model> (<tier>)
        """
        if not self._initialized:
            try:
                self.initialize(warm=False)
            except Exception:
                pass

        work_key = (work or "").strip().lower() or "unknown"
        chosen_tier = (tier or cfg.tier_for_work(work_key)).strip().upper()
        if chosen_tier not in cfg.VALID_TIERS:
            chosen_tier = cfg.TIER_FAST

        tags = self.available_models()
        catalog_primary = cfg.models_for_tier(chosen_tier)[0]

        if force_model:
            decision = RouteDecision(
                tier=chosen_tier,
                model=force_model,
                work=work_key,
                primary=catalog_primary,
                used_fallback=force_model.strip().lower()
                != catalog_primary.strip().lower(),
                reason="forced",
                warm=False,
                ready=True,
            )
            self._commit(decision, log=log)
            return decision

        resolved, used_fb, resolve_reason = self._resolve_model(chosen_tier, tags)
        if not resolved:
            resolved = catalog_primary
            used_fb = False
            reason = "no_installed_candidate"
            ready = False
        else:
            reason = resolve_reason or ("fallback" if used_fb else "primary")
            ready = True
            used_fb = (
                resolved.strip().lower() != catalog_primary.strip().lower()
            )

        warm = False
        if ensure_warm and ready:
            warm = self.ensure_warm(chosen_tier)
            with self._lock:
                w = self.workers.get(chosen_tier)
                if w:
                    w.last_used = time.time()
                    warm = w.warm

        decision = RouteDecision(
            tier=chosen_tier,
            model=resolved,
            work=work_key,
            primary=primary,
            used_fallback=used_fb,
            reason=reason,
            warm=warm,
            ready=ready,
        )
        self._commit(decision, log=log)
        return decision

    def escalate(
        self,
        from_tier: str,
        work: str = "",
        *,
        log: bool = True,
        prior_key: str = "",
    ) -> Optional[RouteDecision]:
        """
        Escalate only when prior result is insufficient.

        FAST → REASONING. CODING → CODING (retry same tier).
        Refuses to re-run the same work_key on another model unless prior_key
        is provided and marked insufficient by the caller (caller clears cache).
        """
        src = (from_tier or "").strip().upper()
        dest = cfg.TIER_ESCALATION.get(src)
        if dest is None:
            return None
        ctx = self.current_request()
        if ctx and prior_key:
            # Same work already has a good cached result — do not redo
            existing = ctx.get(prior_key)
            if existing is not None and not self.result_insufficient(existing):
                if log:
                    self.on_log(
                        f"MODEL DEDUPE: skip escalate {src}→{dest} "
                        f"(work already sufficient)"
                    )
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

    def keep_alive_for(self, model: str) -> Any:
        with self._lock:
            for w in self.workers.values():
                if w.model == model and w.warm:
                    return cfg.KEEP_ALIVE_WARM
        return cfg.KEEP_ALIVE_ACTIVE

    def mark_in_flight(self, model: str, delta: int = 1) -> None:
        with self._lock:
            for w in self.workers.values():
                if w.model == model:
                    w.in_flight = max(0, w.in_flight + delta)
                    if delta > 0:
                        w.last_used = time.time()
            sem = self._model_semaphores.get(model)
        # Semaphores gate concurrency outside the status lock
        if sem is None:
            return
        if delta > 0:
            sem.acquire()
        elif delta < 0:
            try:
                sem.release()
            except ValueError:
                pass

    # ── parallel independent calls ──────────────────────────────────────

    def run_parallel(
        self,
        jobs: list[dict[str, Any]],
        runner: Callable[[dict[str, Any]], Any],
    ) -> list[Any]:
        """
        Run independent model jobs concurrently.

        Each job is a dict passed to runner (caller builds work/tier/prompt).
        Results preserve input order. Not for dependent pipeline stages.
        """
        if not jobs:
            return []
        if len(jobs) == 1:
            return [runner(jobs[0])]

        results: list[Any] = [None] * len(jobs)
        futures = {
            self._executor.submit(runner, job): idx
            for idx, job in enumerate(jobs)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:
                results[idx] = {"ok": False, "error": str(exc)}
        return results

    # ── quality / insufficiency ─────────────────────────────────────────

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
                if len(raw) < 40:
                    return True
        return False

    # ── memory / ps helpers ─────────────────────────────────────────────

    def detect_memory_budget_bytes(self) -> int:
        """
        Estimate bytes available for warm models (GPU VRAM preferred, else RAM).
        Conservative fraction to avoid thrashing.
        """
        vram = self._detect_vram_bytes()
        if vram and vram > 0:
            return int(vram * cfg.WARM_BUDGET_FRACTION)
        ram = self._detect_ram_bytes()
        if ram and ram > 0:
            return int(ram * cfg.WARM_BUDGET_FRACTION)
        # Fallback ~12 GiB soft budget
        return int(12 * 1024**3 * cfg.WARM_BUDGET_FRACTION)

    def loaded_models(self) -> Optional[list[dict[str, Any]]]:
        if self._ps_fn is not None:
            try:
                return self._ps_fn()
            except Exception:
                return None
        return self._http_ps()

    def _detect_vram_bytes(self) -> int:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                timeout=3,
            ).decode("utf-8", errors="replace")
            total = 0
            for line in out.strip().splitlines():
                try:
                    total += int(float(line.strip())) * 1024 * 1024
                except ValueError:
                    continue
            return total
        except Exception:
            return 0

    def _detect_ram_bytes(self) -> int:
        try:
            # Linux MemAvailable
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        parts = line.split()
                        return int(parts[1]) * 1024
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        # Prefer available; stash total as fallback
                        total = int(parts[1]) * 1024
                        return int(total * 0.5)
        except Exception:
            pass
        try:
            page = os.sysconf("SC_PAGE_SIZE")
            phys = os.sysconf("SC_PHYS_PAGES")
            return int(page * phys * 0.5)
        except Exception:
            return 0

    def _model_size_bytes(self, model: str, tags: Optional[list[str]]) -> int:
        if self._size_fn is not None:
            try:
                sz = self._size_fn(model)
                if sz:
                    return int(sz)
            except Exception:
                pass
        # Try loaded ps entry
        for m in self.loaded_models() or []:
            name = str(m.get("name") or m.get("model") or "")
            if name.lower() == (model or "").lower():
                for key in ("size", "size_vram"):
                    if m.get(key):
                        try:
                            return int(m[key])
                        except Exception:
                            pass
        return cfg.size_hint_bytes(model)

    def _http_preload(self, model: str, keep_alive: Any) -> bool:
        try:
            import json
            import urllib.request

            payload = json.dumps(
                {
                    "model": model,
                    "messages": [],
                    "stream": False,
                    "keep_alive": keep_alive,
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                f"{self.host}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                return 200 <= int(resp.status) < 300
        except Exception as exc:
            logger.info("preload HTTP failed (%s): %s", model, exc)
            # generate fallback
            try:
                import json
                import urllib.request

                payload = json.dumps(
                    {
                        "model": model,
                        "prompt": "",
                        "stream": False,
                        "keep_alive": keep_alive,
                    }
                ).encode("utf-8")
                req = urllib.request.Request(
                    f"{self.host}/api/generate",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=45) as resp:
                    return 200 <= int(resp.status) < 300
            except Exception as exc2:
                logger.info("preload generate failed (%s): %s", model, exc2)
                return False

    def _http_ps(self) -> Optional[list[dict[str, Any]]]:
        try:
            import json
            import urllib.request

            with urllib.request.urlopen(f"{self.host}/api/ps", timeout=5) as resp:
                if resp.status != 200:
                    return None
                data = json.loads(resp.read().decode("utf-8"))
            models = data.get("models") if isinstance(data, dict) else None
            return list(models or [])
        except Exception:
            return None

    # ── internals ───────────────────────────────────────────────────────

    def candidates_for_tier(
        self, tier: str, tags: Optional[list[str]] = None
    ) -> list[str]:
        """Catalog + auto-discovered installed models, minus runner failures."""
        with self._lock:
            ban = set(self._failed_models.keys())
        return cfg.discover_models_for_tier(tier, tags, exclude=ban)

    def _resolve_model(
        self, tier: str, tags: Optional[list[str]]
    ) -> tuple[Optional[str], bool, str]:
        """
        Pick best model for tier.

        Returns (model|None, used_fallback, reason).
        Prefers installed catalog entries, then auto-discovered tags.
        """
        catalog = cfg.models_for_tier(tier)
        primary = catalog[0] if catalog else cfg.DEFAULT_MODEL
        candidates = self.candidates_for_tier(tier, tags)
        if not candidates:
            return None, False, "empty_candidates"
        if not self.prefer_installed or tags is None:
            return candidates[0], candidates[0] != primary, (
                "catalog" if candidates[0] == primary else "auto"
            )
        tag_set = {n.strip().lower() for n in tags}
        for i, name in enumerate(candidates):
            if name.strip().lower() in tag_set:
                in_catalog = name.strip().lower() in {
                    c.strip().lower() for c in catalog
                }
                if i == 0 and name.strip().lower() == primary.strip().lower():
                    reason = "primary"
                elif in_catalog:
                    reason = "fallback"
                else:
                    reason = "auto_installed"
                return name, name.strip().lower() != primary.strip().lower(), reason
        return None, False, "no_installed_candidate"

    def _commit(self, decision: RouteDecision, *, log: bool) -> None:
        with self._lock:
            self._last_route = decision
            w = self.workers.get(decision.tier)
            if w and w.model == decision.model:
                w.last_used = time.time()
        if not log:
            return
        warm_mark = " warm" if decision.warm else ""
        msg = f"MODEL ROUTE: {decision.tier} → {decision.model}{warm_mark}"
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

    def close(self) -> None:
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


# Backward-compatible name — ModelRouter IS the concurrent model pool.
class ModelRouter(ModelPool):
    """Alias: tiers are independent pool workers, not pipeline gates."""

    pass
