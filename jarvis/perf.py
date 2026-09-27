"""
Request-scoped performance telemetry for JARVIS.

Tracks stage durations and model-call timings so each cycle can report
where time was spent. Pure Python — no LLM involvement.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional


@dataclass
class PerfSpan:
    name: str
    started_at: float
    ended_at: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ms(self) -> float:
        end = self.ended_at or time.perf_counter()
        return max(0.0, (end - self.started_at) * 1000.0)


@dataclass
class ModelCall:
    work: str
    tier: str
    model: str
    ms: float
    ok: bool = True
    meta: dict[str, Any] = field(default_factory=dict)


class PerfTracker:
    """Thread-safe per-request performance recorder."""

    def __init__(self, request_id: str = "") -> None:
        self.request_id = request_id or ""
        self._lock = threading.RLock()
        self.spans: list[PerfSpan] = []
        self.model_calls: list[ModelCall] = []
        self._open: dict[str, PerfSpan] = {}
        self.t0 = time.perf_counter()

    def reset(self, request_id: str = "") -> None:
        with self._lock:
            self.request_id = request_id or self.request_id
            self.spans.clear()
            self.model_calls.clear()
            self._open.clear()
            self.t0 = time.perf_counter()

    def begin(self, name: str, **meta: Any) -> None:
        with self._lock:
            self._open[name] = PerfSpan(
                name=name, started_at=time.perf_counter(), meta=dict(meta)
            )

    def end(self, name: str, **meta: Any) -> float:
        with self._lock:
            span = self._open.pop(name, None)
            if span is None:
                span = PerfSpan(name=name, started_at=time.perf_counter())
            span.ended_at = time.perf_counter()
            if meta:
                span.meta.update(meta)
            self.spans.append(span)
            return span.ms

    @contextmanager
    def stage(self, name: str, **meta: Any) -> Iterator["PerfTracker"]:
        self.begin(name, **meta)
        try:
            yield self
        finally:
            self.end(name)

    def record_model(
        self,
        *,
        work: str,
        tier: str,
        model: str,
        ms: float,
        ok: bool = True,
        **meta: Any,
    ) -> None:
        with self._lock:
            self.model_calls.append(
                ModelCall(
                    work=str(work or ""),
                    tier=str(tier or ""),
                    model=str(model or ""),
                    ms=float(ms),
                    ok=bool(ok),
                    meta=dict(meta),
                )
            )

    def total_ms(self) -> float:
        return max(0.0, (time.perf_counter() - self.t0) * 1000.0)

    def stage_totals(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        with self._lock:
            for s in self.spans:
                totals[s.name] = totals.get(s.name, 0.0) + s.ms
        return totals

    def model_totals(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        with self._lock:
            for m in self.model_calls:
                key = f"{m.tier}:{m.work}:{m.model}" if m.model else f"{m.tier}:{m.work}"
                totals[key] = totals.get(key, 0.0) + m.ms
        return totals

    def as_dict(self) -> dict[str, Any]:
        stages = self.stage_totals()
        models = self.model_totals()
        with self._lock:
            return {
                "request_id": self.request_id,
                "total_ms": round(self.total_ms(), 1),
                "stages": {k: round(v, 1) for k, v in stages.items()},
                "model_calls": [
                    {
                        "work": m.work,
                        "tier": m.tier,
                        "model": m.model,
                        "ms": round(m.ms, 1),
                        "ok": m.ok,
                    }
                    for m in self.model_calls
                ],
                "model_totals_ms": {k: round(v, 1) for k, v in models.items()},
                "model_call_count": len(self.model_calls),
            }

    def summary_lines(self, limit: int = 12) -> list[str]:
        data = self.as_dict()
        lines = [
            f"PERF total={data['total_ms']:.0f}ms "
            f"model_calls={data['model_call_count']}"
        ]
        stages = sorted(
            (data.get("stages") or {}).items(), key=lambda kv: -kv[1]
        )
        if stages:
            top = ", ".join(f"{n}={ms:.0f}ms" for n, ms in stages[:limit])
            lines.append(f"PERF stages: {top}")
        models = sorted(
            (data.get("model_totals_ms") or {}).items(), key=lambda kv: -kv[1]
        )
        if models:
            top_m = ", ".join(f"{n}={ms:.0f}ms" for n, ms in models[:limit])
            lines.append(f"PERF models: {top_m}")
        return lines


# Process-wide hook brain can call without importing orchestrator.
_TLS = threading.local()


def set_active_tracker(tracker: Optional[PerfTracker]) -> None:
    _TLS.tracker = tracker


def get_active_tracker() -> Optional[PerfTracker]:
    return getattr(_TLS, "tracker", None)
