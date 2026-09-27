"""
Central model / tier configuration for JARVIS concurrent model pool.

FAST / REASONING / CODING are independent worker resources — not a pipeline.
All Ollama model names and work→tier mappings live HERE.
"""

from __future__ import annotations

from typing import Any

# Legacy single-model default (kept as universal fallback end of chain).
DEFAULT_MODEL = "qwen2.5-coder:7b"
DEFAULT_HOST = "http://127.0.0.1:11434"

# Tier labels used in MODEL ROUTE / ESCALATION / FALLBACK / POOL logs.
TIER_FAST = "FAST"
TIER_REASONING = "REASONING"
TIER_CODING = "CODING"

VALID_TIERS = frozenset({TIER_FAST, TIER_REASONING, TIER_CODING})

# Warm order: smallest / most-used first so we always get a hot FAST worker.
WARM_ORDER = (TIER_FAST, TIER_CODING, TIER_REASONING)

# keep_alive for warmed / in-use models (Ollama duration string or seconds).
# Not -1 forever — allows LRU eviction when VRAM is tight.
KEEP_ALIVE_WARM = "30m"
KEEP_ALIVE_ACTIVE = "15m"
# Unload immediately (Ollama: keep_alive=0)
KEEP_ALIVE_UNLOAD = 0

# Soft memory budget fraction of detected free RAM/VRAM for warm set.
WARM_BUDGET_FRACTION = 0.75
# Always try to keep at least this many tier primaries warm when possible.
MIN_WARM_TIERS = 1
# Concurrent in-flight chats per model (Ollama serializes most GPU work anyway).
MAX_INFLIGHT_PER_MODEL = 2

# Approximate sizes (bytes) when /api/tags omits size — used only for warm planning.
# Rough Q4-ish footprints; not task-specific.
MODEL_SIZE_HINTS_BYTES: dict[str, int] = {
    "qwen3:4b": 2_500_000_000,
    "qwen2.5:7b": 4_500_000_000,
    "qwen2.5-coder:7b": 4_500_000_000,
    "qwen2.5:14b": 8_500_000_000,
    "qwen2.5-coder:14b": 8_500_000_000,
    "qwen3:30b": 18_000_000_000,
    "qwen2.5:32b": 19_000_000_000,
    "qwen3-coder:30b": 18_000_000_000,
}

# Primary + fallback chains per tier. First installed/available model wins.
TIER_MODELS: dict[str, dict[str, Any]] = {
    TIER_FAST: {
        "primary": "qwen3:4b",
        "fallbacks": [
            "qwen2.5:7b",
            "qwen2.5-coder:7b",
        ],
    },
    TIER_REASONING: {
        "primary": "qwen3:30b",
        "fallbacks": [
            "qwen2.5:32b",
            "qwen2.5:14b",
            "qwen2.5-coder:14b",
            "qwen2.5-coder:7b",
        ],
    },
    TIER_CODING: {
        "primary": "qwen3-coder:30b",
        "fallbacks": [
            "qwen2.5-coder:14b",
            "qwen2.5-coder:7b",
        ],
    },
}

# Work kind → tier. Direct selection — not a FAST→REASONING→CODING sequence.
WORK_KIND_TIER: dict[str, str] = {
    # FAST — cheap / structured / simple decisions
    "intent": TIER_FAST,
    "converse": TIER_FAST,
    "extract_args": TIER_FAST,
    "query_generation": TIER_FAST,
    "structure": TIER_FAST,
    "plan": TIER_FAST,
    "chat": TIER_FAST,
    # REASONING — hard diagnosis, research synthesis, semantics
    "learning": TIER_REASONING,
    "research": TIER_REASONING,
    "diagnose": TIER_REASONING,
    "semantic": TIER_REASONING,
    "verify_claim": TIER_REASONING,
    # CODING — skill / Python generation & repair only
    "skill_code": TIER_CODING,
    "code_repair": TIER_CODING,
    "code_analysis": TIER_CODING,
}

# Escalation ONLY when prior result is insufficient (not a pipeline gate).
TIER_ESCALATION: dict[str, str | None] = {
    TIER_FAST: TIER_REASONING,
    TIER_REASONING: None,
    TIER_CODING: TIER_CODING,  # same-tier retry with failure context
}


def all_configured_models() -> list[str]:
    """Unique model names in primary+fallback order (stable)."""
    seen: set[str] = set()
    out: list[str] = []
    for tier in (TIER_FAST, TIER_REASONING, TIER_CODING):
        tier_cfg = TIER_MODELS[tier]
        for name in [tier_cfg["primary"], *list(tier_cfg.get("fallbacks") or [])]:
            n = str(name).strip()
            if n and n not in seen:
                seen.add(n)
                out.append(n)
    if DEFAULT_MODEL not in seen:
        out.append(DEFAULT_MODEL)
    return out


def models_for_tier(tier: str) -> list[str]:
    """Ordered candidate models for a tier (primary then fallbacks)."""
    t = (tier or "").strip().upper()
    tier_cfg = TIER_MODELS.get(t) or TIER_MODELS[TIER_FAST]
    names = [str(tier_cfg["primary"]).strip()]
    for fb in tier_cfg.get("fallbacks") or []:
        n = str(fb).strip()
        if n and n not in names:
            names.append(n)
    return names


def tier_for_work(work: str) -> str:
    """Map a work kind to a tier; unknown work defaults to FAST (not 30B)."""
    key = (work or "").strip().lower()
    return WORK_KIND_TIER.get(key, TIER_FAST)


def size_hint_bytes(model: str) -> int:
    """Best-effort size estimate for warm-budget planning."""
    name = (model or "").strip().lower()
    if name in MODEL_SIZE_HINTS_BYTES:
        return MODEL_SIZE_HINTS_BYTES[name]
    # Heuristic from name tokens
    if ":4b" in name or "4b" in name:
        return 2_500_000_000
    if ":7b" in name or "7b" in name:
        return 4_500_000_000
    if ":14b" in name or "14b" in name:
        return 8_500_000_000
    if ":30b" in name or "30b" in name or ":32b" in name or "32b" in name:
        return 18_000_000_000
    return 6_000_000_000
