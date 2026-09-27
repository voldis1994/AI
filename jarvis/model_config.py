"""
Central model / tier configuration for JARVIS Multi-Model Router.

All Ollama model names and work→tier mappings live HERE — not scattered
across call sites. Change models in this file only.
"""

from __future__ import annotations

from typing import Any

# Legacy single-model default (kept as universal fallback end of chain).
DEFAULT_MODEL = "qwen2.5-coder:7b"
DEFAULT_HOST = "http://127.0.0.1:11434"

# Tier labels used in MODEL ROUTE / ESCALATION / FALLBACK logs.
TIER_FAST = "FAST"
TIER_REASONING = "REASONING"
TIER_CODING = "CODING"

VALID_TIERS = frozenset({TIER_FAST, TIER_REASONING, TIER_CODING})

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

# Work kind → tier. Routing is by work type / complexity, never by user task text.
WORK_KIND_TIER: dict[str, str] = {
    # FAST — cheap / structured / simple
    "intent": TIER_FAST,
    "converse": TIER_FAST,
    "extract_args": TIER_FAST,
    "query_generation": TIER_FAST,
    "structure": TIER_FAST,
    # REASONING — planning, learning, diagnosis, semantics
    "plan": TIER_REASONING,
    "learning": TIER_REASONING,
    "research": TIER_REASONING,
    "diagnose": TIER_REASONING,
    "semantic": TIER_REASONING,
    "verify_claim": TIER_REASONING,
    # CODING — skill / Python generation & repair
    "skill_code": TIER_CODING,
    "code_repair": TIER_CODING,
    "code_analysis": TIER_CODING,
}

# Escalation ladder (FAST may escalate to REASONING; CODING retries CODING).
TIER_ESCALATION: dict[str, str | None] = {
    TIER_FAST: TIER_REASONING,
    TIER_REASONING: None,
    TIER_CODING: TIER_CODING,
}


def all_configured_models() -> list[str]:
    """Unique model names in primary+fallback order (stable)."""
    seen: set[str] = set()
    out: list[str] = []
    for tier in (TIER_FAST, TIER_REASONING, TIER_CODING):
        cfg = TIER_MODELS[tier]
        for name in [cfg["primary"], *list(cfg.get("fallbacks") or [])]:
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
    cfg = TIER_MODELS.get(t) or TIER_MODELS[TIER_FAST]
    names = [str(cfg["primary"]).strip()]
    for fb in cfg.get("fallbacks") or []:
        n = str(fb).strip()
        if n and n not in names:
            names.append(n)
    return names


def tier_for_work(work: str) -> str:
    """Map a work kind to a tier; unknown work defaults to REASONING."""
    key = (work or "").strip().lower()
    return WORK_KIND_TIER.get(key, TIER_REASONING)
