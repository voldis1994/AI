"""
Central model / tier configuration for JARVIS concurrent model pool.

FAST / REASONING / CODING are independent worker resources — not a pipeline.
All Ollama model names and work→tier mappings live HERE.

Selection prefers models that are actually installed (Ollama /api/tags).
Configured primaries are preferences; missing ones fall through to fallbacks
and then to auto-discovered installed models that fit the tier.
"""

from __future__ import annotations

import re
from typing import Any, Optional

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
    "qwen3:8b": 5_000_000_000,
    "qwen2.5:7b": 4_500_000_000,
    "qwen2.5-coder:7b": 4_500_000_000,
    "qwen2.5:14b": 8_500_000_000,
    "qwen2.5-coder:14b": 8_500_000_000,
    "qwen3:30b": 18_000_000_000,
    "qwen2.5:32b": 19_000_000_000,
    "qwen3-coder:30b": 18_000_000_000,
}

# Primary + fallback chains per tier. First installed/available model wins.
# Include mid-size local pulls (8b) so machines without 30b still get a good pick.
TIER_MODELS: dict[str, dict[str, Any]] = {
    TIER_FAST: {
        "primary": "qwen3:4b",
        "fallbacks": [
            "qwen3:8b",
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
            "qwen3:8b",
            "qwen2.5:7b",
            "qwen2.5-coder:7b",
        ],
    },
    TIER_CODING: {
        "primary": "qwen3-coder:30b",
        "fallbacks": [
            "qwen2.5-coder:14b",
            "qwen2.5-coder:7b",
            "qwen3:8b",
            "qwen3:4b",
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

# Preferred parameter ranges (billions) when auto-picking from installed tags.
_TIER_PARAM_RANGE: dict[str, tuple[float, float]] = {
    TIER_FAST: (1.0, 10.0),
    TIER_REASONING: (7.0, 70.0),
    TIER_CODING: (3.0, 40.0),
}

_PARAM_RE = re.compile(r"(?i)(?:^|[:\-/_.])(\d+(?:\.\d+)?)\s*b\b")


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
    params = parse_param_billions(name)
    if params is not None:
        # ~0.55–0.65 GiB per billion params for Q4-ish — rough
        return int(params * 600_000_000)
    if ":4b" in name or "4b" in name:
        return 2_500_000_000
    if ":7b" in name or "7b" in name:
        return 4_500_000_000
    if ":8b" in name or "8b" in name:
        return 5_000_000_000
    if ":14b" in name or "14b" in name:
        return 8_500_000_000
    if ":30b" in name or "30b" in name or ":32b" in name or "32b" in name:
        return 18_000_000_000
    return 6_000_000_000


def parse_param_billions(model: str) -> Optional[float]:
    """Extract parameter count in billions from a model tag (e.g. qwen3:8b → 8.0)."""
    name = (model or "").strip().lower()
    if not name:
        return None
    m = _PARAM_RE.search(name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def is_coder_model(model: str) -> bool:
    n = (model or "").lower()
    return "coder" in n or "code" in n.split(":")[0]


def score_model_for_tier(model: str, tier: str) -> float:
    """
    Higher = better fit for tier when auto-picking from installed tags.

    Not a quality judgment of the weights — only size/family fitness so
    FAST stays small, CODING prefers coder tags, REASONING prefers larger.
    """
    t = (tier or "").strip().upper() or TIER_FAST
    name = (model or "").strip().lower()
    if not name:
        return -1e9
    params = parse_param_billions(name)
    lo, hi = _TIER_PARAM_RANGE.get(t, (1.0, 70.0))
    score = 0.0

    if params is None:
        score -= 5.0
    else:
        if lo <= params <= hi:
            score += 20.0
            # Prefer mid-band for the tier
            mid = (lo + hi) / 2.0
            score -= abs(params - mid) * 0.35
        elif params < lo:
            score += 8.0 - (lo - params) * 0.8
        else:
            # Oversized for tier — usable but costly (and crash-prone on weak GPU)
            score += 6.0 - (params - hi) * 0.6

    coder = is_coder_model(name)
    if t == TIER_CODING:
        score += 15.0 if coder else -4.0
    elif t == TIER_REASONING:
        score += 3.0 if not coder else -1.0
        if "qwen3" in name:
            score += 2.0
    elif t == TIER_FAST:
        score += 2.0 if not coder else 0.5
        if params is not None and params <= 8.0:
            score += 4.0

    # Mild preference for known families
    if name.startswith("qwen"):
        score += 1.5
    if "embed" in name or "vision" in name or "vl" in name.split(":")[0]:
        score -= 25.0
    return score


def discover_models_for_tier(
    tier: str,
    installed: Optional[list[str]],
    *,
    exclude: Optional[set[str]] = None,
) -> list[str]:
    """
    Ordered candidates for a tier from catalog + installed Ollama tags.

    1) Configured primary/fallbacks that are installed (catalog order)
    2) Other installed models ranked by tier fitness (auto-select)
    3) If nothing installed is known, return catalog order (caller may still try)
    """
    t = (tier or "").strip().upper() or TIER_FAST
    catalog = models_for_tier(t)
    ban = {x.strip().lower() for x in (exclude or set()) if str(x).strip()}
    tags = [str(x).strip() for x in (installed or []) if str(x).strip()]
    tag_l = {n.lower(): n for n in tags}  # preserve original casing

    out: list[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        key = name.strip().lower()
        if not key or key in seen or key in ban:
            return
        seen.add(key)
        # Prefer installed casing when present
        out.append(tag_l.get(key, name.strip()))

    if tags:
        for name in catalog:
            if name.strip().lower() in tag_l and name.strip().lower() not in ban:
                _add(name)
        # Auto-discover: anything installed that scores reasonably
        extras = []
        for raw in tags:
            key = raw.lower()
            if key in seen or key in ban:
                continue
            sc = score_model_for_tier(raw, t)
            if sc < 0:
                continue
            extras.append((sc, raw))
        extras.sort(key=lambda x: (-x[0], x[1].lower()))
        for _, raw in extras:
            _add(raw)
        if out:
            return out

    # Offline / empty tags — catalog preference order
    for name in catalog:
        _add(name)
    if DEFAULT_MODEL.strip().lower() not in seen:
        _add(DEFAULT_MODEL)
    return out
