"""
JARVIS ontology — canonical layer vocabulary for the whole system.

SKILL / COMPETENCE — knowledge domain (what area JARVIS has learned)
CAPABILITY         — what JARVIS can do (action class under a competence)
TOOL / IMPLEMENTATION — executable code / API (versioned, rollback-capable)
KNOWLEDGE          — verified knowledge attached to a competence
EXPERIENCE         — verified prior results / failures attached to competence/tool

Implementation lifecycle (CANDIDATE→TESTING→ACTIVE, archive/rollback) lives in
``ImplementationRegistry``. Competence tree indexing lives in
``CompetenceRegistry``. Neither may invent USER REQUEST requirements —
that truth is only ``TaskContract``.
"""

from __future__ import annotations

# ── Layer labels ───────────────────────────────────────────────────────
LAYER_SKILL = "skill"
LAYER_COMPETENCE = "competence"  # alias of skill domain
LAYER_CAPABILITY = "capability"
LAYER_TOOL = "tool"
LAYER_IMPLEMENTATION = "implementation"  # alias of tool
LAYER_KNOWLEDGE = "knowledge"
LAYER_EXPERIENCE = "experience"

ALL_LAYERS = frozenset({
    LAYER_SKILL,
    LAYER_COMPETENCE,
    LAYER_CAPABILITY,
    LAYER_TOOL,
    LAYER_IMPLEMENTATION,
    LAYER_KNOWLEDGE,
    LAYER_EXPERIENCE,
})

# Normalize aliases to canonical storage keys used in competence tree / tools.
_CANONICAL = {
    LAYER_SKILL: LAYER_SKILL,
    LAYER_COMPETENCE: LAYER_SKILL,
    LAYER_CAPABILITY: LAYER_CAPABILITY,
    LAYER_TOOL: LAYER_TOOL,
    LAYER_IMPLEMENTATION: LAYER_TOOL,
    LAYER_KNOWLEDGE: LAYER_KNOWLEDGE,
    LAYER_EXPERIENCE: LAYER_EXPERIENCE,
}


def canonicalize_layer(layer: str | None) -> str:
    """Map ontology aliases to the storage layer key."""
    key = (layer or "").strip().lower()
    return _CANONICAL.get(key, key)


def is_executable_layer(layer: str | None) -> bool:
    return canonicalize_layer(layer) == LAYER_TOOL


def is_knowledge_domain(layer: str | None) -> bool:
    return canonicalize_layer(layer) == LAYER_SKILL
