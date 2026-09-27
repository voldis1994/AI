"""
Universal USER REQUEST intent classification.

Each new request is classified independently — never inherits the previous
skill/capability. Intents:

  conversation — answer normally (chat / question)
  learning     — research, verify, and save knowledge (no skill build/repair)
  task         — action that may use/build/repair a skill
"""

from __future__ import annotations

import re
from typing import Any, Optional


# Universal signal lexicons — language-agnostic stems, not task/skill names.
_LEARN_RE = re.compile(
    r"(?i)\b(?:"
    r"learn|learning|study|studying|research|researches|researched|"
    r"memorize|memorise|revise|revision|understand|understanding|"
    r"teach(?:\s+me)?|explain|tutorial|basics|fundamentals|"
    r"knowledge|quiz(?:\s+me)?|flashcards?|"
    r"iem[aā]c(?:ies|īties|ities)?|m[aā]c(?:ies|īties|ibu|ību)?|"
    r"izp[eē]t(?:i|īt)?|iegaum[eē]|izskaidr(?:o|ot)|pamatus|pamati|"
    r"apg[uū]t|apg[uū]sti|zin[aā]šan"
    r")\b"
)

# Strong external-world action (artifact / side-effect). Overrides soft "create tests".
_ACTION_ARTIFACT_RE = re.compile(
    r"(?i)(?:"
    r"\b(?:create|write|make|save|download|fetch|scrape|install|convert|"
    r"upload|delete|remove|rename|move|copy|run|execute|compile|deploy|"
    r"izveido|uzraksti|saglab[aā]|lejupiel[aā]d[eē]|izdz[eē]s|p[aā]rd[eē]v[eē])\b"
    r".{0,40}\b(?:file|files|folder|directory|path|url|http|https|"
    r"failu|fails|mapi|katalog|"
    r"[A-Za-z0-9_.-]+\.[A-Za-z]{1,8})\b"
    r"|"
    r"\b(?:file|failu)\b.{0,40}\b(?:create|write|make|izveido|uzraksti)\b"
    r")"
)

_ACTION_RE = re.compile(
    r"(?i)\b(?:"
    r"create|write|make|build|download|fetch|scrape|install|convert|"
    r"upload|delete|remove|rename|move|copy|run|execute|generate|"
    r"automate|compute|calculate|parse|crawl|"
    r"izveido|uzraksti|izpildi|lejupiel[aā]d[eē]|saglab[aā]|p[aā]rveido|"
    r"izdz[eē]s|ģener[eē]|gener[eē]"
    r")\b"
)

_CONVERSATION_RE = re.compile(
    r"(?i)^\s*(?:"
    r"hi|hello|hey|thanks|thank you|ok|okay|bye|good\s*(?:morning|night)|"
    r"how are you|who are you|what(?:'s| is) your name|"
    r"čau|sveiks|paldies|labr[iī]t|labvakar|"
    r"kas tu esi|kā tevi sauc"
    r")\s*[?.!]?\s*$"
)

_QUESTION_RE = re.compile(
    r"(?i)^\s*(?:what|why|who|when|where|which|how|vai|kas|kāpēc|kad|kur)\b"
)

VALID_INTENTS = frozenset({"conversation", "learning", "task"})


class IntentClassifier:
    """Classify each USER REQUEST independently (no prior-skill inheritance)."""

    @classmethod
    def classify_offline(cls, text: str) -> dict[str, Any]:
        raw = (text or "").strip()
        if not raw:
            return cls._result("conversation", raw, needs_capability=False)

        if _CONVERSATION_RE.match(raw):
            return cls._result("conversation", raw, needs_capability=False)

        has_learn = bool(_LEARN_RE.search(raw))
        has_artifact = bool(_ACTION_ARTIFACT_RE.search(raw))
        has_action = bool(_ACTION_RE.search(raw))

        # Learning request: acquire knowledge. Pedagogical "create tests/quiz"
        # does NOT force the action/skill path unless a real artifact is named.
        if has_learn and not has_artifact:
            return cls._result(
                "learning",
                raw,
                needs_capability=False,
                needs_research=True,
                keywords=cls._keywords(raw),
            )

        # Explicit external action / artifact → skill path
        if has_artifact or (has_action and not has_learn):
            return cls._result(
                "task",
                raw,
                needs_capability=True,
                keywords=cls._keywords(raw),
            )

        # Mixed "learn … and create file X" → task (artifact wins)
        if has_learn and has_artifact:
            return cls._result(
                "task",
                raw,
                needs_capability=True,
                keywords=cls._keywords(raw),
            )

        # Pure questions without action → conversation
        if _QUESTION_RE.search(raw) and not has_action:
            return cls._result("conversation", raw, needs_capability=False)

        if has_action:
            return cls._result(
                "task",
                raw,
                needs_capability=True,
                keywords=cls._keywords(raw),
            )

        return cls._result("conversation", raw, needs_capability=False)

    @classmethod
    def normalize(cls, intent: Optional[dict], user_text: str = "") -> dict[str, Any]:
        """Normalize brain/offline JSON; fall back to offline if intent invalid."""
        text = (user_text or "").strip()
        if not isinstance(intent, dict):
            return cls.classify_offline(text)

        name = str(intent.get("intent") or "").strip().lower()
        # Legacy alias
        if name == "knowledge":
            name = "learning"
        if name not in VALID_INTENTS:
            return cls.classify_offline(text)

        # Safety: if brain said "task" but offline clearly sees learning
        # without an artifact action, prefer learning (prevents create_file inherit).
        offline = cls.classify_offline(text)
        if (
            name == "task"
            and offline.get("intent") == "learning"
            and not _ACTION_ARTIFACT_RE.search(text)
        ):
            name = "learning"

        if (
            name == "conversation"
            and offline.get("intent") in ("learning", "task")
        ):
            # Brain under-classified an actionable/learning ask
            name = str(offline["intent"])

        needs_capability = bool(intent.get("needs_capability"))
        if name == "task":
            needs_capability = True
        elif name in ("learning", "conversation"):
            needs_capability = False

        goal = str(intent.get("goal") or text).strip() or text
        keywords = intent.get("keywords")
        if not isinstance(keywords, list) or not keywords:
            keywords = offline.get("keywords") or cls._keywords(text)

        out = {
            "intent": name,
            "goal": goal,
            "needs_capability": needs_capability,
            "needs_research": bool(
                intent.get("needs_research")
                or name == "learning"
                or offline.get("needs_research")
            ),
            "keywords": [str(k) for k in keywords[:12]],
        }
        return out

    @staticmethod
    def topic_slug(text: str, limit: int = 48) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
        return (slug or "topic")[:limit]

    @staticmethod
    def _keywords(text: str) -> list[str]:
        stop = {
            "a", "an", "the", "and", "or", "to", "for", "with", "from", "in",
            "on", "of", "please", "jarvis", "ar", "un", "ka", "kā", "par",
        }
        out: list[str] = []
        for tok in re.findall(r"[A-Za-z0-9_]{3,}", text or ""):
            if tok.lower() in stop:
                continue
            if tok not in out:
                out.append(tok)
        return out[:12]

    @classmethod
    def _result(
        cls,
        intent: str,
        goal: str,
        *,
        needs_capability: bool,
        needs_research: bool = False,
        keywords: Optional[list] = None,
    ) -> dict[str, Any]:
        return {
            "intent": intent,
            "goal": goal,
            "needs_capability": needs_capability,
            "needs_research": needs_research or intent == "learning",
            "keywords": list(keywords or cls._keywords(goal)),
        }
