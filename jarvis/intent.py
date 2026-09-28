"""
Requirements hints for USER REQUEST (not exclusive lifecycle routing).

IntentClassifier produces a structured requirements hint that may *add*
outcomes. It must never downgrade a semantically derived TaskContract that
already requires artifact/execution into a learning-only path.

Regex/offline heuristics are fallback hints only.
"""

from __future__ import annotations

import re
from typing import Any, Optional


# Fallback hint lexicons — NOT authoritative routing.
_LEARN_RE = re.compile(
    r"(?i)\b(?:"
    r"learn|learning|study|studying|"
    r"memorize|memorise|revise|revision|understand|understanding|"
    r"teach(?:\s+me)?|explain|tutorial|basics|fundamentals|"
    r"knowledge|quiz(?:\s+me)?|flashcards?|"
    r"iem[aā]c(?:ies|īties|ities)?|m[aā]c(?:ies|īties|ibu|ību)?|"
    r"iegaum[eē]|izskaidr(?:o|ot)|pamatus|pamati|"
    r"apg[uū]t|apg[uū]sti|zin[aā]šan"
    r")\b"
)

_RESEARCH_RE = re.compile(
    r"(?i)\b(?:"
    r"research|researches|researched|investigate|investigating|"
    r"izp[eē]t(?:i|īt|īšana)?|petijum|pētījum"
    r")\b"
)

_ACTION_ARTIFACT_RE = re.compile(
    r"(?i)(?:"
    r"\b(?:create|write|make|save|download|fetch|scrape|install|convert|"
    r"upload|delete|remove|rename|move|copy|run|execute|compile|deploy|"
    r"izveido|uzraksti|saglab[aā]|lejupiel[aā]d[eē]|izdz[eē]s|p[aā]rd[eē]v[eē]|"
    r"erstelle|crée|создай)\b"
    r".{0,40}\b(?:file|files|folder|directory|path|url|http|https|"
    r"failu|fails|mapi|katalog|datei|fichier|файл|"
    r"[A-Za-z0-9_.-]+\.[A-Za-z]{1,8})\b"
    r"|"
    r"\b(?:file|failu|datei|fichier|файл)\b.{0,40}\b(?:create|write|make|"
    r"izveido|uzraksti|erstelle|crée|создай)\b"
    r")"
)

_ACTION_RE = re.compile(
    r"(?i)\b(?:"
    r"create|write|make|build|download|fetch|scrape|install|convert|"
    r"upload|delete|remove|rename|move|copy|run|execute|generate|"
    r"automate|compute|calculate|parse|crawl|"
    r"izveido|uzraksti|izpildi|lejupiel[aā]d[eē]|saglab[aā]|p[aā]rveido|"
    r"izdz[eē]s|ģener[eē]|gener[eē]|erstelle|crée|создай"
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

# Legacy exclusive labels — kept for compatibility; routing must use requirements.
VALID_INTENTS = frozenset({"conversation", "learning", "task"})


class IntentClassifier:
    """Produce requirements *hints* per USER REQUEST (no prior-skill inherit)."""

    @classmethod
    def classify_offline(cls, text: str) -> dict[str, Any]:
        raw = (text or "").strip()
        if not raw:
            return cls._requirements_result(
                primary="conversation",
                goal=raw,
                needs_capability=False,
            )

        if _CONVERSATION_RE.match(raw):
            return cls._requirements_result(
                primary="conversation",
                goal=raw,
                needs_capability=False,
            )

        has_learn = bool(_LEARN_RE.search(raw))
        has_research = bool(_RESEARCH_RE.search(raw))
        has_artifact = bool(_ACTION_ARTIFACT_RE.search(raw))
        has_action = bool(_ACTION_RE.search(raw))
        has_path = bool(
            re.search(
                r"(?<![A-Za-z0-9_\"'])([A-Za-z0-9_./\\-]+\.[A-Za-z0-9]{1,12})\b",
                raw,
            )
        )

        needs_learning = has_learn
        needs_research = has_research or has_learn
        # Capability only from grounded action/artifact signals — learning
        # verbs alone must never force a skill lifecycle.
        needs_capability = bool(
            has_artifact or has_path or (has_action and not has_learn and not has_research)
        )
        # Combined: learning/research + artifact → BOTH requirements (not exclusive)
        if (has_learn or has_research) and (has_artifact or has_path):
            needs_capability = True

        if needs_capability and (needs_learning or needs_research):
            primary = "task"  # advisory; requirement flags stay True
        elif needs_learning and not needs_capability:
            primary = "learning"
        elif needs_research and not needs_capability:
            primary = "learning"  # advisory primary; needs_research flag set
        elif needs_capability:
            primary = "task"
        elif _QUESTION_RE.search(raw) and not has_action:
            primary = "conversation"
        elif has_action:
            primary = "task"
            needs_capability = True
        else:
            primary = "conversation"

        return cls._requirements_result(
            primary=primary,
            goal=raw,
            needs_capability=needs_capability,
            needs_research=needs_research,
            needs_learning=needs_learning,
            keywords=cls._keywords(raw),
        )

    @classmethod
    def normalize(cls, intent: Optional[dict], user_text: str = "") -> dict[str, Any]:
        """
        Normalize brain/offline JSON into a requirements hint.

        CRITICAL: must NOT rewrite semantic execution/artifact requirements
        into learning-only. Hints may add learning/research flags; they must
        not clear needs_capability when the request (or brain) signals action.
        """
        text = (user_text or "").strip()
        offline = cls.classify_offline(text)
        if not isinstance(intent, dict):
            return offline

        name = str(intent.get("intent") or "").strip().lower()
        if name == "knowledge":
            name = "learning"
        if name not in VALID_INTENTS:
            return offline

        # Start from brain flags
        needs_capability = bool(intent.get("needs_capability"))
        needs_research = bool(intent.get("needs_research"))
        needs_learning = bool(
            intent.get("needs_learning") or name == "learning"
        )

        # Offline may ADD requirements — never strip capability/artifact
        if offline.get("needs_capability"):
            needs_capability = True
        if offline.get("needs_learning"):
            needs_learning = True
        if offline.get("needs_research"):
            needs_research = True

        # Brain said conversation but offline sees action/learning → upgrade flags
        if name == "conversation" and (
            offline.get("needs_capability") or offline.get("needs_learning")
        ):
            name = str(offline.get("intent") or name)
            needs_capability = needs_capability or bool(
                offline.get("needs_capability")
            )
            needs_learning = needs_learning or bool(offline.get("needs_learning"))

        # NEVER downgrade brain/offline needs_capability → learning-only.
        # Semantic artifact/execution requirements survive normalize.
        if needs_capability:
            if name == "learning":
                name = "task"
        elif name == "task" and (
            offline.get("needs_learning") or offline.get("needs_research")
        ) and not offline.get("needs_capability"):
            # Brain said task without capability; offline sees learn/research only
            # → adopt learning primary, but do NOT invent capability.
            name = "learning"
            needs_learning = needs_learning or bool(offline.get("needs_learning"))
            needs_research = needs_research or bool(offline.get("needs_research"))
        elif needs_learning:
            name = "learning"
        elif name == "task":
            # Bare "task" label without capability flag — do not invent capability
            # unless offline also signals action/artifact.
            if offline.get("needs_capability"):
                needs_capability = True
            else:
                name = str(offline.get("intent") or "conversation")
                needs_learning = needs_learning or bool(offline.get("needs_learning"))
                needs_research = needs_research or bool(offline.get("needs_research"))

        goal = str(intent.get("goal") or text).strip() or text
        keywords = intent.get("keywords")
        if not isinstance(keywords, list) or not keywords:
            keywords = offline.get("keywords") or cls._keywords(text)

        return cls._requirements_result(
            primary=name,
            goal=goal,
            needs_capability=needs_capability,
            needs_research=needs_research or needs_learning,
            needs_learning=needs_learning,
            keywords=[str(k) for k in keywords[:12]],
        )

    @classmethod
    def merge_into_contract_requirements(
        cls,
        requirements: Optional[dict[str, Any]],
        hint: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Union of hint flags into contract.requirements (additive only)."""
        out = dict(requirements or {})
        h = hint or {}
        for key in (
            "needs_capability",
            "needs_research",
            "needs_learning",
            "needs_conversation",
        ):
            if h.get(key):
                out[key] = True
        if h.get("keywords"):
            prev = list(out.get("keywords") or [])
            for k in h["keywords"]:
                if k not in prev:
                    prev.append(k)
            out["keywords"] = prev[:12]
        return out

    @staticmethod
    def topic_slug(text: str, limit: int = 48) -> str:
        """
        Canonical MEMORY/topic key. Unicode letters kept (Latvian/German/…)
        so ``iemācies vīns`` does not collapse to ``iem_cies_v_ns`` and collide
        with every other learn request.
        """
        slug = re.sub(r"[^\w]+", "_", (text or "").lower(), flags=re.UNICODE)
        slug = re.sub(r"_+", "_", slug).strip("_")
        return (slug or "topic")[:limit]

    @staticmethod
    def _keywords(text: str) -> list[str]:
        stop = {
            "a", "an", "the", "and", "or", "to", "for", "with", "from", "in",
            "on", "of", "please", "jarvis", "ar", "un", "ka", "kā", "par",
        }
        try:
            from jarvis.request_items import path_segment_tokens

            path_segs = path_segment_tokens(text or "")
        except Exception:
            path_segs = set()
        out: list[str] = []
        for tok in re.findall(r"[\w]{3,}", text or "", flags=re.UNICODE):
            low = tok.lower()
            if low in stop or low in path_segs:
                continue
            if tok not in out:
                out.append(tok)
        return out[:12]

    @classmethod
    def _requirements_result(
        cls,
        *,
        primary: str,
        goal: str,
        needs_capability: bool,
        needs_research: bool = False,
        needs_learning: bool = False,
        keywords: Optional[list] = None,
    ) -> dict[str, Any]:
        # Trust explicit flags. Advisory primary "learning" must NOT invent
        # needs_learning when the caller only signaled research.
        needs_learning = bool(needs_learning) or (
            primary == "learning" and not bool(needs_research)
        )
        needs_research = bool(needs_research) or needs_learning
        needs_conversation = (
            primary == "conversation"
            and not needs_capability
            and not needs_learning
            and not needs_research
        )
        return {
            # Legacy field — advisory primary only (not exclusive lifecycle)
            "intent": primary,
            "goal": goal,
            "needs_capability": bool(needs_capability),
            "needs_research": needs_research,
            "needs_learning": needs_learning,
            "needs_conversation": needs_conversation,
            "keywords": list(keywords or cls._keywords(goal)),
            "requirements": {
                "needs_capability": bool(needs_capability),
                "needs_research": needs_research,
                "needs_learning": needs_learning,
                "needs_conversation": needs_conversation,
            },
        }

    # Back-compat alias used by older call sites
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
        return cls._requirements_result(
            primary=intent,
            goal=goal,
            needs_capability=needs_capability,
            needs_research=needs_research,
            needs_learning=intent == "learning",
            keywords=keywords,
        )
