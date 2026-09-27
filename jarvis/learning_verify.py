"""
Universal LEARNING VERIFY — semantic goal coverage, not keyword matching.

Compares USER REQUEST → acquired knowledge → whether it answers the user's goal.
If the request also asks for a concrete/practical result, that result must be
produced and checked. No topic-specific hardcoding.
"""

from __future__ import annotations

import ast
import operator
import re
from difflib import SequenceMatcher
from typing import Any, Optional


# Universal signals that the user wants a concrete answer / worked result
# (not topic-specific — applies to math, facts, conversions, demos, etc.).
_PRACTICAL_ASK_RE = re.compile(
    r"(?i)\b(?:"
    r"compute|calculate|solve|evaluate|work\s*out|figure\s*out|"
    r"what(?:'s|\s+is)|find(?:\s+the)?\s+(?:value|answer|result)|"
    r"give\s+(?:me\s+)?(?:the\s+)?(?:answer|result)|"
    r"demonstrate|show\s+(?:me\s+)?(?:how|that|the\s+result)|"
    r"aprēķin|izskaitļ|cik\s+ir|kāds\s+ir\s+rezult|atbildi|"
    r"parādi\s+rezult|atrisi|izrēķin"
    r")\b"
)

# Expression-like fragments (digits + operator) — language-agnostic cue.
_EXPR_RE = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*([\+\-\*/×÷])\s*(\d+(?:\.\d+)?)(?![\w.])"
)

_SKILL_JARGON_RE = re.compile(
    r"(?i)skill must define skill_meta|return concrete result\.path|"
    r"rewrite_skill|skill body not implemented|protect_active|fault_layer"
)

_SKILL_TEST_IDEA_RE = re.compile(
    r"(?i)execute\s+skill\.run|independently verify artifacts"
)

_OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "×": operator.mul,
    "÷": operator.truediv,
}


def knowledge_blob(research: dict[str, Any], entry: Optional[dict] = None) -> str:
    """
    Flatten VERIFY-ready knowledge into a narrative.

    Never includes research `raw`, scrape dumps, skill-repair jargon, or
    diagnostic / failed-approach logs — only clean synthesized fields.
    """
    # Prefer explicit KnowledgeArtifact payload when present
    art = research.get("artifact") if isinstance(research, dict) else None
    if isinstance(art, dict) and (
        art.get("summary") or art.get("concepts") or art.get("explanations")
    ):
        from jarvis.knowledge_artifact import KnowledgeArtifact

        return KnowledgeArtifact.from_dict(art).narrative()

    entry = entry or {}
    parts = [
        str(research.get("approach") or entry.get("approach") or entry.get("summary") or ""),
        str(research.get("summary") or entry.get("summary") or ""),
        " ".join(
            str(x)
            for x in (
                research.get("concepts")
                or research.get("key_apis")
                or entry.get("concepts")
                or entry.get("key_apis")
                or []
            )
        ),
        " ".join(str(x) for x in (research.get("explanations") or entry.get("explanations") or [])),
        " ".join(str(x) for x in (research.get("examples") or entry.get("examples") or [])),
        str(
            research.get("practice")
            or research.get("test_idea")
            or entry.get("practice")
            or entry.get("test_idea")
            or ""
        ),
        # Intentionally NO research["raw"] — logs must not enter VERIFY
    ]
    pr = research.get("practical_result") or entry.get("practical_result")
    if isinstance(pr, dict):
        parts.append(str(pr.get("answer") or ""))
        parts.append(str(pr.get("explanation") or ""))
        parts.append(str(pr.get("result") if pr.get("result") is not None else ""))
    elif pr:
        parts.append(str(pr))
    return "\n".join(p for p in parts if p).strip()


def has_substance(research: dict[str, Any], entry: Optional[dict] = None) -> bool:
    """True when clean synthesized knowledge has real learning content."""
    art = research.get("artifact") if isinstance(research, dict) else None
    if isinstance(art, dict) and (
        art.get("summary") or art.get("concepts") or art.get("explanations")
    ):
        from jarvis.knowledge_artifact import KnowledgeArtifact

        return KnowledgeArtifact.from_dict(art).has_substance()

    entry = entry or {}
    approach = str(
        research.get("approach") or research.get("summary") or entry.get("summary") or ""
    ).strip()
    if _SKILL_JARGON_RE.search(approach):
        return False
    concepts = list(
        research.get("concepts")
        or research.get("key_apis")
        or entry.get("concepts")
        or entry.get("key_apis")
        or []
    )
    explanations = list(research.get("explanations") or entry.get("explanations") or [])
    examples = list(research.get("examples") or entry.get("examples") or [])
    test_idea = str(
        research.get("practice")
        or research.get("test_idea")
        or entry.get("practice")
        or entry.get("test_idea")
        or ""
    )
    learning_test = bool(test_idea) and not _SKILL_TEST_IDEA_RE.search(test_idea)
    # Do NOT use research raw length as substance — that rewarded polluted logs
    if len(approach) >= 40 and (concepts or explanations or examples or learning_test):
        return True
    if len(approach) >= 80:
        return True
    if concepts and (explanations or examples or learning_test):
        return True
    narr = knowledge_blob(research, entry)
    if len(narr) >= 120 and (concepts or explanations):
        return True
    return False


def diagnose_artifact_gaps(
    user_request: str,
    research: dict[str, Any],
    entry: Optional[dict] = None,
) -> list[str]:
    """
    Deterministic gap list when semantic relatedness is high but VERIFY fails.

    Returns concrete KnowledgeArtifact fields to fill — not a full re-research.
    """
    from jarvis.knowledge_artifact import KnowledgeArtifact

    art_data = (research or {}).get("artifact")
    if isinstance(art_data, dict):
        art = KnowledgeArtifact.from_dict(art_data)
    else:
        art = KnowledgeArtifact.from_dict({**(entry or {}), **(research or {})})
    art.user_request = user_request or art.user_request
    gaps = art.missing_fields(user_request)
    # If relatedness is already high, prefer structural gaps over goal_coverage spam
    blob = knowledge_blob(research, entry)
    related = soft_relatedness(user_request, blob) if blob else 0.0
    if related >= 0.45 and "goal_coverage" in gaps:
        gaps = [g for g in gaps if g != "goal_coverage"]
    return gaps or art.missing_fields(user_request)


def requests_practical_result(user_request: str) -> bool:
    """Detect if USER REQUEST also wants a concrete answer / worked result."""
    text = user_request or ""
    if _PRACTICAL_ASK_RE.search(text):
        return True
    if _EXPR_RE.search(text):
        return True
    return False


def extract_arithmetic_tasks(user_request: str) -> list[dict[str, Any]]:
    """Find universal numeric expressions in the request (any operands)."""
    tasks: list[dict[str, Any]] = []
    for m in _EXPR_RE.finditer(user_request or ""):
        a, op, b = m.group(1), m.group(2), m.group(3)
        try:
            left = float(a) if "." in a else int(a)
            right = float(b) if "." in b else int(b)
            value = _OPS[op](left, right)
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            tasks.append({
                "expression": f"{a}{op}{b}",
                "expected": value,
            })
        except Exception:
            continue
    return tasks


_STOP = {
    "a", "an", "the", "and", "or", "to", "for", "with", "from", "into",
    "in", "on", "at", "of", "by", "is", "are", "be", "as", "it", "this",
    "that", "learn", "learning", "study", "studying", "please", "about",
    "basic", "basics", "create", "make", "practical", "tests", "verify",
    "knowledge", "how", "what", "when", "where", "which", "your", "my",
    "iemacies", "macies", "izveido", "par", "un", "ka", "kā", "ar",
}


def soft_relatedness(user_request: str, knowledge_text: str) -> float:
    """
    Fuzzy relatedness in [0, 1] — NOT exact keyword/token equality.

    Combines:
      - whole-text sequence similarity
      - character n-gram overlap
      - per-content-word fuzzy/stem alignment (paraphrases & morphology)
    """
    a = _normalize(user_request)
    b = _normalize(knowledge_text)
    if not a or not b:
        return 0.0
    seq = SequenceMatcher(None, a, b).ratio()
    grams_a = _char_ngrams(a, 3)
    grams_b = _char_ngrams(b, 3)
    if not grams_a or not grams_b:
        ngram = 0.0
    else:
        inter = len(grams_a & grams_b)
        union = len(grams_a | grams_b) or 1
        ngram = inter / union
    word_score = _word_fuzzy_alignment(a, b)
    # Emphasize word-level fuzzy alignment (handles paraphrases better)
    return max(seq, ngram, word_score) * 0.55 + min(seq, ngram, word_score) * 0.15 + word_score * 0.30


def _has_strong_anchor(user_request: str, knowledge_text: str) -> bool:
    """True if at least one content word strongly aligns (stem or fuzzy≥0.82)."""
    a = _normalize(user_request)
    b = _normalize(knowledge_text)
    req_words = [w for w in a.split() if len(w) >= 4 and w not in _STOP]
    know_words = [w for w in b.split() if len(w) >= 3 and w not in _STOP]
    if not req_words or not know_words:
        return False
    know_stems = set()
    for w in know_words:
        know_stems.update(_stems(w))
    for rw in req_words:
        if set(_stems(rw)) & know_stems:
            return True
        for kw in know_words:
            if SequenceMatcher(None, rw, kw).ratio() >= 0.82:
                return True
    return False


def _word_fuzzy_alignment(request: str, knowledge: str) -> float:
    """Average best fuzzy match of each request content word against knowledge words."""
    req_words = [
        w for w in request.split()
        if len(w) >= 4 and w not in _STOP
    ]
    know_words = [
        w for w in knowledge.split()
        if len(w) >= 3 and w not in _STOP
    ]
    if not req_words or not know_words:
        return 0.0
    know_stems = set()
    for w in know_words:
        know_stems.update(_stems(w))
    scores: list[float] = []
    for rw in req_words:
        best = 0.0
        r_stems = set(_stems(rw))
        if r_stems & know_stems:
            best = 1.0
        else:
            for kw in know_words:
                ratio = SequenceMatcher(None, rw, kw).ratio()
                if ratio > best:
                    best = ratio
        scores.append(best)
    return sum(scores) / len(scores)


def _stems(word: str) -> list[str]:
    """Lightweight morphological variants — not an exact-token check."""
    w = (word or "").lower()
    out = {w}
    for suf in (
        "ing", "tion", "ments", "ment", "ness", "ally", "als", "al",
        "ers", "er", "ed", "es", "ies", "s", "ā", "ām", "iem", "us", "as",
    ):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            out.add(w[: -len(suf)])
    # networking → network, fundamentals → fundament
    if w.endswith("ing") and len(w) > 6:
        out.add(w[:-3])
    return list(out)


def _normalize(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"[^\w\s+\-*/×÷.]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _char_ngrams(text: str, n: int) -> set[str]:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < n:
        return {compact} if compact else set()
    return {compact[i : i + n] for i in range(len(compact) - n + 1)}


def produce_practical_offline(
    user_request: str, research: dict[str, Any]
) -> dict[str, Any]:
    """
    Produce a concrete learning result without topic hardcoding.

    - Arithmetic expressions in the request → evaluate safely
    - Otherwise → derive an answer narrative from acquired knowledge
    """
    tasks = extract_arithmetic_tasks(user_request)
    if tasks:
        results = [t["expected"] for t in tasks]
        primary = results[0] if len(results) == 1 else results
        exprs = ", ".join(t["expression"] for t in tasks)
        return {
            "answer": str(primary),
            "result": primary,
            "explanation": (
                f"Evaluated expression(s) from USER REQUEST: {exprs} → {primary}"
            ),
            "source": "offline_expression",
            "tasks": tasks,
        }

    # Narrative answer from knowledge
    approach = str(research.get("approach") or "").strip()
    key_apis = list(research.get("key_apis") or [])[:8]
    test_idea = str(research.get("test_idea") or "").strip()
    answer_parts = []
    if approach:
        answer_parts.append(approach[:600])
    if key_apis:
        answer_parts.append("Key points: " + "; ".join(str(x) for x in key_apis))
    if test_idea and not _SKILL_TEST_IDEA_RE.search(test_idea):
        answer_parts.append("Practice: " + test_idea[:300])
    answer = "\n".join(answer_parts).strip()
    return {
        "answer": answer or "Insufficient knowledge to form a concrete answer yet.",
        "result": answer[:200] if answer else None,
        "explanation": "Derived practical answer from acquired topic knowledge.",
        "source": "offline_knowledge",
        "tasks": [],
    }


def verify_practical_offline(
    user_request: str, practical: Optional[dict[str, Any]]
) -> dict[str, Any]:
    """Check a produced practical result against the USER REQUEST (offline)."""
    if not isinstance(practical, dict) or not (
        practical.get("answer") or practical.get("result") is not None
    ):
        return {
            "ok": False,
            "detail": "practical result missing",
            "missing": ["practical_result"],
        }

    tasks = extract_arithmetic_tasks(user_request)
    if tasks:
        # Compare produced result to expected evaluation(s)
        produced = practical.get("result")
        expected = tasks[0]["expected"] if len(tasks) == 1 else [t["expected"] for t in tasks]
        if _values_match(produced, expected):
            return {
                "ok": True,
                "detail": f"practical result matches evaluated expression(s): {expected}",
                "missing": [],
            }
        # Also accept answer text containing the expected value
        ans = str(practical.get("answer") or "")
        if str(expected) in ans:
            return {
                "ok": True,
                "detail": f"practical answer text contains expected {expected}",
                "missing": [],
            }
        return {
            "ok": False,
            "detail": f"practical result {produced!r} ≠ expected {expected!r}",
            "missing": ["practical_result_mismatch"],
        }

    # Non-arithmetic practical ask: require non-empty answer related to request
    ans = str(practical.get("answer") or "").strip()
    if len(ans) < 20:
        return {
            "ok": False,
            "detail": "practical answer too thin",
            "missing": ["practical_result"],
        }
    rel = soft_relatedness(user_request, ans)
    if rel < 0.08:
        return {
            "ok": False,
            "detail": f"practical answer unrelated to request (relatedness={rel:.2f})",
            "missing": ["practical_goal_mismatch"],
        }
    return {
        "ok": True,
        "detail": f"practical answer present (relatedness={rel:.2f})",
        "missing": [],
    }


def _values_match(produced: Any, expected: Any) -> bool:
    if produced is None:
        return False
    if isinstance(expected, list):
        if isinstance(produced, (list, tuple)) and len(produced) == len(expected):
            return all(_values_match(p, e) for p, e in zip(produced, expected))
        return False
    try:
        if isinstance(expected, (int, float)) and isinstance(produced, (int, float)):
            return abs(float(produced) - float(expected)) < 1e-9
        # produced as string
        ps = str(produced).strip()
        if str(expected) == ps:
            return True
        # try numeric parse of answer
        try:
            return abs(float(ps) - float(expected)) < 1e-9
        except Exception:
            return False
    except Exception:
        return str(produced).strip() == str(expected).strip()


def safe_eval_expression(expr: str) -> Any:
    """Evaluate a simple arithmetic AST (no names/calls)."""
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.Expression):
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            continue
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            continue
        if isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Load, ast.UnaryOp, ast.USub)):
            continue
        raise ValueError(f"disallowed expression node: {type(node).__name__}")
    return eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, {})


def offline_semantic_judgment(
    user_request: str,
    research: dict[str, Any],
    entry: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Offline stand-in for semantic coverage when the model is unavailable.

    Does NOT require exact keyword hits. Uses substance + fuzzy relatedness,
    and practical-result checks when the request asks for one.
    """
    entry = entry or {}
    blob = knowledge_blob(research, entry)
    substance = has_substance(research, entry)
    related = soft_relatedness(user_request, blob) if blob else 0.0
    needs_practical = requests_practical_result(user_request)

    missing: list[str] = []
    # Soft semantic bar: paraphrases pass, unrelated topics fail.
    # Require either a strong word/stem anchor OR higher overall relatedness.
    strong = _has_strong_anchor(user_request, blob)
    covers = substance and (related >= 0.45 or (related >= 0.22 and strong))
    if not substance:
        missing.append("knowledge_substance")
    if substance and not covers:
        missing.append("goal_coverage")

    practical = research.get("practical_result") or entry.get("practical_result")
    practical_check = {"ok": True, "detail": "not required", "missing": []}
    if needs_practical:
        practical_check = verify_practical_offline(user_request, practical)
        if not practical_check["ok"]:
            covers = False
            missing.extend(practical_check.get("missing") or ["practical_result"])

    sources = list(research.get("sources") or entry.get("sources") or [])
    results = list(research.get("results") or [])
    has_sources = bool(sources) or bool(results) or bool(blob)
    if not has_sources:
        covers = False
        missing.append("knowledge_sources")

    return {
        "covers_goal": covers,
        "confidence": min(0.95, 0.35 + related + (0.2 if substance else 0.0)),
        "reason": (
            f"offline semantic: substance={substance} relatedness={related:.2f} "
            f"practical_required={needs_practical} practical_ok={practical_check['ok']}"
        ),
        "missing_aspects": missing,
        "requires_practical_result": needs_practical,
        "relatedness": related,
        "practical_check": practical_check,
    }
