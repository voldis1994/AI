"""
Clean KnowledgeArtifact for JARVIS learning VERIFY.

VERIFY compares USER REQUEST ↔ this artifact only — never research dumps,
skill-repair jargon, failed approaches, diagnostics, or prior-request logs.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from jarvis import learning_verify as lv

# Lines / phrases that must never enter a learning knowledge artifact.
_POLLUTION_RE = re.compile(
    r"(?i)(?:"
    r"skill must define|skill_meta|skill\.run|protect_active|"
    r"rewrite_skill|fault_layer|failed approach|approach_fingerprint|"
    r"verify fail|learning verify|diagnose:|observe:|"
    r"result\.path|result\.directory|unimplemented stub|"
    r"implement python solution|return concrete result|"
    r"independently verify artifacts|candidate\.py|repair_insight|"
    r"previously missing aspect|local capability hints|###\s*local hints|"
    r"prefer python stdlib|never claim success without producing|"
    r"producing real artifacts|ok/result/error/evidence|"
    r"hint:\s*(?:urllib|pathlib|subprocess|json stdlib|prefer python|"
    r"requests|csv stdlib|zipfile|hashlib|datetime|pillow|openpyxl|pypdf|"
    r"beautifulsoup|html\.parser|re stdlib)"
    r")"
)

_HINT_SKILL_RE = re.compile(
    r"(?i)^(?:[-*•]\s*)?(?:"
    r"hint:|###\s*(?:local hints|search|pypi)|prefer python stdlib|"
    r"skill must define|never claim success|return concrete result|"
    r"implement python solution)"
)

_EXAMPLE_CUE_RE = re.compile(
    r"(?i)\b(?:e\.g\.|for example|example|piemērs|piemeram|such as|instance)\b"
)


@dataclass
class KnowledgeArtifact:
    """Structured, VERIFY-ready learning knowledge (universal — no topic hardcode)."""

    request_id: str = ""
    topic: str = ""
    original_request: str = ""
    concepts: list[str] = field(default_factory=list)
    explanations: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    source_evidence: list[dict[str, Any]] = field(default_factory=list)
    practical_result: Optional[dict[str, Any]] = None
    practice: str = ""
    summary: str = ""
    verified: bool = False

    # ── serialization ───────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "KnowledgeArtifact":
        d = dict(data or {})
        return cls(
            request_id=str(d.get("request_id") or ""),
            topic=str(d.get("topic") or ""),
            original_request=str(d.get("original_request") or d.get("user_request") or ""),
            concepts=_clean_str_list(d.get("concepts") or d.get("key_apis") or []),
            explanations=_clean_str_list(d.get("explanations") or []),
            examples=_clean_str_list(d.get("examples") or []),
            source_evidence=list(d.get("source_evidence") or d.get("sources") or [])[:20],
            practical_result=(
                d.get("practical_result")
                if isinstance(d.get("practical_result"), dict)
                else None
            ),
            practice=str(d.get("practice") or d.get("test_idea") or "").strip(),
            summary=str(d.get("summary") or d.get("approach") or "").strip(),
            verified=bool(d.get("verified")),
        )

    def verify_payload(self) -> dict[str, Any]:
        """Fields exposed to VERIFY / semantic judgment — no raw/logs."""
        return {
            "approach": self.summary or " ".join(self.explanations)[:800],
            "summary": self.summary,
            "key_apis": list(self.concepts)[:12],
            "concepts": list(self.concepts)[:12],
            "explanations": list(self.explanations)[:8],
            "examples": list(self.examples)[:6],
            "pitfalls": [],  # never inject diagnostic gap labels
            "test_idea": self.practice,
            "practice": self.practice,
            "practical_result": self.practical_result,
            "sources": list(self.source_evidence)[:15],
            "results": [],  # exclude scrape dumps
            "raw": "",  # NEVER feed research raw into VERIFY
        }

    def research_compat(self) -> dict[str, Any]:
        """Legacy research-shaped view for memory save helpers."""
        payload = self.verify_payload()
        payload["libraries"] = []
        return payload

    # ── quality (deterministic) ─────────────────────────────────────────

    def narrative(self) -> str:
        parts = [
            self.summary,
            " ".join(self.explanations),
            " ".join(self.concepts),
            " ".join(self.examples),
            self.practice,
        ]
        pr = self.practical_result
        if isinstance(pr, dict):
            parts.append(str(pr.get("answer") or ""))
            parts.append(str(pr.get("explanation") or ""))
            if pr.get("result") is not None:
                parts.append(str(pr.get("result")))
        return "\n".join(p for p in parts if p).strip()

    def has_substance(self) -> bool:
        """Clean learning content present (never uses research raw)."""
        if is_polluted(self.summary) or any(is_polluted(x) for x in self.explanations):
            return False
        summary = (self.summary or "").strip()
        concepts = [c for c in self.concepts if c and not is_polluted(c)]
        explanations = [e for e in self.explanations if e and not is_polluted(e)]
        examples = [e for e in self.examples if e and not is_polluted(e)]
        practice = (self.practice or "").strip()
        if lv._SKILL_TEST_IDEA_RE.search(practice):
            practice = ""
        narr = self.narrative()
        # Require conceptual body — practice alone / request echo is not enough
        if len(summary) >= 40 and (concepts or explanations or examples):
            return True
        if concepts and (explanations or examples or (practice and len(summary) >= 40)):
            return True
        if len(narr) >= 120 and (concepts or explanations):
            return True
        return False

    def is_contaminated(self) -> bool:
        texts = [self.summary, self.practice, *self.concepts, *self.explanations, *self.examples]
        return any(is_polluted(t) for t in texts if t)

    def missing_fields(self, user_request: str = "") -> list[str]:
        """
        Deterministic gap list for DIAGNOSE when coverage is weak.

        Used when relatedness is high but substance fails — fill only these.
        """
        gaps: list[str] = []
        if not self.concepts:
            gaps.append("concepts")
        if not self.explanations and len(self.summary or "") < 40:
            gaps.append("explanations")
        if not self.examples:
            # Examples help substance when explanations are thin
            if len(self.summary or "") < 80 or not self.explanations:
                gaps.append("examples")
        if not self.practice or lv._SKILL_TEST_IDEA_RE.search(self.practice):
            gaps.append("practice")
        if not self.summary or len(self.summary) < 40:
            gaps.append("summary")
        if user_request and lv.requests_practical_result(user_request):
            pr = self.practical_result
            if not isinstance(pr, dict) or not (
                pr.get("answer") or pr.get("result") is not None
            ):
                gaps.append("practical_result")
        if self.is_contaminated():
            gaps.append("contamination")
        # Dedupe preserve order
        out: list[str] = []
        for g in gaps:
            if g not in out:
                out.append(g)
        return out or ["explanations"]

    def merge_gap_fill(self, other: "KnowledgeArtifact") -> "KnowledgeArtifact":
        """Merge only additive clean fields from a gap-fill pass (not full reset)."""
        return KnowledgeArtifact(
            request_id=self.request_id or other.request_id,
            topic=self.topic or other.topic,
            original_request=self.original_request or other.original_request,
            concepts=_merge_unique(self.concepts, other.concepts),
            explanations=_merge_unique(self.explanations, other.explanations),
            examples=_merge_unique(self.examples, other.examples),
            source_evidence=_merge_sources(self.source_evidence, other.source_evidence),
            practical_result=other.practical_result or self.practical_result,
            practice=(
                other.practice
                if other.practice and not lv._SKILL_TEST_IDEA_RE.search(other.practice)
                else self.practice
            ),
            summary=other.summary if len(other.summary or "") > len(self.summary or "") else self.summary,
            verified=False,
        )


def is_polluted(text: str) -> bool:
    t = (text or "").strip().lstrip("-*• ").strip()
    if not t:
        return False
    if _HINT_SKILL_RE.search(t) or _HINT_SKILL_RE.search((text or "").strip()):
        return True
    return bool(_POLLUTION_RE.search(t)) or bool(_POLLUTION_RE.search(text or ""))


def scrub_text(text: str) -> str:
    """Drop polluted lines from a free-text block."""
    keep: list[str] = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if _HINT_SKILL_RE.search(s) or is_polluted(s):
            continue
        # Drop entire local-hints / skill-builder section headers
        if re.match(r"(?i)^#{1,3}\s*(local hints|pypi hint)\b", s):
            continue
        keep.append(s)
    return "\n".join(keep).strip()


def _research_has_acquired_body(
    research: dict[str, Any],
    prior: Optional[KnowledgeArtifact] = None,
) -> bool:
    """
    True only when research actually acquired knowledge (extracts / notes).

    Empty SEARCH + invented keyword fluff must NOT count — that was the
    false LEARNING VERIFY PASS bug (0 URLs → still concepts=4 PASS).
    """
    if prior is not None and prior.has_substance():
        # Gap-fill from a prior artifact is allowed only if prior had evidence
        # or a non-echo narrative already verified upstream.
        if prior.source_evidence or len(prior.narrative()) >= 80:
            return True
    if research.get("knowledge_from_extracts"):
        return True
    extracts = research.get("extracts") or []
    if any(isinstance(e, dict) and e.get("ok") for e in extracts):
        return True
    sources = list(
        research.get("source_evidence") or research.get("sources") or []
    )
    if any(isinstance(s, dict) and s.get("extracted") for s in sources):
        return True
    approach = scrub_text(str(research.get("approach") or ""))
    if is_polluted(approach):
        approach = ""
    concepts = list(research.get("concepts") or research.get("key_apis") or [])
    concepts = [c for c in concepts if c and not is_polluted(str(c))]
    raw = scrub_text(str(research.get("raw") or ""))
    # Strip meta headers used when SEARCH finds nothing / local gap-fill seeds
    raw_body = re.sub(r"(?im)^###\s+.*$", "", raw)
    raw_body = re.sub(
        r"(?im)^(search:|gap-fill focus|search titles).*$", "", raw_body
    ).strip()
    if len(approach) >= 40 and (concepts or len(raw_body) >= 40):
        return True
    if len(raw_body) >= 120:
        return True
    if concepts and len(raw_body) >= 40:
        return True
    return False


def synthesize_offline(
    *,
    user_request: str,
    topic: str,
    request_id: str = "",
    research: Optional[dict[str, Any]] = None,
    prior: Optional[KnowledgeArtifact] = None,
    missing: Optional[list[str]] = None,
) -> KnowledgeArtifact:
    """
    Deterministic KnowledgeArtifact from research notes — no 30B required.

    Filters skill/debug pollution and builds concepts/explanations/examples
    from clean text only. Refuses to invent knowledge from the USER REQUEST
    alone when SEARCH/OPEN produced nothing.
    """
    research = dict(research or {})
    missing = [str(m) for m in (missing or []) if m]
    base = prior or KnowledgeArtifact(
        request_id=request_id,
        topic=topic,
        original_request=user_request,
    )

    if not _research_has_acquired_body(research, prior):
        # Honest empty artifact — VERIFY must FAIL (no fake PASS)
        return KnowledgeArtifact(
            request_id=request_id or base.request_id,
            topic=topic or base.topic,
            original_request=user_request or base.original_request,
            concepts=[],
            explanations=[],
            examples=[],
            source_evidence=[],
            practical_result=research.get("practical_result") or base.practical_result,
            practice="",
            summary="",
            verified=False,
        )

    raw_clean = scrub_text(str(research.get("raw") or ""))
    approach = scrub_text(str(research.get("approach") or ""))
    if is_polluted(approach) or not approach:
        approach = ""

    # Prefer existing clean summary
    summary = scrub_text(str(research.get("summary") or approach or base.summary or ""))
    if not summary and raw_clean:
        # First related sentences as summary
        summary = _pick_related_sentences(user_request, raw_clean, limit=3)

    concepts = [c for c in base.concepts if c and not is_polluted(c)]
    for item in research.get("key_apis") or research.get("concepts") or []:
        s = str(item).strip().lstrip("-*• ").strip()
        if s and not is_polluted(s) and s not in concepts:
            # Reject skill tips / off-topic scrape crumbs
            if lv.soft_relatedness(user_request, s) < 0.10 and len(concepts) > 0:
                continue
            if is_polluted(s):
                continue
            concepts.append(s)
    # Derive concepts from clean summary/raw if still empty / gap-fill
    if not concepts or "concepts" in missing:
        for part in re.split(r"[.;\n]", summary or raw_clean):
            part = part.strip().lstrip("#-*• ").strip()
            if part.lower().startswith("search:"):
                part = part.split(":", 1)[-1].strip()
            if 12 <= len(part) <= 140 and not is_polluted(part) and part not in concepts:
                # Require relatedness to USER REQUEST (never ingest random scrape)
                if lv.soft_relatedness(user_request, part) >= 0.12:
                    concepts.append(part)
            if len(concepts) >= 8:
                break

    explanations = list(base.explanations)
    if approach and not is_polluted(approach) and approach not in explanations:
        explanations.insert(0, approach[:800])
    if "explanations" in missing or not explanations:
        for sent in _pick_related_sentences(user_request, raw_clean, limit=5).split("\n"):
            sent = sent.strip()
            if sent and sent not in explanations and not is_polluted(sent):
                explanations.append(sent)

    examples = list(base.examples)
    if "examples" in missing or not examples:
        for line in (raw_clean + "\n" + summary).splitlines():
            if _EXAMPLE_CUE_RE.search(line) or lv._EXPR_RE.search(line):
                s = line.strip()
                if s and not is_polluted(s) and s not in examples:
                    examples.append(s[:240])
            if len(examples) >= 4:
                break
    # If still empty, craft a universal practice example from the request wording
    if not examples and summary:
        examples.append(
            f"Worked illustration of the user goal: {user_request[:160]}"
        )

    practice = scrub_text(str(research.get("test_idea") or research.get("practice") or base.practice or ""))
    if not practice or lv._SKILL_TEST_IDEA_RE.search(practice) or is_polluted(practice):
        practice = (
            "Self-check: restate the topic in your own words, "
            "give one worked example, and answer a practice question "
            "aligned with the USER REQUEST."
        )

    sources = _clean_sources(
        list(research.get("sources") or research.get("source_evidence") or [])
        + list(base.source_evidence)
    )

    if not summary:
        focus = "; ".join(concepts[:4]) if concepts else "core ideas"
        summary = (
            f"Understanding for the user goal — {user_request}. "
            f"Key ideas: {focus}."
        )[:800]

    # Ensure at least some concepts from the request content words (universal)
    if not concepts:
        from jarvis.intent import IntentClassifier

        for tok in IntentClassifier._keywords(user_request)[:6]:
            if tok and tok not in concepts and len(tok) >= 4:
                concepts.append(tok)
        if not concepts:
            concepts.append(f"fundamentals of: {user_request[:80]}")
    if not explanations:
        explanations.append(summary[:400])

    # Final scrub — never persist skill/debug residue
    concepts = [c for c in concepts if c and not is_polluted(c)][:12]
    explanations = [e for e in explanations if e and not is_polluted(e)][:8]
    examples = [e for e in examples if e and not is_polluted(e)][:6]
    if is_polluted(summary):
        summary = scrub_text(summary) or (
            f"Understanding for the user goal — {user_request}."
        )
    if is_polluted(practice) or lv._SKILL_TEST_IDEA_RE.search(practice):
        practice = (
            "Self-check: restate the topic in your own words, "
            "give one worked example, and answer a practice question "
            "aligned with the USER REQUEST."
        )

    art = KnowledgeArtifact(
        request_id=request_id or base.request_id,
        topic=topic or base.topic,
        original_request=user_request or base.original_request,
        concepts=concepts,
        explanations=explanations,
        examples=examples,
        source_evidence=sources[:15],
        practical_result=research.get("practical_result") or base.practical_result,
        practice=practice,
        summary=summary[:1200],
        verified=False,
    )
    if prior:
        art = prior.merge_gap_fill(art)
        art.request_id = request_id or art.request_id
        art.topic = topic or art.topic
        art.original_request = user_request or art.original_request
    return art


def gap_fill_queries(
    user_request: str,
    gaps: list[str],
    *,
    contract: Any = None,
) -> list[str]:
    """
    Targeted queries for missing KnowledgeArtifact fields only.

    Prefer locked TaskContract subject / content_requirements / outcome
    queries over dumping the whole USER REQUEST as a research blob.
    """
    anchor = ""
    outcome_qs: list[str] = []
    if contract is not None:
        try:
            subj = str(getattr(contract, "subject", "") or "")
            if subj and subj not in ("", "unknown", "ambiguous"):
                anchor = subj
            reqs = list(getattr(contract, "content_requirements", ()) or ())
            if not anchor and reqs:
                anchor = " ".join(str(r) for r in reqs[:3])
            # Prefer learning/research outcome queries when present
            for kind in ("learning", "research"):
                if hasattr(contract, "requires") and contract.requires(kind):
                    outcome_qs.extend(
                        list(contract.outcome_research_queries(kind) or [])
                    )
            if not outcome_qs and hasattr(contract, "default_research_queries"):
                outcome_qs = list(contract.default_research_queries() or [])
        except Exception:
            anchor = ""
    if not anchor:
        anchor = (user_request or "").strip()

    queries: list[str] = []
    for g in gaps:
        g = str(g)
        if g in ("concepts", "knowledge_substance", "summary", "explanations"):
            queries.append(f"{anchor} — core concepts and clear explanation")
        elif g in ("examples",):
            queries.append(f"{anchor} — worked examples and illustrations")
        elif g in ("practice",):
            queries.append(f"{anchor} — practice questions for self-check")
        elif g in ("practical_result",):
            queries.append(f"{anchor} — concrete answer / worked result")
        elif g == "contamination":
            queries.append(f"{anchor} — clean conceptual overview")
        elif g not in ("goal_coverage", "knowledge_sources", "no_skill_inherit"):
            queries.append(f"{anchor} — fill gap: {g}")
    # Prepend contract outcome queries (grounded)
    for q in outcome_qs:
        q = " ".join(str(q).split())
        if q and q not in queries:
            queries.insert(0, q)
    out: list[str] = []
    for q in queries:
        q = " ".join(q.split())
        if q and q not in out:
            out.append(q)
    if not out:
        out = [f"{anchor} — deepen understanding"] if anchor else []
    # Ground against contract when available
    if contract is not None and hasattr(contract, "ground_research_queries"):
        try:
            grounded = contract.ground_research_queries(out)
            if grounded:
                return grounded[:5]
        except Exception:
            pass
    return out[:5]


# ── helpers ─────────────────────────────────────────────────────────────

def _clean_str_list(items: list[Any]) -> list[str]:
    out: list[str] = []
    for x in items:
        s = str(x).strip()
        if s and not is_polluted(s) and s not in out:
            out.append(s)
    return out


def _merge_unique(a: list[str], b: list[str]) -> list[str]:
    out = list(a)
    for x in b:
        s = str(x).strip()
        if s and not is_polluted(s) and s not in out:
            out.append(s)
    return out[:16]


def _merge_sources(a: list[dict], b: list[dict]) -> list[dict]:
    return _clean_sources(list(a) + list(b))


def _clean_sources(sources: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for s in sources:
        if not isinstance(s, dict):
            continue
        title = str(s.get("title") or "")
        url = str(s.get("url") or "")
        # Drop local skill-hint pseudo-sources that are not evidence
        if is_polluted(title) or title.lower() in ("local capability hints",):
            continue
        key = url or title
        if not key or key in seen:
            continue
        seen.add(key)
        row = {
            "url": url,
            "title": title,
            "source": s.get("source") or s.get("provider") or "",
            "provider": s.get("provider") or "",
            "query": s.get("query") or "",
        }
        # Preserve OPEN/EXTRACT evidence flags for LEARNING VERIFY
        if s.get("extracted") or s.get("ok"):
            row["extracted"] = True
        if s.get("ok") is not None:
            row["ok"] = bool(s.get("ok"))
        out.append(row)
    return out[:20]


def _pick_related_sentences(user_request: str, text: str, limit: int = 4) -> str:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    scored: list[tuple[float, str]] = []
    for s in sentences:
        s = s.strip()
        if len(s) < 20 or is_polluted(s):
            continue
        rel = lv.soft_relatedness(user_request, s)
        if rel >= 0.08:
            scored.append((rel, s[:400]))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        # Fallback: first clean sentences
        clean = [s.strip()[:400] for s in sentences if len(s.strip()) >= 20 and not is_polluted(s)]
        return "\n".join(clean[:limit])
    return "\n".join(s for _, s in scored[:limit])
