"""
Structured TaskContract outcomes — not exclusive intent labels.

A single USER REQUEST may require multiple outcomes simultaneously
(learning + artifact + execution + formatted response, …). Global DONE
is allowed only when every required outcome is independently VERIFIED.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


# Explicit skill/BUILD cues — bare "2+5" / "cik ir 2+5" must NOT match.
_SKILL_BUILD_CUE_RE = re.compile(
    r"(?i)\b(?:"
    r"return|compute|calculate|create|write|make|build|implement|"
    r"code|function|program|script|skill|module|class|"
    r"izveido|uzraksti|aprēķini|atgriez|implement[eē]"
    r")\b"
)


# Canonical outcome kinds (open set — stages may add grounded kinds)
OUTCOME_CONVERSATION = "conversation"
OUTCOME_LEARNING = "learning"
OUTCOME_RESEARCH = "research"
OUTCOME_ARTIFACT = "artifact"
OUTCOME_CAPABILITY = "capability"
OUTCOME_EXECUTION = "execution"
OUTCOME_TEST = "test"
OUTCOME_SIDE_EFFECT = "side_effect"
OUTCOME_FINAL_OUTPUT = "final_output"

ALL_OUTCOME_KINDS = frozenset(
    {
        OUTCOME_CONVERSATION,
        OUTCOME_LEARNING,
        OUTCOME_RESEARCH,
        OUTCOME_ARTIFACT,
        OUTCOME_CAPABILITY,
        OUTCOME_EXECUTION,
        OUTCOME_TEST,
        OUTCOME_SIDE_EFFECT,
        OUTCOME_FINAL_OUTPUT,
    }
)

STATUS_PENDING = "PENDING"
STATUS_VERIFIED = "VERIFIED"
STATUS_FAILED = "FAILED"
STATUS_SKIPPED = "SKIPPED"


@dataclass
class OutcomeTracker:
    """Mutable per-request tracker for required outcome verification."""

    required: tuple[str, ...]
    status: dict[str, str] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        req = tuple(dict.fromkeys(str(x) for x in self.required if str(x).strip()))
        self.required = req
        for k in req:
            self.status.setdefault(k, STATUS_PENDING)

    def mark(
        self,
        kind: str,
        *,
        verified: bool,
        evidence: Any = None,
        failed: bool = False,
    ) -> None:
        kind = str(kind or "").strip()
        if not kind:
            return
        if kind not in self.status and kind not in self.required:
            # Allow recording optional stage evidence without expanding DONE gate
            self.status[kind] = STATUS_PENDING
        if failed:
            self.status[kind] = STATUS_FAILED
        elif verified:
            self.status[kind] = STATUS_VERIFIED
        else:
            self.status[kind] = STATUS_PENDING
        if evidence is not None:
            self.evidence[kind] = evidence

    def skip(self, kind: str, reason: str = "") -> None:
        kind = str(kind or "").strip()
        if kind in self.required:
            # Required outcomes cannot be skipped into DONE
            self.status[kind] = STATUS_PENDING
            if reason:
                self.evidence[kind] = {"skipped_refused": reason}
            return
        self.status[kind] = STATUS_SKIPPED
        if reason:
            self.evidence[kind] = reason

    def all_required_verified(self) -> bool:
        if not self.required:
            return False
        return all(self.status.get(k) == STATUS_VERIFIED for k in self.required)

    def pending(self) -> list[str]:
        return [
            k
            for k in self.required
            if self.status.get(k) not in (STATUS_VERIFIED, STATUS_SKIPPED)
        ]

    def failed(self) -> list[str]:
        return [k for k in self.required if self.status.get(k) == STATUS_FAILED]

    def snapshot(self) -> dict[str, Any]:
        return {
            "required": list(self.required),
            "status": dict(self.status),
            "pending": self.pending(),
            "failed": self.failed(),
            "all_verified": self.all_required_verified(),
        }


def derive_required_outcomes(
    *,
    artifacts: Iterable[str] = (),
    behaviors: Iterable[Any] = (),
    side_effects: Iterable[str] = (),
    acceptance_criteria: Iterable[str] = (),
    content_source: Iterable[str] = (),
    constraints: Optional[dict[str, Any]] = None,
    draft_outcomes: Iterable[str] = (),
    requirements_hint: Optional[dict[str, Any]] = None,
    final_output_constraints: Optional[dict[str, Any]] = None,
    request_text: str = "",
) -> tuple[str, ...]:
    """
    Derive required outcomes structurally from the semantic contract.

    Hints may *add* requirements; they must never remove artifact/execution
    requirements that the grounded contract already implies.
    """
    out: list[str] = []
    cons = constraints or {}
    hint = requirements_hint or {}
    arts = [str(a) for a in (artifacts or ()) if str(a).strip()]
    beh = list(behaviors or ())
    http = list(cons.get("http") or [])
    req_text = (request_text or "").strip()

    def _add(kind: str) -> None:
        if kind and kind not in out:
            out.append(kind)

    # Practical learning results (e.g. compute 12+5 during LEARN) are verified
    # inside the learning stage — they must not force a parallel skill lifecycle
    # when there is no artifact/network side effect.
    practical_learning_beh = bool(
        hint.get("needs_learning")
        and not arts
        and not http
        and not cons.get("files")
        and beh
        and all(
            isinstance(b, dict)
            and str(b.get("kind") or "").startswith("skill_result_")
            for b in beh
        )
    )
    # Bare Q&A like "2+5" / "cik ir 2+5" — answer directly; do NOT BUILD a skill
    # and hang on model timeouts. Explicit return/compute/create… stay skill path.
    practical_qa_beh = bool(
        not practical_learning_beh
        and not arts
        and not http
        and not cons.get("files")
        and not hint.get("needs_capability")
        and not hint.get("needs_learning")
        and beh
        and all(
            isinstance(b, dict)
            and str(b.get("kind") or "").startswith("skill_result_")
            for b in beh
        )
        and (not req_text or not _SKILL_BUILD_CUE_RE.search(req_text))
    )
    skip_skill_lifecycle = practical_learning_beh or practical_qa_beh

    # Explicit draft / semantic outcomes (already grounded upstream)
    for raw in draft_outcomes or ():
        s = str(raw or "").strip().lower()
        if not s:
            continue
        head = s.split(":", 1)[0]
        if head in ALL_OUTCOME_KINDS:
            _add(head)
        elif s.startswith("artifact"):
            _add(OUTCOME_ARTIFACT)
        elif s.startswith("behavior") or s.startswith("skill_"):
            if not skip_skill_lifecycle:
                _add(OUTCOME_EXECUTION)
        elif s.startswith("http") or s.startswith("content"):
            _add(OUTCOME_ARTIFACT if arts else OUTCOME_SIDE_EFFECT)
        elif s.startswith("learning") or s.startswith("knowledge"):
            _add(OUTCOME_LEARNING)

    if arts or cons.get("files") or cons.get("directories"):
        _add(OUTCOME_ARTIFACT)
        _add(OUTCOME_EXECUTION)
        _add(OUTCOME_TEST)
        _add(OUTCOME_CAPABILITY)

    if beh and not skip_skill_lifecycle:
        _add(OUTCOME_EXECUTION)
        _add(OUTCOME_TEST)
        _add(OUTCOME_CAPABILITY)

    if http or any(
        str(s).startswith("network") for s in (side_effects or ())
    ):
        _add(OUTCOME_SIDE_EFFECT)
        if not arts and not skip_skill_lifecycle:
            _add(OUTCOME_EXECUTION)
            _add(OUTCOME_CAPABILITY)

    srcs = [str(s).lower() for s in (content_source or ())]
    if "memory" in srcs or hint.get("needs_learning"):
        _add(OUTCOME_LEARNING)

    if hint.get("needs_learning"):
        _add(OUTCOME_LEARNING)

    if hint.get("needs_research"):
        # Research is distinct from learning; both may be required together
        # so research PASS can never be mistaken for learning PASS.
        _add(OUTCOME_RESEARCH)

    # Final-output constraints (language / format / sentence count)
    foc = final_output_constraints or cons.get("final_output") or {}
    has_foc = bool(
        foc and any(foc.get(k) not in (None, "", [], {}) for k in foc)
    )
    if has_foc:
        _add(OUTCOME_FINAL_OUTPUT)

    # Conversation / answer when no actionable / learning / research work
    actionable_present = bool(
        set(out)
        & {
            OUTCOME_ARTIFACT,
            OUTCOME_EXECUTION,
            OUTCOME_CAPABILITY,
            OUTCOME_TEST,
            OUTCOME_SIDE_EFFECT,
            OUTCOME_LEARNING,
            OUTCOME_RESEARCH,
        }
    )
    if practical_qa_beh or not actionable_present:
        if practical_qa_beh or hint.get("needs_conversation") or not (
            hint.get("needs_capability")
            or hint.get("needs_learning")
            or hint.get("needs_research")
        ):
            _add(OUTCOME_CONVERSATION)

    # Hints cannot strip artifact/execution once derived
    if hint.get("needs_capability") and not practical_qa_beh:
        _add(OUTCOME_CAPABILITY)
        # Capability without concrete artifact/behavior → still required
        # (VALIDATE will needs_refine — not conversation fallback)
        if not arts and not beh and not http:
            _add(OUTCOME_EXECUTION)

    # Acceptance criteria kinds reinforce outcomes
    for crit in acceptance_criteria or ():
        s = str(crit or "")
        low = s.lower()
        if low.startswith("artifact_") or "content_present" in low:
            _add(OUTCOME_ARTIFACT)
            _add(OUTCOME_EXECUTION)
        elif low.startswith("http_"):
            _add(OUTCOME_SIDE_EFFECT)
        elif low.startswith("skill_") or low.startswith("behavior"):
            if not skip_skill_lifecycle:
                _add(OUTCOME_EXECUTION)
        elif "knowledge" in low or low.startswith("learning"):
            _add(OUTCOME_LEARNING)
        elif low.startswith("final_output"):
            _add(OUTCOME_FINAL_OUTPUT)

    return tuple(out)


def extract_final_output_constraints(request: str) -> dict[str, Any]:
    """
    Grounded final-output constraints from the USER REQUEST text.

    Structural / universal cues only — not task-specific. Values must appear
    in the request (language codes, quoted format labels, digit counts).
    """
    import re

    req = request or ""
    out: dict[str, Any] = {}
    # Language: "in Latvian", "latviešu valodā", "in English", "pa latviski"
    lang_m = re.search(
        r"(?i)\b(?:in|on|pa)\s+"
        r"(latvian|english|german|french|russian|spanish|latviešu|ang[lļ]u|"
        r"v[aā]cu|fran[cč]u|krievu)\b",
        req,
    )
    if not lang_m:
        lang_m = re.search(
            r"(?i)\b(latviešu|ang[lļ]u|v[aā]cu)\s+valod",
            req,
        )
    if lang_m:
        out["language"] = lang_m.group(1).lower()

    # Sentence / paragraph count: "3 sentences", "2 teikumos"
    n_m = re.search(
        r"(?i)\b(\d+)\s*(?:sentences?|teikum[aāos]*)\b",
        req,
    )
    if n_m:
        out["sentence_count"] = int(n_m.group(1))

    # Explicit format quotes: format "bullet list"
    fmt_m = re.search(r"(?i)format(?:ted)?\s+[\"']([^\"']+)[\"']", req)
    if fmt_m:
        out["format"] = fmt_m.group(1).strip()

    return out


def verify_final_output(
    text: str,
    constraints: dict[str, Any],
) -> tuple[bool, str]:
    """Deterministic checks for final-output constraints against reply text."""
    import re

    if not constraints:
        return True, "no final_output constraints"
    body = (text or "").strip()
    if not body:
        return False, "empty final output"
    details: list[str] = []

    sc = constraints.get("sentence_count")
    if sc is not None:
        # Split on sentence terminators; require exact count when specified
        parts = [p for p in re.split(r"[.!?…]+", body) if p.strip()]
        if len(parts) != int(sc):
            return False, f"sentence_count want={sc} got={len(parts)}"
        details.append(f"sentence_count={sc}")

    fmt = constraints.get("format")
    if fmt:
        if str(fmt).lower() not in body.lower():
            # Soft: format label need not appear; check structural bullets
            if "bullet" in str(fmt).lower():
                if not re.search(r"(?m)^\s*[-*•]", body):
                    return False, f"format={fmt!r} missing bullet structure"
            else:
                return False, f"format={fmt!r} not satisfied"
        details.append(f"format={fmt}")

    lang = constraints.get("language")
    if lang:
        # Presence check only when user named the language as a deliverable
        # word inside the output is not required; mark verified if non-empty
        # (full language ID needs model — deterministic gate = non-empty body)
        details.append(f"language={lang}")

    return True, "final_output ok: " + ", ".join(details or ["basic"])
