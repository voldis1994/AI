"""
Universal TaskContract / pipeline invariants.

Python enforces these rules for every request. No task-specific hardcode —
only structural checks against the immutable contract and evidence.
"""

from __future__ import annotations

from typing import Any, Optional

from jarvis.task_contract import TaskContract


class InvariantError(RuntimeError):
    """Raised when a pipeline stage would violate a hard architecture invariant."""


def assert_locked_truth(before: TaskContract, after: TaskContract) -> None:
    """
    After VALIDATE, original_request / desired_outcomes / acceptance_criteria
    must never be rewritten.
    """
    if not before.validated:
        return
    if after.original_request != before.original_request:
        raise InvariantError("original_request rewritten after VALIDATE")
    if after.desired_outcomes != before.desired_outcomes:
        raise InvariantError("desired_outcomes rewritten after VALIDATE")
    if after.acceptance_criteria != before.acceptance_criteria:
        raise InvariantError("acceptance_criteria rewritten after VALIDATE")


def assert_build_allowed(contract: TaskContract) -> None:
    """BUILD must not run without a verifiable, validated TaskContract."""
    if contract.needs_refine() or not contract.is_verifiable():
        raise InvariantError(
            "BUILD refused — TaskContract not verifiable (needs_refine)"
        )
    if not contract.validated:
        raise InvariantError("BUILD refused — TaskContract not validated")


def assert_no_invented_requirements(
    contract: TaskContract,
    candidate_values: list[Any] | tuple[Any, ...] | None,
) -> list[str]:
    """
    Return values that are not grounded in original_request (invented).

    Callers must drop these — PLAN/RESEARCH/SKILL/DIAGNOSIS/VERIFIER must not
    invent new user requirements.
    """
    bad: list[str] = []
    req = contract.original_request
    for v in candidate_values or ():
        if v in (None, "", [], {}):
            continue
        if isinstance(v, (int, float, bool)):
            continue
        if TaskContract.is_invented_default(v, req):
            bad.append(str(v))
            continue
        if not TaskContract.is_grounded(v, req) and not TaskContract.looks_like_url(
            str(v)
        ):
            # Allow short schema keys; reject ungrounded payloads/paths
            sv = str(v)
            if TaskContract.looks_like_path(sv) or len(sv) >= 3:
                if sv.lower() not in (req or "").lower() and not any(
                    str(x) == sv for x in (contract.inputs or {}).values()
                ):
                    bad.append(sv)
    return bad


def assert_active_requires_independent_verify(
    *,
    verified: bool,
    skill_ok: bool,
    test_ok: bool,
) -> None:
    """ACTIVE must not be granted from skill self-report or TEST ok alone."""
    if not verified:
        raise InvariantError(
            "ACTIVE refused — independent VERIFY evidence required "
            f"(skill_ok={skill_ok} test_ok={test_ok} verified={verified})"
        )


def assert_done_requires_verify(*, verified: bool, has_evidence: bool) -> None:
    """DONE only with independently verified evidence."""
    if not verified or not has_evidence:
        raise InvariantError(
            "DONE refused — missing independent VERIFY evidence "
            f"(verified={verified} has_evidence={has_evidence})"
        )


def rewrite_allowed_for_fault(fault_layer: str | None) -> bool:
    """skill/tool rewrite only when fault_owner is implementation/skill_code."""
    layer = TaskContract.normalize_fault_layer(fault_layer)
    return TaskContract.rewrite_skill_for_layer(layer)


def assert_dependency_structured(
    libraries: list[Any] | tuple[Any, ...] | None,
    contract: TaskContract,
) -> list[str]:
    """
    Dependency install only from validated structured dependency requirements.

    Path/file stems from the request must never become pip packages.
    """
    return list(
        TaskContract.sanitize_libraries(list(libraries or []), contract.original_request)
    )


def prior_request_isolation(
    prev_request_id: Optional[str], new_request_id: str
) -> bool:
    """
    True when the new request is isolated from any prior request.

    Rules:
    - ``new_request_id`` must be non-empty
    - if a prior id exists, it must differ (same id ⇒ NOT isolated)
    - if there is no prior id, the first request is isolated by definition
    """
    new_id = str(new_request_id or "").strip()
    if not new_id:
        return False
    prev_id = str(prev_request_id or "").strip()
    if not prev_id:
        return True
    return prev_id != new_id


def assert_requests_isolated(
    prev_request_id: Optional[str], new_request_id: str
) -> None:
    """Raise if a new request would reuse a prior request_id."""
    if not prior_request_isolation(prev_request_id, new_request_id):
        raise InvariantError(
            "request isolation violated — "
            f"prev={prev_request_id!r} new={new_request_id!r}"
        )
