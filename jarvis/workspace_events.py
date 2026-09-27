"""
Lightweight workspace event kinds for GUI visualization.

Not a parallel architecture — Orchestrator._event fans out the same way as
on_log / on_status. Callers must never wait on GUI; fire-and-forget only.

All views (CODE / TERMINAL / WORKSPACE / CHAT) consume these same events —
no simulated terminal, no UI-only code generation.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

# Event kinds (string constants — universal, not task-specific)
PHASE = "phase"
ACTION = "action"
SKILL_FILE = "skill_file"
CODE_DELTA = "code_delta"
WORKSPACE_FILE = "workspace_file"
SUBPROCESS = "subprocess"
RESEARCH = "research"
VERIFY = "verify"
COMMAND = "command"

# View hints the GUI may auto-select
VIEW_CHAT = "CHAT"
VIEW_WORKSPACE = "WORKSPACE"
VIEW_CODE = "CODE"
VIEW_TERMINAL = "TERMINAL"
VIEW_RESEARCH = "RESEARCH"
VIEW_FILES = "FILES"

VIEWS = (
    VIEW_CHAT,
    VIEW_WORKSPACE,
    VIEW_CODE,
    VIEW_TERMINAL,
    VIEW_RESEARCH,
    VIEW_FILES,
)

# Status → preferred view (universal phase map)
STATUS_VIEW = {
    "THINKING": VIEW_CHAT,
    "CONVERSING": VIEW_CHAT,
    "REQUEST": VIEW_CHAT,
    "MEMORY": VIEW_RESEARCH,
    "PLAN": VIEW_WORKSPACE,
    "CHECK_CAPABILITIES": VIEW_WORKSPACE,
    "RESEARCH": VIEW_RESEARCH,
    "LEARN": VIEW_RESEARCH,
    "BUILD_SKILL": VIEW_CODE,
    "REPAIR": VIEW_CODE,
    "SAVE_SKILL": VIEW_CODE,
    "INSTALL_DEPS": VIEW_TERMINAL,
    "TEST": VIEW_TERMINAL,
    "RETEST": VIEW_TERMINAL,
    "EXECUTE": VIEW_TERMINAL,
    "OBSERVE": VIEW_WORKSPACE,
    "DIAGNOSE": VIEW_WORKSPACE,
    "VERIFY": VIEW_WORKSPACE,
    "SAVE_EXPERIENCE": VIEW_CHAT,
    "DONE": VIEW_CHAT,
    "IDLE": VIEW_CHAT,
    "ERROR": VIEW_CHAT,
    "FAIL": VIEW_CHAT,
}

# Short live action lines for CHAT / WORKSPACE (same pipeline phases)
ACTION_SUMMARY = {
    "THINKING": "Thinking...",
    "CONVERSING": "Replying...",
    "REQUEST": "Accepted request.",
    "MEMORY": "Checking memory...",
    "PLAN": "Planning...",
    "CHECK_CAPABILITIES": "Matching capabilities...",
    "RESEARCH": "Researching...",
    "LEARN": "Learning...",
    "BUILD_SKILL": "Generating code...",
    "REPAIR": "Repairing...",
    "SAVE_SKILL": "Saving skill...",
    "INSTALL_DEPS": "Installing dependencies...",
    "TEST": "Testing...",
    "RETEST": "Retesting...",
    "EXECUTE": "Executing...",
    "OBSERVE": "Observing failure...",
    "DIAGNOSE": "Diagnosing...",
    "VERIFY": "Verifying...",
    "SAVE_EXPERIENCE": "Saving experience...",
    "DONE": "Done.",
    "IDLE": "Idle.",
    "ERROR": "Error.",
    "FAIL": "Failed.",
}

EventCallback = Callable[[str, dict[str, Any]], None]


def fire(
    callback: Optional[EventCallback],
    kind: str,
    **payload: Any,
) -> None:
    """Invoke callback without raising; never block the caller."""
    if callback is None:
        return
    try:
        callback(kind, payload)
    except Exception:
        pass


def view_for_status(status: str) -> str:
    return STATUS_VIEW.get(str(status or "").upper(), VIEW_CHAT)


def action_summary_for_status(
    status: str,
    *,
    reason: str = "",
    verified: Optional[bool] = None,
) -> str:
    """Human-short action line from a real pipeline phase / verify outcome."""
    key = str(status or "").upper()
    if verified is False or (
        key == "VERIFY" and reason and "fail" in reason.lower()
    ):
        detail = (reason or "verification failed").strip()
        if len(detail) > 160:
            detail = detail[:157] + "..."
        return f"Verifier failed: {detail}"
    if verified is True and key == "VERIFY":
        return "Verifier passed."
    return ACTION_SUMMARY.get(key, f"{key.title()}..." if key else "")
