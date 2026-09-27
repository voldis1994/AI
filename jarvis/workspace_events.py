"""
Lightweight workspace event kinds for GUI visualization.

Not a parallel architecture — Orchestrator._event fans out the same way as
on_log / on_status. Callers must never wait on GUI; fire-and-forget only.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

# Event kinds (string constants — universal, not task-specific)
PHASE = "phase"
SKILL_FILE = "skill_file"
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
    "OBSERVE": VIEW_TERMINAL,
    "DIAGNOSE": VIEW_WORKSPACE,
    "VERIFY": VIEW_WORKSPACE,
    "SAVE_EXPERIENCE": VIEW_CHAT,
    "DONE": VIEW_CHAT,
    "IDLE": VIEW_CHAT,
    "ERROR": VIEW_CHAT,
    "FAIL": VIEW_CHAT,
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
