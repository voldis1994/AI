"""
Runtime build/commit fingerprint for startup logs.

Reports git commit (and dirty state), optional build id, and package path
so operators can see exactly which code is running.
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    here = Path(__file__).resolve().parent.parent
    if (here / ".git").exists() or (here / "JARVIS.py").exists():
        return here
    return Path.cwd()


@lru_cache(maxsize=1)
def runtime_fingerprint() -> dict[str, Any]:
    root = _repo_root()
    commit = "unknown"
    dirty = False
    branch = "unknown"
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(root),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(root),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=str(root),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        dirty = bool(status)
    except Exception:
        pass

    build_id = (
        os.environ.get("CURSOR_ENVIRONMENT_BUILD_ID")
        or os.environ.get("JARVIS_BUILD_ID")
        or os.environ.get("BUILD_ID")
        or ""
    )
    return {
        "commit": commit + ("-dirty" if dirty else ""),
        "branch": branch,
        "dirty": dirty,
        "build_id": build_id or None,
        "root": str(root),
    }


def fingerprint_line() -> str:
    fp = runtime_fingerprint()
    parts = [
        f"commit={fp['commit']}",
        f"branch={fp['branch']}",
    ]
    if fp.get("build_id"):
        parts.append(f"build={fp['build_id']}")
    return "JARVIS RUNTIME " + " ".join(parts)
