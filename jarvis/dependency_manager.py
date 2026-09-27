"""
JARVIS dependency manager — installs pip packages for the current user.

After every install, re-checks import. No privilege escalation / UAC bypass.
"""

from __future__ import annotations

import importlib
import logging
import subprocess
import sys
from typing import Any, Callable, Optional

logger = logging.getLogger("jarvis.deps")

IMPORT_TO_PIP = {
    "bs4": "beautifulsoup4",
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "yaml": "PyYAML",
    "docx": "python-docx",
    "sklearn": "scikit-learn",
    "dateutil": "python-dateutil",
}

PIP_TO_IMPORT = {v: k for k, v in IMPORT_TO_PIP.items()}


class DependencyManager:
    """Ensures skill dependencies are installed before testing/execution."""

    def __init__(
        self,
        on_log: Optional[Callable[[str], None]] = None,
        on_event: Optional[Callable[[str, dict], None]] = None,
        timeout: float = 180.0,
    ) -> None:
        self.on_log = on_log or (lambda _m: None)
        self.on_event = on_event or (lambda _k, _p: None)
        self.timeout = timeout
        self._installed: set[str] = set()

    def ensure(self, dependencies: list[str]) -> dict:
        """Install missing packages as current user, then re-verify import."""
        if not dependencies:
            return {"ok": True, "installed": [], "failed": [], "details": "no deps"}

        installed: list[str] = []
        failed: list[str] = []
        details: list[str] = []

        for dep in dependencies:
            dep = (dep or "").strip()
            if not dep:
                continue
            pip_name = IMPORT_TO_PIP.get(dep, dep)
            import_name = self._import_name_for(dep, pip_name)

            if self._is_importable(import_name):
                details.append(f"{pip_name}: already importable as {import_name}")
                continue

            self.on_log(f"DEPS: installing {pip_name} (--user, current user)")
            ok, msg, proc_info = self._pip_install_user(pip_name)
            details.append(msg)
            self._emit_pip(proc_info)
            if not ok:
                failed.append(pip_name)
                continue

            # Obligatory re-check after install
            importlib.invalidate_caches()
            if self._is_importable(import_name):
                installed.append(pip_name)
                self._installed.add(pip_name)
                details.append(f"{pip_name}: re-import OK ({import_name})")
            else:
                # Fresh subprocess import check (avoids parent cache issues)
                if self._subprocess_import_check(import_name):
                    installed.append(pip_name)
                    self._installed.add(pip_name)
                    details.append(
                        f"{pip_name}: re-import OK via subprocess ({import_name})"
                    )
                else:
                    failed.append(pip_name)
                    details.append(
                        f"{pip_name}: install reported OK but import FAILED"
                    )

        return {
            "ok": len(failed) == 0,
            "installed": installed,
            "failed": failed,
            "details": "\n".join(details),
        }

    def _emit_pip(self, info: dict[str, Any]) -> None:
        if not info:
            return
        try:
            self.on_event(
                "subprocess",
                {
                    "mode": "install_deps",
                    "command": str(info.get("command") or ""),
                    "stdout": str(info.get("stdout") or "")[:8000],
                    "stderr": str(info.get("stderr") or "")[:8000],
                    "traceback": "",
                    "returncode": info.get("returncode"),
                    "timed_out": bool(info.get("timed_out")),
                    "ok": bool(info.get("ok")),
                    "error": str(info.get("error") or "")[:500],
                    "skill_path": "",
                },
            )
        except Exception:
            pass

    def _import_name_for(self, dep: str, pip_name: str) -> str:
        if dep in IMPORT_TO_PIP:
            return dep
        if pip_name in PIP_TO_IMPORT:
            return PIP_TO_IMPORT[pip_name]
        return pip_name.replace("-", "_").split("[")[0]

    def _is_importable(self, name: str) -> bool:
        mod = name.replace("-", "_").split("[")[0]
        candidates = [mod]
        for imp, pip in IMPORT_TO_PIP.items():
            if pip.lower() == name.lower() or pip.lower() == mod.lower() or imp == mod:
                candidates.append(imp)
        for c in candidates:
            try:
                importlib.import_module(c)
                return True
            except Exception:
                continue
        return False

    def _pip_install_user(self, package: str) -> tuple[bool, str, dict[str, Any]]:
        """Install with current-user rights only. No sudo / UAC bypass."""
        cmd = [sys.executable, "-m", "pip", "install", "--user", package]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            info: dict[str, Any] = {
                "command": " ".join(cmd),
                "stdout": proc.stdout or "",
                "stderr": proc.stderr or "",
                "returncode": proc.returncode,
                "timed_out": False,
                "ok": proc.returncode == 0,
                "error": "",
            }
            if proc.returncode == 0:
                return True, f"{package}: pip install --user OK", info
            err = (proc.stderr or proc.stdout or "")[-500:]
            # Fallback without --user if environment forbids it (e.g. venv)
            if (
                "not on PATH" in err
                or "Can not perform a '--user'" in err
                or proc.returncode != 0
            ):
                cmd2 = [sys.executable, "-m", "pip", "install", package]
                proc2 = subprocess.run(
                    cmd2,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                info2: dict[str, Any] = {
                    "command": " ".join(cmd2),
                    "stdout": proc2.stdout or "",
                    "stderr": proc2.stderr or "",
                    "returncode": proc2.returncode,
                    "timed_out": False,
                    "ok": proc2.returncode == 0,
                    "error": "",
                }
                if proc2.returncode == 0:
                    return True, f"{package}: pip install OK (env user site)", info2
                err2 = (proc2.stderr or proc2.stdout or "")[-500:]
                info2["error"] = err2
                return False, f"{package}: FAILED — {err2}", info2
            info["error"] = err
            return False, f"{package}: FAILED — {err}", info
        except subprocess.TimeoutExpired:
            return (
                False,
                f"{package}: FAILED — timeout",
                {
                    "command": " ".join(cmd),
                    "stdout": "",
                    "stderr": "",
                    "returncode": None,
                    "timed_out": True,
                    "ok": False,
                    "error": "timeout",
                },
            )
        except Exception as exc:
            return (
                False,
                f"{package}: FAILED — {exc}",
                {
                    "command": " ".join(cmd),
                    "stdout": "",
                    "stderr": "",
                    "returncode": None,
                    "timed_out": False,
                    "ok": False,
                    "error": str(exc),
                },
            )

    def _subprocess_import_check(self, name: str) -> bool:
        mod = name.replace("-", "_").split("[")[0]
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    f"import importlib; importlib.import_module({mod!r})",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return proc.returncode == 0
        except Exception:
            return False
