"""
JARVIS dependency manager — installs pip packages for the current user.

After every install, re-checks import. No privilege escalation / UAC bypass.
"""

from __future__ import annotations

import importlib
import logging
import subprocess
import sys
from typing import Callable, Optional

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
        timeout: float = 180.0,
    ) -> None:
        self.on_log = on_log or (lambda _m: None)
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
            ok, msg = self._pip_install_user(pip_name)
            details.append(msg)
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
                    details.append(f"{pip_name}: re-import OK via subprocess ({import_name})")
                else:
                    failed.append(pip_name)
                    details.append(f"{pip_name}: install reported OK but import FAILED")

        return {
            "ok": len(failed) == 0,
            "installed": installed,
            "failed": failed,
            "details": "\n".join(details),
        }

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

    def _pip_install_user(self, package: str) -> tuple[bool, str]:
        """Install with current-user rights only. No sudo / UAC bypass."""
        cmd = [sys.executable, "-m", "pip", "install", "--user", package]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            if proc.returncode == 0:
                return True, f"{package}: pip install --user OK"
            err = (proc.stderr or proc.stdout or "")[-500:]
            # Fallback without --user if environment forbids it (e.g. venv)
            if "not on PATH" in err or "Can not perform a '--user'" in err or proc.returncode != 0:
                proc2 = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                if proc2.returncode == 0:
                    return True, f"{package}: pip install OK (env user site)"
                err2 = (proc2.stderr or proc2.stdout or "")[-500:]
                return False, f"{package}: FAILED — {err2}"
            return False, f"{package}: FAILED — {err}"
        except subprocess.TimeoutExpired:
            return False, f"{package}: FAILED — timeout"
        except Exception as exc:
            return False, f"{package}: FAILED — {exc}"

    def _subprocess_import_check(self, name: str) -> bool:
        mod = name.replace("-", "_").split("[")[0]
        try:
            proc = subprocess.run(
                [sys.executable, "-c", f"import importlib; importlib.import_module({mod!r})"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return proc.returncode == 0
        except Exception:
            return False
