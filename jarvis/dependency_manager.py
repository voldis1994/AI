"""
JARVIS dependency manager — installs pip packages required by skills.
"""

from __future__ import annotations

import importlib
import logging
import subprocess
import sys
from typing import Callable, Optional

logger = logging.getLogger("jarvis.deps")

# Map import names → pip package names when they differ
IMPORT_TO_PIP = {
    "bs4": "beautifulsoup4",
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "yaml": "PyYAML",
    "docx": "python-docx",
    "sklearn": "scikit-learn",
    "dateutil": "python-dateutil",
}


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
        """Install missing packages. Returns {ok, installed, failed, details}."""
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
            if self._is_importable(dep) or self._is_importable(pip_name.replace("-", "_")):
                details.append(f"{pip_name}: already available")
                continue
            if pip_name in self._installed:
                continue

            self.on_log(f"DEPS: installing {pip_name}")
            ok, msg = self._pip_install(pip_name)
            details.append(msg)
            if ok:
                installed.append(pip_name)
                self._installed.add(pip_name)
            else:
                failed.append(pip_name)

        return {
            "ok": len(failed) == 0,
            "installed": installed,
            "failed": failed,
            "details": "\n".join(details),
        }

    def _is_importable(self, name: str) -> bool:
        mod = name.replace("-", "_").split("[")[0]
        # Try common variants
        candidates = [mod, IMPORT_TO_PIP.get(mod, mod)]
        # reverse map
        for imp, pip in IMPORT_TO_PIP.items():
            if pip.lower() == name.lower() or pip.lower() == mod.lower():
                candidates.append(imp)
        for c in candidates:
            try:
                importlib.import_module(c)
                return True
            except Exception:
                continue
        return False

    def _pip_install(self, package: str) -> tuple[bool, str]:
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            if proc.returncode == 0:
                return True, f"{package}: installed OK"
            err = (proc.stderr or proc.stdout or "")[-500:]
            return False, f"{package}: FAILED — {err}"
        except subprocess.TimeoutExpired:
            return False, f"{package}: FAILED — timeout"
        except Exception as exc:
            return False, f"{package}: FAILED — {exc}"
