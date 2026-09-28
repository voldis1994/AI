"""
Universal request-item classification for JARVIS.

Before research / install, extracted tokens are labeled as one of:
  PATH | FILE | LIBRARY | COMMAND | CONTENT | REQUIREMENT

Words from file paths or user prose are NEVER treated as PyPI dependencies.
Only LIBRARY items (and imports actually required by generated skill code)
may be researched or installed.

Generic — no task-specific hardcoding of paths or package names.
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Optional


class ItemKind(str, Enum):
    PATH = "PATH"
    FILE = "FILE"
    LIBRARY = "LIBRARY"
    COMMAND = "COMMAND"
    CONTENT = "CONTENT"
    REQUIREMENT = "REQUIREMENT"


@dataclass(frozen=True)
class RequestItem:
    text: str
    kind: ItemKind
    reason: str = ""


# Common stdlib top-level modules — never pip-install these.
_STDLIB: frozenset[str] = frozenset(
    {
        "abc", "argparse", "array", "ast", "asyncio", "base64", "binascii",
        "bisect", "builtins", "calendar", "cmath", "collections", "concurrent",
        "contextlib", "copy", "csv", "ctypes", "dataclasses", "datetime",
        "decimal", "difflib", "enum", "errno", "fnmatch", "fractions",
        "functools", "gc", "getpass", "gettext", "glob", "gzip", "hashlib",
        "heapq", "hmac", "html", "http", "importlib", "inspect", "io",
        "ipaddress", "itertools", "json", "logging", "math", "mimetypes",
        "multiprocessing", "numbers", "operator", "os", "pathlib", "pickle",
        "platform", "pprint", "queue", "random", "re", "secrets", "select",
        "shlex", "shutil", "signal", "socket", "sqlite3", "ssl", "statistics",
        "string", "struct", "subprocess", "sys", "tempfile", "textwrap",
        "threading", "time", "timeit", "tkinter", "traceback", "types",
        "typing", "unicodedata", "urllib", "uuid", "venv", "warnings",
        "weakref", "xml", "xmlrpc", "zipfile", "zlib", "zoneinfo",
        "__future__",
    }
)

_STOP = {
    "a", "an", "the", "and", "or", "to", "for", "with", "from", "into",
    "in", "on", "at", "of", "by", "is", "are", "be", "as", "it", "this",
    "that", "create", "write", "make", "build", "run", "file", "files",
    "containing", "contains", "named", "called", "please", "jarvis",
    "text", "content", "contents", "data", "value", "values", "using",
    "how", "show", "example", "examples", "demo", "python", "code",
    "script", "program", "output", "path", "paths", "install", "package",
    "library", "module", "dependency", "dependencies",
    "izveido", "uzraksti", "failu", "faila", "ar", "saturu", "satur",
}

_PATH_LIKE = re.compile(
    r"^(?:[A-Za-z]:)?[A-Za-z0-9_./\\-]+\.[A-Za-z0-9]{1,12}$"
)
_CMD_RE = re.compile(
    r"(?i)^\s*(?:python3?|pip3?|node|npm|cargo|go|ruby|bash|sh|curl|wget)\b"
)
_LIB_CUE = re.compile(
    r"(?i)\b(?:pip\s+install|import|using\s+(?:the\s+)?(?:library|package)|"
    r"depends?\s+on|dependency|from\s+pypi)\b"
)


def looks_like_path(value: str) -> bool:
    sv = str(value or "").strip()
    if not sv or sv.lower().startswith(("http://", "https://")):
        return False
    if _PATH_LIKE.match(sv):
        return True
    if ("/" in sv or "\\" in sv) and " " not in sv:
        return True
    return False


def looks_like_file(value: str) -> bool:
    """Bare filename with extension (may also be a relative PATH)."""
    sv = str(value or "").strip()
    if not sv or " " in sv:
        return False
    return bool(re.match(r"^[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,12}$", sv))


def path_segment_tokens(
    request: str,
    artifacts: Optional[Iterable[str]] = None,
) -> set[str]:
    """
    Directory parts + stems from path-like tokens in the request / artifacts.

    Example: 'workspace/calculator.py' → {'workspace', 'calculator', 'calculator.py'}
    These must never become LIBRARY / pip targets / polluted arg keys.
    """
    out: set[str] = set()
    candidates: list[str] = []
    for m in re.finditer(
        r"(?<![A-Za-z0-9_\"'])([A-Za-z0-9_./\\-]+\.[A-Za-z0-9]{1,12})\b",
        request or "",
    ):
        candidates.append(m.group(1))
    for a, b in re.findall(r"\"([^\"]+)\"|'([^']+)'", request or ""):
        tok = (a or b).strip()
        if looks_like_path(tok):
            candidates.append(tok)
    for art in artifacts or ():
        if art:
            candidates.append(str(art))
    # Also catch path-shaped tokens without requiring a match boundary again
    for tok in re.findall(r"[A-Za-z0-9_./\\-]{2,}", request or ""):
        if looks_like_path(tok):
            candidates.append(tok)

    for raw in candidates:
        sv = str(raw).strip().strip("\"'")
        if not sv:
            continue
        pure = PureWindowsPath(sv) if "\\" in sv else PurePosixPath(sv)
        for part in pure.parts:
            if part in (".", "..", "/", "\\"):
                continue
            low = part.lower()
            if low in _STOP:
                continue
            out.add(low)
            stem = PurePosixPath(part).stem.lower()
            if stem and stem not in _STOP and len(stem) >= 2:
                out.add(stem)
        # Full relative path lowercased
        out.add(sv.replace("\\", "/").lower())
    return out


def classify_token(
    token: str,
    *,
    request: str = "",
    path_segments: Optional[set[str]] = None,
) -> ItemKind:
    """Classify a single extracted token (generic heuristics)."""
    sv = str(token or "").strip()
    if not sv:
        return ItemKind.CONTENT
    low = sv.lower()
    segs = path_segments if path_segments is not None else path_segment_tokens(request)

    if _CMD_RE.match(sv):
        return ItemKind.COMMAND
    if looks_like_path(sv):
        return ItemKind.PATH if ("/" in sv or "\\" in sv) else ItemKind.FILE
    if looks_like_file(sv):
        return ItemKind.FILE
    if low in segs or PurePosixPath(sv).stem.lower() in segs:
        # Bare stem/dir of an output path — not a library
        return ItemKind.PATH
    if low in _STDLIB:
        return ItemKind.CONTENT  # stdlib mention ≠ install target
    if low in _STOP:
        return ItemKind.CONTENT
    # Explicit library cues near the token in the request
    if request and _LIB_CUE.search(request):
        # Only if token itself looks like a package name and not a path segment
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{1,63}", sv) and low not in segs:
            return ItemKind.LIBRARY
    # Default: prose / content / soft requirement — never auto-LIBRARY
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{2,63}", sv):
        return ItemKind.CONTENT
    return ItemKind.CONTENT


def classify_request_items(
    request: str,
    *,
    grounded_args: Optional[dict[str, Any]] = None,
    artifacts: Optional[Iterable[str]] = None,
) -> list[RequestItem]:
    """
    Extract and classify items from the user request (+ grounded args).

    PATH/FILE cover output artifacts; CONTENT covers payloads; LIBRARY is
    reserved for explicit package cues (never path stems).
    """
    req = (request or "").strip()
    arts = list(artifacts or [])
    items: list[RequestItem] = []
    seen: set[tuple[str, str]] = set()

    def _add(text: str, kind: ItemKind, reason: str) -> None:
        key = (text.lower(), kind.value)
        if not text or key in seen:
            return
        seen.add(key)
        items.append(RequestItem(text=text, kind=kind, reason=reason))

    segs = path_segment_tokens(req, arts)

    # Paths / files from request
    for m in re.finditer(
        r"(?<![A-Za-z0-9_\"'])([A-Za-z0-9_./\\-]+\.[A-Za-z0-9]{1,12})\b",
        req,
    ):
        tok = m.group(1)
        kind = (
            ItemKind.PATH
            if ("/" in tok or "\\" in tok)
            else ItemKind.FILE
        )
        _add(tok, kind, "extension-bearing path/file in request")
        if tok not in arts:
            arts.append(tok)

    for a, b in re.findall(r"\"([^\"]+)\"|'([^']+)'", req):
        tok = (a or b).strip()
        if not tok:
            continue
        if looks_like_path(tok) or looks_like_file(tok):
            kind = (
                ItemKind.PATH
                if ("/" in tok or "\\" in tok)
                else ItemKind.FILE
            )
            _add(tok, kind, "quoted path/file")
            if tok not in arts:
                arts.append(tok)
        else:
            _add(tok, ItemKind.CONTENT, "quoted content")

    # Grounded args
    for k, v in (grounded_args or {}).items():
        sv = str(v).strip()
        if not sv:
            continue
        if looks_like_path(sv) or looks_like_file(sv):
            kind = (
                ItemKind.PATH
                if ("/" in sv or "\\" in sv)
                else ItemKind.FILE
            )
            _add(sv, kind, f"grounded arg {k}")
        else:
            _add(sv, ItemKind.CONTENT, f"grounded arg {k}")

    # Requirement cues
    for m in re.finditer(
        r"(?i)\b(?:must|should|need(?:s)?\s+to|required\s+to)\s+([^.;\n]{3,80})",
        req,
    ):
        _add(m.group(1).strip(), ItemKind.REQUIREMENT, "requirement cue")

    # Explicit library cues only
    if _LIB_CUE.search(req):
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,63}", req):
            low = tok.lower()
            if low in _STOP or low in segs or low in _STDLIB:
                continue
            if classify_token(tok, request=req, path_segments=segs) == ItemKind.LIBRARY:
                _add(tok, ItemKind.LIBRARY, "explicit library cue")

    # Soft content leftovers (never LIBRARY)
    for tok in re.findall(r"[A-Za-z0-9_./\\-]{3,}", req):
        low = tok.lower()
        if low in _STOP or low in segs:
            continue
        if looks_like_path(tok) or looks_like_file(tok):
            continue
        if any(i.text.lower() == low for i in items):
            continue
        # Skip if already covered; don't invent LIBRARY
        kind = classify_token(tok, request=req, path_segments=segs)
        if kind in (ItemKind.CONTENT, ItemKind.REQUIREMENT):
            # Don't flood — only keep a few significant content tokens
            if len([i for i in items if i.kind == ItemKind.CONTENT]) < 6:
                _add(tok, ItemKind.CONTENT, "request token")

    return items


def is_installable_library(
    name: str,
    *,
    request: str = "",
    artifacts: Optional[Iterable[str]] = None,
) -> bool:
    """True only when name can safely be treated as a third-party package."""
    sv = str(name or "").strip()
    if not sv or " " in sv:
        return False
    low = sv.lower().replace("-", "_").split("[")[0]
    if looks_like_path(sv) or "/" in sv or "\\" in sv or sv.endswith(".py"):
        return False
    if low in _STDLIB or low in _STOP:
        return False
    segs = path_segment_tokens(request, artifacts)
    if low in segs or PurePosixPath(sv).stem.lower() in segs:
        return False
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{1,63}", sv):
        return False
    return True


def sanitize_libraries(
    names: Optional[Iterable[Any]],
    request: str = "",
    *,
    artifacts: Optional[Iterable[str]] = None,
    skill_code: Optional[str] = None,
) -> list[str]:
    """
    Drop path/file/content tokens from a dependency list.

    When skill_code is provided, keep only packages that the code actually
    imports (external). Never invent installs from prose/path stems.
    """
    raw = [str(x).strip() for x in (names or []) if str(x).strip()]
    code_imports = extract_external_imports(skill_code) if skill_code else []
    if skill_code is not None:
        # Code is authority: only install what it imports externally
        return list(code_imports)

    out: list[str] = []
    for name in raw:
        if is_installable_library(name, request=request, artifacts=artifacts):
            if name not in out:
                out.append(name)
    return out


def extract_external_imports(code: Optional[str]) -> list[str]:
    """Parse skill source; return third-party top-level import names."""
    src = code or ""
    if not src.strip():
        return []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        # Fallback: regex import lines
        names = re.findall(
            r"(?m)^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_]*)",
            src,
        )
        return [
            n for n in dict.fromkeys(names)
            if n.lower() not in _STDLIB and n.lower() not in _STOP
        ]

    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = (alias.name or "").split(".")[0]
                if top:
                    found.append(top)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # relative
            if node.module:
                top = node.module.split(".")[0]
                if top:
                    found.append(top)

    # Prefer stdlib check via sys.stdlib_module_names when available (3.10+)
    std = set(_STDLIB)
    try:
        std |= {m.lower() for m in getattr(sys, "stdlib_module_names", ())}
    except Exception:
        pass

    out: list[str] = []
    for name in found:
        low = name.lower()
        if low in std or low in _STOP:
            continue
        if low not in out:
            out.append(low)
    return out


def library_tokens_for_pypi(
    query: str,
    *,
    request: str = "",
    artifacts: Optional[Iterable[str]] = None,
) -> list[str]:
    """
    Tokens safe to look up on PyPI for a research query.

    Never returns path segments, filenames, or stopwords.
    """
    req = request or query
    segs = path_segment_tokens(req, artifacts)
    items = classify_request_items(req, artifacts=artifacts)
    libs = [i.text for i in items if i.kind == ItemKind.LIBRARY]
    if libs:
        return libs[:3]
    # No explicit LIBRARY items → do not invent candidates from prose/paths
    # (path segments / files are intentionally excluded)
    _ = segs  # used via classify / is_installable
    return []
