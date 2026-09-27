"""
JARVIS research system — gathers knowledge before building a skill or learning.

Modes:
  skill    — local skill hints, optional \" python\" DDG bias, brain notes if online
  learning — conceptual focus only; never skill-contract hints; never brain skill-repair notes

Network and brain synthesis are independently gateable (gap-fill = local only).
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional

logger = logging.getLogger("jarvis.research")

# Keep network waits short — slow endpoints must not dominate the loop.
DEFAULT_TIMEOUT = 10.0
PYPI_TIMEOUT = 5.0
MAX_QUERIES_SKILL = 5
MAX_QUERIES_LEARNING = 3


class ResearchSystem:
    """Collects information needed to learn a new capability or topic."""

    def __init__(
        self,
        brain: Any = None,
        on_log: Optional[Callable[[str], None]] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.brain = brain
        self.on_log = on_log or (lambda _m: None)
        self.timeout = timeout

    def research(
        self,
        queries: list[str],
        goal: str = "",
        *,
        mode: str = "skill",
        network: bool = True,
        use_brain: Optional[bool] = None,
        max_queries: Optional[int] = None,
        append_python: Optional[bool] = None,
    ) -> dict[str, Any]:
        """
        Gather research notes.

        mode=\"learning\": no skill local hints, no skill-repair brain synthesis,
        no forced \" python\" query suffix (unless append_python=True).
        network=False: skip DuckDuckGo/PyPI (true gap-fill / offline path).
        use_brain: None → only for skill mode when brain.is_available().
        """
        mode_key = (mode or "skill").strip().lower()
        if mode_key not in ("skill", "learning"):
            mode_key = "skill"
        limit = max_queries
        if limit is None:
            limit = MAX_QUERIES_LEARNING if mode_key == "learning" else MAX_QUERIES_SKILL
        limit = max(1, min(int(limit), 5))
        do_python = (
            bool(append_python)
            if append_python is not None
            else (mode_key == "skill")
        )
        brain_ok = False
        if use_brain is None:
            brain_ok = (
                mode_key == "skill"
                and self.brain is not None
                and bool(getattr(self.brain, "is_available", lambda: False)())
            )
        else:
            brain_ok = bool(use_brain) and self.brain is not None

        self.on_log(
            f"RESEARCH: starting ({len(queries)} queries) mode={mode_key} "
            f"network={network} brain={brain_ok}"
        )
        gathered_parts: list[str] = []
        results: list[dict[str, Any]] = []

        if mode_key == "learning":
            gathered_parts.append(
                self._learning_local_notes(goal or " ".join(queries))
            )
            results.append({
                "url": "",
                "title": "Learning focus",
                "source": "local",
                "provider": "jarvis.research",
                "timestamp": time.time(),
                "query": goal or (queries[0] if queries else ""),
                "snippet": gathered_parts[-1][:500],
            })
        else:
            gathered_parts.append(self._local_hints(goal or " ".join(queries)))
            results.append({
                "url": "",
                "title": "Local capability hints",
                "source": "local",
                "provider": "jarvis.research",
                "timestamp": time.time(),
                "query": goal or (queries[0] if queries else ""),
                "snippet": gathered_parts[-1][:500],
            })

        query_list = list(queries[:limit])
        for q in query_list:
            self.on_log(f"RESEARCH: query → {q}")

        if network and query_list:
            def _fetch_one(q: str) -> tuple[str, list[str], list[dict[str, Any]]]:
                parts: list[str] = []
                local_results: list[dict[str, Any]] = []
                html_notes, ddg_results = self._duckduckgo_search(
                    q, append_python=do_python
                )
                if html_notes:
                    parts.append(f"### Search: {q}\n{html_notes}")
                local_results.extend(ddg_results)
                # Learning: skip PyPI package noise (not conceptual evidence)
                if mode_key == "skill":
                    pypi_notes, pypi_results = self._pypi_lookup(q)
                    if pypi_notes:
                        parts.append(f"### PyPI hint: {q}\n{pypi_notes}")
                    local_results.extend(pypi_results)
                return q, parts, local_results

            if len(query_list) <= 1:
                for q in query_list:
                    _, parts, local_results = _fetch_one(q)
                    gathered_parts.extend(parts)
                    results.extend(local_results)
            else:
                by_query: dict[str, tuple[list[str], list[dict[str, Any]]]] = {}
                with ThreadPoolExecutor(max_workers=min(4, len(query_list))) as pool:
                    futs = {pool.submit(_fetch_one, q): q for q in query_list}
                    for fut in as_completed(futs):
                        q, parts, local_results = fut.result()
                        by_query[q] = (parts, local_results)
                for q in query_list:
                    parts, local_results = by_query.get(q, ([], []))
                    gathered_parts.extend(parts)
                    results.extend(local_results)
        elif not network:
            # Gap-fill / local: keep query text as synthesis seed (no skill jargon)
            gathered_parts.append(
                "### Gap-fill focus\n" + "\n".join(f"- {q}" for q in query_list)
            )

        gathered = "\n\n".join(gathered_parts)

        sources = [
            {
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "source": r.get("source") or r.get("provider", ""),
                "provider": r.get("provider", ""),
                "timestamp": r.get("timestamp"),
                "query": r.get("query", ""),
            }
            for r in results
            if r.get("url") or r.get("source") == "local"
        ]

        default_test = (
            "Self-check: restate concepts, give one example, answer a practice question."
            if mode_key == "learning"
            else "Execute skill.run and independently verify artifacts"
        )
        notes: dict[str, Any] = {
            "approach": "",
            "libraries": [],
            "key_apis": [],
            "pitfalls": [],
            "test_idea": default_test,
            "raw": gathered[:12000],
            "sources": sources[:30],
            "results": results[:30],
            "mode": mode_key,
            "network": bool(network),
            "brain_used": False,
        }

        if brain_ok:
            try:
                research_blob = gathered + "\n\nSTRUCTURED RESULTS:\n" + json.dumps(
                    sources[:15], ensure_ascii=False, default=str
                )
                synthesized = self.brain.research_notes(
                    goal or (queries[0] if queries else ""),
                    research_blob,
                )
                notes.update(
                    {
                        k: v
                        for k, v in synthesized.items()
                        if k not in ("raw", "results", "sources", "mode", "network")
                    }
                )
                notes["brain_used"] = True
            except Exception as exc:
                logger.warning("brain research_notes failed: %s", exc)
                notes["approach"] = gathered[:800]

        if not notes.get("approach"):
            if mode_key == "learning":
                notes["approach"] = (
                    gathered[:800]
                    or f"Study and explain: {goal or (queries[0] if queries else '')}"
                )
            else:
                notes["approach"] = (
                    gathered[:800] or f"Implement Python solution for: {goal}"
                )

        self.on_log(
            f"RESEARCH: done — {len(notes.get('libraries') or [])} libs, "
            f"{len(results)} results mode={mode_key} brain={notes['brain_used']}"
        )
        return notes

    def _learning_local_notes(self, text: str) -> str:
        return (
            "### Learning focus\n"
            f"- USER REQUEST / topic: {text}\n"
            "- Extract concepts, clear explanations, worked examples, "
            "and a practice self-check aligned with the request.\n"
            "- Do not include skill contracts, pip installs, or repair diagnostics."
        )

    def _local_hints(self, text: str) -> str:
        t = text.lower()
        hints = [
            "Prefer Python stdlib when possible.",
            "Skill must define SKILL_META and run(context)->dict with ok/result/error/evidence.",
            "Return concrete result.path / result.directory so the verifier can re-check.",
            "Never claim success without producing real artifacts.",
        ]
        mapping = [
            (("http", "url", "download", "fetch", "api", "request"), "urllib.request or requests"),
            (("json",), "json stdlib"),
            (("csv",), "csv stdlib"),
            (("file", "read", "write", "folder", "directory"), "pathlib / os"),
            (("zip", "archive"), "zipfile stdlib"),
            (("hash", "md5", "sha"), "hashlib stdlib"),
            (("regex", "parse text"), "re stdlib"),
            (("time", "date", "schedule"), "datetime stdlib"),
            (("subprocess", "shell", "command"), "subprocess stdlib"),
            (("image", "png", "jpg"), "Pillow (pip: Pillow)"),
            (("excel", "xlsx"), "openpyxl (pip: openpyxl)"),
            (("pdf",), "pypdf (pip: pypdf)"),
            (("scrape", "html", "beautifulsoup"), "html.parser or beautifulsoup4"),
        ]
        for keys, tip in mapping:
            if any(k in t for k in keys):
                hints.append(f"Hint: {tip}")
        return "### Local hints\n" + "\n".join(f"- {h}" for h in hints)

    def _duckduckgo_search(
        self, query: str, *, append_python: bool = True
    ) -> tuple[str, list[dict[str, Any]]]:
        results: list[dict[str, Any]] = []
        ts = time.time()
        try:
            q_text = query + (" python" if append_python else "")
            q = urllib.parse.quote_plus(q_text)
            url = f"https://html.duckduckgo.com/html/?q={q}"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "JARVIS-Research/1.0"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                html = resp.read().decode("utf-8", errors="replace")
            titles = re.findall(
                r'class="result__a"[^>]*>(.*?)</a>', html, flags=re.I | re.S
            )
            snippets = re.findall(
                r'class="result__snippet"[^>]*>(.*?)</(?:a|td|div)',
                html, flags=re.I | re.S,
            )
            hrefs = re.findall(
                r'class="result__a"[^>]*href="([^"]+)"', html, flags=re.I
            )
            clean = []
            for i, title in enumerate(titles[:5]):
                t = re.sub(r"<[^>]+>", "", title).strip()
                sn = ""
                if i < len(snippets):
                    sn = re.sub(r"<[^>]+>", "", snippets[i]).strip()
                href = hrefs[i] if i < len(hrefs) else ""
                m = re.search(r"uddg=([^&]+)", href)
                if m:
                    href = urllib.parse.unquote(m.group(1))
                clean.append(f"- {t}: {sn}")
                results.append({
                    "url": href,
                    "title": t,
                    "source": "web",
                    "provider": "duckduckgo",
                    "timestamp": ts,
                    "query": query,
                    "snippet": sn,
                })
            return ("\n".join(clean) if clean else ""), results
        except Exception as exc:
            logger.info("DDG search failed: %s", exc)
            results.append({
                "url": "",
                "title": "DuckDuckGo unavailable",
                "source": "web",
                "provider": "duckduckgo",
                "timestamp": ts,
                "query": query,
                "snippet": str(exc),
            })
            return f"(web search unavailable: {exc})", results

    def _pypi_lookup(self, query: str) -> tuple[str, list[dict[str, Any]]]:
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", query.lower())
        candidates = []
        skip = {
            "python", "how", "the", "and", "for", "with", "from", "that",
            "this", "create", "make", "write", "file", "using",
        }
        for tok in tokens:
            if tok in skip:
                continue
            candidates.append(tok)
            if len(candidates) >= 3:
                break
        if not candidates:
            return "", []

        notes: list[str] = []
        results: list[dict[str, Any]] = []
        ts = time.time()

        def _one(pkg: str) -> Optional[tuple[str, dict[str, Any]]]:
            try:
                url = f"https://pypi.org/pypi/{urllib.parse.quote(pkg)}/json"
                req = urllib.request.Request(
                    url, headers={"User-Agent": "JARVIS-Research/1.0"}
                )
                with urllib.request.urlopen(req, timeout=PYPI_TIMEOUT) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                info = data.get("info", {})
                summary = (info.get("summary") or "")[:200]
                name = info.get("name") or pkg
                project_url = (
                    info.get("project_url") or f"https://pypi.org/project/{name}/"
                )
                return (
                    f"- {name}: {summary}",
                    {
                        "url": project_url,
                        "title": f"PyPI: {name}",
                        "source": "pypi",
                        "provider": "pypi.org",
                        "timestamp": ts,
                        "query": query,
                        "snippet": summary,
                    },
                )
            except Exception:
                return None

        # Parallel package lookups (was serial — major latency)
        with ThreadPoolExecutor(max_workers=min(3, len(candidates))) as pool:
            futs = [pool.submit(_one, pkg) for pkg in candidates]
            for fut in as_completed(futs):
                item = fut.result()
                if item:
                    note, row = item
                    notes.append(note)
                    results.append(row)
        return "\n".join(notes), results
