"""
JARVIS research system — gathers knowledge before building a new skill.

Each result stores: URL, title, source/provider, timestamp, query.
Results feed skill building via research notes.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.parse
import urllib.request
from typing import Any, Callable, Optional

logger = logging.getLogger("jarvis.research")


class ResearchSystem:
    """Collects information needed to learn a new capability."""

    def __init__(
        self,
        brain: Any = None,
        on_log: Optional[Callable[[str], None]] = None,
        timeout: float = 20.0,
    ) -> None:
        self.brain = brain
        self.on_log = on_log or (lambda _m: None)
        self.timeout = timeout

    def research(self, queries: list[str], goal: str = "") -> dict[str, Any]:
        self.on_log(f"RESEARCH: starting ({len(queries)} queries)")
        gathered_parts: list[str] = []
        results: list[dict[str, Any]] = []

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

        for q in queries[:5]:
            self.on_log(f"RESEARCH: query → {q}")
            html_notes, ddg_results = self._duckduckgo_search(q)
            if html_notes:
                gathered_parts.append(f"### Search: {q}\n{html_notes}")
            results.extend(ddg_results)

            pypi_notes, pypi_results = self._pypi_lookup(q)
            if pypi_notes:
                gathered_parts.append(f"### PyPI hint: {q}\n{pypi_notes}")
            results.extend(pypi_results)

        gathered = "\n\n".join(gathered_parts)

        # Compact source list for builders / prompts
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

        notes: dict[str, Any] = {
            "approach": "",
            "libraries": [],
            "key_apis": [],
            "pitfalls": [],
            "test_idea": "Execute skill.run and independently verify artifacts",
            "raw": gathered[:12000],
            "sources": sources[:30],
            "results": results[:30],
        }

        if self.brain is not None:
            try:
                # Feed structured results into synthesis
                research_blob = gathered + "\n\nSTRUCTURED RESULTS:\n" + json.dumps(
                    sources[:15], ensure_ascii=False, default=str
                )
                synthesized = self.brain.research_notes(
                    goal or (queries[0] if queries else ""),
                    research_blob,
                )
                notes.update({k: v for k, v in synthesized.items() if k not in ("raw", "results", "sources")})
            except Exception as exc:
                logger.warning("brain research_notes failed: %s", exc)
                notes["approach"] = gathered[:800]

        if not notes.get("approach"):
            notes["approach"] = gathered[:800] or f"Implement Python solution for: {goal}"

        self.on_log(
            f"RESEARCH: done — {len(notes.get('libraries') or [])} libs, "
            f"{len(results)} results"
        )
        return notes

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

    def _duckduckgo_search(self, query: str) -> tuple[str, list[dict[str, Any]]]:
        results: list[dict[str, Any]] = []
        ts = time.time()
        try:
            q = urllib.parse.quote_plus(query + " python")
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
                html,
                flags=re.I | re.S,
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
        notes = []
        results: list[dict[str, Any]] = []
        ts = time.time()
        for pkg in candidates:
            try:
                url = f"https://pypi.org/pypi/{urllib.parse.quote(pkg)}/json"
                req = urllib.request.Request(
                    url, headers={"User-Agent": "JARVIS-Research/1.0"}
                )
                with urllib.request.urlopen(req, timeout=8) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                info = data.get("info", {})
                summary = (info.get("summary") or "")[:200]
                name = info.get("name") or pkg
                project_url = info.get("project_url") or f"https://pypi.org/project/{name}/"
                notes.append(f"- {name}: {summary}")
                results.append({
                    "url": project_url,
                    "title": f"PyPI: {name}",
                    "source": "pypi",
                    "provider": "pypi.org",
                    "timestamp": ts,
                    "query": query,
                    "snippet": summary,
                })
            except Exception:
                continue
        return "\n".join(notes), results
