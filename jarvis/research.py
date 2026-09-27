"""
JARVIS research system — gathers knowledge before building a new skill.

Uses local docs, web search (DuckDuckGo HTML), and optional Ollama synthesis.
Does not claim success; only returns gathered notes.
"""

from __future__ import annotations

import json
import logging
import re
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
        sources: list[dict[str, str]] = []

        # Always include stdlib / local knowledge hints
        gathered_parts.append(self._local_hints(goal or " ".join(queries)))

        for q in queries[:5]:
            self.on_log(f"RESEARCH: query → {q}")
            html_notes, urls = self._duckduckgo_search(q)
            if html_notes:
                gathered_parts.append(f"### Search: {q}\n{html_notes}")
                sources.extend(urls)
            pypi = self._pypi_lookup(q)
            if pypi:
                gathered_parts.append(f"### PyPI hint: {q}\n{pypi}")

        gathered = "\n\n".join(gathered_parts)
        notes: dict[str, Any] = {
            "approach": "",
            "libraries": [],
            "key_apis": [],
            "pitfalls": [],
            "test_idea": "Execute skill.run and assert ok/evidence",
            "raw": gathered[:12000],
            "sources": sources[:20],
        }

        if self.brain is not None:
            try:
                synthesized = self.brain.research_notes(
                    goal or (queries[0] if queries else ""),
                    gathered,
                )
                notes.update({k: v for k, v in synthesized.items() if k != "raw"})
            except Exception as exc:
                logger.warning("brain research_notes failed: %s", exc)
                notes["approach"] = gathered[:800]

        if not notes.get("approach"):
            notes["approach"] = gathered[:800] or f"Implement Python solution for: {goal}"

        self.on_log(
            f"RESEARCH: done — {len(notes.get('libraries') or [])} libs, "
            f"{len(sources)} sources"
        )
        return notes

    def _local_hints(self, text: str) -> str:
        t = text.lower()
        hints = [
            "Prefer Python stdlib when possible.",
            "Skill must define SKILL_META and run(context)->dict with ok/result/error/evidence.",
            "Never claim success without producing evidence (file exists, output bytes, etc.).",
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
            (("subprocess", "shell", "command"), "subprocess stdlib — careful with shell=True"),
            (("image", "png", "jpg"), "Pillow (pip: Pillow)"),
            (("excel", "xlsx"), "openpyxl (pip: openpyxl)"),
            (("pdf",), "pypdf (pip: pypdf)"),
            (("scrape", "html", "beautifulsoup"), "html.parser or beautifulsoup4"),
        ]
        for keys, tip in mapping:
            if any(k in t for k in keys):
                hints.append(f"Hint: {tip}")
        return "### Local hints\n" + "\n".join(f"- {h}" for h in hints)

    def _duckduckgo_search(self, query: str) -> tuple[str, list[dict[str, str]]]:
        urls: list[dict[str, str]] = []
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
            # Extract result snippets
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
                # DuckDuckGo redirect URLs
                m = re.search(r"uddg=([^&]+)", href)
                if m:
                    href = urllib.parse.unquote(m.group(1))
                clean.append(f"- {t}: {sn}")
                if href:
                    urls.append({"title": t, "url": href})
            return ("\n".join(clean) if clean else ""), urls
        except Exception as exc:
            logger.info("DDG search failed: %s", exc)
            return f"(web search unavailable: {exc})", urls

    def _pypi_lookup(self, query: str) -> str:
        # Extract likely package tokens
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
                notes.append(f"- {name}: {summary}")
            except Exception:
                continue
        return "\n".join(notes)
