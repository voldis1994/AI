"""
Universal source pipeline for JARVIS research.

SEARCH hits are candidates only — snippets are NOT knowledge.
Quality ranking → OPEN/FETCH full pages → EXTRACT text → COMPARE → feed SYNTHESIZE.

No hardcoded sites (no Wikipedia-only bias). Ranking is topic-relative and
structural (https, substance length, relatedness, low-spam signals).
"""

from __future__ import annotations

import html as html_lib
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional

from jarvis import learning_verify as lv

logger = logging.getLogger("jarvis.source_pipeline")

DEFAULT_FETCH_TIMEOUT = 12.0
MAX_OPEN_SOURCES = 4
MIN_EXTRACT_CHARS = 180
MAX_EXTRACT_CHARS = 6000

_UA = "JARVIS-Research/1.0 (+local; knowledge pipeline)"

# Structural spam / non-content signals (universal — not site allowlists)
_LOW_QUALITY_TITLE_RE = re.compile(
    r"(?i)\b(?:"
    r"unavailable|error|404|login|sign[\s-]?in|subscribe|buy now|"
    r"cookie|captcha|access denied|just a moment"
    r")\b"
)
_SPAM_HOST_HINT_RE = re.compile(
    r"(?i)(?:"
    r"clickbait|free-download|torrent|porn|casino|betting|"
    r"\.xyz/|\.tk/|bit\.ly|tinyurl"
    r")"
)


def rank_search_hits(
    hits: list[dict[str, Any]],
    *,
    goal: str,
    limit: int = MAX_OPEN_SOURCES,
) -> list[dict[str, Any]]:
    """
    Rank raw SEARCH hits for OPEN. Snippets alone never become knowledge —
    they only help choose which URLs to fetch.
    """
    scored: list[tuple[float, dict[str, Any]]] = []
    seen_urls: set[str] = set()
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        url = str(hit.get("url") or "").strip()
        title = str(hit.get("title") or "").strip()
        snippet = str(hit.get("snippet") or "").strip()
        if not url or not url.lower().startswith(("http://", "https://")):
            continue
        # Normalize URL key (strip fragment)
        key = urllib.parse.urldefrag(url)[0].rstrip("/")
        if key in seen_urls:
            continue
        seen_urls.add(key)
        if _LOW_QUALITY_TITLE_RE.search(title):
            continue
        if _SPAM_HOST_HINT_RE.search(url) or _SPAM_HOST_HINT_RE.search(title):
            continue

        blob = f"{title}\n{snippet}"
        rel = lv.soft_relatedness(goal, blob) if goal else 0.0
        score = rel * 2.0
        if url.lower().startswith("https://"):
            score += 0.15
        if len(title) >= 12:
            score += 0.08
        if len(snippet) >= 40:
            score += 0.08
        # Prefer content-ish paths over bare homepages / pure tracking
        path = urllib.parse.urlparse(url).path or ""
        if path not in ("", "/"):
            score += 0.05
        if score < 0.08 and rel < 0.08:
            continue
        scored.append((score, {**hit, "url": url, "rank_score": round(score, 4)}))

    scored.sort(key=lambda x: -x[0])
    return [h for _, h in scored[: max(1, limit)]]


def fetch_url(
    url: str,
    *,
    timeout: float = DEFAULT_FETCH_TIMEOUT,
) -> dict[str, Any]:
    """OPEN a URL and return status + raw body (html/text)."""
    out: dict[str, Any] = {
        "url": url,
        "ok": False,
        "status": 0,
        "content_type": "",
        "body": "",
        "error": "",
    }
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": _UA,
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
            },
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            ctype = str(resp.headers.get("Content-Type") or "")
            charset = "utf-8"
            m = re.search(r"charset=([\w-]+)", ctype, flags=re.I)
            if m:
                charset = m.group(1)
            try:
                body = raw.decode(charset, errors="replace")
            except Exception:
                body = raw.decode("utf-8", errors="replace")
            out.update({
                "ok": True,
                "status": int(getattr(resp, "status", 200) or 200),
                "content_type": ctype,
                "body": body[:250_000],
            })
            return out
    except urllib.error.HTTPError as exc:
        out["status"] = int(exc.code or 0)
        out["error"] = str(exc)
        try:
            out["body"] = (exc.read() or b"").decode("utf-8", errors="replace")[:50_000]
        except Exception:
            pass
        return out
    except Exception as exc:
        out["error"] = str(exc)
        logger.info("SOURCE OPEN failed %s: %s", url[:120], exc)
        return out


def extract_text(body: str, *, content_type: str = "") -> str:
    """EXTRACT readable text from HTML/plain — drop scripts/nav chrome lightly."""
    if not body:
        return ""
    ct = (content_type or "").lower()
    if "json" in ct and "<" not in body[:200]:
        return body.strip()[:MAX_EXTRACT_CHARS]
    if "html" in ct or "<html" in body[:500].lower() or "<!doctype" in body[:200].lower():
        text = body
        text = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", text)
        text = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", text)
        text = re.sub(r"(?is)<noscript[^>]*>.*?</noscript>", " ", text)
        text = re.sub(r"(?is)<!--.*?-->", " ", text)
        # Drop common chrome tags' contents coarsely
        text = re.sub(r"(?is)<(nav|footer|header|aside)[^>]*>.*?</\1>", " ", text)
        text = re.sub(r"(?is)<br\s*/?>", "\n", text)
        text = re.sub(r"(?is)</p\s*>", "\n\n", text)
        text = re.sub(r"(?is)<[^>]+>", " ", text)
        text = html_lib.unescape(text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()[:MAX_EXTRACT_CHARS]
    return body.strip()[:MAX_EXTRACT_CHARS]


def extract_from_fetch(fetched: dict[str, Any], *, goal: str = "") -> dict[str, Any]:
    """Build an EXTRACT record from an OPEN result."""
    body = str(fetched.get("body") or "")
    text = extract_text(body, content_type=str(fetched.get("content_type") or ""))
    # Prefer goal-related paragraphs
    if goal and text:
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) >= 40]
        scored = sorted(
            ((lv.soft_relatedness(goal, p), p) for p in paras),
            key=lambda x: -x[0],
        )
        if scored and scored[0][0] >= 0.08:
            keep = [p for rel, p in scored[:12] if rel >= 0.06]
            if keep:
                text = "\n\n".join(keep)[:MAX_EXTRACT_CHARS]
    return {
        "url": fetched.get("url") or "",
        "ok": bool(fetched.get("ok")) and len(text) >= MIN_EXTRACT_CHARS,
        "text": text,
        "char_count": len(text),
        "relatedness": lv.soft_relatedness(goal, text) if goal and text else 0.0,
        "error": fetched.get("error") or "",
        "status": fetched.get("status") or 0,
        "title": "",
    }


def compare_extracts(
    extracts: list[dict[str, Any]],
    *,
    goal: str,
) -> dict[str, Any]:
    """
    COMPARE opened sources — find consensus ideas vs thin/singleton noise.
    Deterministic; no LLM required.
    """
    usable = [e for e in extracts if e.get("ok") and e.get("text")]
    if not usable:
        return {
            "consensus": [],
            "support_count": {},
            "source_count": 0,
            "compared": False,
        }
    # Tokenize content words from each extract
    token_sets: list[set[str]] = []
    for e in usable:
        toks = {
            t for t in re.findall(r"[a-zA-Zāčēģīķļņšūž]{4,}", (e.get("text") or "").lower())
            if t not in lv._STOP
        }
        token_sets.append(toks)
    # Goal terms preferred
    goal_toks = {
        t for t in re.findall(r"[a-zA-Zāčēģīķļņšūž]{4,}", (goal or "").lower())
        if t not in lv._STOP
    }
    support: dict[str, int] = {}
    for ts in token_sets:
        for t in ts:
            if goal_toks and t not in goal_toks and lv.soft_relatedness(goal, t) < 0.05:
                # still count shared technical terms across sources
                pass
            support[t] = support.get(t, 0) + 1
    # Consensus = terms appearing in ≥2 sources (or top goal-aligned if single source)
    consensus = [t for t, c in support.items() if c >= 2]
    if not consensus and usable:
        # Single solid source — take top related sentences as "supported"
        text = usable[0].get("text") or ""
        sents = re.split(r"(?<=[.!?])\s+", text)
        consensus_sents = []
        for s in sents:
            s = s.strip()
            if len(s) < 40:
                continue
            if lv.soft_relatedness(goal, s) >= 0.12:
                consensus_sents.append(s[:240])
            if len(consensus_sents) >= 6:
                break
        return {
            "consensus": consensus_sents,
            "support_count": {k: v for k, v in sorted(support.items(), key=lambda x: -x[1])[:40]},
            "source_count": len(usable),
            "compared": True,
            "single_source": True,
        }
    # Build short consensus phrases from shared terms + related sentences
    phrases: list[str] = []
    for e in usable:
        for s in re.split(r"(?<=[.!?])\s+", e.get("text") or ""):
            s = s.strip()
            if len(s) < 40:
                continue
            hits = sum(1 for t in consensus if t in s.lower())
            if hits >= 2 and lv.soft_relatedness(goal, s) >= 0.10:
                if s not in phrases:
                    phrases.append(s[:280])
            if len(phrases) >= 8:
                break
    return {
        "consensus": phrases or sorted(consensus, key=lambda t: -support.get(t, 0))[:12],
        "support_count": {k: v for k, v in sorted(support.items(), key=lambda x: -x[1])[:40]},
        "source_count": len(usable),
        "compared": True,
        "single_source": False,
    }


def open_and_extract(
    candidates: list[dict[str, Any]],
    *,
    goal: str,
    on_log: Optional[Callable[[str], None]] = None,
    timeout: float = DEFAULT_FETCH_TIMEOUT,
    max_open: int = MAX_OPEN_SOURCES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    SOURCE OPEN + SOURCE EXTRACT for top-ranked candidates (parallel).
    Returns (extracts_with_meta, comparison).
    """
    log = on_log or (lambda _m: None)
    ranked = rank_search_hits(candidates, goal=goal, limit=max_open)
    extracts: list[dict[str, Any]] = []

    def _one(hit: dict[str, Any]) -> dict[str, Any]:
        url = str(hit.get("url") or "")
        log(f"SOURCE OPEN: {url[:160]}")
        fetched = fetch_url(url, timeout=timeout)
        if fetched.get("ok"):
            log(f"SOURCE OPEN: ok status={fetched.get('status')} url={url[:120]}")
        else:
            log(
                f"SOURCE OPEN: fail url={url[:120]} "
                f"err={(fetched.get('error') or '')[:80]}"
            )
        ext = extract_from_fetch(fetched, goal=goal)
        ext["title"] = str(hit.get("title") or "")
        ext["snippet"] = str(hit.get("snippet") or "")  # kept for audit, not knowledge
        ext["provider"] = hit.get("provider") or hit.get("source") or ""
        ext["query"] = hit.get("query") or ""
        ext["rank_score"] = hit.get("rank_score")
        if ext.get("ok"):
            log(
                f"SOURCE EXTRACT: ok chars={ext.get('char_count')} "
                f"rel={ext.get('relatedness', 0):.2f} title={(ext.get('title') or '')[:60]}"
            )
        else:
            log(
                f"SOURCE EXTRACT: thin/fail chars={ext.get('char_count')} "
                f"url={url[:100]}"
            )
        return ext

    if not ranked:
        return [], compare_extracts([], goal=goal)

    with ThreadPoolExecutor(max_workers=min(4, len(ranked))) as pool:
        futs = [pool.submit(_one, h) for h in ranked]
        for fut in as_completed(futs):
            try:
                extracts.append(fut.result())
            except Exception as exc:
                logger.warning("open_and_extract worker failed: %s", exc)

    # Stable order by relatedness / rank
    extracts.sort(
        key=lambda e: (-(e.get("relatedness") or 0), -(e.get("char_count") or 0))
    )
    comparison = compare_extracts(extracts, goal=goal)
    return extracts, comparison


def knowledge_from_extracts(
    extracts: list[dict[str, Any]],
    comparison: dict[str, Any],
    *,
    goal: str,
) -> str:
    """
    Build SYNTHESIZE input from EXTRACTED full content — never from snippets alone.
    """
    parts: list[str] = []
    consensus = comparison.get("consensus") or []
    if consensus:
        parts.append("### Cross-source consensus")
        for c in consensus[:8]:
            parts.append(f"- {c}")
    for e in extracts:
        if not e.get("ok"):
            continue
        title = e.get("title") or e.get("url") or "source"
        parts.append(f"### Source: {title}")
        parts.append(f"URL: {e.get('url')}")
        parts.append(str(e.get("text") or "")[:3500])
    blob = "\n\n".join(parts).strip()
    if not blob:
        return ""
    # Guard: reject if somehow only snippet-length
    if len(blob) < MIN_EXTRACT_CHARS:
        return ""
    return blob[:14000]
