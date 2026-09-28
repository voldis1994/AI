"""
Semantic UNDERSTAND helpers for TaskContract.

The model (or a universal offline literal/structure pass) proposes contract
fields. Deterministic code only grounds and validates — it does not invent
task-specific verb vocabularies or hardcode particular goals.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Optional


# Language-agnostic function words — NOT task verbs. Used only to drop
# fillers when selecting distinctive content tokens as behavior needles.
_FILLER = frozenset({
    "a", "an", "the", "and", "or", "to", "for", "with", "from", "into",
    "in", "on", "at", "of", "by", "is", "are", "be", "as", "it", "this",
    "that", "please", "jarvis", "me", "us", "them", "him", "her", "my",
    "your", "our", "their", "i", "you", "we", "they", "do", "does", "did",
    "will", "would", "can", "could", "should", "must", "may", "might",
    "not", "no", "yes", "ok", "just", "only", "also", "very", "really",
    "now", "here", "there", "then", "than", "so", "if", "when", "while",
    "about", "after", "before", "over", "under", "again", "once", "all",
    "any", "some", "such", "own", "same", "other", "into", "out", "up",
    "down", "off", "via", "per", "each", "every", "both", "few", "more",
    "most", "other", "into", "using", "use", "used", "based", "need",
    "needs", "wanted", "want", "make", "made", "get", "got", "set",
    # Common LV fillers (not action hardcode — stopwords only)
    "un", "vai", "ar", "no", "uz", "par", "ka", "kā", "lai", "man",
    "lūdzu", "ludzu", "šis", "šo", "tas", "to", "tie", "tās", "es",
    "tu", "viņš", "viņa", "mēs", "jūs",
})

_URL_RE = re.compile(r"https?://\S+", re.I)
_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_\"'])([A-Za-z0-9_./\\-]+\.[A-Za-z0-9]{1,12})\b"
)
_NUM_RE = re.compile(
    r"(?<![\w.])(-?\d+(?:\.\d+)?)(?![\w.])"
)
_LIST_RE = re.compile(r"\[[^\[\]]{1,200}\]")
_ARITH_RE = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*([\+\-\*/×÷])\s*(\d+(?:\.\d+)?)(?![\w.])"
)
_INVENTED_RE = re.compile(
    r"(?i)\b(?:user[_-]?provided(?:[_-]?\w+)?|"
    r"default(?:[_-]?(?:path|file|name|content|value|dir|output))?|"
    r"placeholder|changeme|your[_-]?name|todo|tbd|xxx+|dummy|"
    r"sample(?:[_-]?(?:path|file))|"
    r"example(?:[_-]?(?:path|file))|"
    r"temp[_-]?file|untitled)\b"
)


def _strip_url_punct(u: str) -> str:
    return (u or "").rstrip(".,);]\"'")


def extract_grounded_literals(request: str) -> dict[str, list[str]]:
    """
    Universal literal harvest from the request text.

    Categories are structural (url / path / quote / number / sequence),
    never task-type labels. Every value is a substring of the request.
    """
    req = request or ""
    urls: list[str] = []
    for m in _URL_RE.finditer(req):
        u = _strip_url_punct(m.group(0))
        if u and u not in urls and not _INVENTED_RE.search(u):
            urls.append(u)

    quotes: list[str] = []
    path_like_quotes: list[str] = []
    for a, b in re.findall(r"\"([^\"]+)\"|'([^']+)'", req):
        tok = (a or b).strip()
        if not tok or _INVENTED_RE.search(tok):
            continue
        # Quoted path/filename is an artifact, not a content payload
        if re.search(r"\.[A-Za-z0-9]{1,12}$", tok) and "://" not in tok and " " not in tok:
            if tok not in path_like_quotes:
                path_like_quotes.append(tok)
            continue
        if tok not in quotes:
            quotes.append(tok)

    paths: list[str] = []
    url_blob = " ".join(urls).lower()
    for tok in path_like_quotes:
        if tok not in paths:
            paths.append(tok)
    for m in _PATH_RE.finditer(req):
        tok = m.group(1)
        if _INVENTED_RE.search(tok):
            continue
        if "://" in tok or tok.lower() in url_blob:
            continue
        if any(tok.lower() in u.lower() for u in urls):
            continue
        if tok not in paths:
            paths.append(tok)

    numbers: list[str] = []
    for m in _NUM_RE.finditer(req):
        n = m.group(1)
        if n not in numbers:
            numbers.append(n)

    sequences: list[str] = []
    for m in _LIST_RE.finditer(req):
        lit = m.group(0)
        if lit not in sequences:
            sequences.append(lit)

    return {
        "urls": urls,
        "quotes": quotes,
        "paths": paths,
        "numbers": numbers,
        "sequences": sequences,
    }


def _distinctive_tokens(request: str, exclude: set[str]) -> list[str]:
    """Content-bearing tokens in the request, minus fillers and excluded literals."""
    out: list[str] = []
    for tok in re.findall(r"[A-Za-z0-9_]{2,}", request or ""):
        low = tok.lower()
        if low in _FILLER or low in exclude:
            continue
        if _INVENTED_RE.search(tok):
            continue
        if tok not in out:
            out.append(tok)
    return out


def _arith_result(request: str) -> Optional[str]:
    m = _ARITH_RE.search(request or "")
    if not m:
        return None
    a_s, op, b_s = m.group(1), m.group(2), m.group(3)
    try:
        a = float(a_s) if "." in a_s else int(a_s)
        b = float(b_s) if "." in b_s else int(b_s)
        if op == "+":
            val: Any = a + b
        elif op == "-":
            val = a - b
        elif op in ("*", "×"):
            val = a * b
        elif op in ("/", "÷"):
            if b == 0:
                return None
            val = a / b
        else:
            return None
        if isinstance(val, float) and val.is_integer():
            val = int(val)
        return str(val)
    except Exception:
        return None


def _reverse_or_sort(request: str) -> Optional[str]:
    """If a sequence literal appears with reverse/sort cue word in request."""
    req = request or ""
    low = req.lower()
    for lit in _LIST_RE.findall(req):
        try:
            val = ast.literal_eval(lit)
        except Exception:
            continue
        if not isinstance(val, list):
            continue
        # Cue must be present as a surface word in the request (any language
        # spelling the user used) — we only check that some token near the
        # list signals transform; accept common international stems.
        if re.search(r"(?i)\b(reverse|reversed|otpa[kc]y|apgriez)\b", low):
            return repr(list(reversed(val)))
        if re.search(r"(?i)\b(sort|sorted|k[aā]rtot)\b", low):
            try:
                return repr(sorted(val))
            except Exception:
                return None
    return None


def offline_semantic_draft(
    request: str,
    args: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Universal offline UNDERSTAND draft when no model is available.

    Builds outcomes/criteria from grounded literal *forms* (URL, path, quote,
    number, sequence) plus optional schema args. Does not consult task-verb
    tables. Unknown tasks without grounded literals yield empty criteria
    (VALIDATE → needs_refine).
    """
    from jarvis.request_items import ItemKind, classify_request_items

    req = (request or "").strip()
    lit = extract_grounded_literals(req)
    args = dict(args or {})

    # Merge grounded schema args into literal pools by form
    for _k, val in args.items():
        sv = str(val).strip()
        if not sv or _INVENTED_RE.search(sv):
            continue
        if sv.startswith(("http://", "https://")):
            if sv not in lit["urls"]:
                lit["urls"].append(sv)
        elif re.search(r"\.[A-Za-z0-9]{1,12}$", sv) and "://" not in sv:
            if sv not in lit["paths"]:
                lit["paths"].append(sv)
        elif sv not in lit["quotes"] and (len(sv) >= 2):
            # treat as content payload candidate
            if sv not in lit["quotes"]:
                lit["quotes"].append(sv)

    # Universal request-item classification (FILE / CONTENT / …) — not verbs
    items = classify_request_items(req)
    url_blob = " ".join(lit["urls"]).lower()
    for it in items:
        if it.kind in (ItemKind.FILE, ItemKind.PATH):
            tok = it.text.strip()
            if not tok or _INVENTED_RE.search(tok):
                continue
            # Never promote URL host/path fragments to workspace artifacts
            if tok.startswith("//") or "://" in tok:
                continue
            if tok.lower() in url_blob or any(
                tok.lower() in u.lower() for u in lit["urls"]
            ):
                continue
            if tok not in lit["paths"]:
                lit["paths"].append(tok)
        elif it.kind == ItemKind.CONTENT and it.reason == "quoted content":
            if it.text not in lit["quotes"]:
                lit["quotes"].append(it.text)

    # Unquoted CONTENT payloads associated with files (e.g. …containing E2E_OK).
    # Only payload-like tokens — never every prose word in the sentence.
    unquoted_content: list[str] = []
    path_blob = " ".join(lit["paths"]).lower()
    for p in lit["paths"]:
        path_blob += " " + Path_stem(p)
    prose_candidates: list[str] = []
    for it in items:
        if it.kind != ItemKind.CONTENT:
            continue
        if it.reason == "quoted content":
            continue
        tok = it.text.strip()
        if not tok or _INVENTED_RE.search(tok):
            continue
        if tok.lower() in path_blob or tok.lower() in _FILLER:
            continue
        if not re.fullmatch(r"[A-Za-z0-9_]{2,64}", tok):
            continue
        if tok in lit["quotes"]:
            continue
        # ALL_CAPS / snake_payload markers are clear file contents
        if re.fullmatch(r"[A-Z][A-Z0-9_]{1,63}", tok) or "_" in tok:
            if tok not in unquoted_content:
                unquoted_content.append(tok)
        else:
            prose_candidates.append(tok)
    # Single remaining unquoted identifier after a path → content (e.g. containing ok)
    if (
        not unquoted_content
        and lit["paths"]
        and not lit["quotes"]
        and len(prose_candidates) == 1
    ):
        unquoted_content.append(prose_candidates[0])

    files: list[dict[str, Any]] = []
    http: list[dict[str, Any]] = []
    must_contain: list[str] = []
    behaviors: list[dict[str, Any]] = []
    acceptance: list[str] = []
    plan: list[dict[str, Any]] = []
    outcomes: list[str] = []
    outputs: list[str] = []
    side_effects: list[str] = []
    artifacts: list[str] = []

    primary_content = lit["quotes"][0] if lit["quotes"] else (
        unquoted_content[0] if unquoted_content else None
    )
    # Only use unquoted CONTENT as file payload when a path exists —
    # pathless prose tokens stay empty offline (model UNDERSTAND covers them).
    if primary_content in unquoted_content and not lit["paths"]:
        primary_content = None

    for i, p in enumerate(lit["paths"]):
        entry: dict[str, Any] = {"path": p, "min_bytes": 1}
        if i == 0 and primary_content is not None:
            entry["contains"] = primary_content
        files.append(entry)
        artifacts.append(p)
        outcomes.append(f"artifact:{p}")
        outputs.append(p)
        acceptance.append(f"artifact_exists:{p}@workspace_stat")
        plan.append({"kind": "artifact_exists", "target": p, "how": "workspace_stat"})
        if i == 0 and primary_content is not None:
            acceptance.append(f"content_present:{primary_content}@workspace_read")
            plan.append(
                {
                    "kind": "content_present",
                    "target": primary_content,
                    "how": "workspace_read",
                }
            )
            outcomes.append(f"content:{primary_content}")
        side_effects.append("filesystem_write")

    used_primary = bool(files and primary_content is not None)
    quote_start = 1 if used_primary and lit["quotes"] and primary_content == lit["quotes"][0] else 0
    for extra in lit["quotes"][quote_start:]:
        if files:
            if extra not in must_contain and extra != primary_content:
                must_contain.append(extra)
                acceptance.append(f"content_present:{extra}@workspace_read")
                plan.append(
                    {
                        "kind": "content_present",
                        "target": extra,
                        "how": "workspace_read",
                    }
                )
        else:
            behaviors.append(
                {
                    "kind": "skill_output_contains",
                    "target": extra,
                    "how": "skill_stdout_or_result",
                }
            )
            acceptance.append(
                f"skill_output_contains:{extra}@skill_stdout_or_result"
            )
            plan.append(
                {
                    "kind": "skill_output_contains",
                    "target": extra,
                    "how": "skill_stdout_or_result",
                }
            )
            outcomes.append(f"behavior:skill_output_contains:{extra}")
            outputs.append(f"result:skill_output_contains:{extra}")

    # Extra unquoted payloads on multi-content file requests
    if files and primary_content is not None:
        for extra in unquoted_content:
            if extra == primary_content:
                continue
            if extra not in must_contain:
                must_contain.append(extra)
                acceptance.append(f"content_present:{extra}@workspace_read")
                plan.append(
                    {
                        "kind": "content_present",
                        "target": extra,
                        "how": "workspace_read",
                    }
                )

    # Pathless primary quote → behavior (output), not workspace scan
    if lit["quotes"] and not files:
        primary_content = lit["quotes"][0]
        if not any(b.get("target") == primary_content for b in behaviors):
            behaviors.append(
                {
                    "kind": "skill_output_contains",
                    "target": primary_content,
                    "how": "skill_stdout_or_result",
                }
            )
            acceptance.append(
                f"skill_output_contains:{primary_content}@skill_stdout_or_result"
            )
            plan.append(
                {
                    "kind": "skill_output_contains",
                    "target": primary_content,
                    "how": "skill_stdout_or_result",
                }
            )
            outcomes.append(f"behavior:skill_output_contains:{primary_content}")
            outputs.append(f"result:skill_output_contains:{primary_content}")

    for u in lit["urls"]:
        http.append({"url": u})
        acceptance.append(f"http_ok:{u}@http_get")
        plan.append({"kind": "http_ok", "target": u, "how": "http_get"})
        outcomes.append(f"http:{u}")
        side_effects.append("network_http")

    # Arithmetic / sequence transforms — only when operands/literals are in request
    arith = _arith_result(req)
    if arith is not None:
        acceptance.append(f"skill_result_equals:{arith}@skill_result")
        plan.append(
            {"kind": "skill_result_equals", "target": arith, "how": "skill_result"}
        )
        behaviors.append(
            {"kind": "skill_result_equals", "target": arith, "how": "skill_result"}
        )
        outcomes.append(f"behavior:skill_result_equals:{arith}")
        outputs.append(f"result:skill_result_equals:{arith}")

    transformed = _reverse_or_sort(req)
    if transformed is not None:
        acceptance.append(f"skill_result_equals:{transformed}@skill_result")
        plan.append(
            {
                "kind": "skill_result_equals",
                "target": transformed,
                "how": "skill_result",
            }
        )
        behaviors.append(
            {
                "kind": "skill_result_equals",
                "target": transformed,
                "how": "skill_result",
            }
        )
        outcomes.append(f"behavior:skill_result_equals:{transformed}")

    # Single grounded numeric literal with no artifact/http/quote context →
    # treat as expected equality result (model UNDERSTAND covers richer cases).
    # Skip when the digit is a final-output count ("2 sentences", "3 teikumos")
    # — that belongs to final_output constraints, not skill_result_equals.
    _count_phrase = bool(
        re.search(
            r"(?i)\b\d+\s*(?:sentences?|teikum[aāos]*|paragraphs?|words?|"
            r"lines?|bullets?)\b",
            req,
        )
    )
    if (
        not acceptance
        and len(lit["numbers"]) == 1
        and not files
        and not http
        and not lit["quotes"]
        and not lit["sequences"]
        and not _count_phrase
    ):
        n = lit["numbers"][0]
        acceptance.append(f"skill_result_equals:{n}@skill_result")
        plan.append(
            {"kind": "skill_result_equals", "target": n, "how": "skill_result"}
        )
        behaviors.append(
            {"kind": "skill_result_equals", "target": n, "how": "skill_result"}
        )
        outcomes.append(f"behavior:skill_result_equals:{n}")
        outputs.append(f"result:skill_result_equals:{n}")

    # Type words literally present in the request (schema vocabulary, not tasks)
    if not acceptance and not files and not http:
        type_map = {
            "integer": "int",
            "int": "int",
            "string": "str",
            "str": "str",
            "list": "list",
            "dict": "dict",
            "bool": "bool",
            "boolean": "bool",
            "float": "float",
        }
        low = req.lower()
        for word, py_t in type_map.items():
            if re.search(rf"\b{re.escape(word)}\b", low):
                acceptance.append(f"skill_result_type:{py_t}@skill_result")
                plan.append(
                    {
                        "kind": "skill_result_type",
                        "target": py_t,
                        "how": "skill_result",
                    }
                )
                behaviors.append(
                    {
                        "kind": "skill_result_type",
                        "target": py_t,
                        "how": "skill_result",
                    }
                )
                outcomes.append(f"behavior:skill_result_type:{py_t}")
                outputs.append(f"result:skill_result_type:{py_t}")
                break

    # Without grounded literals / derived equals, leave criteria empty —
    # VALIDATE marks needs_refine. Model UNDERSTAND covers free-form semantics.

    # Subject: leftover distinctive tokens (no invented sense)
    exclude_subj = {x.lower() for x in (
        lit["paths"] + [Path_stem(p) for p in lit["paths"]]
        + lit["quotes"] + lit["numbers"]
    )}
    for u in lit["urls"]:
        for part in re.findall(r"[A-Za-z0-9_]{3,}", u):
            if part.lower() not in ("http", "https", "www"):
                exclude_subj.add(part.lower())
    subj_toks = _distinctive_tokens(req, exclude_subj)
    if len(subj_toks) >= 2:
        subject = " ".join(subj_toks[:10])
    elif len(subj_toks) == 1:
        subject = "ambiguous"
    else:
        subject = "unknown"

    return {
        "intent": "task",
        "desired_outcomes": outcomes[:16],
        "artifacts": artifacts[:16],
        "behaviors": behaviors[:16],
        "constraints": {
            "files": files,
            "directories": [],
            "imports": [],
            "http": http,
            "must_contain": must_contain,
            "check_process": True,
            "source": "semantic_offline",
            "checks": plan[:24],
        },
        "inputs": dict(args),
        "outputs": outputs[:16],
        "side_effects": list(dict.fromkeys(side_effects))[:16],
        "acceptance_criteria": acceptance[:16],
        "verification_plan": plan[:24],
        "subject": subject,
        "actions": [],
        "content_source": list(lit["urls"])[:8],
        "content_requirements": (
            [primary_content] if primary_content and files else []
        ),
    }


def Path_stem(p: str) -> str:
    from pathlib import Path as _P

    return _P(p).stem.lower()


def ground_semantic_draft(
    draft: dict[str, Any],
    request: str,
) -> dict[str, Any]:
    """
    Deterministic grounding filter — drop invented values; keep request-true ones.

    This is validation, not semantics generation.
    """
    req = request or ""
    out = dict(draft or {})

    def _ok(val: Any) -> bool:
        if val in (None, ""):
            return False
        if isinstance(val, (int, float, bool)):
            return True
        sv = str(val)
        if _INVENTED_RE.search(sv):
            return False
        # Must appear in request, or be a derived numeric/sequence result of
        # literals that themselves appear in the request.
        if sv in req or sv.lower() in req.lower():
            return True
        # Derived equals targets (arith/reverse) — allow if all digits/brackets
        # come from request literals
        if re.fullmatch(r"-?\d+(?:\.\d+)?", sv):
            return any(sv == n or n in req for n in extract_grounded_literals(req)["numbers"]) or (
                _arith_result(req) == sv
            )
        if sv.startswith("[") and sv.endswith("]"):
            return _reverse_or_sort(req) == sv or sv in req
        return False

    def _filter_str_list(items: Any) -> list[str]:
        return [str(x) for x in (items or []) if _ok(x)]

    out["desired_outcomes"] = _filter_str_list(out.get("desired_outcomes"))
    out["artifacts"] = _filter_str_list(out.get("artifacts"))
    out["outputs"] = _filter_str_list(out.get("outputs"))
    out["side_effects"] = [
        str(x) for x in (out.get("side_effects") or [])
        if str(x) in (
            "filesystem_write", "directory_create", "network_http",
            "import_dependency", "env_install",
        ) or _ok(x)
    ]
    out["content_source"] = _filter_str_list(out.get("content_source"))
    out["content_requirements"] = _filter_str_list(out.get("content_requirements"))

    beh_out: list[dict[str, Any]] = []
    for b in out.get("behaviors") or []:
        if not isinstance(b, dict):
            continue
        target = str(b.get("target") or "")
        kind = str(b.get("kind") or "").strip()
        if not kind or not target:
            continue
        if kind.startswith("skill_result_equals") and (
            _arith_result(req) == target or _reverse_or_sort(req) == target or _ok(target)
        ):
            beh_out.append(
                {
                    "kind": kind,
                    "target": target,
                    "how": str(b.get("how") or "skill_result"),
                }
            )
        elif _ok(target):
            beh_out.append(
                {
                    "kind": kind,
                    "target": target,
                    "how": str(b.get("how") or "skill_stdout_or_result"),
                }
            )
    out["behaviors"] = beh_out

    cons = dict(out.get("constraints") or {})
    files = []
    for f in cons.get("files") or []:
        if not isinstance(f, dict):
            continue
        p = str(f.get("path") or "")
        if not _ok(p):
            continue
        entry = {"path": p, "min_bytes": int(f.get("min_bytes") or 1)}
        c = f.get("contains")
        if c is not None and _ok(c):
            entry["contains"] = str(c)
        files.append(entry)
    cons["files"] = files
    cons["http"] = [
        {"url": str((h or {}).get("url"))}
        for h in (cons.get("http") or [])
        if isinstance(h, dict) and _ok((h or {}).get("url"))
    ]
    cons["must_contain"] = _filter_str_list(cons.get("must_contain"))
    cons["directories"] = [
        {"path": str((d or {}).get("path"))}
        for d in (cons.get("directories") or [])
        if isinstance(d, dict) and _ok((d or {}).get("path"))
    ]
    cons["imports"] = _filter_str_list(cons.get("imports"))
    cons.setdefault("check_process", True)
    cons["source"] = cons.get("source") or "semantic_grounded"

    # Rebuild checks / acceptance from grounded pieces if model omitted them
    checks: list[dict[str, Any]] = []
    acceptance: list[str] = []
    for f in files:
        p = f["path"]
        checks.append({"kind": "artifact_exists", "target": p, "how": "workspace_stat"})
        acceptance.append(f"artifact_exists:{p}@workspace_stat")
        if f.get("contains") is not None:
            c = str(f["contains"])
            checks.append({"kind": "content_present", "target": c, "how": "workspace_read"})
            acceptance.append(f"content_present:{c}@workspace_read")
    for m in cons.get("must_contain") or []:
        if files:
            checks.append({"kind": "content_present", "target": str(m), "how": "workspace_read"})
            acceptance.append(f"content_present:{m}@workspace_read")
        else:
            checks.append(
                {
                    "kind": "skill_output_contains",
                    "target": str(m),
                    "how": "skill_stdout_or_result",
                }
            )
            acceptance.append(f"skill_output_contains:{m}@skill_stdout_or_result")
    for h in cons.get("http") or []:
        u = h["url"]
        checks.append({"kind": "http_ok", "target": u, "how": "http_get"})
        acceptance.append(f"http_ok:{u}@http_get")
    for b in beh_out:
        kind = b["kind"]
        target = b["target"]
        how = b.get("how") or ""
        checks.append({"kind": kind, "target": target, "how": how})
        label = f"{kind}:{target}" + (f"@{how}" if how else "")
        if label not in acceptance:
            acceptance.append(label)

    # Prefer model-provided acceptance if every entry grounds; else rebuilt
    model_acc = []
    for raw in out.get("acceptance_criteria") or []:
        s = str(raw).strip()
        if not s or s in ("needs_refine", "skill_ok_and_verified"):
            continue
        body = s.split("@", 1)[0]
        if ":" not in body:
            continue
        _kind, target = body.split(":", 1)
        if _ok(target) or _arith_result(req) == target or _reverse_or_sort(req) == target:
            model_acc.append(s)
    if model_acc:
        out["acceptance_criteria"] = model_acc[:16]
        # Keep / merge plan
        plan = []
        for s in model_acc:
            how = ""
            body = s
            if "@" in s:
                body, how = s.rsplit("@", 1)
            kind, target = body.split(":", 1)
            plan.append({"kind": kind, "target": target, "how": how})
        cons["checks"] = plan[:24]
        out["verification_plan"] = plan[:24]
    else:
        out["acceptance_criteria"] = acceptance[:16]
        cons["checks"] = checks[:24]
        out["verification_plan"] = checks[:24]

    out["constraints"] = cons

    # Inputs: only grounded
    inputs = {}
    for k, v in (out.get("inputs") or {}).items():
        if v in (None, ""):
            continue
        if isinstance(v, (int, float, bool)) or _ok(v):
            inputs[str(k)] = v
    out["inputs"] = inputs

    subj = str(out.get("subject") or "").strip()
    if subj and subj not in ("unknown", "ambiguous"):
        # Keep subject tokens that appear in request
        kept = [
            t for t in re.findall(r"[A-Za-z0-9_]{3,}", subj)
            if t.lower() in req.lower() and t.lower() not in _FILLER
        ]
        out["subject"] = " ".join(kept[:10]) if len(kept) >= 2 else (
            "ambiguous" if len(kept) == 1 else "unknown"
        )
    else:
        out.setdefault("subject", "unknown")

    out["actions"] = [
        str(a).lower() for a in (out.get("actions") or [])
        if str(a).strip() and str(a).lower() in req.lower()
    ][:8]
    out["intent"] = str(out.get("intent") or "task").strip() or "task"
    return out
