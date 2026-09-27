"""
JARVIS Brain — multi-model Ollama interface via concurrent ModelPool.

Work kind → direct FAST / REASONING / CODING worker (independent resources).
Never invents "DONE" — verification is external (Python checks, not LLM trust).
"""

from __future__ import annotations

import hashlib
import json
import re
import logging
import time
from typing import Any, Callable, Optional

from jarvis.model_config import DEFAULT_HOST, DEFAULT_MODEL, KEEP_ALIVE_ACTIVE
from jarvis.model_router import ModelPool, ModelRouter, RequestWorkContext
from jarvis.perf import get_active_tracker

logger = logging.getLogger("jarvis.brain")

# Re-export for callers that import from jarvis.brain
__all__ = ["Brain", "DEFAULT_MODEL", "DEFAULT_HOST", "ModelPool", "ModelRouter"]

# Shorter than historical 180s — hung Ollama must not dominate repair loops.
DEFAULT_CHAT_TIMEOUT = 90.0

_CONN_FAIL_MARKERS = (
    "connection refused",
    "failed to connect",
    "connect call failed",
    "connection reset",
    "name or service not known",
    "nodename nor servname",
    "network is unreachable",
    "timed out",
    "timeout",
    "ollama unreachable",
)


def _is_connection_failure(exc: Any) -> bool:
    if isinstance(
        exc,
        (
            ConnectionRefusedError,
            ConnectionResetError,
            BrokenPipeError,
            TimeoutError,
            ConnectionError,
        ),
    ):
        return True
    msg = str(exc).lower()
    # urllib wraps OSError / URLError — match message markers
    return any(m in msg for m in _CONN_FAIL_MARKERS)


class Brain:
    """Ollama API wrapper with universal Multi-Model routing."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: float = DEFAULT_CHAT_TIMEOUT,
        on_log: Optional[Callable[[str], None]] = None,
        on_event: Optional[Callable[[str, dict], None]] = None,
        router: Optional[ModelRouter] = None,
    ) -> None:
        # Legacy single-model hint / forced override (tests may pin a model)
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._client = None
        self._force_single_model = False
        self.on_log = on_log or (lambda _m: None)
        # Same pipeline event bus as Orchestrator (code_delta, etc.)
        self.on_event = on_event or (lambda _k, _p: None)
        self.router = router or ModelRouter(
            list_models=self._list_model_names,
            on_log=self._route_log,
            host=self.host,
        )
        # Keep router logger in sync if Brain logger changes later
        self.router.set_logger(self._route_log)
        self.pool: ModelPool = self.router  # alias — router IS the pool
        self._pool_ready = False

    def set_logger(self, on_log: Callable[[str], None]) -> None:
        self.on_log = on_log or (lambda _m: None)
        self.router.set_logger(self._route_log)

    def set_event_callback(
        self, on_event: Optional[Callable[[str, dict], None]]
    ) -> None:
        self.on_event = on_event or (lambda _k, _p: None)

    def _emit_code_delta(
        self,
        code: str,
        *,
        stage: str = "delta",
        name: str = "",
    ) -> None:
        """Fire-and-forget live code stream for CODE tab (never blocks)."""
        try:
            self.on_event(
                "code_delta",
                {
                    "code": code,
                    "stage": stage,
                    "name": name,
                },
            )
        except Exception:
            pass

    def ensure_pool_ready(self, *, warm: bool = True) -> dict[str, Any]:
        """
        Initialize Ollama connection and warm models that fit GPU/RAM.
        Called at Orchestrator boot — tiers become independent ready workers.
        """
        if self._force_single_model:
            self._pool_ready = True
            return {"online": False, "forced_single": True}
        report = self.pool.initialize(warm=warm)
        self._pool_ready = True
        return report

    def begin_request_pool(self, request_id: str = "") -> RequestWorkContext:
        return self.pool.begin_request(request_id)

    def end_request_pool(self) -> Optional[dict[str, Any]]:
        return self.pool.end_request()

    def _route_log(self, msg: str) -> None:
        try:
            self.on_log(msg)
        except Exception:
            pass
        if msg.startswith("MODEL "):
            logger.info(msg)

    def _get_client(self):
        if self._client is None:
            try:
                import ollama

                self._client = ollama.Client(host=self.host, timeout=self.timeout)
            except Exception as exc:
                logger.warning("ollama client init failed: %s", exc)
                self._client = None
        return self._client

    def is_available(self) -> bool:
        """True when Ollama is up AND at least one configured model is installed."""
        return self.model_status() == "ONLINE"

    def model_status(self) -> str:
        """
        Return:
          ONLINE        — server reachable and ≥1 configured model present
          MODEL MISSING — server reachable but none of the catalog installed
          OFFLINE       — server unreachable
        """
        if self._force_single_model:
            tags = self._list_model_names()
            if tags is None:
                return "OFFLINE"
            if self.router.model_in_tags(self.model, tags):
                return "ONLINE"
            return "MODEL MISSING"
        return self.router.status()

    def model_catalog_summary(self) -> str:
        """Short multi-model pool summary for GUI/CLI (ready/warm marks)."""
        detail = self.router.status_detail()
        parts = []
        for tier, info in (detail.get("tiers") or {}).items():
            resolved = info.get("resolved") or info.get("primary")
            if info.get("warm"):
                mark = "♨"  # warm in memory
            elif info.get("online") or info.get("ready"):
                mark = "✓"
            else:
                mark = "·"
            parts.append(f"{tier}:{resolved}{mark}")
        return " | ".join(parts)

    def _list_model_names(self) -> Optional[list[str]]:
        """Return model name list, or None if server unreachable."""
        try:
            client = self._get_client()
            if client is not None:
                data = client.list()
                models = data.get("models") if isinstance(data, dict) else getattr(data, "models", None)
                names: list[str] = []
                for m in models or []:
                    if isinstance(m, dict):
                        names.append(str(m.get("name") or m.get("model") or ""))
                    else:
                        names.append(str(getattr(m, "name", None) or getattr(m, "model", "") or ""))
                return [n for n in names if n]
        except Exception as exc:
            logger.info("ollama client list failed: %s", exc)

        # HTTP fallback
        try:
            import urllib.request

            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=5) as resp:
                if resp.status != 200:
                    return None
                payload = json.loads(resp.read().decode("utf-8"))
            names = []
            for m in payload.get("models") or []:
                names.append(str(m.get("name") or m.get("model") or ""))
            return [n for n in names if n]
        except Exception:
            return None

    def chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        temperature: float = 0.3,
        *,
        work: str = "",
        tier: Optional[str] = None,
        allow_escalate: bool = False,
        expect_json: bool = False,
        expect_code: bool = False,
        escalate_error_only: bool = False,
        model: Optional[str] = None,
        on_stream: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Chat via concurrent model pool.

        work/tier selects the worker directly (FAST|REASONING|CODING) — never
        walks a mandatory pipeline. Escalation runs only when the prior result
        is truly insufficient. Same work is not re-executed on another model
        within the request context unless escalating after insufficiency.

        on_stream: optional callback receiving accumulated text as Ollama
        tokens arrive (used for live CODE tab; not a separate model).
        """
        full: list[dict[str, str]] = []
        if system:
            full.append({"role": "system", "content": system})
        full.extend(messages)

        if self._force_single_model and not model:
            model = self.model

        work_key_name = work or ("forced" if model else "chat")
        prompt_fp = self._prompt_fingerprint(full)
        ctx = self.pool.current_request()
        dedupe_key = RequestWorkContext.work_key(
            work_key_name, prompt_fp, tier or ""
        )
        if ctx is not None:
            cached = ctx.get(dedupe_key)
            if cached is not None and not ModelRouter.result_insufficient(
                cached,
                expect_json=expect_json,
                expect_code=expect_code,
                error_only=escalate_error_only,
            ):
                self.on_log(
                    f"MODEL DEDUPE: reuse {work_key_name} "
                    f"(skip re-run on another model)"
                )
                if on_stream is not None:
                    try:
                        on_stream(str(cached))
                    except Exception:
                        pass
                return cached

        decision = self.pool.acquire(
            work=work_key_name,
            tier=tier,
            force_model=model,
            ensure_warm=not self._force_single_model,
        )
        t0 = time.perf_counter()
        self.pool.mark_in_flight(decision.model, +1)
        try:
            if on_stream is not None:
                text = self._chat_on_model_stream(
                    full,
                    temperature,
                    decision.model,
                    on_stream=on_stream,
                    keep_alive=self.pool.keep_alive_for(decision.model),
                )
            else:
                text = self._chat_on_model(
                    full,
                    temperature,
                    decision.model,
                    keep_alive=self.pool.keep_alive_for(decision.model),
                )
        finally:
            self.pool.mark_in_flight(decision.model, -1)
        self._record_model_timing(
            work=work or decision.work,
            tier=decision.tier,
            model=decision.model,
            ms=(time.perf_counter() - t0) * 1000.0,
            ok=not str(text).startswith("[BRAIN ERROR]"),
            warm=decision.warm,
        )
        if ctx is not None:
            ctx.put(dedupe_key, text, decision)

        if allow_escalate and ModelRouter.result_insufficient(
            text,
            expect_json=expect_json,
            expect_code=expect_code,
            error_only=escalate_error_only,
        ):
            # Connection failures: do not burn another 30B / HTTP timeout
            if text.startswith("[BRAIN ERROR]") and _is_connection_failure(text):
                try:
                    self.router.invalidate_cache()
                except Exception:
                    pass
                return text
            # Clear insufficient cache so escalate may run once
            if ctx is not None:
                with ctx._lock:
                    ctx.results.pop(dedupe_key, None)
            esc = self.pool.escalate(
                decision.tier,
                work=work or decision.work,
                prior_key=dedupe_key,
            )
            if esc is not None and (
                esc.model != decision.model or esc.tier != decision.tier
            ):
                t1 = time.perf_counter()
                self.pool.mark_in_flight(esc.model, +1)
                try:
                    text2 = self._chat_on_model(
                        full,
                        temperature,
                        esc.model,
                        keep_alive=self.pool.keep_alive_for(esc.model),
                    )
                finally:
                    self.pool.mark_in_flight(esc.model, -1)
                self._record_model_timing(
                    work=work or decision.work,
                    tier=esc.tier,
                    model=esc.model,
                    ms=(time.perf_counter() - t1) * 1000.0,
                    ok=not str(text2).startswith("[BRAIN ERROR]"),
                    escalated=True,
                )
                if ctx is not None:
                    ctx.put(dedupe_key, text2, esc)
                if not ModelRouter.result_insufficient(
                    text2,
                    expect_json=expect_json,
                    expect_code=expect_code,
                    error_only=escalate_error_only,
                ):
                    return text2
                if text2.strip() and (
                    not text.strip() or text.startswith("[BRAIN ERROR]")
                ):
                    return text2
            elif esc is not None and esc.tier == decision.tier:
                # CODING same-tier retry with failure context
                t1 = time.perf_counter()
                self.pool.mark_in_flight(esc.model, +1)
                try:
                    text2 = self._chat_on_model(
                        full,
                        temperature,
                        esc.model,
                        keep_alive=self.pool.keep_alive_for(esc.model),
                    )
                finally:
                    self.pool.mark_in_flight(esc.model, -1)
                self._record_model_timing(
                    work=work or decision.work,
                    tier=esc.tier,
                    model=esc.model,
                    ms=(time.perf_counter() - t1) * 1000.0,
                    ok=not str(text2).startswith("[BRAIN ERROR]"),
                    retry=True,
                )
                if text2.strip() and not text2.startswith("[BRAIN ERROR]"):
                    if ctx is not None:
                        ctx.put(dedupe_key, text2, esc)
                    return text2
        return text

    @staticmethod
    def _prompt_fingerprint(messages: list[dict[str, str]]) -> str:
        blob = json.dumps(messages, ensure_ascii=False, default=str)[:2000]
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def parallel_generate(
        self,
        jobs: list[dict[str, Any]],
    ) -> list[str]:
        """
        Run independent generate jobs concurrently on the pool.

        Each job: {prompt, system?, temperature?, work?, tier?, ...}
        Dependent pipeline stages must NOT use this.
        """
        def _run(job: dict[str, Any]) -> str:
            return self.generate(
                str(job.get("prompt") or ""),
                system=job.get("system"),
                temperature=float(job.get("temperature") or 0.3),
                work=str(job.get("work") or ""),
                tier=job.get("tier"),
                allow_escalate=bool(job.get("allow_escalate", False)),
                expect_json=bool(job.get("expect_json", False)),
                expect_code=bool(job.get("expect_code", False)),
                escalate_error_only=bool(job.get("escalate_error_only", False)),
                model=job.get("model"),
            )

        return list(self.pool.run_parallel(jobs, _run))

    @staticmethod
    def _record_model_timing(
        *,
        work: str,
        tier: str,
        model: str,
        ms: float,
        ok: bool = True,
        **meta: Any,
    ) -> None:
        tracker = get_active_tracker()
        if tracker is None:
            return
        try:
            tracker.record_model(
                work=work, tier=tier, model=model, ms=ms, ok=ok, **meta
            )
        except Exception:
            pass

    def _chat_on_model(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        model: str,
        *,
        keep_alive: Any = None,
    ) -> str:
        ka = KEEP_ALIVE_ACTIVE if keep_alive is None else keep_alive
        client = self._get_client()
        if client is not None:
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "options": {"temperature": temperature},
                }
                # ollama-python accepts keep_alive on chat
                try:
                    resp = client.chat(**kwargs, keep_alive=ka)
                except TypeError:
                    resp = client.chat(**kwargs)
                content = resp.get("message", {}).get("content", "")
                return (content or "").strip()
            except Exception as exc:
                logger.error("ollama chat failed (%s): %s", model, exc)
                # Connection / timeout: fail fast — do NOT wait another full HTTP timeout
                if _is_connection_failure(exc):
                    try:
                        self.router.invalidate_cache()
                    except Exception:
                        pass
                    return f"[BRAIN ERROR] Ollama unreachable: {exc}"
                return self._http_chat(
                    messages, temperature, model=model, keep_alive=ka
                )

        return self._http_chat(messages, temperature, model=model, keep_alive=ka)

    def _chat_on_model_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        model: str,
        *,
        on_stream: Callable[[str], None],
        keep_alive: Any = None,
    ) -> str:
        """Stream tokens from Ollama; invoke on_stream(accumulated) as they arrive."""
        ka = KEEP_ALIVE_ACTIVE if keep_alive is None else keep_alive
        client = self._get_client()
        if client is not None:
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "options": {"temperature": temperature},
                    "stream": True,
                }
                try:
                    stream = client.chat(**kwargs, keep_alive=ka)
                except TypeError:
                    stream = client.chat(**kwargs)
                accumulated = ""
                for chunk in stream:
                    piece = ""
                    if isinstance(chunk, dict):
                        piece = str(
                            (chunk.get("message") or {}).get("content") or ""
                        )
                    else:
                        try:
                            piece = str(
                                (getattr(chunk, "message", None) or {}).get(
                                    "content"
                                )
                                or getattr(
                                    getattr(chunk, "message", None),
                                    "content",
                                    "",
                                )
                                or ""
                            )
                        except Exception:
                            piece = ""
                    if piece:
                        accumulated += piece
                        try:
                            on_stream(accumulated)
                        except Exception:
                            pass
                return accumulated.strip()
            except Exception as exc:
                logger.error("ollama stream chat failed (%s): %s", model, exc)
                if _is_connection_failure(exc):
                    try:
                        self.router.invalidate_cache()
                    except Exception:
                        pass
                    return f"[BRAIN ERROR] Ollama unreachable: {exc}"
                return self._http_chat_stream(
                    messages,
                    temperature,
                    model=model,
                    on_stream=on_stream,
                    keep_alive=ka,
                )
        return self._http_chat_stream(
            messages,
            temperature,
            model=model,
            on_stream=on_stream,
            keep_alive=ka,
        )

    def _http_chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        model: Optional[str] = None,
        *,
        on_stream: Callable[[str], None],
        keep_alive: Any = None,
    ) -> str:
        use_model = model or self.model
        ka = KEEP_ALIVE_ACTIVE if keep_alive is None else keep_alive
        try:
            import urllib.request

            payload = json.dumps(
                {
                    "model": use_model,
                    "messages": messages,
                    "stream": True,
                    "keep_alive": ka,
                    "options": {"temperature": temperature},
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                f"{self.host}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            accumulated = ""
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                while True:
                    line = resp.readline()
                    if not line:
                        break
                    line = line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue
                    piece = str(
                        (data.get("message") or {}).get("content") or ""
                    )
                    if piece:
                        accumulated += piece
                        try:
                            on_stream(accumulated)
                        except Exception:
                            pass
                    if data.get("done"):
                        break
            return accumulated.strip()
        except Exception as exc:
            logger.error("HTTP stream chat failed (%s): %s", use_model, exc)
            if _is_connection_failure(exc):
                try:
                    self.router.invalidate_cache()
                except Exception:
                    pass
                return f"[BRAIN ERROR] Ollama unreachable: {exc}"
            # Fall back to non-streaming once
            text = self._http_chat(
                messages, temperature, model=use_model, keep_alive=ka
            )
            if text and on_stream is not None:
                try:
                    on_stream(text)
                except Exception:
                    pass
            return text

    def _http_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        model: Optional[str] = None,
        *,
        keep_alive: Any = None,
    ) -> str:
        use_model = model or self.model
        ka = KEEP_ALIVE_ACTIVE if keep_alive is None else keep_alive
        try:
            import urllib.request

            payload = json.dumps(
                {
                    "model": use_model,
                    "messages": messages,
                    "stream": False,
                    "keep_alive": ka,
                    "options": {"temperature": temperature},
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                f"{self.host}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return (data.get("message", {}) or {}).get("content", "").strip()
        except Exception as exc:
            logger.error("HTTP chat failed (%s): %s", use_model, exc)
            if _is_connection_failure(exc):
                try:
                    self.router.invalidate_cache()
                except Exception:
                    pass
            return f"[BRAIN ERROR] Ollama unreachable: {exc}"

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        *,
        work: str = "",
        tier: Optional[str] = None,
        allow_escalate: bool = False,
        expect_json: bool = False,
        expect_code: bool = False,
        escalate_error_only: bool = False,
        model: Optional[str] = None,
        on_stream: Optional[Callable[[str], None]] = None,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.chat(
            messages,
            system=system,
            temperature=temperature,
            work=work,
            tier=tier,
            allow_escalate=allow_escalate,
            expect_json=expect_json,
            expect_code=expect_code,
            escalate_error_only=escalate_error_only,
            model=model,
            on_stream=on_stream,
        )

    # ── Structured helpers ──────────────────────────────────────────────

    def converse(self, user_text: str, history: Optional[list[dict[str, str]]] = None) -> str:
        """Casual conversation path: answer THIS user message only."""
        system = (
            "Tu esi JARVIS — autonomais AI asistents. Atbildi skaidri un noderīgi. "
            "Runā latviešu valodā, ja lietotājs raksta latviski, citādi angļu. "
            "CRITICAL: Answer ONLY the latest user message. "
            "Do NOT repeat, paste, or reuse any previous learning/task DONE result, "
            "skill output, verifier summary, or earlier final_result. "
            "Neizliecies, ka esi izpildījis darbību, ja tu to neesi izpildījis."
        )
        # Keep history short — conversation turns only (caller should pre-filter)
        messages = list(history or [])[-8:]
        messages.append({"role": "user", "content": user_text})
        return self.chat(
            messages,
            system=system,
            temperature=0.5,
            work="converse",
            # Soft chat: stay on FAST; escalate only hard errors (not weak prose)
            allow_escalate=True,
            escalate_error_only=True,
        )

    def classify_intent(self, user_text: str) -> dict[str, Any]:
        """
        Classify THIS user request independently (never inherit prior skill).

        conversation — greetings / Q&A / chat
        learning     — research + verify + save knowledge (no skill build)
        task         — external action that may use/build/repair a skill
        """
        from jarvis.intent import IntentClassifier

        system = (
            "Classify the user message independently. Reply ONLY with JSON:\n"
            '{"intent":"conversation"|"learning"|"task","goal":"<short goal>",'
            '"needs_capability":true|false,"needs_research":true|false,'
            '"keywords":["..."]}\n'
            "Rules:\n"
            "- conversation: greetings, opinions, simple Q&A with no research/action.\n"
            "- learning: user wants to learn/study/research/understand a topic, "
            "build knowledge, quizzes/tests of knowledge. needs_research=true, "
            "needs_capability=false. Do NOT treat this as a skill/capability task.\n"
            "- task: user wants an external action "
            "(create/write a file, fetch URL, convert, scrape, install, automate, etc.). "
            "needs_capability=true.\n"
            "- Classify only the CURRENT message. Never reuse or inherit a previous "
            "skill/capability name.\n"
            "- Pedagogical 'create tests/quiz to verify knowledge' is learning, "
            "not a file/skill task, unless a concrete file/path/URL artifact is requested."
        )
        raw = self.generate(
            user_text,
            system=system,
            temperature=0.1,
            work="intent",
            # Offline IntentClassifier covers weak JSON — do not burn 30B
            allow_escalate=False,
            expect_json=True,
        )
        parsed = self._parse_json(raw, IntentClassifier.classify_offline(user_text))
        return IntentClassifier.normalize(parsed, user_text)

    def plan(self, goal: str, known_capabilities: list[str]) -> dict[str, Any]:
        """Create an execution plan and decide if research/learning is needed."""
        caps = "\n".join(f"- {c}" for c in known_capabilities) or "(none yet)"
        system = (
            "You are JARVIS planner for an ACTION task. Reply ONLY with JSON:\n"
            '{"steps":["..."],"can_reuse":["skill_name"],'
            '"missing":["what we cannot do"],"needs_research":true|false,'
            '"needs_new_skill":true|false,"skill_name":"snake_case_name",'
            '"skill_description":"...","research_queries":["..."],'
            '"args":{"any_key":"value extracted from the user goal"},'
            '"required_args":["arg_names_the_skill_will_need"]}\n'
            "Rules:\n"
            "- Only put a skill in can_reuse if it clearly performs THIS goal's action. "
            "Never reuse an unrelated prior skill just because it is ACTIVE.\n"
            "- If no existing skill fits, needs_new_skill=true and skill_name MUST be a "
            "new snake_case name derived from THIS goal (not an old unrelated skill).\n"
            "- Extract structured args from the user goal into args — do not leave them "
            "empty when the goal clearly contains parameters "
            "(filenames, text, urls, numbers, etc.). "
            "Arg names should match what a Python skill would read from context['args'].\n"
            "- NEVER invent one arg key per word in the request. Use a small schema "
            "(typically path/content/url-style keys), not sentence tokens."
        )
        prompt = f"GOAL:\n{goal}\n\nKNOWN CAPABILITIES:\n{caps}"
        raw = self.generate(
            prompt,
            system=system,
            temperature=0.2,
            work="plan",
            allow_escalate=False,
            expect_json=True,
        )
        return self._parse_json(raw, {
            "steps": [f"Handle: {goal}"],
            "can_reuse": [],
            "missing": [goal],
            "needs_research": True,
            "needs_new_skill": True,
            "skill_name": self._slug(goal)[:40] or "new_skill",
            "skill_description": goal,
            "research_queries": [goal],
            "args": {},
            "required_args": [],
        })

    def extract_task_args(
        self,
        goal: str,
        skill_meta: Optional[dict] = None,
        prior_args: Optional[dict] = None,
        diagnosis: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Universally convert a user goal into structured args for context['args'].

        No fixed schema — keys come from the goal / skill meta / diagnosis hints.
        """
        system = (
            "Extract structured arguments for a JARVIS skill from the user goal. "
            "Reply ONLY with JSON object of argument key→value pairs "
            '(optionally wrap as {"args":{...}}).\n'
            "Rules:\n"
            "- Prefer keys from skill_meta.required_args / diagnosis missing_args.\n"
            "- Do NOT invent a new key for every word in the sentence.\n"
            "- Do NOT use request tokens (individual words) as argument names.\n"
            "- Do NOT invent values not implied by the goal.\n"
            "- If diagnosis lists missing/required args, prioritize filling those keys "
            "from the goal text.\n"
            "- Values may be strings, numbers, bools, or lists.\n"
            "- If nothing extractable, return {}."
        )
        payload = {
            "goal": goal,
            "skill_meta": skill_meta or {},
            "prior_args": prior_args or {},
            "diagnosis_hints": {
                "missing_args": (diagnosis or {}).get("missing_args"),
                "required_args": (diagnosis or {}).get("required_args"),
                "suggested_args": (diagnosis or {}).get("suggested_args"),
                "root_cause": (diagnosis or {}).get("root_cause"),
            },
        }
        raw = self.generate(
            json.dumps(payload, ensure_ascii=False, default=str)[:8000],
            system=system,
            temperature=0.1,
            work="extract_args",
            # ContextBuilder has deterministic arg grounding — avoid 30B escalate
            allow_escalate=False,
            expect_json=True,
        )
        parsed = self._parse_json(raw, {})
        if isinstance(parsed.get("args"), dict):
            return parsed["args"]
        # If model returned a flat dict of args
        banned = {
            "diagnosis", "root_cause", "approach", "needs_research",
            "test_plan", "fault_layer", "rewrite_skill",
        }
        return {k: v for k, v in parsed.items() if k not in banned and not str(k).startswith("_")}

    def research_notes(self, query: str, gathered: str) -> dict[str, Any]:
        """Summarize research into actionable learning notes."""
        system = (
            "Summarize research for building or REPAIRING a Python skill. "
            "Reply ONLY with JSON:\n"
            '{"approach":"NEW strategy label distinct from prior failures",'
            '"libraries":["pip-package"],'
            '"key_apis":["..."],"pitfalls":["..."],"test_idea":"...",'
            '"repair_insight":"what specifically to change in the next repair"}'
        )
        prompt = f"QUERY: {query}\n\nGATHERED INFO:\n{gathered[:8000]}"
        raw = self.generate(
            prompt,
            system=system,
            temperature=0.2,
            work="research",
            expect_json=True,
        )
        return self._parse_json(raw, {
            "approach": gathered[:500],
            "libraries": [],
            "key_apis": [],
            "pitfalls": [],
            "test_idea": "run skill and check return value",
            "repair_insight": "",
        })

    def generate_research_queries(
        self,
        observation: dict[str, Any],
        diagnosis: Optional[dict[str, Any]] = None,
        failed_approaches: Optional[list] = None,
    ) -> list[str]:
        """
        Generate adaptive research questions from a repair-cycle failure.

        Universal — no task-specific hardcoding. Uses observation, diagnosis,
        traceback/error, failed approaches, and verifier failure.
        """
        system = (
            "Generate focused web/research queries to unblock a failing JARVIS skill repair. "
            "Reply ONLY with JSON: {\"research_queries\":[\"...\"]}\n"
            "Rules:\n"
            "- 2-5 concrete queries grounded in the failure data.\n"
            "- Cover root cause, APIs/patterns to try, and how to satisfy verifier constraints.\n"
            "- Do NOT invent a specific user filename/task; stay general to the failure class.\n"
            "- Prefer Python stdlib / common library angles."
        )
        payload = {
            "goal": observation.get("goal"),
            "phase": observation.get("phase"),
            "exception": str(observation.get("exception") or "")[:800],
            "traceback": str(observation.get("traceback") or "")[:1200],
            "stderr": str(observation.get("stderr") or "")[:600],
            "verifier_result": observation.get("verifier_result"),
            "diagnosis": {
                k: (diagnosis or {}).get(k)
                for k in (
                    "root_cause", "fault_layer", "what_to_change", "approach",
                    "test_plan", "missing_args", "expected_artifacts",
                )
            },
            "failed_approaches": [
                {
                    "approach": a.get("approach"),
                    "last_error": str(a.get("last_error") or "")[:200],
                }
                for a in (failed_approaches or [])[:8]
            ],
            "context_args_keys": list(
                ((observation.get("context") or {}).get("args") or {}).keys()
            ),
        }
        raw = self.generate(
            json.dumps(payload, ensure_ascii=False, default=str)[:10000],
            system=system,
            temperature=0.3,
            work="query_generation",
            allow_escalate=True,
            escalate_error_only=True,
            expect_json=True,
        )
        parsed = self._parse_json(raw, {})
        queries = parsed.get("research_queries")
        if isinstance(queries, list) and queries:
            return [str(q).strip() for q in queries if str(q).strip()][:5]
        return self._offline_research_queries(observation, diagnosis, failed_approaches)

    @staticmethod
    def _offline_research_queries(
        observation: dict[str, Any],
        diagnosis: Optional[dict[str, Any]] = None,
        failed_approaches: Optional[list] = None,
    ) -> list[str]:
        """Deterministic query synthesis when the model is unavailable."""
        diagnosis = diagnosis or {}
        goal = str(observation.get("goal") or "")[:160]
        err = str(observation.get("exception") or "")[:200]
        root = str(diagnosis.get("root_cause") or err)[:200]
        tb = str(observation.get("traceback") or "")
        tb_line = ""
        for line in tb.splitlines():
            s = line.strip()
            if s and not s.startswith("File ") and "Traceback" not in s:
                tb_line = s[:160]
                break
        vr = observation.get("verifier_result") or {}
        vr_reason = ""
        if isinstance(vr, dict):
            vr_reason = str(vr.get("reason") or "")[:200]
        queries = [
            f"python repair: {root}",
            f"python {goal} error: {err}",
        ]
        if tb_line:
            queries.append(f"python traceback: {tb_line}")
        if vr_reason:
            queries.append(f"skill verification failure: {vr_reason}")
        for a in (failed_approaches or [])[:3]:
            ap = str(a.get("approach") or "").strip()
            if ap:
                queries.append(f"python alternative after failed approach {ap[:80]}")
        layer = str(diagnosis.get("fault_layer") or "")
        if layer:
            queries.append(f"python fix {layer} for: {root[:120]}")
        # Dedupe preserving order
        out: list[str] = []
        for q in queries:
            q = " ".join(q.split())
            if q and q not in out:
                out.append(q)
        return out[:5]

    def write_skill_code(
        self,
        skill_name: str,
        description: str,
        research: dict[str, Any],
        previous_code: Optional[str] = None,
        error_log: Optional[str] = None,
        diagnosis: Optional[dict[str, Any]] = None,
        failed_approaches: Optional[list] = None,
        test_plan: Optional[str] = None,
        *,
        user_request: Optional[str] = None,
        task_goal: Optional[dict[str, Any]] = None,
        grounded_args: Optional[dict[str, Any]] = None,
        artifacts: Optional[list] = None,
        missing_requirements: Optional[list] = None,
        constraints: Optional[dict[str, Any]] = None,
    ) -> str:
        """Generate or repair a complete Python skill module from diagnosis/research."""
        system = (
            "You write JARVIS skill modules. Output ONLY valid Python code, no markdown fences.\n"
            "Every skill MUST define:\n"
            "  SKILL_META = {'name': str, 'description': str, 'capabilities': [str], "
            "'dependencies': [str], 'version': int, 'required_args': [str]}\n"
            "  def run(context: dict) -> dict:\n"
            "      '''Execute the skill. context has 'goal', 'args', 'workspace'. "
            "Return {'ok': bool, 'result': Any, 'error': str|None, 'evidence': str}. "
            "Put independently checkable fields in result "
            "(path/file/directory/url/imports/expect_status/contains).'''\n"
            "CRITICAL argument rules:\n"
            "- Read parameters ONLY from context['args'] (and goal/workspace as needed).\n"
            "- NEVER invent/guess missing argument values.\n"
            "- NEVER use placeholder/default paths or default content when the goal/args "
            "specify real values — implement exactly what the user requested.\n"
            "- If required args are missing from context['args'], return ok=False with "
            "error listing the missing keys — do not fabricate them.\n"
            "Use only stdlib + declared dependencies. Be concrete and correct. "
            "Do not pretend success — set ok=False on failure.\n"
            "FORBIDDEN: do not leave a TODO/stub body and do not return "
            "'Skill body not implemented yet'. You MUST implement a real run() from the "
            "USER REQUEST / TaskGoal / RESEARCH / DIAGNOSIS provided.\n"
            "Honor the immutable USER REQUEST and TaskGoal constraints exactly. "
            "If a DIAGNOSIS is provided and fault_layer is skill_code, implement the fix. "
            "Do not repeat failed approaches listed below."
        )
        parts = [
            f"Skill name: {skill_name}",
            f"Description: {description}",
        ]
        if user_request:
            parts.append(f"USER REQUEST (immutable TaskGoal source):\n{user_request}")
        if task_goal:
            parts.append(
                "TASK GOAL (structured):\n"
                + json.dumps(task_goal, ensure_ascii=False, default=str)[:4000]
            )
        elif constraints:
            parts.append(
                "CONSTRAINTS / SUCCESS CRITERIA:\n"
                + json.dumps(constraints, ensure_ascii=False, default=str)[:3000]
            )
        if grounded_args is not None:
            parts.append(
                "GROUNDED ARGS (skill contract — read these from context['args']):\n"
                + json.dumps(grounded_args, ensure_ascii=False, default=str)[:2000]
            )
        if artifacts:
            parts.append(
                "WORKSPACE ARTIFACTS (what already exists):\n"
                + json.dumps(artifacts, ensure_ascii=False, default=str)[:2000]
            )
        if missing_requirements:
            parts.append(
                "MISSING REQUIREMENTS (must satisfy next):\n"
                + json.dumps(missing_requirements, ensure_ascii=False, default=str)[:2000]
            )
        parts.append(f"Research: {json.dumps(research, ensure_ascii=False)[:6000]}")
        if research.get("knowledge_history"):
            parts.append(
                "SAVED KNOWLEDGE HISTORY (reuse these insights):\n"
                + json.dumps(research["knowledge_history"], ensure_ascii=False, default=str)[:4000]
            )
        if diagnosis:
            parts.append(
                "DIAGNOSIS (follow this):\n"
                + json.dumps(diagnosis, ensure_ascii=False, default=str)[:4000]
            )
        if failed_approaches:
            parts.append(
                "FAILED APPROACHES — do NOT repeat these:\n"
                + json.dumps(failed_approaches, ensure_ascii=False, default=str)[:3000]
            )
        if test_plan:
            parts.append(f"TEST PLAN for next version:\n{test_plan}")
        if previous_code:
            parts.append(f"PREVIOUS / EXISTING CODE:\n{previous_code[:8000]}")
        if error_log:
            parts.append(
                "VERIFIER / ERROR SUMMARY (fix this):\n" + error_log[:4000]
            )
        # Slightly higher temperature on repair to encourage approach change
        is_repair = bool(diagnosis or error_log or previous_code)
        temperature = 0.35 if is_repair else 0.15
        work = "code_repair" if is_repair else "skill_code"

        def _stream_to_ui(accumulated: str) -> None:
            # Show exact model output as it arrives (same CODE editor).
            # Prefer extracted python when fences appear; otherwise raw text.
            shown = self._extract_python(accumulated) or accumulated
            self._emit_code_delta(shown, stage="delta", name=skill_name)

        self._emit_code_delta("", stage="start", name=skill_name)
        raw = self.generate(
            "\n\n".join(parts),
            system=system,
            temperature=temperature,
            work=work,
            expect_code=True,
            on_stream=_stream_to_ui,
        )
        code = self._extract_python(raw)
        # Failed CODING → retry CODING with failure context (same tier, not REASONING)
        needs_retry = (
            ModelRouter.result_insufficient(raw, expect_code=True)
            or ModelRouter.result_insufficient(code, expect_code=True)
            or ("def run" not in (code or "") and "SKILL_META" not in (code or ""))
        )
        if needs_retry:
            esc = self.router.escalate("CODING", work="code_repair")
            if esc is not None:
                retry_parts = list(parts) + [
                    "PREVIOUS MODEL OUTPUT WAS INSUFFICIENT — regenerate complete "
                    "valid Python skill with SKILL_META and def run(context)."
                ]
                if error_log:
                    retry_parts.append(f"FAILURE CONTEXT:\n{str(error_log)[:4000]}")
                self._emit_code_delta("", stage="start", name=skill_name)
                raw2 = self.generate(
                    "\n\n".join(retry_parts),
                    system=system,
                    temperature=0.4,
                    work="code_repair",
                    model=esc.model,
                    expect_code=True,
                    on_stream=_stream_to_ui,
                )
                code2 = self._extract_python(raw2)
                if code2 and ("def run" in code2 or "SKILL_META" in code2):
                    self._emit_code_delta(code2, stage="done", name=skill_name)
                    return code2
        if code:
            self._emit_code_delta(code, stage="done", name=skill_name)
        return code

    def diagnose(
        self,
        observation: dict[str, Any],
        failed_approaches: Optional[list] = None,
        prior_solutions: Optional[list] = None,
    ) -> dict[str, Any]:
        """
        Universal diagnosis from a full observation package.

        Ollama decides root cause, what to change, research/deps needs,
        next test plan, and a NEW approach if previous ones failed.
        """
        system = (
            "You are JARVIS diagnostician for a self-learning agent. "
            "You receive a full OBSERVATION of a failed skill attempt. "
            "Reply ONLY with JSON:\n"
            "{\n"
            '  "root_cause": "...",\n'
            '  "fault_layer": "goal_parsing"|"context_mapping"|"skill_code"|"execution"|"environment"|"verifier",\n'
            '  "rewrite_skill": true|false,\n'
            '  "what_to_change": "...",\n'
            '  "approach": "short label of the NEW strategy to try",\n'
            '  "approach_changed": true|false,\n'
            '  "needs_research": true|false,\n'
            '  "missing_knowledge": ["only if a real knowledge gap blocks repair"],\n'
            '  "research_queries": ["..."],\n'
            '  "needs_new_deps": ["pip-or-import-name"],\n'
            '  "missing_args": ["arg_names"],\n'
            '  "required_args": ["arg_names"],\n'
            '  "suggested_args": {"arg": "value from USER REQUEST if present"},\n'
            '  "test_plan": "how to validate the next version",\n'
            '  "expected_artifacts": ["what verifier should find"],\n'
            '  "is_unfixable": false,\n'
            '  "diagnosis": "one-paragraph summary"\n'
            "}\n"
            "Rules:\n"
            "- The original USER REQUEST is an immutable TaskGoal for the whole cycle. "
            "Never invent values that are not in the request.\n"
            "- Use exception, traceback, stdout, stderr, returncode, code, "
            "context (especially context.args + context.user_request), dependencies, artifacts.\n"
            "- goal_parsing: cannot derive constraints from the USER REQUEST.\n"
            "- context_mapping: context.args empty/missing keys, OR args contain invented "
            "defaults / values not grounded in the USER REQUEST. rewrite_skill=false. "
            "Fill suggested_args ONLY from values present in the request.\n"
            "- skill_code: skill logic wrong; used defaults; created artifacts that do not "
            "match TaskGoal. rewrite_skill=true ONLY for this layer.\n"
            "- execution: subprocess/runner/harness failed to pass context or crashed.\n"
            "- environment: import/install/dependency failures.\n"
            "- verifier: verifier harness itself is wrong (not when skill output fails "
            "user constraints).\n"
            "- Legacy aliases context_args→context_mapping, test_harness→execution, "
            "dependency→environment are accepted but prefer canonical names.\n"
            "- If the same approach already failed, set approach_changed=true "
            "and propose a meaningfully different approach.\n"
            "- needs_research=true ONLY when missing_knowledge is non-empty "
            "(a real knowledge gap). Do NOT set needs_research for ordinary "
            "skill_code bugs, VERIFY mismatches, or repeated identical failures — "
            "those go to CODING rewrite. Research is never a universal fallback.\n"
            "- Do not claim the task is done; only diagnose.\n"
            "- Prefer concrete, testable next steps."
        )
        payload = {
            "observation": {
                k: observation.get(k)
                for k in (
                    "goal", "skill_name", "version", "phase",
                    "exception", "traceback", "stdout", "stderr",
                    "returncode", "timed_out", "crash", "killed",
                    "context", "dependencies", "artifacts",
                    "skill_result", "verifier_result", "code_fingerprint",
                )
            },
            "skill_code": (observation.get("skill_code") or "")[:8000],
            "failed_approaches": failed_approaches or observation.get("prior_approaches") or [],
            "prior_solutions": prior_solutions or [],
        }
        raw = self.generate(
            json.dumps(payload, ensure_ascii=False, default=str)[:14000],
            system=system,
            temperature=0.25,
            work="diagnose",
            expect_json=True,
        )
        from jarvis.task_goal import TaskGoal
        from jarvis.context_builder import ContextBuilder

        ctx = observation.get("context") or {}
        ctx_args = ctx.get("args") if isinstance(ctx, dict) else {}
        if not isinstance(ctx_args, dict):
            ctx_args = {}
        request = str(
            (ctx.get("user_request") if isinstance(ctx, dict) else None)
            or observation.get("user_request")
            or observation.get("goal")
            or ""
        )
        empty_args = len(ctx_args) == 0
        invented_args = any(
            TaskGoal.is_invented_default(v, request)
            or (
                isinstance(v, str)
                and v
                and not TaskGoal.is_grounded(v, request)
            )
            for v in ctx_args.values()
        )
        err = str(observation.get("exception") or "").lower()
        phase = str(observation.get("phase") or "").upper()
        args_fault = (empty_args or invented_args) and any(
            tok in err for tok in (
                "argument", "args", "missing", "required", "keyerror",
                "default", "untrusted", "claim_aligns", "user_provided",
            )
        )
        # VERIFY mismatch / defaults with grounded args → rewrite skill
        goal_mismatch = phase == "VERIFY" and any(
            tok in err
            for tok in (
                "default", "placeholder", "untrusted", "claim_aligns",
                "not in user", "user constraints", "user request",
                "contains(", "missing from expected", "reject_defaults",
            )
        ) and not invented_args and not empty_args
        no_constraints = phase == "VERIFY" and "no user-derived constraints" in err
        env_fault = any(
            tok in err for tok in ("modulenotfound", "no module named", "importerror", "pip ")
        )
        exec_fault = phase in ("TEST", "RETEST", "EXECUTE", "INSTALL_DEPS") and any(
            tok in err for tok in ("timed_out", "timeout", "crash", "returncode")
        )
        if no_constraints and empty_args:
            fb_layer = "goal_parsing"
        elif args_fault and not goal_mismatch:
            fb_layer = "context_mapping"
        elif env_fault:
            fb_layer = "environment"
        elif exec_fault and not goal_mismatch:
            fb_layer = "execution"
        elif goal_mismatch:
            fb_layer = "skill_code"
        else:
            fb_layer = "skill_code"
        fb_layer = TaskGoal.normalize_fault_layer(fb_layer)
        fallback = {
            "root_cause": str(observation.get("exception") or "unknown")[:500],
            "fault_layer": fb_layer,
            "rewrite_skill": TaskGoal.rewrite_skill_for_layer(fb_layer),
            "what_to_change": (
                "Map TaskGoal → skill args from USER REQUEST (no invented defaults)"
                if fb_layer == "context_mapping"
                else (
                    "Re-parse immutable TaskGoal / USER REQUEST into constraints"
                    if fb_layer == "goal_parsing"
                    else (
                        "Rewrite skill to satisfy original TaskGoal "
                        "(no default/placeholder artifacts)"
                        if goal_mismatch
                        else (
                            "Fix environment / install missing dependencies"
                            if fb_layer == "environment"
                            else (
                                "Fix execution/harness context passing"
                                if fb_layer == "execution"
                                else "Revise skill logic based on stderr/traceback"
                            )
                        )
                    )
                )
            ),
            "approach": (
                "context_mapping"
                if fb_layer == "context_mapping"
                else (
                    "goal_parsing_repair"
                    if fb_layer == "goal_parsing"
                    else (
                        "honor_user_request"
                        if goal_mismatch
                        else f"alt_approach_v{(observation.get('version') or 0) + 1}"
                    )
                )
            ),
            "approach_changed": True,
            # Research only for genuine knowledge gaps — not every skill_code fail
            "needs_research": False,
            "missing_knowledge": [],
            "research_queries": [],
            "needs_new_deps": [],
            "missing_args": [],
            "required_args": [],
            "suggested_args": {},
            "test_plan": (
                "Retest with grounded args from TaskGoal; VERIFY must match "
                "original USER REQUEST (reject defaults / skill self-proof)"
            ),
            "expected_artifacts": [],
            "is_unfixable": False,
            "diagnosis": str(observation.get("exception") or "failure")[:500],
        }
        result = self._parse_json(raw, fallback)
        # Normalize fault_layer (canonical + legacy aliases)
        layer = TaskGoal.normalize_fault_layer(
            result.get("fault_layer") or fallback["fault_layer"]
        )
        # Force skill rewrite when VERIFY proves goal mismatch / defaults
        # with already-grounded args (not a context mapping problem).
        if goal_mismatch:
            layer = "skill_code"
        result["fault_layer"] = layer
        result["rewrite_skill"] = TaskGoal.rewrite_skill_for_layer(layer)
        if not isinstance(result.get("suggested_args"), dict):
            result["suggested_args"] = {}
        if not isinstance(result.get("missing_args"), list):
            result["missing_args"] = list(result.get("required_args") or [])
        if not isinstance(result.get("missing_knowledge"), list):
            result["missing_knowledge"] = []
        # Research requires an explicit knowledge gap — never universal
        if result.get("needs_research") and not result.get("missing_knowledge"):
            result["needs_research"] = False

        # Enrich missing/suggested args from error + original request
        result = ContextBuilder.enrich_diagnosis_args(
            result,
            observation,
            str(observation.get("goal") or ""),
            user_request=request,
        )
        layer = TaskGoal.normalize_fault_layer(
            result.get("fault_layer") or layer
        )
        if goal_mismatch and layer not in ("context_mapping", "goal_parsing"):
            layer = "skill_code"
        result["fault_layer"] = layer
        result["rewrite_skill"] = TaskGoal.rewrite_skill_for_layer(layer)

        # Enforce approach change when fingerprint collided with failed list
        failed_fps = {
            str(a.get("approach_fingerprint") or a.get("fingerprint") or "")
            for a in (failed_approaches or [])
        }
        approach = str(result.get("approach") or fallback["approach"])
        from jarvis.observer import Observer

        fp = Observer.fingerprint_approach(approach)
        result["approach"] = approach
        result["approach_fingerprint"] = fp
        if fp in failed_fps and failed_fps and layer == "skill_code":
            result["approach_changed"] = True
            result["approach"] = f"{approach} | divergent-{observation.get('version')}"
            result["approach_fingerprint"] = Observer.fingerprint_approach(result["approach"])
            result["what_to_change"] = (
                str(result.get("what_to_change") or "")
                + " | prior approach fingerprint collided — force new strategy"
            )
        if "diagnosis" not in result:
            result["diagnosis"] = result.get("root_cause") or fallback["diagnosis"]
        return result

    def analyze_failure(self, skill_code: str, error_log: str, evidence: str) -> dict[str, Any]:
        """Backward-compatible thin wrapper around diagnose()."""
        observation = {
            "goal": "",
            "skill_name": "",
            "version": 0,
            "phase": "TEST",
            "exception": error_log,
            "traceback": error_log,
            "stdout": "",
            "stderr": evidence or "",
            "returncode": None,
            "skill_code": skill_code,
            "context": {},
            "dependencies": [],
            "artifacts": [],
            "skill_result": {"error": error_log, "evidence": evidence},
        }
        return self.diagnose(observation)

    def verify_claim(self, goal: str, result: dict[str, Any], evidence: str) -> dict[str, Any]:
        """Ask the model to judge verification — but orchestrator still requires evidence."""
        system = (
            "Judge whether the goal was ACTUALLY achieved based on evidence. "
            "Reply ONLY with JSON: "
            '{"achieved":true|false,"confidence":0.0-1.0,"reason":"..."}\n'
            "If evidence is missing or vague, achieved must be false."
        )
        prompt = (
            f"GOAL: {goal}\nRESULT: {json.dumps(result, ensure_ascii=False, default=str)}\n"
            f"EVIDENCE:\n{evidence[:4000]}"
        )
        raw = self.generate(
            prompt,
            system=system,
            temperature=0.1,
            work="verify_claim",
            expect_json=True,
        )
        return self._parse_json(raw, {
            "achieved": False,
            "confidence": 0.0,
            "reason": "Could not parse verification response",
        })

    def judge_learning_coverage(
        self,
        user_request: str,
        knowledge: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Semantically judge whether acquired knowledge answers the USER REQUEST.

        Do NOT require exact keyword/token matches — judge meaning and goal coverage.
        Advisory only — orchestrator still applies deterministic Python checks.
        """
        system = (
            "You verify LEARNING results for JARVIS. "
            "Semantically compare the USER REQUEST to the acquired knowledge. "
            "Reply ONLY with JSON:\n"
            "{\n"
            '  "covers_goal": true|false,\n'
            '  "confidence": 0.0-1.0,\n'
            '  "reason": "why it does or does not answer the user goal",\n'
            '  "missing_aspects": ["semantic gaps, not keywords"],\n'
            '  "requires_practical_result": true|false,\n'
            '  "practical_ok": true|false|null,\n'
            '  "practical_feedback": "..."\n'
            "}\n"
            "Rules:\n"
            "- covers_goal=true only if the knowledge would let the user achieve "
            "their learning goal (concepts, explanation, practice as asked).\n"
            "- Do NOT require exact keyword overlap; paraphrases and synonyms count.\n"
            "- If the user also asked for a concrete answer / worked result / "
            "computation / demonstration, set requires_practical_result=true and "
            "judge practical_result inside knowledge.\n"
            "- If practical result is required but missing or wrong, covers_goal=false.\n"
            "- No topic hardcoding; stay grounded in THIS request + knowledge."
        )
        prompt = (
            f"USER REQUEST:\n{user_request}\n\n"
            f"KNOWLEDGE:\n{json.dumps(knowledge, ensure_ascii=False, default=str)[:9000]}"
        )
        raw = self.generate(
            prompt,
            system=system,
            temperature=0.1,
            work="semantic",
            expect_json=True,
        )
        fallback = {
            "covers_goal": False,
            "confidence": 0.0,
            "reason": "Could not parse learning judgment",
            "missing_aspects": ["goal_coverage"],
            "requires_practical_result": False,
            "practical_ok": None,
            "practical_feedback": "",
        }
        result = self._parse_json(raw, fallback)
        result["covers_goal"] = bool(result.get("covers_goal"))
        try:
            result["confidence"] = float(result.get("confidence") or 0.0)
        except Exception:
            result["confidence"] = 0.0
        if not isinstance(result.get("missing_aspects"), list):
            result["missing_aspects"] = []
        return result

    def produce_learning_answer(
        self,
        user_request: str,
        knowledge: dict[str, Any],
    ) -> dict[str, Any]:
        """Produce a concrete answer / practical result from acquired knowledge."""
        system = (
            "Using the acquired learning knowledge, produce the concrete answer "
            "or practical result the USER REQUEST asks for. "
            "Reply ONLY with JSON:\n"
            '{"answer":"...","result":<string|number|object|null>,'
            '"explanation":"brief why this satisfies the request"}\n'
            "Rules:\n"
            "- If the request includes a computable expression or concrete question, "
            "put the final value in result and explain briefly.\n"
            "- Stay grounded in the knowledge + request. Do not invent unrelated topics.\n"
            "- No skill code or file creation."
        )
        prompt = (
            f"USER REQUEST:\n{user_request}\n\n"
            f"KNOWLEDGE:\n{json.dumps(knowledge, ensure_ascii=False, default=str)[:8000]}"
        )
        raw = self.generate(
            prompt,
            system=system,
            temperature=0.15,
            work="learning",
            expect_json=True,
        )
        parsed = self._parse_json(raw, {
            "answer": "",
            "result": None,
            "explanation": "",
        })
        if not isinstance(parsed, dict):
            return {"answer": "", "result": None, "explanation": "", "source": "brain"}
        parsed["source"] = "brain"
        return parsed

    # ── Utilities ───────────────────────────────────────────────────────

    @staticmethod
    def _slug(text: str) -> str:
        s = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
        return s or "skill"

    @staticmethod
    def _extract_python(text: str) -> str:
        text = text.strip()
        fence = re.search(r"```(?:python)?\s*([\s\S]*?)```", text)
        if fence:
            return fence.group(1).strip()
        # Drop leading prose if model ignored instructions
        if "SKILL_META" in text:
            idx = text.find("SKILL_META")
            # include imports before SKILL_META
            head = text[:idx]
            last_import = max(head.rfind("\nimport "), head.rfind("\nfrom "), head.rfind("import "))
            if last_import >= 0:
                start = last_import if head[last_import] != "\n" else last_import + 1
                return text[start:].strip()
            return text[idx:].strip()
        return text

    @staticmethod
    def _parse_json(text: str, fallback: dict[str, Any]) -> dict[str, Any]:
        text = text.strip()
        # Strip fences
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fence:
            text = fence.group(1).strip()
        # Find first { ... }
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return dict(fallback)
        try:
            data = json.loads(match.group(0))
            if not isinstance(data, dict):
                return dict(fallback)
            out = dict(fallback)
            out.update(data)
            return out
        except json.JSONDecodeError:
            # try to fix trailing commas
            cleaned = re.sub(r",\s*}", "}", match.group(0))
            cleaned = re.sub(r",\s*]", "]", cleaned)
            try:
                data = json.loads(cleaned)
                out = dict(fallback)
                out.update(data)
                return out
            except Exception:
                return dict(fallback)
