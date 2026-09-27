"""
JARVIS Brain — Ollama interface (qwen2.5-coder:7b).

Provides structured prompting for planning, coding, research, repair,
and casual conversation. Never invents "DONE" — verification is external.
"""

from __future__ import annotations

import json
import re
import logging
from typing import Any, Optional

logger = logging.getLogger("jarvis.brain")

DEFAULT_MODEL = "qwen2.5-coder:7b"
DEFAULT_HOST = "http://127.0.0.1:11434"


class Brain:
    """Thin wrapper around the Ollama HTTP/Python API."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: float = 180.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._client = None

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
        """Return True if Ollama is reachable and the model can be listed."""
        try:
            client = self._get_client()
            if client is None:
                return self._http_ping()
            client.list()
            return True
        except Exception:
            return self._http_ping()

    def _http_ping(self) -> bool:
        try:
            import urllib.request

            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        temperature: float = 0.3,
    ) -> str:
        """Send a chat completion request. Returns assistant text or an error string."""
        full: list[dict[str, str]] = []
        if system:
            full.append({"role": "system", "content": system})
        full.extend(messages)

        client = self._get_client()
        if client is not None:
            try:
                resp = client.chat(
                    model=self.model,
                    messages=full,
                    options={"temperature": temperature},
                )
                content = resp.get("message", {}).get("content", "")
                return (content or "").strip()
            except Exception as exc:
                logger.error("ollama chat failed: %s", exc)
                return self._http_chat(full, temperature)

        return self._http_chat(full, temperature)

    def _http_chat(self, messages: list[dict[str, str]], temperature: float) -> str:
        try:
            import urllib.request

            payload = json.dumps(
                {
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
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
            logger.error("HTTP chat failed: %s", exc)
            return f"[BRAIN ERROR] Ollama unreachable: {exc}"

    def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.3) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, system=system, temperature=temperature)

    # ── Structured helpers ──────────────────────────────────────────────

    def converse(self, user_text: str, history: Optional[list[dict[str, str]]] = None) -> str:
        """Casual conversation path: USER → Ollama → real reply."""
        system = (
            "Tu esi JARVIS — autonomais AI asistents. Atbildi skaidri un noderīgi. "
            "Runā latviešu valodā, ja lietotājs raksta latviski, citādi angļu. "
            "Neizliecies, ka esi izpildījis darbību, ja tu to neesi izpildījis."
        )
        messages = list(history or [])
        messages.append({"role": "user", "content": user_text})
        return self.chat(messages, system=system, temperature=0.5)

    def classify_intent(self, user_text: str) -> dict[str, Any]:
        """Decide if the request is conversation or a task to execute/learn."""
        system = (
            "Classify the user message. Reply ONLY with JSON:\n"
            '{"intent":"conversation"|"task","goal":"<short goal>",'
            '"needs_capability":true|false,"keywords":["..."]}\n'
            "intent=conversation for greetings, questions, chat.\n"
            "intent=task when the user wants an action done "
            "(create file, fetch data, convert, scrape, compute, automate, etc.)."
        )
        raw = self.generate(user_text, system=system, temperature=0.1)
        return self._parse_json(raw, {
            "intent": "conversation",
            "goal": user_text,
            "needs_capability": False,
            "keywords": [],
        })

    def plan(self, goal: str, known_capabilities: list[str]) -> dict[str, Any]:
        """Create an execution plan and decide if research/learning is needed."""
        caps = "\n".join(f"- {c}" for c in known_capabilities) or "(none yet)"
        system = (
            "You are JARVIS planner. Reply ONLY with JSON:\n"
            '{"steps":["..."],"can_reuse":["skill_name"],'
            '"missing":["what we cannot do"],"needs_research":true|false,'
            '"needs_new_skill":true|false,"skill_name":"snake_case_name",'
            '"skill_description":"...","research_queries":["..."]}\n'
            "Prefer reusing existing capabilities. Only request a new skill if none fit."
        )
        prompt = f"GOAL:\n{goal}\n\nKNOWN CAPABILITIES:\n{caps}"
        raw = self.generate(prompt, system=system, temperature=0.2)
        return self._parse_json(raw, {
            "steps": [f"Handle: {goal}"],
            "can_reuse": [],
            "missing": [goal],
            "needs_research": True,
            "needs_new_skill": True,
            "skill_name": self._slug(goal)[:40] or "new_skill",
            "skill_description": goal,
            "research_queries": [goal],
        })

    def research_notes(self, query: str, gathered: str) -> dict[str, Any]:
        """Summarize research into actionable learning notes."""
        system = (
            "Summarize research for building a Python skill. Reply ONLY with JSON:\n"
            '{"approach":"...","libraries":["pip-package"],'
            '"key_apis":["..."],"pitfalls":["..."],"test_idea":"..."}'
        )
        prompt = f"QUERY: {query}\n\nGATHERED INFO:\n{gathered[:8000]}"
        raw = self.generate(prompt, system=system, temperature=0.2)
        return self._parse_json(raw, {
            "approach": gathered[:500],
            "libraries": [],
            "key_apis": [],
            "pitfalls": [],
            "test_idea": "run skill and check return value",
        })

    def write_skill_code(
        self,
        skill_name: str,
        description: str,
        research: dict[str, Any],
        previous_code: Optional[str] = None,
        error_log: Optional[str] = None,
    ) -> str:
        """Generate a complete Python skill module."""
        system = (
            "You write JARVIS skill modules. Output ONLY valid Python code, no markdown fences.\n"
            "Every skill MUST define:\n"
            "  SKILL_META = {'name': str, 'description': str, 'capabilities': [str], "
            "'dependencies': [str], 'version': int}\n"
            "  def run(context: dict) -> dict:\n"
            "      '''Execute the skill. context has 'goal', 'args', 'workspace'. "
            "Return {'ok': bool, 'result': Any, 'error': str|None, 'evidence': str}'''\n"
            "Use only stdlib + declared dependencies. Be concrete and correct. "
            "Do not pretend success — set ok=False on failure."
        )
        parts = [
            f"Skill name: {skill_name}",
            f"Description: {description}",
            f"Research: {json.dumps(research, ensure_ascii=False)}",
        ]
        if previous_code:
            parts.append(f"PREVIOUS CODE:\n{previous_code}")
        if error_log:
            parts.append(f"ERRORS TO FIX:\n{error_log}")
            parts.append("Repair the skill so tests pass.")
        raw = self.generate("\n\n".join(parts), system=system, temperature=0.15)
        return self._extract_python(raw)

    def analyze_failure(self, skill_code: str, error_log: str, evidence: str) -> dict[str, Any]:
        system = (
            "Analyze why a skill failed. Reply ONLY with JSON:\n"
            '{"diagnosis":"...","fix_plan":"...","needs_new_deps":["..."],'
            '"is_unfixable":false}'
        )
        prompt = f"CODE:\n{skill_code[:6000]}\n\nERROR:\n{error_log}\n\nEVIDENCE:\n{evidence}"
        raw = self.generate(prompt, system=system, temperature=0.2)
        return self._parse_json(raw, {
            "diagnosis": error_log[:500],
            "fix_plan": "Inspect and repair the failing logic",
            "needs_new_deps": [],
            "is_unfixable": False,
        })

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
        raw = self.generate(prompt, system=system, temperature=0.1)
        return self._parse_json(raw, {
            "achieved": False,
            "confidence": 0.0,
            "reason": "Could not parse verification response",
        })

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
