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
        """True only when Ollama is up AND the exact model exists."""
        return self.model_status() == "ONLINE"

    def model_status(self) -> str:
        """
        Return:
          ONLINE        — server reachable and exact model present
          MODEL MISSING — server reachable but model not installed
          OFFLINE       — server unreachable
        """
        tags = self._list_model_names()
        if tags is None:
            return "OFFLINE"
        if self._model_in_tags(tags):
            return "ONLINE"
        return "MODEL MISSING"

    def _model_in_tags(self, names: list[str]) -> bool:
        target = self.model.strip().lower()
        # Exact match preferred; also accept name without tag if identical base+tag listed
        normalized = {n.strip().lower() for n in names}
        if target in normalized:
            return True
        # ollama sometimes lists "qwen2.5-coder:7b" and "qwen2.5-coder:7b-..." variants
        # Require exact model string match only (user requirement).
        return False

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
            '"skill_description":"...","research_queries":["..."],'
            '"args":{"any_key":"value extracted from the user goal"},'
            '"required_args":["arg_names_the_skill_will_need"]}\n'
            "Prefer reusing existing capabilities. Only request a new skill if none fit.\n"
            "Extract structured args from the user goal into args — do not leave them empty "
            "when the goal clearly contains parameters (filenames, text, urls, numbers, etc.). "
            "Arg names should match what a Python skill would read from context['args']."
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
            "- Infer useful keys from the goal (whatever the task needs).\n"
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
        diagnosis: Optional[dict[str, Any]] = None,
        failed_approaches: Optional[list] = None,
        test_plan: Optional[str] = None,
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
            "If a DIAGNOSIS is provided and fault_layer is skill_code, implement the fix. "
            "Do not repeat failed approaches listed below."
        )
        parts = [
            f"Skill name: {skill_name}",
            f"Description: {description}",
            f"Research: {json.dumps(research, ensure_ascii=False)[:6000]}",
        ]
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
            parts.append(f"PREVIOUS CODE:\n{previous_code[:8000]}")
        if error_log:
            parts.append(f"LAST ERROR / OBSERVATION SUMMARY:\n{error_log[:4000]}")
        # Slightly higher temperature on repair to encourage approach change
        temperature = 0.35 if (diagnosis or error_log) else 0.15
        raw = self.generate("\n\n".join(parts), system=system, temperature=temperature)
        return self._extract_python(raw)

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
            '  "fault_layer": "skill_code"|"context_args"|"test_harness"|"dependency"|"verifier",\n'
            '  "rewrite_skill": true|false,\n'
            '  "what_to_change": "...",\n'
            '  "approach": "short label of the NEW strategy to try",\n'
            '  "approach_changed": true|false,\n'
            '  "needs_research": true|false,\n'
            '  "research_queries": ["..."],\n'
            '  "needs_new_deps": ["pip-or-import-name"],\n'
            '  "missing_args": ["arg_names"],\n'
            '  "required_args": ["arg_names"],\n'
            '  "suggested_args": {"arg": "value from goal if present"},\n'
            '  "test_plan": "how to validate the next version",\n'
            '  "expected_artifacts": ["what verifier should find"],\n'
            '  "is_unfixable": false,\n'
            '  "diagnosis": "one-paragraph summary"\n'
            "}\n"
            "Rules:\n"
            "- Use exception, traceback, stdout, stderr, returncode, code, "
            "context (especially context.args), dependencies, artifacts.\n"
            "- If context.args is empty/missing keys the skill needs, fault_layer MUST be "
            "context_args and rewrite_skill=false. Fill suggested_args from the goal.\n"
            "- If VERIFY failed because the skill used default/placeholder values or created "
            "artifacts that do not match the USER REQUEST / goal constraints, fault_layer MUST "
            "be skill_code and rewrite_skill=true — the skill must implement the real request.\n"
            "- Only set fault_layer=verifier when the verifier harness itself is wrong "
            "(not when the skill output simply fails user constraints).\n"
            "- dependency: import/install failures. "
            "test_harness: runner did not pass context correctly.\n"
            "- If the same approach already failed, set approach_changed=true "
            "and propose a meaningfully different approach.\n"
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
        )
        ctx = observation.get("context") or {}
        ctx_args = ctx.get("args") if isinstance(ctx, dict) else {}
        empty_args = not isinstance(ctx_args, dict) or len(ctx_args) == 0
        err = str(observation.get("exception") or "").lower()
        phase = str(observation.get("phase") or "").upper()
        args_fault = empty_args and any(
            tok in err for tok in ("argument", "args", "missing", "required", "keyerror")
        )
        # VERIFY mismatch / defaults → skill must be rewritten to honor USER REQUEST
        goal_mismatch = phase == "VERIFY" and any(
            tok in err
            for tok in (
                "default", "placeholder", "untrusted", "claim_aligns",
                "not in user", "user constraints", "user request",
                "contains(", "missing from expected", "reject_defaults",
            )
        )
        if args_fault and not goal_mismatch:
            fb_layer = "context_args"
        elif goal_mismatch:
            fb_layer = "skill_code"
        else:
            fb_layer = "skill_code"
        fallback = {
            "root_cause": str(observation.get("exception") or "unknown")[:500],
            "fault_layer": fb_layer,
            "rewrite_skill": fb_layer == "skill_code",
            "what_to_change": (
                "Prepare structured context['args'] from the user goal and retest"
                if fb_layer == "context_args"
                else (
                    "Rewrite skill to satisfy USER REQUEST constraints "
                    "(no default/placeholder artifacts)"
                    if goal_mismatch
                    else "Revise skill logic based on stderr/traceback"
                )
            ),
            "approach": (
                "context_args_prep"
                if fb_layer == "context_args"
                else (
                    "honor_user_request"
                    if goal_mismatch
                    else f"alt_approach_v{(observation.get('version') or 0) + 1}"
                )
            ),
            "approach_changed": True,
            "needs_research": fb_layer == "skill_code" and not goal_mismatch,
            "research_queries": [
                f"python {observation.get('goal', '')}",
                str(observation.get("exception") or "")[:120],
            ],
            "needs_new_deps": [],
            "missing_args": [],
            "required_args": [],
            "suggested_args": {},
            "test_plan": (
                "Retest with prepared args; VERIFY must match USER REQUEST "
                "(reject defaults / skill self-proof)"
            ),
            "expected_artifacts": [],
            "is_unfixable": False,
            "diagnosis": str(observation.get("exception") or "failure")[:500],
        }
        result = self._parse_json(raw, fallback)
        # Normalize fault_layer
        layer = str(result.get("fault_layer") or fallback["fault_layer"]).lower()
        valid_layers = {
            "skill_code", "context_args", "test_harness", "dependency", "verifier",
        }
        if layer not in valid_layers:
            layer = fallback["fault_layer"]
        # Force skill rewrite when VERIFY proves goal mismatch / defaults
        if goal_mismatch:
            layer = "skill_code"
            result["rewrite_skill"] = True
        result["fault_layer"] = layer
        if "rewrite_skill" not in result:
            result["rewrite_skill"] = layer == "skill_code"
        if layer == "context_args":
            result["rewrite_skill"] = False
        if not isinstance(result.get("suggested_args"), dict):
            result["suggested_args"] = {}
        if not isinstance(result.get("missing_args"), list):
            result["missing_args"] = list(result.get("required_args") or [])

        # Enrich missing/suggested args from error + goal (universal, no hardcoding)
        from jarvis.context_builder import ContextBuilder

        result = ContextBuilder.enrich_diagnosis_args(
            result, observation, str(observation.get("goal") or "")
        )
        layer = str(result.get("fault_layer") or layer).lower()
        if layer not in valid_layers:
            layer = fallback["fault_layer"]
        if goal_mismatch:
            layer = "skill_code"
            result["rewrite_skill"] = True
        result["fault_layer"] = layer
        if layer == "context_args":
            result["rewrite_skill"] = False

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
        if fp in failed_fps and failed_fps:
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
