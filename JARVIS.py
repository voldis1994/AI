#!/usr/bin/env python3
"""
JARVIS — Autonomous self-learning AI agent.

Main entry point. Brain: Ollama multi-model router
(FAST / REASONING / CODING — see jarvis/model_config.py).
Learning happens by creating/repairing Python skills — not by hardcoding
every future capability into this core.

Cycle:
  REQUEST → PLAN → check capabilities → research → learn → build skill →
  install deps → test → repair → verify → save ACTIVE skill → execute →
  save experience → DONE  (DONE only after factual verification)

Usage:
  python JARVIS.py              # GUI
  python JARVIS.py --cli        # terminal mode
  python JARVIS.py --check      # syntax/import/self-check
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path
from typing import Optional

# tkinter is optional until GUI mode; keep CLI/--check usable headless
try:
    import tkinter as tk
    from tkinter import scrolledtext
except ModuleNotFoundError:  # pragma: no cover
    tk = None  # type: ignore
    scrolledtext = None  # type: ignore

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

(ROOT / "data").mkdir(exist_ok=True)
(ROOT / "skills").mkdir(exist_ok=True)

from jarvis.brain import Brain, DEFAULT_MODEL
from jarvis.model_config import TIER_MODELS, TIER_FAST, TIER_REASONING, TIER_CODING
from jarvis.orchestrator import Orchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(ROOT / "data" / "jarvis.log", encoding="utf-8"),
    ],
)

# ── Matrix palette ──────────────────────────────────────────────────────
BG = "#000000"
BG_PANEL = "#0a0f0a"
BG_INPUT = "#001a00"
FG = "#00ff41"
FG_DIM = "#00aa2a"
FG_BRIGHT = "#66ff99"
FG_WARN = "#ffcc00"
FG_ERR = "#ff3333"
FG_TITLE = "#00ff66"
BORDER = "#003300"
FONT_MONO = ("Consolas", 11)
FONT_TITLE = ("Consolas", 28, "bold")
FONT_SMALL = ("Consolas", 9)


class JarvisGUI:
    """Dark Matrix-style interface for JARVIS."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.busy = False
        # Monotonic id so late worker threads cannot paint a stale reply
        self._request_seq = 0

        self.win = tk.Tk()
        self.win.title("JARVIS")
        self.win.configure(bg=BG)
        self.win.geometry("980x700")
        self.win.minsize(720, 520)

        self._build_ui()
        self._bind_keys()

        self.orch = Orchestrator(
            root=root_dir,
            brain=Brain(),
            on_log=self._ui_log,
            on_status=self._ui_status,
        )
        self._refresh_stats()
        self._boot_banner()

        # Periodic stats refresh
        self.win.after(4000, self._tick_stats)

    # ── UI construction ─────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Title
        title_frame = tk.Frame(self.win, bg=BG)
        title_frame.pack(fill=tk.X, padx=16, pady=(14, 4))

        self.title_lbl = tk.Label(
            title_frame,
            text="JARVIS",
            fg=FG_TITLE,
            bg=BG,
            font=FONT_TITLE,
        )
        self.title_lbl.pack(side=tk.LEFT)

        self.subtitle = tk.Label(
            title_frame,
            text="  autonomous · self-learning · verified",
            fg=FG_DIM,
            bg=BG,
            font=FONT_SMALL,
        )
        self.subtitle.pack(side=tk.LEFT, pady=(12, 0))

        # Status bar under title
        status_frame = tk.Frame(self.win, bg=BG_PANEL, highlightbackground=BORDER, highlightthickness=1)
        status_frame.pack(fill=tk.X, padx=16, pady=(4, 8))

        self.status_var = tk.StringVar(value="STATUS: BOOT")
        self.status_lbl = tk.Label(
            status_frame,
            textvariable=self.status_var,
            fg=FG,
            bg=BG_PANEL,
            font=FONT_MONO,
            anchor="w",
        )
        self.status_lbl.pack(side=tk.LEFT, padx=10, pady=6)

        self.brain_var = tk.StringVar(value="BRAIN: …")
        self.brain_lbl = tk.Label(
            status_frame,
            textvariable=self.brain_var,
            fg=FG_DIM,
            bg=BG_PANEL,
            font=FONT_SMALL,
            anchor="e",
        )
        self.brain_lbl.pack(side=tk.RIGHT, padx=10, pady=6)

        # Main body: log + side panel
        body = tk.Frame(self.win, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=16, pady=4)

        # Conversation / log zone
        log_frame = tk.Frame(body, bg=BG_PANEL, highlightbackground=BORDER, highlightthickness=1)
        log_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        log_header = tk.Label(
            log_frame,
            text="▸ CONVERSATION / LOG",
            fg=FG_DIM,
            bg=BG_PANEL,
            font=FONT_SMALL,
            anchor="w",
        )
        log_header.pack(fill=tk.X, padx=8, pady=(6, 0))

        self.log = scrolledtext.ScrolledText(
            log_frame,
            bg=BG,
            fg=FG,
            insertbackground=FG,
            font=FONT_MONO,
            wrap=tk.WORD,
            relief=tk.FLAT,
            borderwidth=0,
            state=tk.DISABLED,
            highlightthickness=0,
        )
        self.log.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.log.tag_configure("user", foreground=FG_BRIGHT)
        self.log.tag_configure("jarvis", foreground=FG)
        self.log.tag_configure("system", foreground=FG_DIM)
        self.log.tag_configure("error", foreground=FG_ERR)
        self.log.tag_configure("warn", foreground=FG_WARN)

        # Side panel: memory / stats / skills
        side = tk.Frame(body, bg=BG_PANEL, width=260, highlightbackground=BORDER, highlightthickness=1)
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        side.pack_propagate(False)

        side_header = tk.Label(
            side,
            text="▸ MEMORY / STATS",
            fg=FG_DIM,
            bg=BG_PANEL,
            font=FONT_SMALL,
            anchor="w",
        )
        side_header.pack(fill=tk.X, padx=8, pady=(6, 0))

        self.stats_text = tk.Text(
            side,
            bg=BG,
            fg=FG,
            font=FONT_SMALL,
            relief=tk.FLAT,
            height=12,
            wrap=tk.WORD,
            state=tk.DISABLED,
            highlightthickness=0,
        )
        self.stats_text.pack(fill=tk.X, padx=6, pady=6)

        skills_header = tk.Label(
            side,
            text="▸ SKILLS REGISTRY",
            fg=FG_DIM,
            bg=BG_PANEL,
            font=FONT_SMALL,
            anchor="w",
        )
        skills_header.pack(fill=tk.X, padx=8, pady=(4, 0))

        self.skills_text = scrolledtext.ScrolledText(
            side,
            bg=BG,
            fg=FG,
            font=FONT_SMALL,
            relief=tk.FLAT,
            wrap=tk.WORD,
            state=tk.DISABLED,
            highlightthickness=0,
        )
        self.skills_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Input zone
        input_frame = tk.Frame(self.win, bg=BG)
        input_frame.pack(fill=tk.X, padx=16, pady=(4, 14))

        prompt = tk.Label(input_frame, text="›", fg=FG, bg=BG, font=("Consolas", 16, "bold"))
        prompt.pack(side=tk.LEFT, padx=(0, 6))

        self.entry = tk.Entry(
            input_frame,
            bg=BG_INPUT,
            fg=FG,
            insertbackground=FG,
            font=FONT_MONO,
            relief=tk.FLAT,
            highlightbackground=BORDER,
            highlightcolor=FG,
            highlightthickness=1,
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8)

        self.send_btn = tk.Button(
            input_frame,
            text="SEND",
            command=self._on_send,
            bg=BG_PANEL,
            fg=FG,
            activebackground=BORDER,
            activeforeground=FG_BRIGHT,
            font=FONT_SMALL,
            relief=tk.FLAT,
            padx=14,
            pady=6,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        self.send_btn.pack(side=tk.LEFT, padx=(8, 0))

    def _bind_keys(self) -> None:
        # Single send path — Return must not also trigger button-default double fire
        self.entry.bind("<Return>", self._on_return)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_return(self, _event=None):
        self._on_send()
        return "break"

    # ── Boot / display helpers ──────────────────────────────────────────

    def _boot_banner(self) -> None:
        status = self.orch.brain.model_status()
        fast = TIER_MODELS[TIER_FAST]["primary"]
        reason = TIER_MODELS[TIER_REASONING]["primary"]
        coding = TIER_MODELS[TIER_CODING]["primary"]
        hint = {
            "ONLINE": "CONNECTED",
            "MODEL MISSING": f"MODEL MISSING — ollama pull {fast}",
            "OFFLINE": "OFFLINE (start ollama serve)",
        }.get(status, status)
        catalog = ""
        if hasattr(self.orch.brain, "model_catalog_summary"):
            catalog = self.orch.brain.model_catalog_summary()
        # Explicit + required: adjacent literals + trailing "═" * 56 would parse as
        # (...banner body including "═") * 56 and spam the log dozens of times.
        banner = (
            ("═" * 56)
            + "\n  JARVIS online. Core stable. Skills learn themselves.\n"
            + f"  Brain: Ollama multi-model — {hint}\n"
            + f"  FAST={fast} · REASONING={reason} · CODING={coding}\n"
            + (f"  {catalog}\n" if catalog else "")
            + "  Type a message, a task, or /help\n"
            + ("═" * 56)
        )
        self._append("system", banner)
        self._ui_status("IDLE")
        self.brain_var.set(f"BRAIN: {status} · multi-model")

    def _append(self, tag: str, text: str) -> None:
        """Append one line to the conversation/log widget (thread-safe)."""
        def _do() -> None:
            self.log.configure(state=tk.NORMAL)
            self.log.insert(tk.END, text.rstrip() + "\n\n", tag)
            self.log.see(tk.END)
            self.log.configure(state=tk.DISABLED)

        if threading.current_thread() is threading.main_thread():
            _do()
        else:
            self.win.after(0, _do)

    def _ui_log(self, msg: str) -> None:
        """Process/cycle log sink only — not for USER input or final JARVIS replies.

        Conversation turns are rendered once by _on_send from the return value.
        """
        tag = "system"
        upper = msg.upper()
        if "FAIL" in upper or "ERROR" in upper or "EXCEPTION" in upper:
            tag = "error"
        elif "WARN" in upper or "BROKEN" in upper:
            tag = "warn"
        self._append(tag, f"· {msg}")

    def _ui_status(self, status: str) -> None:
        def _do() -> None:
            self.status_var.set(f"STATUS: {status}")
            color = FG
            if status in ("ERROR", "FAIL"):
                color = FG_ERR
            elif status in ("DONE", "IDLE"):
                color = FG_BRIGHT
            elif status in ("REPAIR", "REPAIRING", "BROKEN"):
                color = FG_WARN
            self.status_lbl.configure(fg=color)

        self.win.after(0, _do)

    def _set_stats_text(self, widget: tk.Text, content: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, content)
        widget.configure(state=tk.DISABLED)

    def _refresh_stats(self) -> None:
        try:
            dash = self.orch.get_dashboard_stats()
        except Exception as exc:
            self._set_stats_text(self.stats_text, f"stats error: {exc}")
            return

        mem = dash["memory"]
        sk = dash["skills"]
        led = dash["ledger"]
        brain = dash.get("brain_status") or (
            "ONLINE" if dash["brain_online"] else "OFFLINE"
        )
        catalog = dash.get("model_catalog") or "multi-model"
        stats = (
            f"Brain   {brain}\n"
            f"Router  {catalog}\n"
            f"──────────────\n"
            f"Conv    {mem['conversations']}\n"
            f"Exper.  {mem['experiences']} (ok {mem['successes']})\n"
            f"Facts   {mem['facts']}\n"
            f"Events  {mem['skill_events']}\n"
            f"──────────────\n"
            f"Tasks   {led['tasks']}\n"
            f"Done    {led['done']}\n"
            f"Fail    {led['failed']}\n"
            f"Actions {led['actions']}\n"
            f"──────────────\n"
            f"Skills  {sk['total']}\n"
            f"ACTIVE  {sk.get('ACTIVE', 0)}\n"
            f"TEST    {sk.get('TESTING', 0)}\n"
            f"CAND    {sk.get('CANDIDATE', 0)}\n"
            f"BROKEN  {sk.get('BROKEN', 0)}\n"
            f"REPAIR  {sk.get('REPAIRING', 0)}\n"
        )
        self._set_stats_text(self.stats_text, stats)

        lines = []
        for s in dash.get("skill_list") or []:
            lines.append(f"[{s['status'][:4]}] {s['name']} v{s['version']}")
        self._set_stats_text(
            self.skills_text,
            "\n".join(lines) if lines else "(none yet — give JARVIS a task)",
        )
        self.brain_var.set(f"BRAIN: {brain} · multi-model")

    def _tick_stats(self) -> None:
        if not self.busy:
            self._refresh_stats()
        self.win.after(4000, self._tick_stats)

    # ── Interaction ─────────────────────────────────────────────────────

    def _on_send(self) -> None:
        """Single owner for YOU › and JARVIS › display (one each per turn)."""
        if self.busy:
            return
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, tk.END)
        self.busy = True
        self._request_seq += 1
        seq = self._request_seq
        self.send_btn.configure(state=tk.DISABLED)
        self._append("user", f"YOU › {text}")
        self._ui_status("WORKING")

        def worker() -> None:
            try:
                result = self.orch.handle_user_message(text)
                reply = result.get("reply") or str(result)
                tag = "jarvis"
                if result.get("type") == "task" and not result.get("success"):
                    tag = "error"
                elif result.get("success") is False:
                    tag = "error"

                def show_reply() -> None:
                    # Drop stale workers — never show a prior request's final_result
                    if seq != self._request_seq:
                        return
                    self._append(tag, f"JARVIS › {reply}")

                self.win.after(0, show_reply)
            except Exception as exc:
                def show_err() -> None:
                    if seq != self._request_seq:
                        return
                    self._append("error", f"JARVIS › Internal error: {exc}")

                self.win.after(0, show_err)
            finally:
                def done() -> None:
                    if seq != self._request_seq:
                        return
                    self.busy = False
                    self.send_btn.configure(state=tk.NORMAL)
                    self._refresh_stats()
                    self.entry.focus_set()

                self.win.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def _on_close(self) -> None:
        try:
            self.orch.close()
        except Exception:
            pass
        self.win.destroy()

    def run(self) -> None:
        self.entry.focus_set()
        self.win.mainloop()


# ── CLI mode ────────────────────────────────────────────────────────────

def run_cli(root: Path) -> int:
    print("JARVIS CLI — type /quit to exit, /help for commands")
    logs: list[str] = []

    def on_log(m: str) -> None:
        print(f"  · {m}")
        logs.append(m)

    def on_status(s: str) -> None:
        print(f"  [{s}]")

    orch = Orchestrator(
        root=root,
        brain=Brain(),
        on_log=on_log,
        on_status=on_status,
    )
    catalog = ""
    if hasattr(orch.brain, "model_catalog_summary"):
        catalog = orch.brain.model_catalog_summary()
    print(f"Brain: {orch.brain.model_status()} (multi-model)")
    if catalog:
        print(f"Models: {catalog}")
    try:
        while True:
            try:
                text = input("YOU › ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            if text.lower() in ("/quit", "/exit", "quit", "exit"):
                break
            result = orch.handle_user_message(text)
            print(f"JARVIS › {result.get('reply')}\n")
    finally:
        orch.close()
    return 0


def run_check(root: Path) -> int:
    """Verify syntax, imports, subprocess isolation, and e2e learn→reuse."""
    import ast
    import importlib
    import shutil
    import traceback

    errors: list[str] = []
    print("=== JARVIS self-check ===")

    # 1) Parse all Python sources
    py_files = [root / "JARVIS.py"] + sorted((root / "jarvis").glob("*.py"))
    for path in py_files:
        try:
            src = path.read_text(encoding="utf-8")
            ast.parse(src, filename=str(path))
            print(f"  OK syntax  {path.relative_to(root)}")
        except SyntaxError as exc:
            msg = f"SYNTAX {path}: {exc}"
            print(f"  FAIL {msg}")
            errors.append(msg)

    # 2) Import package modules
    modules = [
        "jarvis",
        "jarvis.brain",
        "jarvis.model_config",
        "jarvis.model_router",
        "jarvis.recovery",
        "jarvis.memory",
        "jarvis.ledger",
        "jarvis.capability_registry",
        "jarvis.research",
        "jarvis.dependency_manager",
        "jarvis.skill_builder",
        "jarvis.skill_tester",
        "jarvis.skill_runner",
        "jarvis.observer",
        "jarvis.verifier",
        "jarvis.skill_loader",
        "jarvis.context_builder",
        "jarvis.orchestrator",
    ]
    for name in modules:
        try:
            importlib.import_module(name)
            print(f"  OK import  {name}")
        except Exception as exc:
            msg = f"IMPORT {name}: {exc}"
            print(f"  FAIL {msg}")
            errors.append(msg)

    # 3) Subprocess tester + independent verifier + ACTIVE protection
    print("  — subprocess test + verifier separation + ACTIVE protect —")
    try:
        from jarvis.skill_builder import SkillBuilder
        from jarvis.capability_registry import CapabilityRegistry
        from jarvis.skill_tester import SkillTester
        from jarvis.verifier import Verifier
        from jarvis.dependency_manager import DependencyManager
        from jarvis.brain import Brain as RealBrain

        tmp = root / "data" / "_selfcheck"
        if tmp.exists():
            shutil.rmtree(tmp)
        skills = tmp / "skills"
        ws = tmp / "ws"
        skills.mkdir(parents=True)
        ws.mkdir(parents=True)
        db = tmp / "check.db"

        skill_code = '''
SKILL_META = {
    "name": "write_hello_file",
    "description": "Write a hello.txt file into workspace",
    "capabilities": ["write hello file", "create text file"],
    "dependencies": [],
    "version": 1,
}

def run(context: dict) -> dict:
    from pathlib import Path
    workspace = Path(context.get("workspace") or ".")
    path = workspace / "hello.txt"
    path.write_text("Hello from JARVIS skill\\n", encoding="utf-8")
    return {
        "ok": True,
        "result": {"path": str(path), "bytes": path.stat().st_size, "contains": "Hello"},
        "error": None,
        "evidence": f"Wrote {path} exists={path.exists()} size={path.stat().st_size}",
    }
'''
        skill_path = skills / "write_hello_file.py"
        skill_path.write_text(skill_code, encoding="utf-8")

        reg = CapabilityRegistry(db)
        reg.register_candidate(
            "write_hello_file", "Write hello.txt", str(skill_path),
            ["write hello file"], [], 1,
        )
        reg.set_status("write_hello_file", "TESTING")
        tester = SkillTester(ws)
        test_res = tester.test(skill_path, goal="Create hello.txt in workspace")
        assert test_res.get("returncode") is not None, test_res
        assert "stdout" in test_res and "stderr" in test_res
        assert test_res.get("ok"), test_res
        assert "skill_result" in test_res
        # Crash isolation: hang skill must timeout without killing parent
        hang = skills / "hang_skill.py"
        hang.write_text(
            "SKILL_META={'name':'hang_skill','description':'hang','capabilities':[],"
            "'dependencies':[],'version':1}\n"
            "def run(context):\n"
            "    import time\n"
            "    time.sleep(999)\n"
            "    return {'ok': True, 'result': None, 'error': None, 'evidence': 'no'}\n",
            encoding="utf-8",
        )
        hang_res = tester.test(hang, goal="hang", timeout_hint=2.0)
        assert hang_res.get("timed_out"), hang_res
        assert hang_res.get("killed"), hang_res

        ver = Verifier(ws)
        vres = ver.verify("Create hello.txt", test_res)
        assert vres["verified"], vres
        assert "skill_result" in vres and "verifier_result" in vres
        assert vres["verifier_result"]["pass"]

        # Defaults / skill self-proof must NOT earn PASS against a different USER REQUEST
        (ws / "default_path.txt").write_text("default content\n", encoding="utf-8")
        bad_sr = {
            "ok": True,
            "result": {
                "path": str(ws / "default_path.txt"),
                "contains": "default content",
            },
            "evidence": f"wrote {ws / 'default_path.txt'}",
            "error": None,
            "returncode": 0,
        }
        bad = ver.verify(
            'Create "notes.txt" containing "hello world"',
            bad_sr,
            args={"dest": "notes.txt", "body": "hello world"},
            user_request='Create "notes.txt" containing "hello world"',
        )
        assert not bad.get("verified"), bad
        assert any(
            c.get("name") == "reject_defaults" and not c.get("ok")
            for c in bad.get("checks") or []
        ), bad.get("checks")

        # Promote path
        reg.promote_candidate(
            "write_hello_file", str(skill_path), 1,
            "Write hello.txt", ["write hello file"], [],
            archive_dir=skills / "archive",
        )
        assert reg.get_skill("write_hello_file")["status"] == "ACTIVE"
        active_file = Path(reg.get_skill("write_hello_file")["file_path"])
        active_before = active_file.read_text(encoding="utf-8")

        # Build v2 while ACTIVE protected — must not overwrite ACTIVE file
        builder = SkillBuilder(skills)
        built = builder.build(
            "write_hello_file",
            "Write hello.txt v2",
            {"libraries": [], "key_apis": ["write hello file"], "approach": "rewrite"},
            version=2,
            protect_active_path=str(active_file),
        )
        assert built["ok"] and built.get("protected_active")
        assert Path(built["path"]).name.endswith(".candidate.py")
        assert active_file.read_text(encoding="utf-8") == active_before
        print("  OK subprocess timeout/kill + SKILL≠VERIFIER + ACTIVE protect")

        # Brain model status API
        b = RealBrain()
        st = b.model_status()
        assert st in ("ONLINE", "MODEL MISSING", "OFFLINE"), st
        print(f"  OK brain     model_status={st}")

        dm = DependencyManager()
        assert dm.ensure([])["ok"]

        reg.close()
    except Exception as exc:
        msg = f"CYCLE: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # 4) E2E: missing skill → fail once → OBSERVE/DIAGNOSE/REPAIR → ACTIVE → reuse
    # FakeBrain only simulates Ollama; core stays universal (no hardcoded skill logic).
    print("  — e2e observe/diagnose/repair → ACTIVE → restart → reuse —")
    try:
        e2e_root = root / "data" / "_e2e"
        if e2e_root.exists():
            shutil.rmtree(e2e_root)
        e2e_root.mkdir(parents=True)

        class FakeBrain(Brain):
            """Simulates Ollama decisions from observations — not a core hardcode."""

            def __init__(self) -> None:
                super().__init__()
                self.build_count = 0

            def model_status(self) -> str:
                return "ONLINE"

            def is_available(self) -> bool:
                return True

            def classify_intent(self, user_text: str) -> dict:
                return {
                    "intent": "task",
                    "goal": user_text,
                    "needs_capability": True,
                    "keywords": user_text.split()[:8],
                }

            def plan(self, goal: str, known_capabilities: list[str]) -> dict:
                reuse = []
                for line in known_capabilities:
                    if "write_e2e_marker" in line:
                        reuse.append("write_e2e_marker")
                return {
                    "steps": ["accomplish goal via skill"],
                    "can_reuse": reuse,
                    "missing": [] if reuse else [goal],
                    "needs_research": not bool(reuse),
                    "needs_new_skill": not bool(reuse),
                    "skill_name": "write_e2e_marker",
                    "skill_description": goal,
                    "research_queries": [goal],
                    "args": {},
                    "required_args": [],
                }

            def extract_task_args(
                self, goal, skill_meta=None, prior_args=None, diagnosis=None
            ):
                # Existing e2e skill hardcodes its artifact; no args required.
                return {}

            def research_notes(self, query: str, gathered: str) -> dict:
                return {
                    "approach": "initial_broken_then_repair",
                    "libraries": [],
                    "key_apis": [query],
                    "pitfalls": [],
                    "test_idea": "independent artifact verification",
                }

            def write_skill_code(
                self,
                skill_name,
                description,
                research,
                previous_code=None,
                error_log=None,
                diagnosis=None,
                failed_approaches=None,
                test_plan=None,
            ) -> str:
                self.build_count += 1
                # First build intentionally fails — exercises OBSERVE→DIAGNOSE→REPAIR
                if self.build_count == 1 and not diagnosis:
                    return f'''
SKILL_META = {{
    "name": "{skill_name}",
    "description": "{description}",
    "capabilities": ["e2e"],
    "dependencies": [],
    "version": 1,
}}

def run(context: dict) -> dict:
    return {{
        "ok": False,
        "result": None,
        "error": "intentional first-attempt failure for repair pipeline test",
        "evidence": "",
    }}
'''
                # After diagnosis: different approach that actually works
                return f'''
from pathlib import Path

SKILL_META = {{
    "name": "{skill_name}",
    "description": "{description}",
    "capabilities": ["e2e"],
    "dependencies": [],
    "version": 2,
}}

def run(context: dict) -> dict:
    workspace = Path(context.get("workspace") or ".")
    path = workspace / "jarvis_e2e_marker.txt"
    path.write_text("E2E_OK\\n", encoding="utf-8")
    return {{
        "ok": True,
        "result": {{"path": str(path), "contains": "E2E_OK"}},
        "error": None,
        "evidence": f"created {{path}} size={{path.stat().st_size}}",
    }}
'''

            def diagnose(self, observation, failed_approaches=None, prior_solutions=None):
                from jarvis.observer import Observer
                approach = "pathlib_write_after_observe"
                # Must differ from failed approaches
                fps = {
                    a.get("approach_fingerprint")
                    for a in (failed_approaches or [])
                }
                fp = Observer.fingerprint_approach(approach)
                if fp in fps:
                    approach = approach + "_v2"
                    fp = Observer.fingerprint_approach(approach)
                return {
                    "root_cause": str(observation.get("exception") or "test failed"),
                    "fault_layer": "skill_code",
                    "rewrite_skill": True,
                    "what_to_change": "Implement real artifact creation based on goal",
                    "approach": approach,
                    "approach_fingerprint": fp,
                    "approach_changed": True,
                    "needs_research": False,
                    "research_queries": [],
                    "needs_new_deps": [],
                    "missing_args": [],
                    "required_args": [],
                    "suggested_args": {},
                    "test_plan": "subprocess run + verifier checks file exists with content",
                    "expected_artifacts": ["jarvis_e2e_marker.txt"],
                    "is_unfixable": False,
                    "diagnosis": "First approach returned ok=False; switch strategy",
                }

            def verify_claim(self, goal, result, evidence) -> dict:
                return {"achieved": True, "confidence": 0.9, "reason": "advisory"}

            def converse(self, user_text, history=None) -> str:
                return "ok"

        phases: list[str] = []
        orch = Orchestrator(
            root=e2e_root,
            brain=FakeBrain(),
            on_status=lambda s: phases.append(s),
            on_log=lambda m: print(f"    · {m}") if any(
                k in m for k in (
                    "REQUEST", "RESEARCH", "BUILD", "TEST", "RETEST", "OBSERVE",
                    "DIAGNOSE", "REPAIR", "VERIFY", "ACTIVE", "DONE", "EXECUTE",
                    "SKILL RESULT", "VERIFIER",
                )
            ) else None,
        )
        assert orch.registry.get_skill("write_e2e_marker") is None

        goal = "Create file jarvis_e2e_marker.txt containing E2E_OK"
        result1 = orch.run_cycle(goal)
        assert result1.get("success"), result1
        assert orch.registry.get_skill("write_e2e_marker")["status"] == "ACTIVE"
        marker = e2e_root / "workspace_runtime" / "jarvis_e2e_marker.txt"
        assert marker.exists() and "E2E_OK" in marker.read_text(encoding="utf-8")

        # Universal repair pipeline phases exercised
        for required in ("TEST", "OBSERVE", "DIAGNOSE", "REPAIR", "RETEST", "VERIFY", "DONE"):
            assert required in phases, f"missing phase {required} in {phases}"

        # Failures / diagnoses / solutions persisted
        assert orch.memory.stats()["learning_failures"] >= 1
        assert orch.memory.stats()["learning_diagnoses"] >= 1
        assert orch.memory.stats()["learning_solutions"] >= 1
        assert orch.memory.get_failed_approaches("write_e2e_marker")

        fact = orch.memory.get_fact("research:write_e2e_marker")
        assert fact and "results" in fact
        orch.close()
        print("  OK e2e      BUILD→TEST→OBSERVE→DIAGNOSE→REPAIR→RETEST→VERIFY→ACTIVE")

        phases2: list[str] = []
        orch2 = Orchestrator(
            root=e2e_root,
            brain=FakeBrain(),
            on_status=lambda s: phases2.append(s),
        )
        assert orch2.registry.get_skill("write_e2e_marker")["status"] == "ACTIVE"
        if marker.exists():
            marker.unlink()
        result2 = orch2.run_cycle(goal)
        assert result2.get("success"), result2
        assert marker.exists()
        assert result2.get("skill") == "write_e2e_marker"
        assert "EXECUTE" in phases2 and "VERIFY" in phases2 and "DONE" in phases2
        # Reuse should not need another OBSERVE/DIAGNOSE cycle
        assert "OBSERVE" not in phases2
        orch2.close()
        print("  OK reuse    restart → ACTIVE skill reused (no re-learn)")
    except Exception as exc:
        msg = f"E2E: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # 4b) E2E: empty args → DIAGNOSE context_mapping → RETEST with args (no skill rewrite)
    print("  — e2e context_mapping fault_layer repair (no skill rewrite) —")
    try:
        from jarvis.context_builder import ContextBuilder
        from jarvis.task_goal import TaskGoal
        from jarvis.verifier import Verifier

        # Unit: immutable TaskGoal + grounding (no invented defaults / no token→file noise)
        tg = TaskGoal.from_request("Izveido failu test.txt ar tekstu DARBOJAS")
        assert tg.user_request == "Izveido failu test.txt ar tekstu DARBOJAS"
        paths = [f.get("path") for f in tg.constraints.get("files") or []]
        assert paths == ["test.txt"], paths
        assert (tg.constraints.get("files") or [{}])[0].get("contains") == "DARBOJAS"
        grounded = TaskGoal.ground_args(
            {"path": "user_provided_path", "content": "DARBOJAS"},
            tg.user_request,
        )
        assert "path" not in grounded and grounded.get("content") == "DARBOJAS", grounded
        noisy_v = Verifier(root / "data" / "_tg_verify_ws")
        (root / "data" / "_tg_verify_ws").mkdir(parents=True, exist_ok=True)
        built = noisy_v.extract_constraints(
            tg.user_request,
            {"path": "user_provided_path", "content": "user_provided_content"},
        )
        file_paths = [f.get("path") for f in built.get("files") or []]
        assert file_paths == ["test.txt"], built
        # Content token stays as contains — never a second invented file path
        assert "DARBOJAS" not in file_paths, built
        assert (built.get("files") or [{}])[0].get("contains") == "DARBOJAS", built
        # user_provided_* must not become file constraints
        assert not any(
            "user_provided" in str(f.get("path") or "").lower()
            for f in built.get("files") or []
        ), built

        # Unit: universal missing-arg parse + offline fill (no hardcoded key names)
        parsed = ContextBuilder.parse_missing_arg_names(
            "The skill is missing required arguments 'alpha' and 'beta' when invoked."
        )
        assert parsed == ["alpha", "beta"], parsed
        filled = ContextBuilder._offline_extract(
            'Do work alpha_file.dat with "PAYLOAD_Z"',
            diagnosis={"missing_args": ["alpha", "beta"], "required_args": ["alpha", "beta"]},
        )
        assert filled.get("alpha") == "alpha_file.dat", filled
        assert filled.get("beta") == "PAYLOAD_Z", filled
        # Context mapper rejects invented defaults even if diagnosis suggests them
        cb = ContextBuilder()
        mapped = cb.build(
            tg.user_request,
            str(root / "data" / "_tg_verify_ws"),
            prior_args={"path": "user_provided_path", "content": "user_provided_content"},
            diagnosis={
                "missing_args": ["path", "content"],
                "required_args": ["path", "content"],
                "suggested_args": {
                    "path": "user_provided_path",
                    "content": "user_provided_content",
                },
                "fault_layer": "context_mapping",
            },
            user_request=tg.user_request,
            task_goal=tg,
        )
        assert mapped["args"].get("path") == "test.txt", mapped["args"]
        assert mapped["args"].get("content") == "DARBOJAS", mapped["args"]
        assert TaskGoal.normalize_fault_layer("context_args") == "context_mapping"
        assert TaskGoal.rewrite_skill_for_layer("context_mapping") is False
        assert TaskGoal.rewrite_skill_for_layer("skill_code") is True

        args_root = root / "data" / "_e2e_args"
        if args_root.exists():
            shutil.rmtree(args_root)
        args_root.mkdir(parents=True)

        class ArgsFaultBrain(Brain):
            """Skill is correct; PLAN/TEST start with empty args — repair must fill them."""

            def __init__(self) -> None:
                super().__init__()
                self.build_count = 0

            def model_status(self) -> str:
                return "ONLINE"

            def is_available(self) -> bool:
                return True

            def classify_intent(self, user_text: str) -> dict:
                return {
                    "intent": "task",
                    "goal": user_text,
                    "needs_capability": True,
                    "keywords": user_text.split()[:8],
                }

            def plan(self, goal: str, known_capabilities: list[str]) -> dict:
                reuse = []
                for line in known_capabilities:
                    if "args_echo_skill" in line:
                        reuse.append("args_echo_skill")
                return {
                    "steps": ["run skill with structured args"],
                    "can_reuse": reuse,
                    "missing": [] if reuse else [goal],
                    "needs_research": not bool(reuse),
                    "needs_new_skill": not bool(reuse),
                    "skill_name": "args_echo_skill",
                    "skill_description": goal,
                    "research_queries": [goal],
                    # Simulate the bug: planner forgot to extract args
                    "args": {},
                    "required_args": [],
                }

            def extract_task_args(
                self, goal, skill_meta=None, prior_args=None, diagnosis=None
            ):
                # Without diagnosis, pretend brain extracted nothing (empty args bug).
                # With diagnosis missing_args, fill universally from the goal.
                if diagnosis and (
                    diagnosis.get("missing_args")
                    or diagnosis.get("required_args")
                    or diagnosis.get("suggested_args")
                ):
                    return ContextBuilder._offline_extract(goal, diagnosis=diagnosis)
                return {}

            def research_notes(self, query: str, gathered: str) -> dict:
                return {
                    "approach": "read_context_args",
                    "libraries": [],
                    "key_apis": [query],
                    "pitfalls": ["never invent missing args"],
                    "test_idea": "pass structured args into run(context)",
                }

            def write_skill_code(
                self,
                skill_name,
                description,
                research,
                previous_code=None,
                error_log=None,
                diagnosis=None,
                failed_approaches=None,
                test_plan=None,
            ) -> str:
                self.build_count += 1
                # Correct skill from the start — fails only when context.args incomplete
                import json as _json
                desc_lit = _json.dumps(description or "")
                return f'''
from pathlib import Path

SKILL_META = {{
    "name": "{skill_name}",
    "description": {desc_lit},
    "capabilities": ["args_echo"],
    "dependencies": [],
    "version": {self.build_count},
    "required_args": ["dest", "body"],
}}

def run(context: dict) -> dict:
    args = context.get("args") or {{}}
    required = list(SKILL_META.get("required_args") or [])
    missing = [k for k in required if args.get(k) in (None, "")]
    if missing:
        return {{
            "ok": False,
            "result": None,
            "error": (
                "The skill is missing required arguments "
                + " and ".join(repr(m) for m in missing)
                + " when invoked."
            ),
            "evidence": f"args_keys={{list(args.keys())}}",
        }}
    workspace = Path(context.get("workspace") or ".")
    path = workspace / str(args["dest"])
    path.write_text(str(args["body"]) + "\\n", encoding="utf-8")
    return {{
        "ok": True,
        "result": {{"path": str(path), "contains": str(args["body"])}},
        "error": None,
        "evidence": f"wrote {{path}} size={{path.stat().st_size}}",
    }}
'''

            def diagnose(self, observation, failed_approaches=None, prior_solutions=None):
                from jarvis.observer import Observer

                err = str(observation.get("exception") or "")
                phase = str(observation.get("phase") or "")
                parsed = ContextBuilder.parse_missing_arg_names(err)
                if parsed:
                    d = {
                        "root_cause": "context.args missing required keys",
                        "fault_layer": "context_mapping",
                        "rewrite_skill": False,
                        "what_to_change": "Map TaskGoal → skill args from USER REQUEST",
                        "approach": "context_mapping",
                        "approach_changed": True,
                        "needs_research": False,
                        "research_queries": [],
                        "needs_new_deps": [],
                        "missing_args": parsed,
                        "required_args": parsed,
                        "suggested_args": {},
                        "test_plan": "retest with grounded context.args",
                        "expected_artifacts": [],
                        "is_unfixable": False,
                        "diagnosis": err[:300],
                    }
                    d["approach_fingerprint"] = Observer.fingerprint_approach(
                        d["approach"]
                    )
                    return ContextBuilder.enrich_diagnosis_args(
                        d, observation, str(observation.get("goal") or "")
                    )
                # Verifier issues are not skill-code faults in this scenario
                if phase == "VERIFY" or "VERIFIER" in err.upper():
                    approach = "verifier_align"
                    fp = Observer.fingerprint_approach(approach)
                    return {
                        "root_cause": err[:500],
                        "fault_layer": "verifier",
                        "rewrite_skill": False,
                        "what_to_change": "Keep skill; rely on result.path evidence",
                        "approach": approach,
                        "approach_fingerprint": fp,
                        "approach_changed": True,
                        "needs_research": False,
                        "research_queries": [],
                        "needs_new_deps": [],
                        "missing_args": [],
                        "required_args": [],
                        "suggested_args": {},
                        "test_plan": "retest with same args",
                        "expected_artifacts": [],
                        "is_unfixable": False,
                        "diagnosis": err[:300],
                    }
                approach = "unexpected_skill_fix"
                fp = Observer.fingerprint_approach(approach)
                return {
                    "root_cause": err[:500],
                    "fault_layer": "skill_code",
                    "rewrite_skill": True,
                    "what_to_change": "fix skill",
                    "approach": approach,
                    "approach_fingerprint": fp,
                    "approach_changed": True,
                    "needs_research": False,
                    "research_queries": [],
                    "needs_new_deps": [],
                    "missing_args": [],
                    "required_args": [],
                    "suggested_args": {},
                    "test_plan": "retest",
                    "expected_artifacts": [],
                    "is_unfixable": False,
                    "diagnosis": err[:300],
                }

            def verify_claim(self, goal, result, evidence) -> dict:
                return {"achieved": True, "confidence": 0.9, "reason": "advisory"}

            def converse(self, user_text, history=None) -> str:
                return "ok"

        phases_a: list[str] = []
        logs_a: list[str] = []
        brain_a = ArgsFaultBrain()
        orch_a = Orchestrator(
            root=args_root,
            brain=brain_a,
            on_status=lambda s: phases_a.append(s),
            on_log=lambda m: logs_a.append(m),
        )
        # Quoted values only — arg *names* come from skill error / diagnosis,
        # not from hardcoded path/content keys in the goal.
        goal_a = 'Create "args_e2e_out.txt" containing "ARGS_OK"'
        result_a = orch_a.run_cycle(goal_a)
        assert result_a.get("success"), result_a
        out = args_root / "workspace_runtime" / "args_e2e_out.txt"
        assert out.exists() and "ARGS_OK" in out.read_text(encoding="utf-8"), out
        # Skill written once — context_mapping repair must not rewrite skill
        assert brain_a.build_count == 1, brain_a.build_count
        assert "OBSERVE" in phases_a and "DIAGNOSE" in phases_a and "RETEST" in phases_a
        assert any(
            "context_mapping" in m or "context_args" in m or "CONTEXT repair" in m
            for m in logs_a
        ), logs_a[-20:]
        assert any("Skip skill rewrite" in m for m in logs_a), logs_a[-20:]
        assert any("TaskGoal" in m or "REQUEST:" in m for m in logs_a), logs_a[:10]
        orch_a.close()
        print("  OK args     empty args → context_mapping diagnose → retest (no rewrite)")
    except Exception as exc:
        msg = f"E2E_ARGS: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # 4c) E2E: default artifact must VERIFY FAIL → skill rewrite → honor USER REQUEST
    print("  — e2e verifier rejects defaults; repair honors USER REQUEST —")
    try:
        def_root = root / "data" / "_e2e_defaults"
        if def_root.exists():
            shutil.rmtree(def_root)
        def_root.mkdir(parents=True)

        class DefaultTrapBrain(Brain):
            def __init__(self) -> None:
                super().__init__()
                self.build_count = 0

            def model_status(self) -> str:
                return "ONLINE"

            def is_available(self) -> bool:
                return True

            def classify_intent(self, user_text: str) -> dict:
                return {
                    "intent": "task",
                    "goal": user_text,
                    "needs_capability": True,
                    "keywords": user_text.split()[:8],
                }

            def plan(self, goal: str, known_capabilities: list[str]) -> dict:
                return {
                    "steps": ["create requested artifact"],
                    "can_reuse": [],
                    "missing": [goal],
                    "needs_research": True,
                    "needs_new_skill": True,
                    "skill_name": "make_artifact",
                    "skill_description": goal,
                    "research_queries": [goal],
                    "args": {"dest": "user_note.txt", "body": "REAL_PAYLOAD"},
                    "required_args": ["dest", "body"],
                }

            def extract_task_args(
                self, goal, skill_meta=None, prior_args=None, diagnosis=None
            ):
                return {"dest": "user_note.txt", "body": "REAL_PAYLOAD"}

            def research_notes(self, query: str, gathered: str) -> dict:
                return {
                    "approach": "default_trap_then_honor_request",
                    "libraries": [],
                    "key_apis": [query],
                    "pitfalls": ["never use default_path / default content"],
                    "test_idea": "verifier must match USER REQUEST",
                }

            def write_skill_code(
                self,
                skill_name,
                description,
                research,
                previous_code=None,
                error_log=None,
                diagnosis=None,
                failed_approaches=None,
                test_plan=None,
            ) -> str:
                self.build_count += 1
                import json as _json
                desc_lit = _json.dumps(description or "")
                # First build: intentionally writes defaults (must VERIFY FAIL)
                if self.build_count == 1 and not (
                    diagnosis and diagnosis.get("rewrite_skill")
                ):
                    return f'''
from pathlib import Path
SKILL_META = {{
    "name": "{skill_name}",
    "description": {desc_lit},
    "capabilities": ["make_artifact"],
    "dependencies": [],
    "version": 1,
    "required_args": ["dest", "body"],
}}
def run(context: dict) -> dict:
    workspace = Path(context.get("workspace") or ".")
    path = workspace / "default_path.txt"
    path.write_text("default content\\n", encoding="utf-8")
    return {{
        "ok": True,
        "result": {{"path": str(path), "contains": "default content"}},
        "error": None,
        "evidence": f"wrote {{path}}",
    }}
'''
                # Repair: honor context args / user request
                return f'''
from pathlib import Path
SKILL_META = {{
    "name": "{skill_name}",
    "description": {desc_lit},
    "capabilities": ["make_artifact"],
    "dependencies": [],
    "version": {self.build_count},
    "required_args": ["dest", "body"],
}}
def run(context: dict) -> dict:
    args = context.get("args") or {{}}
    required = list(SKILL_META.get("required_args") or [])
    missing = [k for k in required if args.get(k) in (None, "")]
    if missing:
        return {{
            "ok": False,
            "result": None,
            "error": "missing required arguments: " + repr(missing),
            "evidence": "",
        }}
    workspace = Path(context.get("workspace") or ".")
    path = workspace / str(args["dest"])
    path.write_text(str(args["body"]) + "\\n", encoding="utf-8")
    return {{
        "ok": True,
        "result": {{"path": str(path), "contains": str(args["body"])}},
        "error": None,
        "evidence": f"wrote {{path}}",
    }}
'''

            def diagnose(self, observation, failed_approaches=None, prior_solutions=None):
                from jarvis.observer import Observer
                from jarvis.context_builder import ContextBuilder

                err = str(observation.get("exception") or "")
                phase = str(observation.get("phase") or "")
                if phase == "VERIFY" or "default" in err.lower() or "VERIFIER" in err:
                    approach = "honor_user_request"
                    fp = Observer.fingerprint_approach(approach)
                    d = {
                        "root_cause": "skill used default artifact instead of USER REQUEST",
                        "fault_layer": "skill_code",
                        "rewrite_skill": True,
                        "what_to_change": "Write dest/body from context args; no defaults",
                        "approach": approach,
                        "approach_fingerprint": fp,
                        "approach_changed": True,
                        "needs_research": False,
                        "research_queries": [],
                        "needs_new_deps": [],
                        "missing_args": [],
                        "required_args": ["dest", "body"],
                        "suggested_args": {
                            "dest": "user_note.txt",
                            "body": "REAL_PAYLOAD",
                        },
                        "test_plan": "VERIFY against USER REQUEST",
                        "expected_artifacts": ["user_note.txt"],
                        "is_unfixable": False,
                        "diagnosis": err[:300],
                    }
                    return ContextBuilder.enrich_diagnosis_args(
                        d, observation, str(observation.get("goal") or "")
                    )
                approach = "generic_fix"
                fp = Observer.fingerprint_approach(approach)
                return {
                    "root_cause": err[:500],
                    "fault_layer": "skill_code",
                    "rewrite_skill": True,
                    "what_to_change": "fix skill",
                    "approach": approach,
                    "approach_fingerprint": fp,
                    "approach_changed": True,
                    "needs_research": False,
                    "research_queries": [],
                    "needs_new_deps": [],
                    "missing_args": [],
                    "required_args": [],
                    "suggested_args": {},
                    "test_plan": "retest",
                    "expected_artifacts": [],
                    "is_unfixable": False,
                    "diagnosis": err[:300],
                }

            def verify_claim(self, goal, result, evidence) -> dict:
                return {"achieved": True, "confidence": 0.9, "reason": "advisory"}

            def converse(self, user_text, history=None) -> str:
                return "ok"

        phases_d: list[str] = []
        logs_d: list[str] = []
        brain_d = DefaultTrapBrain()
        orch_d = Orchestrator(
            root=def_root,
            brain=brain_d,
            on_status=lambda s: phases_d.append(s),
            on_log=lambda m: logs_d.append(m),
        )
        goal_d = 'Create "user_note.txt" containing "REAL_PAYLOAD"'
        result_d = orch_d.run_cycle(goal_d)
        assert result_d.get("success"), result_d
        note = def_root / "workspace_runtime" / "user_note.txt"
        assert note.exists() and "REAL_PAYLOAD" in note.read_text(encoding="utf-8"), note
        assert not (def_root / "workspace_runtime" / "default_path.txt").exists() or (
            "REAL_PAYLOAD" in note.read_text(encoding="utf-8")
        )
        assert brain_d.build_count >= 2, brain_d.build_count
        assert "VERIFY" in phases_d and "OBSERVE" in phases_d and "DIAGNOSE" in phases_d
        assert any("reject_defaults" in m or "VERIFIER FAIL" in m or "default" in m.lower()
                   for m in logs_d), logs_d[-40:]
        orch_d.close()
        print("  OK defaults VERIFY FAIL → rewrite → USER REQUEST honored")
    except Exception as exc:
        msg = f"E2E_DEFAULTS: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # 4d) E2E: repeated VERIFY fail → CODING repair (research NOT universal fallback)
    print("  — e2e progress-aware repair (CODING, no research spam) —")
    try:
        from jarvis.observer import Observer as _Obs

        ar_root = root / "data" / "_e2e_adaptive_research"
        if ar_root.exists():
            shutil.rmtree(ar_root)
        ar_root.mkdir(parents=True)

        class CodingFirstRepairBrain(Brain):
            def __init__(self) -> None:
                super().__init__()
                self.build_count = 0
                self.research_notes_calls = 0
                self.gen_query_calls = 0
                self.diagnose_approaches: list[str] = []

            def model_status(self) -> str:
                return "ONLINE"

            def is_available(self) -> bool:
                return True

            def classify_intent(self, user_text: str) -> dict:
                return {
                    "intent": "task",
                    "goal": user_text,
                    "needs_capability": True,
                    "keywords": user_text.split()[:8],
                }

            def plan(self, goal: str, known_capabilities: list[str]) -> dict:
                return {
                    "steps": ["build skill that honors args"],
                    "can_reuse": [],
                    "missing": [goal],
                    "needs_research": False,  # CODING-first; no entry RESEARCH tax
                    "needs_new_skill": True,
                    "skill_name": "repeat_fail_skill",
                    "skill_description": goal,
                    "research_queries": [],
                    "args": {"dest": "adapt_out.txt", "body": "ADAPT_OK"},
                    "required_args": ["dest", "body"],
                }

            def extract_task_args(
                self, goal, skill_meta=None, prior_args=None, diagnosis=None
            ):
                return {"dest": "adapt_out.txt", "body": "ADAPT_OK"}

            def generate_research_queries(
                self, observation, diagnosis=None, failed_approaches=None
            ):
                self.gen_query_calls += 1
                return ["SHOULD_NOT_BE_CALLED_WITHOUT_KNOWLEDGE_GAP"]

            def research_notes(self, query: str, gathered: str) -> dict:
                self.research_notes_calls += 1
                return {
                    "approach": "should_not_research",
                    "libraries": [],
                    "key_apis": [],
                    "pitfalls": [],
                    "test_idea": "",
                    "repair_insight": "",
                }

            def write_skill_code(
                self,
                skill_name,
                description,
                research,
                previous_code=None,
                error_log=None,
                diagnosis=None,
                failed_approaches=None,
                test_plan=None,
            ) -> str:
                self.build_count += 1
                import json as _json
                desc_lit = _json.dumps(description or "")
                # First two builds: same VERIFY failure class (empty file)
                if self.build_count <= 2:
                    return f'''
from pathlib import Path
SKILL_META = {{
    "name": "{skill_name}",
    "description": {desc_lit},
    "capabilities": ["repeat_fail"],
    "dependencies": [],
    "version": {self.build_count},
    "required_args": ["dest", "body"],
}}
def run(context: dict) -> dict:
    args = context.get("args") or {{}}
    workspace = Path(context.get("workspace") or ".")
    dest = str(args.get("dest") or "adapt_out.txt")
    path = workspace / dest
    path.write_text("", encoding="utf-8")
    return {{
        "ok": True,
        "result": {{"path": str(path), "contains": ""}},
        "error": None,
        "evidence": f"touched {{path}}",
    }}
'''
                # After stagnation forces new CODING approach: honor args
                return f'''
from pathlib import Path
SKILL_META = {{
    "name": "{skill_name}",
    "description": {desc_lit},
    "capabilities": ["repeat_fail"],
    "dependencies": [],
    "version": {self.build_count},
    "required_args": ["dest", "body"],
}}
def run(context: dict) -> dict:
    args = context.get("args") or {{}}
    workspace = Path(context.get("workspace") or ".")
    path = workspace / str(args["dest"])
    path.write_text(str(args["body"]) + "\\n", encoding="utf-8")
    return {{
        "ok": True,
        "result": {{"path": str(path), "contains": str(args["body"])}},
        "error": None,
        "evidence": f"wrote {{path}}",
    }}
'''

            def diagnose(self, observation, failed_approaches=None, prior_solutions=None):
                err = str(observation.get("exception") or "")
                approach = "empty_touch_coding_repair"
                fps = {
                    a.get("approach_fingerprint")
                    for a in (failed_approaches or [])
                }
                fp = _Obs.fingerprint_approach(approach)
                if fp in fps:
                    approach = approach + f"_v{len(fps)+1}"
                    fp = _Obs.fingerprint_approach(approach)
                self.diagnose_approaches.append(approach)
                return {
                    "root_cause": "file created empty; content missing for VERIFY",
                    "fault_layer": "skill_code",
                    "rewrite_skill": True,
                    "what_to_change": "Write body from args into dest",
                    "approach": approach,
                    "approach_fingerprint": fp,
                    "approach_changed": True,
                    "needs_research": False,
                    "missing_knowledge": [],
                    "research_queries": [],
                    "needs_new_deps": [],
                    "missing_args": [],
                    "required_args": ["dest", "body"],
                    "suggested_args": {
                        "dest": "adapt_out.txt",
                        "body": "ADAPT_OK",
                    },
                    "test_plan": "VERIFY dest contains body",
                    "expected_artifacts": ["adapt_out.txt"],
                    "is_unfixable": False,
                    "diagnosis": err[:300],
                }

            def verify_claim(self, goal, result, evidence) -> dict:
                return {"achieved": True, "confidence": 0.9, "reason": "advisory"}

            def converse(self, user_text, history=None) -> str:
                return "ok"

        phases_r: list[str] = []
        logs_r: list[str] = []
        brain_r = CodingFirstRepairBrain()
        orch_r = Orchestrator(
            root=ar_root,
            brain=brain_r,
            on_status=lambda s: phases_r.append(s),
            on_log=lambda m: logs_r.append(m),
        )
        goal_r = 'Create "adapt_out.txt" containing "ADAPT_OK"'
        result_r = orch_r.run_cycle(goal_r)
        assert result_r.get("success"), result_r
        out_r = ar_root / "workspace_runtime" / "adapt_out.txt"
        assert out_r.exists() and "ADAPT_OK" in out_r.read_text(encoding="utf-8")
        assert brain_r.build_count >= 3, brain_r.build_count
        # Research must NOT be the universal fallback for repeated VERIFY fails
        assert brain_r.gen_query_calls == 0, brain_r.gen_query_calls
        assert not any("KNOWLEDGE-GAP RESEARCH" in m for m in logs_r), logs_r[-40:]
        assert not any("ADAPTIVE RESEARCH" in m for m in logs_r), logs_r[-40:]
        assert any("STAGNATION" in m or "CODING" in m for m in logs_r), logs_r[-50:]
        assert phases_r.count("RESEARCH") == 0, phases_r
        orch_r.close()
        print("  OK progress-aware CODING repair — no research spam on VERIFY repeat")
    except Exception as exc:
        msg = f"E2E_ADAPTIVE_RESEARCH: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # 4d2) E2E: identical verifier failure must NOT repeat an identical cycle
    print("  — e2e stagnation: identical VERIFY FAIL ≠ identical research cycle —")
    try:
        from jarvis.observer import Observer as _Obs2
        from jarvis.recovery import ProgressAwareRecovery

        st_root = root / "data" / "_e2e_stagnation"
        if st_root.exists():
            shutil.rmtree(st_root)
        st_root.mkdir(parents=True)

        class StickyVerifyFailBrain(Brain):
            """Always emits the same broken skill → identical VERIFY failure."""

            def __init__(self) -> None:
                super().__init__()
                self.build_count = 0
                self.research_notes_calls = 0
                self.approaches: list[str] = []

            def model_status(self) -> str:
                return "ONLINE"

            def is_available(self) -> bool:
                return True

            def classify_intent(self, user_text: str) -> dict:
                return {
                    "intent": "task",
                    "goal": user_text,
                    "needs_capability": True,
                    "keywords": ["create", "file"],
                }

            def plan(self, goal: str, known_capabilities: list[str]) -> dict:
                return {
                    "steps": ["build"],
                    "can_reuse": [],
                    "missing": [goal],
                    "needs_research": False,
                    "needs_new_skill": True,
                    "skill_name": "sticky_fail_skill",
                    "skill_description": goal,
                    "research_queries": [],
                    "args": {"dest": "sticky.txt", "body": "STICKY_OK"},
                    "required_args": ["dest", "body"],
                }

            def extract_task_args(
                self, goal, skill_meta=None, prior_args=None, diagnosis=None
            ):
                return {"dest": "sticky.txt", "body": "STICKY_OK"}

            def research_notes(self, query: str, gathered: str) -> dict:
                self.research_notes_calls += 1
                return {
                    "approach": "identical_research_cycle",
                    "libraries": [],
                    "key_apis": [],
                    "pitfalls": [],
                    "test_idea": "",
                    "repair_insight": "noop",
                }

            def generate_research_queries(
                self, observation, diagnosis=None, failed_approaches=None
            ):
                return ["identical_query_should_not_loop"]

            def write_skill_code(
                self,
                skill_name,
                description,
                research,
                previous_code=None,
                error_log=None,
                diagnosis=None,
                failed_approaches=None,
                test_plan=None,
            ) -> str:
                self.build_count += 1
                import json as _json
                desc_lit = _json.dumps(description or "")
                # Identical broken body every time (same failure_signature)
                return f'''
from pathlib import Path
SKILL_META = {{
    "name": "{skill_name}",
    "description": {desc_lit},
    "capabilities": ["sticky"],
    "dependencies": [],
    "version": {self.build_count},
    "required_args": ["dest", "body"],
}}
def run(context: dict) -> dict:
    args = context.get("args") or {{}}
    workspace = Path(context.get("workspace") or ".")
    path = workspace / str(args.get("dest") or "sticky.txt")
    path.write_text("", encoding="utf-8")
    return {{
        "ok": True,
        "result": {{"path": str(path), "contains": ""}},
        "error": None,
        "evidence": "empty",
    }}
'''

            def diagnose(self, observation, failed_approaches=None, prior_solutions=None):
                # Intentionally sticky approach label — recovery must diverge it
                approach = "always_same_approach"
                self.approaches.append(approach)
                return {
                    "root_cause": "identical empty artifact every time",
                    "fault_layer": "skill_code",
                    "rewrite_skill": True,
                    "what_to_change": "write body",
                    "approach": approach,
                    "approach_fingerprint": _Obs2.fingerprint_approach(approach),
                    "approach_changed": False,
                    "needs_research": True,  # must be ignored without missing_knowledge
                    "missing_knowledge": [],
                    "research_queries": ["identical_query_should_not_loop"],
                    "needs_new_deps": [],
                    "missing_args": [],
                    "required_args": ["dest", "body"],
                    "suggested_args": {"dest": "sticky.txt", "body": "STICKY_OK"},
                    "test_plan": "VERIFY",
                    "expected_artifacts": [],
                    "is_unfixable": False,
                    "diagnosis": "sticky",
                }

            def verify_claim(self, goal, result, evidence) -> dict:
                return {"achieved": False, "confidence": 0.1, "reason": "no"}

            def converse(self, user_text, history=None) -> str:
                return "ok"

        logs_s: list[str] = []
        phases_s: list[str] = []
        brain_s = StickyVerifyFailBrain()
        orch_s = Orchestrator(
            root=st_root,
            brain=brain_s,
            on_log=lambda m: logs_s.append(m),
            on_status=lambda s: phases_s.append(s),
        )
        result_s = orch_s.run_cycle('Create "sticky.txt" containing "STICKY_OK"')
        assert not result_s.get("success"), result_s
        # Must stop with unresolved report — not spin forever
        assert any("RECOVERY STOP" in m or "RECOVERY BUDGET STOP" in m for m in logs_s), logs_s[-60:]
        assert any("STAGNATION" in m for m in logs_s), logs_s[-60:]
        # needs_research without missing_knowledge must NOT trigger research loop
        assert brain_s.research_notes_calls == 0, brain_s.research_notes_calls
        assert phases_s.count("RESEARCH") == 0, phases_s
        assert not any("KNOWLEDGE-GAP RESEARCH" in m for m in logs_s)
        assert not any("ADAPTIVE RESEARCH" in m for m in logs_s)
        # Approaches must diverge after stagnation (not identical cycle)
        diag_approaches = [
            (d.get("diagnosis") or {}).get("approach")
            if isinstance(d.get("diagnosis"), dict)
            else d.get("approach")
            for d in orch_s.memory.recent_diagnoses("sticky_fail_skill", 10)
        ]
        diag_approaches = [a for a in diag_approaches if a]
        assert any("coding_diverge" in str(a) for a in diag_approaches), diag_approaches
        # Failure signatures persisted
        fails = orch_s.memory.recent_failures("sticky_fail_skill", 10)
        assert fails
        assert any(
            (f.get("failure_signature") or (f.get("observation") or {}).get("failure_signature"))
            for f in fails
        )
        # Unit: ProgressAwareRecovery itself rejects identical no-progress loops
        tracker = ProgressAwareRecovery(max_attempts=4, stagnation_limit=2, max_research=1)
        sig = "abc123sig"
        for i in range(3):
            tracker.record(
                fault_layer="skill_code",
                failure_signature=sig,
                attempted_solution="same",
                model_used="test",
                result="VERIFY FAIL same",
                artifacts=[{"path": "x", "exists": True, "size": 0}],
                verifier_failure={"reason": "contains missing"},
                code_fingerprint="deadbeef",
                approach_fingerprint="samefp",
            )
            dec = tracker.evaluate(
                failure_signature=sig,
                fault_layer="skill_code",
                diagnosis_needs_research=True,
                missing_knowledge=[],
            )
        assert dec.stagnated
        assert not dec.allow_research  # no knowledge gap
        assert dec.prefer_coding
        orch_s.close()
        print("  OK stagnation — identical VERIFY FAIL does not repeat research cycle")
    except Exception as exc:
        msg = f"E2E_STAGNATION: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # 4e) E2E: brain stub → knowledge synthesis; research knowledge persisted
    print("  — e2e research→knowledge→code (no unimplemented stub) + save —")
    try:
        from jarvis.skill_builder import SkillBuilder as _SB

        kn_root = root / "data" / "_e2e_knowledge"
        if kn_root.exists():
            shutil.rmtree(kn_root)
        kn_root.mkdir(parents=True)

        class StubThenUselessBrain(Brain):
            """Always returns unimplemented stub — builder must synthesize."""

            def __init__(self) -> None:
                super().__init__()
                self.write_calls = 0
                self.research_notes_calls = 0

            def model_status(self) -> str:
                return "ONLINE"

            def is_available(self) -> bool:
                return True

            def classify_intent(self, user_text: str) -> dict:
                return {
                    "intent": "task",
                    "goal": user_text,
                    "needs_capability": True,
                    "keywords": user_text.split()[:8],
                }

            def plan(self, goal: str, known_capabilities: list[str]) -> dict:
                return {
                    "steps": ["synthesize skill from research"],
                    "can_reuse": [],
                    "missing": [goal],
                    "needs_research": True,
                    "needs_new_skill": True,
                    "skill_name": "knowledge_skill",
                    "skill_description": goal,
                    "research_queries": [goal, "python pathlib write file"],
                    "args": {"dest": "know_out.txt", "body": "KNOW_OK"},
                    "required_args": ["dest", "body"],
                }

            def extract_task_args(
                self, goal, skill_meta=None, prior_args=None, diagnosis=None
            ):
                return {"dest": "know_out.txt", "body": "KNOW_OK"}

            def research_notes(self, query: str, gathered: str) -> dict:
                self.research_notes_calls += 1
                return {
                    "approach": "pathlib_write_from_args",
                    "libraries": [],
                    "key_apis": ["pathlib", "write_text"],
                    "pitfalls": ["never ship unimplemented stub"],
                    "test_idea": "verifier checks dest+body",
                    "repair_insight": "write body into dest under workspace",
                }

            def write_skill_code(self, *a, **kw) -> str:
                self.write_calls += 1
                # Classic failure mode from the screenshot
                return '''
SKILL_META = {"name": "knowledge_skill", "description": "x", "capabilities": [],
 "dependencies": [], "version": 1}
def run(context: dict) -> dict:
    return {
        "ok": False,
        "result": None,
        "error": "Skill body not implemented yet",
        "evidence": "TODO",
    }
'''

            def diagnose(self, observation, failed_approaches=None, prior_solutions=None):
                from jarvis.observer import Observer
                approach = "force_knowledge_synthesis"
                fp = Observer.fingerprint_approach(approach)
                return {
                    "root_cause": "unimplemented stub instead of researched code",
                    "fault_layer": "skill_code",
                    "rewrite_skill": True,
                    "what_to_change": "Turn research knowledge into working run()",
                    "approach": approach,
                    "approach_fingerprint": fp,
                    "approach_changed": True,
                    "needs_research": True,
                    "research_queries": ["python pathlib write file from dict args"],
                    "needs_new_deps": [],
                    "missing_args": [],
                    "required_args": ["dest", "body"],
                    "suggested_args": {"dest": "know_out.txt", "body": "KNOW_OK"},
                    "test_plan": "VERIFY know_out.txt contains KNOW_OK",
                    "expected_artifacts": ["know_out.txt"],
                    "is_unfixable": False,
                    "diagnosis": str(observation.get("exception") or "")[:300],
                }

            def verify_claim(self, goal, result, evidence) -> dict:
                return {"achieved": True, "confidence": 0.9, "reason": "advisory"}

            def converse(self, user_text, history=None) -> str:
                return "ok"

        logs_k: list[str] = []
        brain_k = StubThenUselessBrain()
        orch_k = Orchestrator(
            root=kn_root,
            brain=brain_k,
            on_log=lambda m: logs_k.append(m),
        )
        goal_k = 'Create "know_out.txt" containing "KNOW_OK"'
        result_k = orch_k.run_cycle(goal_k)
        assert result_k.get("success"), result_k
        out_k = kn_root / "workspace_runtime" / "know_out.txt"
        assert out_k.exists() and "KNOW_OK" in out_k.read_text(encoding="utf-8"), out_k
        # Brain returned stub, but builder must not ship it
        assert brain_k.write_calls >= 1
        assert any("knowledge_synthesis" in m or "synthesizing from research" in m
                   for m in logs_k), logs_k[-40:]
        skill_code = Path(orch_k.registry.get_skill("knowledge_skill")["file_path"]).read_text(
            encoding="utf-8"
        )
        assert not _SB.is_unimplemented_stub(skill_code)
        assert "Skill body not implemented yet" not in skill_code
        # Knowledge must be saved and reloadable
        hist = orch_k.memory.get_research_knowledge("knowledge_skill")
        assert hist, hist
        assert orch_k.memory.get_latest_research("knowledge_skill")
        assert any("KNOWLEDGE SAVED" in m for m in logs_k), logs_k[-30:]
        orch_k.close()
        print("  OK knowledge stub→synthesis + research knowledge persisted")
    except Exception as exc:
        msg = f"E2E_KNOWLEDGE: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # 4f) E2E: learning request must NOT inherit prior skill (e.g. create_file)
    print("  — e2e learning intent isolated from prior ACTIVE skills —")
    try:
        from jarvis.intent import IntentClassifier

        # Unit: universal classification (no skill-name hardcoding)
        assert IntentClassifier.classify_offline(
            "Learn algorithms and create practical tests to verify knowledge"
        )["intent"] == "learning"
        assert IntentClassifier.classify_offline(
            "Iemācies tēmu un izveido praktiskus testus zināšanu pārbaudei"
        )["intent"] == "learning"
        assert IntentClassifier.classify_offline(
            'Create file notes.txt containing "hello"'
        )["intent"] == "task"
        assert IntentClassifier.classify_offline("What is recursion?")["intent"] == "conversation"
        assert IntentClassifier.normalize(
            {"intent": "task", "needs_capability": True},
            "Research how caching works",
        )["intent"] == "learning"

        learn_root = root / "data" / "_e2e_learning_intent"
        if learn_root.exists():
            shutil.rmtree(learn_root)
        learn_root.mkdir(parents=True)

        class OfflineBrain(Brain):
            def model_status(self) -> str:
                return "OFFLINE"

            def is_available(self) -> bool:
                return False

        logs_l: list[str] = []
        phases_l: list[str] = []
        orch_l = Orchestrator(
            root=learn_root,
            brain=OfflineBrain(),
            on_log=lambda m: logs_l.append(m),
            on_status=lambda s: phases_l.append(s),
        )
        # Tempt inheritance: ACTIVE unrelated skill already present
        bait = learn_root / "skills" / "create_file.py"
        bait.parent.mkdir(parents=True, exist_ok=True)
        bait.write_text(
            "SKILL_META={'name':'create_file','description':'create file',"
            "'capabilities':['create file'],'dependencies':[],'version':27}\n"
            "def run(context):\n"
            "    return {'ok': True, 'result': {}, 'error': None, 'evidence': 'x'}\n",
            encoding="utf-8",
        )
        orch_l.registry.register_candidate(
            "create_file",
            "create file",
            str(bait),
            ["create file"],
            version=27,
            protect_active=False,
        )
        orch_l.registry.set_status("create_file", "ACTIVE")
        assert orch_l.registry.get_skill("create_file")["status"] == "ACTIVE"

        req_l = "Learn network basics and create practical tests to verify knowledge"
        result_l = orch_l.handle_user_message(req_l)
        assert result_l.get("type") == "learning", result_l
        assert result_l.get("success"), result_l
        assert result_l.get("topic"), result_l
        assert "BUILD_SKILL" not in phases_l, phases_l
        assert "REPAIR" not in phases_l, phases_l
        assert not any("generating skill" in m for m in logs_l), logs_l[-30:]
        assert not any("KNOWLEDGE SAVED: create_file" in m for m in logs_l), logs_l[-30:]
        assert any("topic:" in m or "LEARNING" in m for m in logs_l), logs_l[-30:]
        # Topic knowledge saved; skill-bound research for create_file untouched
        assert orch_l.memory.get_topic_knowledge(result_l["topic"])
        assert not orch_l.memory.get_research_knowledge("create_file")
        # create_file must remain untouched at same version
        assert orch_l.registry.get_skill("create_file")["version"] == 27
        orch_l.close()
        print("  OK learning isolated — no skill inherit / no BUILD_SKILL")
    except Exception as exc:
        msg = f"E2E_LEARNING_INTENT: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # 4g) E2E: learning self-correct — VERIFY fail → OBSERVE → new research → PASS
    print("  — e2e learning self-correct (OBSERVE→research→VERIFY loop) —")
    try:
        sc_root = root / "data" / "_e2e_learning_self_correct"
        if sc_root.exists():
            shutil.rmtree(sc_root)
        sc_root.mkdir(parents=True)

        class OfflineBrainSC(Brain):
            def model_status(self) -> str:
                return "OFFLINE"

            def is_available(self) -> bool:
                return False

        class SelfCorrectOrch(Orchestrator):
            """First enrich leaves thin notes so VERIFY fails once, then recovers."""

            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self._enrich_calls = 0

            def _enrich_learning_research(self, research, **kw):
                self._enrich_calls += 1
                if self._enrich_calls == 1:
                    thin = self._learning_strip_skill_defaults(dict(research))
                    thin["approach"] = ""
                    thin["key_apis"] = []
                    thin["pitfalls"] = []
                    thin["test_idea"] = (
                        "Execute skill.run and independently verify artifacts"
                    )
                    return thin
                return super()._enrich_learning_research(research, **kw)

        logs_sc: list[str] = []
        phases_sc: list[str] = []
        orch_sc = SelfCorrectOrch(
            root=sc_root,
            brain=OfflineBrainSC(),
            on_log=lambda m: logs_sc.append(m),
            on_status=lambda s: phases_sc.append(s),
        )
        req_sc = "Learn graph basics and create practical tests to verify knowledge"
        result_sc = orch_sc.handle_user_message(req_sc)
        assert result_sc.get("type") == "learning", result_sc
        assert result_sc.get("success"), result_sc
        assert (result_sc.get("attempts") or 0) >= 2, result_sc
        assert orch_sc._enrich_calls >= 2
        assert "OBSERVE" in phases_sc and "DIAGNOSE" in phases_sc
        assert phases_sc.count("RESEARCH") >= 2
        assert phases_sc.count("VERIFY") >= 2
        assert any("LEARNING VERIFY: FAIL" in m for m in logs_sc), logs_sc[-40:]
        assert any("LEARNING VERIFY: PASS" in m for m in logs_sc), logs_sc[-40:]
        assert any("LEARNING DIAGNOSE" in m for m in logs_sc), logs_sc[-40:]
        topic_sc = result_sc["topic"]
        failed = orch_sc.memory.get_failed_approaches(f"learning:{topic_sc}")
        assert failed, failed
        # Failed approach must not be repeated as the success path label
        failed_labels = {str(a.get("approach") or "") for a in failed}
        assert "initial_topic_research" in failed_labels
        verified_hist = [
            e for e in orch_sc.memory.get_topic_knowledge(topic_sc)
            if isinstance(e, dict) and e.get("verified")
        ]
        assert verified_hist, orch_sc.memory.get_topic_knowledge(topic_sc)

        # Reuse verified knowledge on a fresh orchestrator (same DB root)
        logs_ru: list[str] = []
        orch_ru = Orchestrator(
            root=sc_root,
            brain=OfflineBrainSC(),
            on_log=lambda m: logs_ru.append(m),
        )
        result_ru = orch_ru.handle_user_message(req_sc)
        assert result_ru.get("success"), result_ru
        assert (result_ru.get("attempts") or 1) == 1, result_ru
        assert any("Reusing" in m and "verified" in m for m in logs_ru), logs_ru[:30]
        orch_sc.close()
        orch_ru.close()
        print("  OK learning self-correct FAIL→OBSERVE→RESEARCH→PASS + reuse")
    except Exception as exc:
        msg = f"E2E_LEARNING_SELF_CORRECT: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # 4h) E2E: LEARNING VERIFY is semantic (no token hits) + practical result
    print("  — e2e learning semantic VERIFY + practical result —")
    try:
        from jarvis import learning_verify as lv

        # Unit: paraphrase covers goal without exact keyword equality
        paraphrase_knowledge = {
            "approach": (
                "Study how devices exchange data: addressing, routing, and "
                "protocols that move packets across links on a network."
            ),
            "key_apis": ["addressing models", "routing basics", "protocol layers"],
            "pitfalls": ["confusing local and wide links"],
            "test_idea": "Self-check: explain packet delivery in your own words.",
            "raw": "Notes about communication between computers.",
            "sources": [{"title": "local"}],
            "results": [{"title": "x"}],
        }
        j_ok = lv.offline_semantic_judgment(
            "Learn networking fundamentals", paraphrase_knowledge
        )
        assert j_ok["covers_goal"], j_ok
        j_bad = lv.offline_semantic_judgment(
            "Learn networking fundamentals",
            {
                **paraphrase_knowledge,
                "approach": "Bake sourdough bread with long fermentation.",
                "key_apis": ["flour hydration", "starter feeding"],
                "raw": "bakery notes",
            },
        )
        assert not j_bad["covers_goal"], j_bad

        # Unit: practical ask produces + verifies a concrete result (any expression)
        expr_req = "Learn arithmetic and compute 12+5"
        assert lv.requests_practical_result(expr_req)
        produced = lv.produce_practical_offline(expr_req, paraphrase_knowledge)
        assert produced.get("result") == 17, produced
        assert lv.verify_practical_offline(expr_req, produced)["ok"]

        sem_root = root / "data" / "_e2e_learning_semantic"
        if sem_root.exists():
            shutil.rmtree(sem_root)
        sem_root.mkdir(parents=True)

        class OfflineBrainSem(Brain):
            def model_status(self) -> str:
                return "OFFLINE"

            def is_available(self) -> bool:
                return False

        logs_sem: list[str] = []
        orch_sem = Orchestrator(
            root=sem_root,
            brain=OfflineBrainSem(),
            on_log=lambda m: logs_sem.append(m),
        )
        # Universal practical learning request — expression not hardcoded in product code
        req_sem = "Learn arithmetic basics and compute 12+5"
        result_sem = orch_sem.handle_user_message(req_sem)
        assert result_sem.get("type") == "learning", result_sem
        assert result_sem.get("success"), result_sem
        ver = result_sem.get("verification") or {}
        check_names = [c.get("name") for c in (ver.get("checks") or [])]
        assert "semantic_goal_coverage" in check_names, check_names
        assert "practical_result" in check_names, check_names
        assert not any("token_hits" in str(c) for c in (ver.get("checks") or []))
        assert (ver.get("practical_result") or {}).get("result") == 17, ver
        assert "Practical result" in (result_sem.get("reply") or "")
        orch_sem.close()
        print("  OK learning semantic VERIFY + practical result checked")
    except Exception as exc:
        msg = f"E2E_LEARNING_SEMANTIC: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # 4i) E2E: each USER REQUEST isolated — conversation ≠ prior learning final_result
    print("  — e2e request routing isolation (conversation ≠ prior DONE) —")
    try:
        from jarvis.intent import IntentClassifier
        from jarvis.memory import Memory

        route_root = root / "data" / "_e2e_request_routing"
        if route_root.exists():
            shutil.rmtree(route_root)
        route_root.mkdir(parents=True)

        prior_done = (
            "DONE. Learning complete — knowledge verified & saved.\n"
            "Topic: sample_topic\n"
            "Attempts: 1\n"
            "Approach: initial_topic_research\n"
            "Summary: prior learning final_result that must not leak\n"
            "Sources: 0"
        )

        class HistoryLeakBrain(Brain):
            """Simulates the bug: if prior DONE is in history, echo it as the reply."""

            def model_status(self) -> str:
                return "ONLINE"

            def is_available(self) -> bool:
                return True

            def classify_intent(self, text: str) -> dict:
                return IntentClassifier.classify_offline(text)

            def converse(self, user_text: str, history=None) -> str:
                for m in history or []:
                    content = str((m or {}).get("content") or "")
                    if Memory._looks_like_cycle_outcome(content):
                        return content  # stale reuse of prior final_result
                return f"Fresh conversational answer about: {user_text}"

        orch_r = Orchestrator(
            root=route_root,
            brain=HistoryLeakBrain(),
            on_log=lambda m: None,
            on_status=lambda s: None,
        )
        # Plant a completed learning turn in conversation memory
        orch_r.memory.add_message(
            "user",
            "Learn sample topic thoroughly",
            meta={"request_id": "oldreq01", "intent": "learning", "phase": "received"},
        )
        orch_r.memory.add_message(
            "assistant",
            prior_done,
            meta={"request_id": "oldreq01", "intent": "learning", "task_id": 99},
        )
        assert orch_r.memory.last_cycle_assistant_reply() == prior_done
        # Filtered history must NOT include the learning DONE pair
        hist = orch_r.memory.conversation_history_for_llm(8)
        assert not any(
            Memory._looks_like_cycle_outcome(m.get("content") or "") for m in hist
        ), hist
        assert not any("sample topic" in (m.get("content") or "").lower() for m in hist), hist

        chat_q = "What is photosynthesis in simple terms?"
        assert IntentClassifier.classify_offline(chat_q)["intent"] == "conversation"
        result_c = orch_r.handle_user_message(chat_q)
        assert result_c.get("type") == "conversation", result_c
        assert result_c.get("request_id"), result_c
        assert result_c.get("request_id") != "oldreq01"
        reply_c = result_c.get("reply") or ""
        assert reply_c.strip() != prior_done.strip(), reply_c
        assert "DONE. Learning complete" not in reply_c, reply_c
        assert "prior learning final_result" not in reply_c, reply_c
        assert "photosynthesis" in reply_c.lower() or "Fresh conversational" in reply_c, reply_c
        # Second request still isolated — sticky stale brain path
        orch_r.close()

        class StickyStaleBrain(Brain):
            """Always returns prior DONE — _ensure_fresh_reply must reject it."""

            def model_status(self) -> str:
                return "ONLINE"

            def is_available(self) -> bool:
                return True

            def classify_intent(self, text: str) -> dict:
                return IntentClassifier.classify_offline(text)

            def converse(self, user_text: str, history=None) -> str:
                return prior_done

        orch_s = Orchestrator(
            root=route_root,
            brain=StickyStaleBrain(),
            on_log=lambda m: None,
            on_status=lambda s: None,
        )
        orch_s.memory.add_message(
            "assistant",
            prior_done,
            meta={"request_id": "oldreq02", "intent": "learning", "task_id": 100},
        )
        chat_q2 = "How are you today?"
        result_s = orch_s.handle_user_message(chat_q2)
        assert result_s.get("type") == "conversation", result_s
        assert result_s.get("request_id")
        reply_s = result_s.get("reply") or ""
        assert reply_s.strip() != prior_done.strip(), reply_s
        assert "DONE. Learning complete" not in reply_s, reply_s
        assert chat_q2[:20] in reply_s or "jautājumu" in reply_s.lower() or "How are you" in reply_s
        # Distinct request_ids across turns
        assert result_c.get("request_id") != result_s.get("request_id")
        orch_s.close()
        print("  OK request routing isolation — conversation ≠ prior learning DONE")
    except Exception as exc:
        msg = f"E2E_REQUEST_ROUTING: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # 4j) E2E: Multi-Model Router — work kind → tier → model + fallback + escalate
    print("  — e2e multi-model router (FAST/REASONING/CODING) —")
    try:
        from jarvis.model_config import (
            TIER_MODELS,
            TIER_FAST,
            TIER_REASONING,
            TIER_CODING,
            tier_for_work,
            models_for_tier,
        )
        from jarvis.model_router import ModelRouter

        assert tier_for_work("intent") == TIER_FAST
        assert tier_for_work("extract_args") == TIER_FAST
        assert tier_for_work("query_generation") == TIER_FAST
        assert tier_for_work("converse") == TIER_FAST
        assert tier_for_work("plan") == TIER_REASONING
        assert tier_for_work("learning") == TIER_REASONING
        assert tier_for_work("research") == TIER_REASONING
        assert tier_for_work("diagnose") == TIER_REASONING
        assert tier_for_work("semantic") == TIER_REASONING
        assert tier_for_work("skill_code") == TIER_CODING
        assert tier_for_work("code_repair") == TIER_CODING

        assert TIER_MODELS[TIER_FAST]["primary"] == "qwen3:4b"
        assert TIER_MODELS[TIER_REASONING]["primary"] == "qwen3:30b"
        assert TIER_MODELS[TIER_CODING]["primary"] == "qwen3-coder:30b"
        # qwen2.5 kept as fallback
        assert "qwen2.5-coder:7b" in models_for_tier(TIER_CODING)
        assert "qwen2.5-coder:7b" in models_for_tier(TIER_FAST)

        route_logs: list[str] = []

        # Only legacy coding model installed → FAST/REASONING/CODING all fallback
        installed = ["qwen2.5-coder:7b"]

        def _list() -> list[str]:
            return list(installed)

        router = ModelRouter(list_models=_list, on_log=lambda m: route_logs.append(m))
        d_fast = router.route(work="intent")
        assert d_fast.tier == TIER_FAST
        assert d_fast.model == "qwen2.5-coder:7b"
        assert d_fast.used_fallback is True
        assert any(m.startswith("MODEL ROUTE: FAST →") for m in route_logs), route_logs
        assert any("MODEL FALLBACK:" in m for m in route_logs), route_logs

        route_logs.clear()
        d_code = router.route(work="skill_code")
        assert d_code.tier == TIER_CODING
        assert d_code.model == "qwen2.5-coder:7b"
        assert any(m.startswith("MODEL ROUTE: CODING →") for m in route_logs), route_logs

        # Primary available — no fallback
        installed = ["qwen3:4b", "qwen3:30b", "qwen3-coder:30b", "qwen2.5-coder:7b"]
        router.invalidate_cache()
        route_logs.clear()
        d_r = router.route(work="plan")
        assert d_r.tier == TIER_REASONING
        assert d_r.model == "qwen3:30b"
        assert d_r.used_fallback is False
        assert any(m == "MODEL ROUTE: REASONING → qwen3:30b" for m in route_logs), route_logs

        route_logs.clear()
        esc = router.escalate(TIER_FAST, work="intent")
        assert esc is not None
        assert esc.tier == TIER_REASONING
        assert any(m == "MODEL ESCALATION: FAST → REASONING" for m in route_logs), route_logs

        route_logs.clear()
        retry = router.escalate(TIER_CODING, work="code_repair")
        assert retry is not None
        assert retry.tier == TIER_CODING
        assert any("MODEL RETRY: CODING" in m for m in route_logs), route_logs

        assert ModelRouter.result_insufficient("")
        assert ModelRouter.result_insufficient("[BRAIN ERROR] boom")
        assert ModelRouter.result_insufficient("sorry no", expect_json=True)
        assert not ModelRouter.result_insufficient('{"ok": true}', expect_json=True)

        # Brain.chat uses router + escalates FAST → REASONING on empty
        calls: list[str] = []

        class RouterProbeBrain(Brain):
            def _list_model_names(self):
                return ["qwen3:4b", "qwen3:30b", "qwen2.5-coder:7b"]

            def _chat_on_model(self, messages, temperature, model):
                calls.append(model)
                # FAST primary returns empty → escalate to REASONING
                if model == "qwen3:4b":
                    return ""
                return '{"intent":"conversation","goal":"hi","needs_capability":false}'

        probe_logs: list[str] = []
        probe = RouterProbeBrain(on_log=lambda m: probe_logs.append(m))
        out = probe.generate(
            "hi",
            system="Reply JSON",
            work="intent",
            allow_escalate=True,
            expect_json=True,
        )
        assert "qwen3:4b" in calls and "qwen3:30b" in calls, calls
        assert "intent" in out or "conversation" in out
        assert any("MODEL ESCALATION: FAST → REASONING" in m for m in probe_logs), probe_logs
        assert any("MODEL ROUTE: FAST → qwen3:4b" in m for m in probe_logs), probe_logs

        print("  OK multi-model router — route/fallback/escalate + work→tier map")
    except Exception as exc:
        msg = f"E2E_MULTI_MODEL_ROUTER: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # 5) Orchestrator boots + conversation logs must not echo USER/JARVIS replies
    try:
        echoed: list[str] = []
        orch = Orchestrator(
            root=root,
            brain=Brain(),
            on_log=lambda m: echoed.append(m),
        )
        stats = orch.get_dashboard_stats()
        assert "brain_status" in stats
        r = orch.handle_user_message("/status")
        assert "reply" in r
        # /status reply is returned to UI — must not also be pushed via on_log
        status_reply = r.get("reply") or ""
        assert not any(status_reply and status_reply in m for m in echoed), echoed
        assert not any(m.startswith("USER:") for m in echoed), echoed
        assert not any(m.startswith("JARVIS:") for m in echoed), echoed
        orch.close()
        print("  OK orch     boot + /status (no USER/JARVIS log echo)")
    except Exception as exc:
        msg = f"ORCH: {exc}"
        print(f"  FAIL {msg}")
        errors.append(msg)

    print("===", "PASS" if not errors else f"FAIL ({len(errors)})", "===")
    return 1 if errors else 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="JARVIS autonomous agent")
    parser.add_argument("--cli", action="store_true", help="Run in terminal mode")
    parser.add_argument("--check", action="store_true", help="Run self-check and exit")
    parser.add_argument(
        "--root",
        type=str,
        default=str(ROOT),
        help="Project root (skills/, data/)",
    )
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(exist_ok=True)
    (root / "skills").mkdir(exist_ok=True)

    if args.check:
        return run_check(root)
    if args.cli:
        return run_cli(root)

    # GUI default
    if tk is None:
        print("tkinter not installed; falling back to CLI", file=sys.stderr)
        print("Install with: sudo apt-get install python3-tk", file=sys.stderr)
        return run_cli(root)
    try:
        app = JarvisGUI(root)
        app.run()
        return 0
    except Exception as exc:
        # TclError or display missing
        print(f"GUI unavailable ({exc}); falling back to CLI", file=sys.stderr)
        return run_cli(root)


if __name__ == "__main__":
    raise SystemExit(main())
