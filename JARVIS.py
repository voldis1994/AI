#!/usr/bin/env python3
"""
JARVIS — Autonomous self-learning AI agent.

Main entry point. Brain: Ollama (qwen2.5-coder:7b).
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

        self.win = tk.Tk()
        self.win.title("JARVIS")
        self.win.configure(bg=BG)
        self.win.geometry("980x700")
        self.win.minsize(720, 520)

        self._build_ui()
        self._bind_keys()

        self.orch = Orchestrator(
            root=root_dir,
            brain=Brain(model=DEFAULT_MODEL),
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
        self.entry.bind("<Return>", lambda _e: self._on_send())
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Boot / display helpers ──────────────────────────────────────────

    def _boot_banner(self) -> None:
        online = self.orch.brain.is_available()
        self._append(
            "system",
            "═" * 56 + "\n"
            "  JARVIS online. Core stable. Skills learn themselves.\n"
            f"  Brain: Ollama / {DEFAULT_MODEL} — "
            f"{'CONNECTED' if online else 'OFFLINE (start ollama serve)'}\n"
            "  Type a message, a task, or /help\n"
            "═" * 56,
        )
        self._ui_status("IDLE")
        self.brain_var.set(
            f"BRAIN: {'ONLINE' if online else 'OFFLINE'} · {DEFAULT_MODEL}"
        )

    def _append(self, tag: str, text: str) -> None:
        def _do() -> None:
            self.log.configure(state=tk.NORMAL)
            self.log.insert(tk.END, text.rstrip() + "\n\n", tag)
            self.log.see(tk.END)
            self.log.configure(state=tk.DISABLED)

        self.win.after(0, _do)

    def _ui_log(self, msg: str) -> None:
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
        brain = "ONLINE" if dash["brain_online"] else "OFFLINE"
        stats = (
            f"Brain   {brain}\n"
            f"Model   {dash['model']}\n"
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
        self.brain_var.set(f"BRAIN: {brain} · {dash['model']}")

    def _tick_stats(self) -> None:
        if not self.busy:
            self._refresh_stats()
        self.win.after(4000, self._tick_stats)

    # ── Interaction ─────────────────────────────────────────────────────

    def _on_send(self) -> None:
        if self.busy:
            return
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, tk.END)
        self._append("user", f"YOU › {text}")
        self.busy = True
        self.send_btn.configure(state=tk.DISABLED)
        self._ui_status("WORKING")

        def worker() -> None:
            try:
                result = self.orch.handle_user_message(text)
                reply = result.get("reply") or str(result)
                tag = "jarvis" if result.get("success", True) or result.get("type") == "conversation" else "error"
                if result.get("type") == "task" and not result.get("success"):
                    tag = "error"
                self._append(tag, f"JARVIS › {reply}")
            except Exception as exc:
                self._append("error", f"JARVIS › Internal error: {exc}")
            finally:
                def done() -> None:
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
        brain=Brain(model=DEFAULT_MODEL),
        on_log=on_log,
        on_status=on_status,
    )
    online = orch.brain.is_available()
    print(f"Brain: {'ONLINE' if online else 'OFFLINE'} ({DEFAULT_MODEL})")
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
    """Verify syntax, imports, and main execution paths without needing Ollama."""
    import ast
    import importlib

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
        "jarvis.memory",
        "jarvis.ledger",
        "jarvis.capability_registry",
        "jarvis.research",
        "jarvis.dependency_manager",
        "jarvis.skill_builder",
        "jarvis.skill_tester",
        "jarvis.verifier",
        "jarvis.skill_loader",
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

    # 3) Exercise core paths with a deterministic built-in skill (no Ollama required)
    print("  — exercising core cycle with fixture skill —")
    try:
        from jarvis.skill_builder import SkillBuilder
        from jarvis.capability_registry import CapabilityRegistry
        from jarvis.skill_tester import SkillTester
        from jarvis.verifier import Verifier
        from jarvis.memory import Memory
        from jarvis.ledger import Ledger
        from jarvis.dependency_manager import DependencyManager

        tmp = root / "data" / "_selfcheck"
        skills = tmp / "skills"
        ws = tmp / "ws"
        skills.mkdir(parents=True, exist_ok=True)
        ws.mkdir(parents=True, exist_ok=True)
        db = tmp / "check.db"
        if db.exists():
            db.unlink()

        # Write a real minimal skill by hand (simulates successful builder output)
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
        "result": {"path": str(path), "bytes": path.stat().st_size},
        "error": None,
        "evidence": f"Wrote {path} exists={path.exists()} size={path.stat().st_size}",
    }
'''
        skill_path = skills / "write_hello_file.py"
        skill_path.write_text(skill_code, encoding="utf-8")

        reg = CapabilityRegistry(db)
        mem = Memory(db)
        led = Ledger(db)
        reg.register_candidate(
            "write_hello_file",
            "Write hello.txt",
            str(skill_path),
            ["write hello file"],
            [],
            1,
        )
        reg.set_status("write_hello_file", "TESTING")
        tester = SkillTester(ws)
        test_res = tester.test(skill_path, goal="Create hello.txt in workspace")
        assert test_res["passed"], test_res
        ver = Verifier(ws)
        vres = ver.verify("Create hello.txt", test_res)
        assert vres["verified"], vres
        reg.set_status("write_hello_file", "ACTIVE")
        skill = reg.get_skill("write_hello_file")
        assert skill and skill["status"] == "ACTIVE"

        # Ledger cycle sketch
        tid = led.start_task("Create hello.txt")
        led.log(tid, "PLAN", "use write_hello_file")
        led.log(tid, "EXECUTE", "ok")
        led.log(tid, "VERIFY", "verified")
        led.finish(tid, True, {"path": str(ws / "hello.txt")})
        mem.save_experience("Create hello.txt", "DONE", True, "write_hello_file")
        mem.add_message("user", "hi")
        mem.add_message("assistant", "hello")

        # Builder syntax validation path
        builder = SkillBuilder(skills)
        bad = builder.build(
            "noop_placeholder",
            "placeholder",
            {"libraries": [], "key_apis": ["noop"], "approach": "n/a"},
            version=1,
        )
        assert bad["ok"] and "def run" in bad["code"]

        # Deps manager no-op
        dm = DependencyManager()
        assert dm.ensure([])["ok"]

        # Status transitions BROKEN → REPAIRING → TESTING → ACTIVE
        reg.set_status("write_hello_file", "BROKEN", "simulated")
        reg.set_status("write_hello_file", "REPAIRING")
        reg.set_status("write_hello_file", "TESTING")
        reg.set_status("write_hello_file", "ACTIVE")
        assert reg.get_skill("write_hello_file")["status"] == "ACTIVE"

        print("  OK cycle    CANDIDATE→TESTING→ACTIVE + VERIFY + ledger + memory")
        print("  OK repair   BROKEN→REPAIRING→TESTING→ACTIVE")
        print(f"  OK evidence {(ws / 'hello.txt').read_text().strip()!r}")

        reg.close()
        mem.close()
        led.close()
    except Exception as exc:
        import traceback
        msg = f"CYCLE: {exc}"
        print(f"  FAIL {msg}")
        traceback.print_exc()
        errors.append(msg)

    # 4) Orchestrator boots
    try:
        orch = Orchestrator(root=root, brain=Brain())
        stats = orch.get_dashboard_stats()
        assert "memory" in stats
        # Conversation path without requiring Ollama
        r = orch.handle_user_message("/status")
        assert "reply" in r
        orch.close()
        print("  OK orch     boot + /status")
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
