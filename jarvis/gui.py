"""
JARVIS cyber-HUD GUI — visual shell with integrated dynamic Workspace.

Views (same window, no extra processes): CHAT | WORKSPACE | CODE | TERMINAL |
RESEARCH | FILES. GUI only mirrors real backend events asynchronously.
"""

from __future__ import annotations

import queue
import re
import threading
from pathlib import Path
from typing import Any, Optional

try:
    import tkinter as tk
    from tkinter import font as tkfont
    from tkinter import scrolledtext
except ModuleNotFoundError:  # pragma: no cover
    tk = None  # type: ignore
    tkfont = None  # type: ignore
    scrolledtext = None  # type: ignore

from jarvis.brain import Brain
from jarvis.model_config import TIER_MODELS, TIER_FAST, TIER_REASONING, TIER_CODING
from jarvis.orchestrator import Orchestrator
from jarvis.workspace_events import (
    VIEWS,
    VIEW_CHAT,
    VIEW_CODE,
    VIEW_FILES,
    VIEW_RESEARCH,
    VIEW_TERMINAL,
    VIEW_WORKSPACE,
    view_for_status,
)

# ── Cyber HUD palette ───────────────────────────────────────────────────
BG = "#030106"
BG_DEEP = "#0a0410"
BG_PANEL = "#10051a"
BG_PANEL_2 = "#1a0a24"
BG_INPUT = "#0c0414"
FG = "#f5e6ff"
FG_DIM = "#c44dff"
FG_BRIGHT = "#ff7af0"
FG_MAGENTA = "#ff2bd6"
FG_GLOW = "#ff4de8"
FG_USER = "#5ec8ff"
FG_OK = "#39ff14"
FG_WARN = "#ffcc00"
FG_ERR = "#ff3355"
FG_TITLE = "#ff3dd4"
FG_KW = "#ff7af0"
FG_STR = "#7dffb3"
FG_CMT = "#6b145c"
FG_NUM = "#5ec8ff"
BORDER = "#ff2bd6"
BORDER_DIM = "#6b145c"

FONT_MONO = ("Consolas", 11)
FONT_TITLE = ("Consolas", 36, "bold")
FONT_SUB = ("Consolas", 9)
FONT_SMALL = ("Consolas", 9)
FONT_STAT = ("Consolas", 10)
FONT_CODE = ("Consolas", 10)

_PY_KW = {
    "and", "as", "assert", "async", "await", "break", "class", "continue",
    "def", "del", "elif", "else", "except", "False", "finally", "for",
    "from", "global", "if", "import", "in", "is", "lambda", "None",
    "nonlocal", "not", "or", "pass", "raise", "return", "True", "try",
    "while", "with", "yield",
}


def _mono() -> str:
    if tk is None or tkfont is None:
        return "Consolas"
    try:
        families = {f.lower() for f in tkfont.families()}
    except Exception:
        return "Consolas"
    for name in (
        "Consolas",
        "Cascadia Mono",
        "JetBrains Mono",
        "Fira Code",
        "Courier New",
        "DejaVu Sans Mono",
        "Monospace",
    ):
        if name.lower() in families or any(name.lower() in f for f in families):
            return name
    return "TkFixedFont"


def _panel(parent: tk.Misc, **pack_kw) -> tk.Frame:
    outer = tk.Frame(parent, bg=BORDER, highlightthickness=0, bd=0)
    if pack_kw:
        outer.pack(**pack_kw)
    inner = tk.Frame(outer, bg=BG_PANEL, highlightthickness=0, bd=0)
    inner.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
    outer.body = inner  # type: ignore[attr-defined]
    return outer


class JarvisGUI:
    """Magenta cyber-HUD with integrated dynamic Workspace views."""

    def __init__(self, root_dir: Path) -> None:
        if tk is None:
            raise RuntimeError("tkinter is required for GUI mode")
        self.root_dir = Path(root_dir)
        self.busy = False
        self._request_seq = 0
        self._dot_phase = 0
        self._fullscreen = True
        self._active_view = VIEW_CHAT
        self._manual_view = False
        self._event_q: queue.SimpleQueue = queue.SimpleQueue()
        self._code_path: Optional[Path] = None
        self._code_mtime: float = 0.0
        self._phase = "IDLE"
        self._research_lines: list[str] = []
        self._view_btns: dict[str, tk.Button] = {}

        self.win = tk.Tk()
        self.win.title("JARVIS")
        self.win.configure(bg=BG)
        self.win.minsize(1000, 680)

        family = _mono()
        global FONT_MONO, FONT_TITLE, FONT_SUB, FONT_SMALL, FONT_STAT, FONT_CODE
        FONT_MONO = (family, 11)
        FONT_TITLE = (family, 32, "bold")
        FONT_SUB = (family, 9)
        FONT_SMALL = (family, 9)
        FONT_STAT = (family, 10)
        FONT_CODE = (family, 10)

        self._build_ui()
        self._bind_keys()
        self._go_fullscreen()

        self.orch = Orchestrator(
            root=root_dir,
            brain=Brain(),
            on_log=self._ui_log,
            on_status=self._ui_status,
            on_event=self._ui_event,
        )
        self._refresh_stats()
        self._boot_banner()
        self._show_view(VIEW_CHAT, manual=False)
        self.win.after(80, self._animate_status_dots)
        self.win.after(50, self._drain_events)
        self.win.after(200, self._focus_entry)
        self.win.after(1200, self._poll_active_file)
        self.win.after(4000, self._tick_stats)

    # ── UI construction ─────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root_border = tk.Frame(self.win, bg=BORDER, bd=0)
        root_border.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        shell = tk.Frame(root_border, bg=BG, bd=0)
        shell.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.shell = shell

        # Header
        header = tk.Frame(shell, bg=BG)
        header.pack(fill=tk.X, padx=12, pady=(8, 2))
        tk.Label(header, text="JARVIS", fg=FG_TITLE, bg=BG, font=FONT_TITLE).pack()
        tk.Label(
            header,
            text="AUTONOMOUS  •  SELF-LEARNING  •  VERIFIED",
            fg=FG_DIM,
            bg=BG,
            font=FONT_SUB,
        ).pack()

        # Status strip
        status_outer = _panel(shell, fill=tk.X, padx=12, pady=(2, 4))
        sf = status_outer.body  # type: ignore[attr-defined]
        left_st = tk.Frame(sf, bg=BG_PANEL)
        left_st.pack(side=tk.LEFT, fill=tk.Y)
        self.status_var = tk.StringVar(value="STATUS: BOOT")
        self.status_lbl = tk.Label(
            left_st, textvariable=self.status_var, fg=FG_GLOW, bg=BG_PANEL,
            font=FONT_MONO, anchor="w",
        )
        self.status_lbl.pack(side=tk.LEFT, padx=(10, 0), pady=5)
        self._status_dots = tk.Label(
            left_st, text="", fg=FG_MAGENTA, bg=BG_PANEL, font=FONT_MONO
        )
        self._status_dots.pack(side=tk.LEFT)
        self.phase_var = tk.StringVar(value="")
        tk.Label(
            left_st, textvariable=self.phase_var, fg=FG_DIM, bg=BG_PANEL, font=FONT_SUB
        ).pack(side=tk.LEFT, padx=(12, 0))

        right_st = tk.Frame(sf, bg=BG_PANEL)
        right_st.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        self.brain_dot = tk.Label(
            right_st, text="●", fg=FG_OK, bg=BG_PANEL, font=FONT_MONO
        )
        self.brain_dot.pack(side=tk.LEFT, pady=5)
        self.brain_var = tk.StringVar(value="BRAIN: … | multi-model")
        self.brain_lbl = tk.Label(
            right_st, textvariable=self.brain_var, fg=FG_OK, bg=BG_PANEL,
            font=FONT_SMALL, anchor="e",
        )
        self.brain_lbl.pack(side=tk.LEFT, padx=(4, 0), pady=5)

        # Input bar (BOTTOM first)
        input_outer = tk.Frame(shell, bg=BORDER, height=58)
        input_outer.pack(fill=tk.X, padx=12, pady=(4, 10), side=tk.BOTTOM)
        input_outer.pack_propagate(False)
        input_frame = tk.Frame(input_outer, bg=BG_PANEL)
        input_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        tk.Label(
            input_frame, text="›", fg=FG_MAGENTA, bg=BG_PANEL,
            font=(FONT_MONO[0], 18, "bold"),
        ).pack(side=tk.LEFT, padx=(10, 6))
        self.entry = tk.Entry(
            input_frame, bg=BG_INPUT, fg=FG, insertbackground=FG_GLOW,
            font=FONT_MONO, relief=tk.FLAT, highlightbackground=BORDER_DIM,
            highlightcolor=FG_MAGENTA, highlightthickness=1,
        )
        self.entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=6, ipady=3)
        self.send_btn = tk.Button(
            input_frame, text="SEND", command=self._on_send, bg=BG_PANEL_2,
            fg=FG_GLOW, activebackground=FG_MAGENTA, activeforeground=BG,
            font=(FONT_MONO[0], 11, "bold"), relief=tk.FLAT, padx=18, pady=4,
            highlightbackground=BORDER, highlightthickness=2, cursor="hand2",
        )
        self.send_btn.pack(side=tk.LEFT, padx=(8, 10), pady=6)

        # Persistent TERMINAL strip above input
        term_outer = _panel(shell, fill=tk.X, padx=12, pady=(2, 2), side=tk.BOTTOM)
        term_body = term_outer.body  # type: ignore[attr-defined]
        th = tk.Frame(term_body, bg=BG_PANEL)
        th.pack(fill=tk.X, padx=4, pady=(2, 0))
        tk.Label(
            th, text="›  TERMINAL", fg=FG_BRIGHT, bg=BG_PANEL, font=FONT_SMALL
        ).pack(side=tk.LEFT)
        self.term_status = tk.Label(
            th, text="idle", fg=BORDER_DIM, bg=BG_PANEL, font=FONT_SUB
        )
        self.term_status.pack(side=tk.RIGHT, padx=6)
        self.term = scrolledtext.ScrolledText(
            term_body, bg=BG_DEEP, fg=FG, font=FONT_CODE, relief=tk.FLAT,
            height=7, wrap=tk.WORD, state=tk.DISABLED, highlightthickness=0,
            takefocus=0, padx=6, pady=4,
        )
        self.term.pack(fill=tk.X, expand=False, padx=4, pady=4)
        self.term.tag_configure("cmd", foreground=FG_MAGENTA)
        self.term.tag_configure("out", foreground=FG)
        self.term.tag_configure("err", foreground=FG_ERR)
        self.term.tag_configure("ok", foreground=FG_OK)
        self.term.tag_configure("dim", foreground=FG_DIM)

        # View tabs
        tabs = tk.Frame(shell, bg=BG)
        tabs.pack(fill=tk.X, padx=12, pady=(2, 2))
        for name in VIEWS:
            btn = tk.Button(
                tabs,
                text=name,
                command=lambda n=name: self._show_view(n, manual=True),
                bg=BG_PANEL,
                fg=FG_DIM,
                activebackground=FG_MAGENTA,
                activeforeground=BG,
                font=FONT_SMALL,
                relief=tk.FLAT,
                padx=10,
                pady=3,
                highlightthickness=1,
                highlightbackground=BORDER_DIM,
                cursor="hand2",
            )
            btn.pack(side=tk.LEFT, padx=(0, 4))
            self._view_btns[name] = btn

        # Body: center stack + side
        body = tk.Frame(shell, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=2)

        center_outer = _panel(body)
        center_outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        center = center_outer.body  # type: ignore[attr-defined]
        self.center = center

        # Stack frames for each view
        self.view_frames: dict[str, tk.Frame] = {}
        for name in VIEWS:
            fr = tk.Frame(center, bg=BG_PANEL)
            self.view_frames[name] = fr

        self._build_chat_view(self.view_frames[VIEW_CHAT])
        self._build_workspace_view(self.view_frames[VIEW_WORKSPACE])
        self._build_code_view(self.view_frames[VIEW_CODE])
        self._build_terminal_view(self.view_frames[VIEW_TERMINAL])
        self._build_research_view(self.view_frames[VIEW_RESEARCH])
        self._build_files_view(self.view_frames[VIEW_FILES])

        # Side column
        side_col = tk.Frame(body, bg=BG, width=280)
        side_col.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        side_col.pack_propagate(False)

        mem_outer = _panel(side_col, fill=tk.X, pady=(0, 8))
        mem = mem_outer.body  # type: ignore[attr-defined]
        tk.Label(
            mem, text="▣  MEMORY / STATS", fg=FG_BRIGHT, bg=BG_PANEL,
            font=FONT_SMALL, anchor="w",
        ).pack(fill=tk.X, padx=8, pady=(6, 0))
        self.stats_text = tk.Text(
            mem, bg=BG_DEEP, fg=FG, font=FONT_STAT, relief=tk.FLAT, height=12,
            wrap=tk.WORD, state=tk.DISABLED, highlightthickness=0, takefocus=0,
            width=26,
        )
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.stats_text.tag_configure("ok", foreground=FG_OK)
        self.stats_text.tag_configure("dim", foreground=FG_DIM)
        self.stats_text.tag_configure("hi", foreground=FG_BRIGHT)

        sk_outer = _panel(side_col, fill=tk.BOTH, expand=True)
        sk = sk_outer.body  # type: ignore[attr-defined]
        tk.Label(
            sk, text="☰  SKILLS REGISTRY", fg=FG_BRIGHT, bg=BG_PANEL,
            font=FONT_SMALL, anchor="w",
        ).pack(fill=tk.X, padx=8, pady=(6, 0))
        self.skills_text = scrolledtext.ScrolledText(
            sk, bg=BG_DEEP, fg=FG, font=FONT_SMALL, relief=tk.FLAT, wrap=tk.WORD,
            state=tk.DISABLED, highlightthickness=0, takefocus=0,
        )
        self.skills_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.skills_text.tag_configure("cand", foreground=FG_MAGENTA)
        self.skills_text.tag_configure("brok", foreground=FG_ERR)
        self.skills_text.tag_configure("actv", foreground=FG_OK)
        self.skills_text.tag_configure("dim", foreground=FG_DIM)

    def _build_chat_view(self, parent: tk.Frame) -> None:
        hdr = tk.Frame(parent, bg=BG_PANEL)
        hdr.pack(fill=tk.X, padx=6, pady=(4, 0))
        tk.Label(
            hdr, text="›  CONVERSATION / LOG", fg=FG_BRIGHT, bg=BG_PANEL,
            font=FONT_SMALL,
        ).pack(side=tk.LEFT)
        tk.Label(
            hdr, text="CHAT", fg=BORDER_DIM, bg=BG_PANEL, font=FONT_SUB
        ).pack(side=tk.RIGHT, padx=6)
        self.log = scrolledtext.ScrolledText(
            parent, bg=BG_DEEP, fg=FG, font=FONT_MONO, wrap=tk.WORD,
            relief=tk.FLAT, state=tk.DISABLED, highlightthickness=0, takefocus=0,
            padx=10, pady=8,
        )
        self.log.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.log.bind("<Button-1>", lambda _e: self._focus_entry())
        for tag, col in (
            ("user", FG_USER), ("jarvis", FG), ("system", FG_DIM),
            ("error", FG_ERR), ("warn", FG_WARN), ("ok", FG_OK),
        ):
            self.log.tag_configure(tag, foreground=col)

    def _build_workspace_view(self, parent: tk.Frame) -> None:
        hdr = tk.Frame(parent, bg=BG_PANEL)
        hdr.pack(fill=tk.X, padx=6, pady=(4, 0))
        tk.Label(
            hdr, text="›  WORKSPACE", fg=FG_BRIGHT, bg=BG_PANEL, font=FONT_SMALL
        ).pack(side=tk.LEFT)
        self.ws_phase = tk.Label(
            hdr, text="IDLE", fg=FG_OK, bg=BG_PANEL, font=FONT_SMALL
        )
        self.ws_phase.pack(side=tk.RIGHT, padx=6)
        self.ws_summary = scrolledtext.ScrolledText(
            parent, bg=BG_DEEP, fg=FG, font=FONT_MONO, wrap=tk.WORD,
            relief=tk.FLAT, state=tk.DISABLED, highlightthickness=0, takefocus=0,
            padx=8, pady=6,
        )
        self.ws_summary.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.ws_summary.tag_configure("hi", foreground=FG_BRIGHT)
        self.ws_summary.tag_configure("ok", foreground=FG_OK)
        self.ws_summary.tag_configure("err", foreground=FG_ERR)
        self.ws_summary.tag_configure("dim", foreground=FG_DIM)

    def _build_code_view(self, parent: tk.Frame) -> None:
        hdr = tk.Frame(parent, bg=BG_PANEL)
        hdr.pack(fill=tk.X, padx=6, pady=(4, 0))
        tk.Label(
            hdr, text="›  CODE WORKSPACE", fg=FG_BRIGHT, bg=BG_PANEL, font=FONT_SMALL
        ).pack(side=tk.LEFT)
        self.code_path_var = tk.StringVar(value="(no active file)")
        tk.Label(
            hdr, textvariable=self.code_path_var, fg=FG_DIM, bg=BG_PANEL, font=FONT_SUB
        ).pack(side=tk.RIGHT, padx=6)
        self.code_status = tk.Label(
            parent, text="STATUS: —", fg=FG_MAGENTA, bg=BG_PANEL, font=FONT_SMALL,
            anchor="w",
        )
        self.code_status.pack(fill=tk.X, padx=10, pady=(2, 0))
        self.code_view = scrolledtext.ScrolledText(
            parent, bg=BG_DEEP, fg=FG, font=FONT_CODE, wrap=tk.NONE,
            relief=tk.FLAT, state=tk.DISABLED, highlightthickness=0, takefocus=0,
            padx=8, pady=6,
        )
        self.code_view.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.code_view.tag_configure("kw", foreground=FG_KW)
        self.code_view.tag_configure("str", foreground=FG_STR)
        self.code_view.tag_configure("cmt", foreground=FG_CMT)
        self.code_view.tag_configure("num", foreground=FG_NUM)

    def _build_terminal_view(self, parent: tk.Frame) -> None:
        hdr = tk.Frame(parent, bg=BG_PANEL)
        hdr.pack(fill=tk.X, padx=6, pady=(4, 0))
        tk.Label(
            hdr, text="›  TERMINAL (full)", fg=FG_BRIGHT, bg=BG_PANEL, font=FONT_SMALL
        ).pack(side=tk.LEFT)
        self.term_full = scrolledtext.ScrolledText(
            parent, bg=BG_DEEP, fg=FG, font=FONT_CODE, wrap=tk.WORD,
            relief=tk.FLAT, state=tk.DISABLED, highlightthickness=0, takefocus=0,
            padx=8, pady=6,
        )
        self.term_full.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        for tag, col in (
            ("cmd", FG_MAGENTA), ("out", FG), ("err", FG_ERR),
            ("ok", FG_OK), ("dim", FG_DIM),
        ):
            self.term_full.tag_configure(tag, foreground=col)

    def _build_research_view(self, parent: tk.Frame) -> None:
        hdr = tk.Frame(parent, bg=BG_PANEL)
        hdr.pack(fill=tk.X, padx=6, pady=(4, 0))
        tk.Label(
            hdr, text="›  RESEARCH", fg=FG_BRIGHT, bg=BG_PANEL, font=FONT_SMALL
        ).pack(side=tk.LEFT)
        self.research_view = scrolledtext.ScrolledText(
            parent, bg=BG_DEEP, fg=FG, font=FONT_MONO, wrap=tk.WORD,
            relief=tk.FLAT, state=tk.DISABLED, highlightthickness=0, takefocus=0,
            padx=8, pady=6,
        )
        self.research_view.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.research_view.tag_configure("hi", foreground=FG_BRIGHT)
        self.research_view.tag_configure("ok", foreground=FG_OK)
        self.research_view.tag_configure("dim", foreground=FG_DIM)
        self.research_view.tag_configure("url", foreground=FG_USER)

    def _build_files_view(self, parent: tk.Frame) -> None:
        hdr = tk.Frame(parent, bg=BG_PANEL)
        hdr.pack(fill=tk.X, padx=6, pady=(4, 0))
        tk.Label(
            hdr, text="›  FILES", fg=FG_BRIGHT, bg=BG_PANEL, font=FONT_SMALL
        ).pack(side=tk.LEFT)
        tk.Button(
            hdr, text="REFRESH", command=self._refresh_files, bg=BG_PANEL_2,
            fg=FG_GLOW, font=FONT_SUB, relief=tk.FLAT, padx=8, cursor="hand2",
        ).pack(side=tk.RIGHT, padx=6)
        self.files_view = scrolledtext.ScrolledText(
            parent, bg=BG_DEEP, fg=FG, font=FONT_MONO, wrap=tk.WORD,
            relief=tk.FLAT, state=tk.DISABLED, highlightthickness=0, takefocus=0,
            padx=8, pady=6,
        )
        self.files_view.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.files_view.tag_configure("hi", foreground=FG_BRIGHT)
        self.files_view.tag_configure("dim", foreground=FG_DIM)
        self.files_view.tag_configure("path", foreground=FG_USER)

    # ── View switching ──────────────────────────────────────────────────

    def _show_view(self, name: str, *, manual: bool) -> None:
        if name not in self.view_frames:
            return
        if manual:
            self._manual_view = True
        self._active_view = name
        for n, fr in self.view_frames.items():
            fr.pack_forget()
        self.view_frames[name].pack(fill=tk.BOTH, expand=True)
        for n, btn in self._view_btns.items():
            if n == name:
                btn.configure(fg=FG_GLOW, highlightbackground=BORDER, bg=BG_PANEL_2)
            else:
                btn.configure(fg=FG_DIM, highlightbackground=BORDER_DIM, bg=BG_PANEL)
        if name == VIEW_FILES:
            self._refresh_files()
        if name == VIEW_CODE and self._code_path:
            self._load_code_file(self._code_path, force=True)

    def _auto_view(self, name: str) -> None:
        if self._manual_view:
            return
        if name != self._active_view:
            self._show_view(name, manual=False)

    # ── Event intake (async — never blocks backend) ─────────────────────

    def _ui_event(self, kind: str, payload: dict) -> None:
        try:
            self._event_q.put_nowait((kind, dict(payload or {})))
        except Exception:
            pass

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self._event_q.get_nowait()
                self._handle_event(kind, payload)
        except queue.Empty:
            pass
        self.win.after(50, self._drain_events)

    def _handle_event(self, kind: str, p: dict[str, Any]) -> None:
        if kind == "phase":
            status = str(p.get("status") or "")
            self._phase = status
            self.phase_var.set(f"· {status}" if status else "")
            self.ws_phase.configure(text=status or "IDLE")
            self.code_status.configure(text=f"STATUS: {status or '—'}")
            view = view_for_status(status)
            if status in ("DONE", "IDLE"):
                self._manual_view = False
                if p.get("status") == "DONE" or status == "DONE":
                    # VERIFY PASS / cycle end → return to CHAT
                    self._auto_view(VIEW_CHAT)
                else:
                    self._auto_view(view)
            else:
                self._auto_view(view)
            self._ws_line(f"PHASE → {status}", "hi")
            return

        if kind == "skill_file":
            path = p.get("path")
            code = p.get("code") or ""
            if path:
                self._code_path = Path(str(path))
                self.code_path_var.set(str(self._code_path))
                if code:
                    self._set_code_text(str(code))
                else:
                    self._load_code_file(self._code_path, force=True)
            self._auto_view(VIEW_CODE)
            self._ws_line(
                f"FILE {(p.get('source') or 'write')}: {path} "
                f"v{p.get('version', '?')} ok={p.get('ok')}",
                "ok" if p.get("ok") else "err",
            )
            self._refresh_files()
            return

        if kind == "subprocess":
            mode = str(p.get("mode") or "run")
            cmd = str(p.get("command") or mode)
            self._term_write(f"$ {cmd}\n", "cmd")
            self._term_write_full(f"$ {cmd}\n", "cmd")
            out = str(p.get("stdout") or "")
            err = str(p.get("stderr") or "")
            tb = str(p.get("traceback") or "")
            if out:
                self._term_write(out if out.endswith("\n") else out + "\n", "out")
                self._term_write_full(out if out.endswith("\n") else out + "\n", "out")
            if err:
                self._term_write(err if err.endswith("\n") else err + "\n", "err")
                self._term_write_full(err if err.endswith("\n") else err + "\n", "err")
            if tb:
                self._term_write(tb if tb.endswith("\n") else tb + "\n", "err")
                self._term_write_full(tb if tb.endswith("\n") else tb + "\n", "err")
            rc = p.get("returncode")
            tag = "ok" if p.get("ok") else "err"
            line = f"[rc={rc}] ok={p.get('ok')} timed_out={p.get('timed_out')}\n"
            if p.get("error"):
                line += f"error: {p.get('error')}\n"
            self._term_write(line, tag)
            self._term_write_full(line, tag)
            self.term_status.configure(
                text=f"{mode} rc={rc}", fg=FG_OK if p.get("ok") else FG_ERR
            )
            self._auto_view(VIEW_TERMINAL)
            self._ws_line(f"SUBPROCESS {mode} rc={rc} ok={p.get('ok')}", tag)
            return

        if kind == "research":
            stage = str(p.get("stage") or "")
            if stage == "start":
                self._research_lines.clear()
                self._research_set("RESEARCH START\n", "hi")
                qs = p.get("queries") or []
                for q in qs:
                    self._research_add(f"  query: {q}\n", "dim")
                self._auto_view(VIEW_RESEARCH)
            elif stage == "search":
                self._research_add(f"SEARCH → {p.get('query')}\n", "hi")
                self._auto_view(VIEW_RESEARCH)
            elif stage == "source":
                self._research_add(
                    f"SOURCE  {(p.get('title') or '')[:80]}\n"
                    f"        {p.get('url')}\n",
                    "url",
                )
            elif stage == "extract":
                self._research_add(
                    f"READING {(p.get('title') or '')[:80]} "
                    f"chars={p.get('chars')} ok={p.get('ok')}\n",
                    "ok" if p.get("ok") else "dim",
                )
                preview = str(p.get("preview") or "").strip()
                if preview:
                    self._research_add(f"  evid: {preview[:300]}\n", "dim")
            elif stage == "done":
                self._research_add("RESEARCH DONE\n", "ok")
                for s in (p.get("sources") or [])[:10]:
                    if isinstance(s, dict):
                        self._research_add(
                            f"  · {(s.get('title') or '')[:70]} | "
                            f"{(s.get('url') or '')[:100]}\n",
                            "url",
                        )
                for e in (p.get("extracts") or [])[:6]:
                    if isinstance(e, dict) and e.get("preview"):
                        self._research_add(
                            f"  proof: {str(e.get('preview'))[:280]}\n", "dim"
                        )
                self._auto_view(VIEW_RESEARCH)
            self._ws_line(f"RESEARCH/{stage}", "hi")
            return

        if kind == "verify":
            verified = bool(p.get("verified"))
            reason = str(p.get("reason") or "")
            tag = "ok" if verified else "err"
            msg = f"VERIFY {'PASS' if verified else 'FAIL'} — {reason}\n"
            self._term_write(msg, tag)
            self._term_write_full(msg, tag)
            self._ws_line(msg.strip(), tag)
            if verified:
                self._manual_view = False
                self._auto_view(VIEW_CHAT)
            else:
                self._auto_view(VIEW_TERMINAL)
            return

    # ── Code / files (real disk) ────────────────────────────────────────

    def _set_code_text(self, code: str) -> None:
        self.code_view.configure(state=tk.NORMAL)
        self.code_view.delete("1.0", tk.END)
        self.code_view.insert(tk.END, code)
        self._highlight_python()
        self.code_view.see(tk.END)
        self.code_view.configure(state=tk.DISABLED)

    def _load_code_file(self, path: Path, *, force: bool = False) -> None:
        try:
            p = Path(path)
            if not p.is_file():
                return
            mtime = p.stat().st_mtime
            if not force and mtime == self._code_mtime and p == self._code_path:
                return
            text = p.read_text(encoding="utf-8", errors="replace")
            self._code_path = p
            self._code_mtime = mtime
            self.code_path_var.set(str(p))
            self._set_code_text(text)
        except Exception as exc:
            self.code_path_var.set(f"read error: {exc}")

    def _poll_active_file(self) -> None:
        if self._code_path and self._active_view in (
            VIEW_CODE, VIEW_WORKSPACE
        ):
            self._load_code_file(self._code_path, force=False)
        self.win.after(1200, self._poll_active_file)

    def _highlight_python(self) -> None:
        text = self.code_view.get("1.0", tk.END)
        for tag in ("kw", "str", "cmt", "num"):
            self.code_view.tag_remove(tag, "1.0", tk.END)
        for m in re.finditer(r"#.*?$", text, re.M):
            self.code_view.tag_add("cmt", f"1.0+{m.start()}c", f"1.0+{m.end()}c")
        for m in re.finditer(r"('''[\s\S]*?'''|\"\"\"[\s\S]*?\"\"\"|'[^'\n]*'|\"[^\"\n]*\")", text):
            self.code_view.tag_add("str", f"1.0+{m.start()}c", f"1.0+{m.end()}c")
        for m in re.finditer(r"\b\d+\.?\d*\b", text):
            self.code_view.tag_add("num", f"1.0+{m.start()}c", f"1.0+{m.end()}c")
        for m in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text):
            if m.group(0) in _PY_KW:
                self.code_view.tag_add("kw", f"1.0+{m.start()}c", f"1.0+{m.end()}c")

    def _refresh_files(self) -> None:
        roots = []
        try:
            roots = [
                ("skills", Path(self.orch.skills_dir)),
                ("workspace", Path(self.orch.workspace)),
            ]
        except Exception:
            roots = [
                ("skills", self.root_dir / "skills"),
                ("workspace", self.root_dir / "workspace_runtime"),
            ]
        lines: list[tuple[str, str]] = []
        for label, root in roots:
            lines.append((f"[{label}] {root}\n", "hi"))
            if not root.exists():
                lines.append(("  (missing)\n", "dim"))
                continue
            try:
                files = sorted(root.rglob("*"))
            except Exception as exc:
                lines.append((f"  error: {exc}\n", "dim"))
                continue
            n = 0
            for f in files:
                if not f.is_file():
                    continue
                if f.name.startswith(".") or f.suffix == ".pyc":
                    continue
                if "__pycache__" in f.parts:
                    continue
                rel = f.relative_to(root)
                try:
                    sz = f.stat().st_size
                except Exception:
                    sz = 0
                lines.append((f"  {rel}  ({sz} B)\n", "path"))
                n += 1
                if n >= 80:
                    lines.append(("  …\n", "dim"))
                    break
            if n == 0:
                lines.append(("  (empty)\n", "dim"))
        self.files_view.configure(state=tk.NORMAL)
        self.files_view.delete("1.0", tk.END)
        for text, tag in lines:
            self.files_view.insert(tk.END, text, tag)
        self.files_view.configure(state=tk.DISABLED)

    # ── Terminal / research helpers ─────────────────────────────────────

    def _term_write(self, text: str, tag: str = "out") -> None:
        self.term.configure(state=tk.NORMAL)
        self.term.insert(tk.END, text, tag)
        self.term.see(tk.END)
        self.term.configure(state=tk.DISABLED)

    def _term_write_full(self, text: str, tag: str = "out") -> None:
        self.term_full.configure(state=tk.NORMAL)
        self.term_full.insert(tk.END, text, tag)
        self.term_full.see(tk.END)
        self.term_full.configure(state=tk.DISABLED)

    def _ws_line(self, text: str, tag: str = "dim") -> None:
        self.ws_summary.configure(state=tk.NORMAL)
        self.ws_summary.insert(tk.END, text.rstrip() + "\n", tag)
        self.ws_summary.see(tk.END)
        self.ws_summary.configure(state=tk.DISABLED)

    def _research_set(self, text: str, tag: str = "dim") -> None:
        self.research_view.configure(state=tk.NORMAL)
        self.research_view.delete("1.0", tk.END)
        self.research_view.insert(tk.END, text, tag)
        self.research_view.configure(state=tk.DISABLED)

    def _research_add(self, text: str, tag: str = "dim") -> None:
        self.research_view.configure(state=tk.NORMAL)
        self.research_view.insert(tk.END, text, tag)
        self.research_view.see(tk.END)
        self.research_view.configure(state=tk.DISABLED)

    # ── Window / keys ───────────────────────────────────────────────────

    def _go_fullscreen(self) -> None:
        try:
            self.win.state("zoomed")
        except Exception:
            try:
                self.win.attributes("-zoomed", True)
            except Exception:
                self.win.geometry("1280x800")
        try:
            self.win.attributes("-fullscreen", True)
            self._fullscreen = True
        except Exception:
            self._fullscreen = False

    def _toggle_fullscreen(self, _event=None):
        self._fullscreen = not self._fullscreen
        try:
            self.win.attributes("-fullscreen", self._fullscreen)
        except Exception:
            pass
        if not self._fullscreen:
            try:
                self.win.state("zoomed")
            except Exception:
                pass
        return "break"

    def _exit_fullscreen(self, _event=None):
        if self._fullscreen:
            self._fullscreen = False
            try:
                self.win.attributes("-fullscreen", False)
            except Exception:
                pass
            try:
                self.win.state("zoomed")
            except Exception:
                pass
        return "break"

    def _focus_entry(self, _event=None):
        try:
            self.entry.focus_set()
        except Exception:
            pass
        return "break"

    def _bind_keys(self) -> None:
        self.entry.bind("<Return>", self._on_return)
        self.win.bind("<F11>", self._toggle_fullscreen)
        self.win.bind("<Escape>", self._exit_fullscreen)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_return(self, _event=None):
        self._on_send()
        return "break"

    def _animate_status_dots(self) -> None:
        status = self.status_var.get()
        if any(x in status for x in ("THINKING", "WORKING", "REPAIR", "CODING", "TEST")):
            self._dot_phase = (self._dot_phase + 1) % 4
            self._status_dots.configure(text="." * self._dot_phase)
        else:
            self._status_dots.configure(text="")
        self.win.after(400, self._animate_status_dots)

    # ── Boot / chat log ─────────────────────────────────────────────────

    def _boot_banner(self) -> None:
        status = self.orch.brain.model_status()
        # Show RESOLVED (auto-selected installed) models — not catalog primaries
        catalog = ""
        if hasattr(self.orch.brain, "model_catalog_summary"):
            catalog = self.orch.brain.model_catalog_summary()
        pull_hint = TIER_MODELS[TIER_FAST]["primary"]
        hint = {
            "ONLINE": "CONNECTED (auto-select installed)",
            "MODEL MISSING": f"MODEL MISSING — ollama pull {pull_hint}",
            "OFFLINE": "OFFLINE (start ollama serve)",
        }.get(status, status)
        try:
            from jarvis.runtime_fingerprint import fingerprint_line

            runtime_line = fingerprint_line()
        except Exception:
            runtime_line = "JARVIS RUNTIME commit=unknown"
        banner = (
            "[SYSTEM] JARVIS cyber-core online — integrated Workspace active.\n"
            f"[SYSTEM] {runtime_line}\n"
            f"[SYSTEM] Ollama multi-model — {hint}\n"
            + (
                f"[SYSTEM] ACTIVE {catalog}\n"
                if catalog
                else (
                    f"[SYSTEM] FAST:{TIER_MODELS[TIER_FAST]['primary']}  "
                    f"REASONING:{TIER_MODELS[TIER_REASONING]['primary']}  "
                    f"CODING:{TIER_MODELS[TIER_CODING]['primary']}\n"
                )
            )
            + "[SYSTEM] Views: CHAT WORKSPACE CODE TERMINAL RESEARCH FILES\n"
            + "[SYSTEM] Type a goal below · F11 fullscreen · Esc exit FS"
        )
        self._append("system", banner)
        self._term_write("[terminal ready — live subprocess output]\n", "dim")
        self._ui_status("IDLE")
        online_col = FG_OK if status == "ONLINE" else FG_ERR
        self.brain_lbl.configure(fg=online_col)
        self.brain_dot.configure(fg=online_col)
        self.brain_var.set(f"BRAIN: {status} | multi-model")

    def _append(self, tag: str, text: str) -> None:
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
        tag = "system"
        upper = msg.upper()
        if "FAIL" in upper or "ERROR" in upper or "EXCEPTION" in upper:
            tag = "error"
        elif "WARN" in upper or "BROKEN" in upper:
            tag = "warn"
        elif "PASS" in upper or "DONE" in upper or "ONLINE" in upper:
            tag = "ok"
        self._append(tag, f"[SYSTEM] {msg}")

    def _ui_status(self, status: str) -> None:
        def _do() -> None:
            self.status_var.set(f"STATUS: {status}")
            color = FG_GLOW
            if status in ("ERROR", "FAIL"):
                color = FG_ERR
            elif status in ("DONE", "IDLE"):
                color = FG_OK
            elif status in ("REPAIR", "REPAIRING", "BROKEN"):
                color = FG_WARN
            elif status in (
                "THINKING", "WORKING", "PLAN", "RESEARCH",
                "BUILD_SKILL", "TEST", "VERIFY",
            ):
                color = FG_MAGENTA
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
        led = dash["ledger"]
        brain = dash.get("brain_status") or (
            "ONLINE" if dash["brain_online"] else "OFFLINE"
        )
        catalog = dash.get("model_catalog") or "multi-model"
        tiers = (dash.get("model_pool") or {}).get("tiers") or dash.get("model_tiers") or {}

        def tier_line(name: str) -> str:
            info = tiers.get(name) or {}
            model = info.get("resolved") or info.get("primary") or "?"
            mark = "✓" if info.get("online") or info.get("ready") else "·"
            warm = " ♨" if info.get("warm") else ""
            short = {"FAST": "Fast", "REASONING": "Reasoning", "CODING": "Coding"}.get(
                name, name.title()
            )
            return f"{short:<10}{model} {mark}{warm}"

        self.stats_text.configure(state=tk.NORMAL)
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert(tk.END, "Brain     ", "dim")
        self.stats_text.insert(
            tk.END,
            f"{brain} ✓\n" if brain == "ONLINE" else f"{brain}\n",
            "ok" if brain == "ONLINE" else "hi",
        )
        self.stats_text.insert(tk.END, f"Router    {catalog}\n", "dim")
        for tname in ("FAST", "REASONING", "CODING"):
            self.stats_text.insert(tk.END, tier_line(tname) + "\n", "hi")
        self.stats_text.insert(tk.END, "──────────────\n", "dim")
        self.stats_text.insert(
            tk.END,
            f"Conv      {mem['conversations']}\n"
            f"Exper.    {mem['experiences']} (ok {mem['successes']})\n"
            f"Facts     {mem['facts']}\n"
            f"Events    {mem['skill_events']}\n"
            f"──────────────\n"
            f"Tasks     {led['tasks']}\n"
            f"Done      {led['done']}\n"
            f"Fail      {led['failed']}\n",
            "hi",
        )
        self.stats_text.configure(state=tk.DISABLED)

        self.skills_text.configure(state=tk.NORMAL)
        self.skills_text.delete("1.0", tk.END)
        skill_list = dash.get("skill_list") or []
        if not skill_list:
            self.skills_text.insert(
                tk.END, "(none yet — give JARVIS a task)", "dim"
            )
        else:
            for s in skill_list:
                st = str(s.get("status") or "")
                tag = "dim"
                short = st[:4].upper()
                if "BROK" in st.upper() or st == "BROKEN":
                    tag = "brok"
                    short = "BROK"
                elif "CAND" in st.upper() or st == "CANDIDATE":
                    tag = "cand"
                    short = "CAND"
                elif st == "ACTIVE":
                    tag = "actv"
                    short = "ACTV"
                line = f"[{short}] {s['name']} v{s['version']}\n"
                self.skills_text.insert(tk.END, line, tag)
        self.skills_text.configure(state=tk.DISABLED)

        online_col = FG_OK if brain == "ONLINE" else FG_ERR
        self.brain_lbl.configure(fg=online_col)
        self.brain_dot.configure(fg=online_col)
        self.brain_var.set(f"BRAIN: {brain} | multi-model")

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
            self._focus_entry()
            return
        self.entry.delete(0, tk.END)
        self.busy = True
        self._request_seq += 1
        seq = self._request_seq
        self._manual_view = False
        self.send_btn.configure(state=tk.DISABLED)
        self._append("user", f"[USER] {text}")
        self._show_view(VIEW_CHAT, manual=False)
        self._ui_status("THINKING")

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
                    if seq != self._request_seq:
                        return
                    self._append(tag, f"[JARVIS] {reply}")
                    self._show_view(VIEW_CHAT, manual=False)

                self.win.after(0, show_reply)
            except Exception as exc:
                def show_err() -> None:
                    if seq != self._request_seq:
                        return
                    self._append("error", f"[JARVIS] Internal error: {exc}")

                self.win.after(0, show_err)
            finally:
                def done() -> None:
                    if seq != self._request_seq:
                        return
                    self.busy = False
                    self.send_btn.configure(state=tk.NORMAL)
                    self._refresh_stats()
                    self._refresh_files()
                    self._focus_entry()
                    self._ui_status("IDLE")

                self.win.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def _on_close(self) -> None:
        try:
            self.orch.close()
        except Exception:
            pass
        self.win.destroy()

    def run(self) -> None:
        self._focus_entry()
        self.win.mainloop()
