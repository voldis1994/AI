"""
JARVIS cyber-HUD GUI — visual shell only.

Magenta neon aesthetic. Layout uses pack only (no place-based panels)
so the input bar cannot collapse off-screen on Windows.
"""

from __future__ import annotations

import threading
from pathlib import Path

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
BORDER = "#ff2bd6"
BORDER_DIM = "#6b145c"

FONT_MONO = ("Consolas", 11)
FONT_TITLE = ("Consolas", 36, "bold")
FONT_SUB = ("Consolas", 9)
FONT_SMALL = ("Consolas", 9)
FONT_STAT = ("Consolas", 10)


def _mono() -> str:
    """Pick first available monospace family (requires an existing Tk root)."""
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
    """Magenta-bordered panel that packs correctly (no place geometry)."""
    outer = tk.Frame(
        parent,
        bg=BORDER,
        highlightthickness=0,
        bd=0,
    )
    if pack_kw:
        outer.pack(**pack_kw)
    inner = tk.Frame(outer, bg=BG_PANEL, highlightthickness=0, bd=0)
    inner.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
    outer.body = inner  # type: ignore[attr-defined]
    return outer


class JarvisGUI:
    """Magenta cyber-HUD interface for JARVIS (visual shell)."""

    def __init__(self, root_dir: Path) -> None:
        if tk is None:
            raise RuntimeError("tkinter is required for GUI mode")
        self.root_dir = root_dir
        self.busy = False
        self._request_seq = 0
        self._dot_phase = 0
        self._fullscreen = True

        self.win = tk.Tk()
        self.win.title("JARVIS")
        self.win.configure(bg=BG)
        self.win.minsize(900, 600)

        family = _mono()
        global FONT_MONO, FONT_TITLE, FONT_SUB, FONT_SMALL, FONT_STAT
        FONT_MONO = (family, 11)
        FONT_TITLE = (family, 36, "bold")
        FONT_SUB = (family, 9)
        FONT_SMALL = (family, 9)
        FONT_STAT = (family, 10)

        self._build_ui()
        self._bind_keys()
        self._go_fullscreen()

        self.orch = Orchestrator(
            root=root_dir,
            brain=Brain(),
            on_log=self._ui_log,
            on_status=self._ui_status,
        )
        self._refresh_stats()
        self._boot_banner()
        self.win.after(80, self._animate_status_dots)
        self.win.after(200, self._focus_entry)
        self.win.after(4000, self._tick_stats)

    def _build_ui(self) -> None:
        root_border = tk.Frame(self.win, bg=BORDER, bd=0)
        root_border.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        shell = tk.Frame(root_border, bg=BG, bd=0)
        shell.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.shell = shell

        header = tk.Frame(shell, bg=BG)
        header.pack(fill=tk.X, padx=12, pady=(10, 4))

        self.title_lbl = tk.Label(
            header,
            text="JARVIS",
            fg=FG_TITLE,
            bg=BG,
            font=FONT_TITLE,
        )
        self.title_lbl.pack()
        tk.Label(
            header,
            text="AUTONOMOUS  •  SELF-LEARNING  •  VERIFIED",
            fg=FG_DIM,
            bg=BG,
            font=FONT_SUB,
        ).pack()

        status_outer = _panel(shell, fill=tk.X, padx=12, pady=(4, 6))
        sf = status_outer.body  # type: ignore[attr-defined]

        left_st = tk.Frame(sf, bg=BG_PANEL)
        left_st.pack(side=tk.LEFT, fill=tk.Y)
        self.status_var = tk.StringVar(value="STATUS: BOOT")
        self.status_lbl = tk.Label(
            left_st,
            textvariable=self.status_var,
            fg=FG_GLOW,
            bg=BG_PANEL,
            font=FONT_MONO,
            anchor="w",
        )
        self.status_lbl.pack(side=tk.LEFT, padx=(10, 0), pady=6)
        self._status_dots = tk.Label(
            left_st, text="", fg=FG_MAGENTA, bg=BG_PANEL, font=FONT_MONO
        )
        self._status_dots.pack(side=tk.LEFT)

        right_st = tk.Frame(sf, bg=BG_PANEL)
        right_st.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        self.brain_dot = tk.Label(
            right_st, text="●", fg=FG_OK, bg=BG_PANEL, font=FONT_MONO
        )
        self.brain_dot.pack(side=tk.LEFT, pady=6)
        self.brain_var = tk.StringVar(value="BRAIN: … | multi-model")
        self.brain_lbl = tk.Label(
            right_st,
            textvariable=self.brain_var,
            fg=FG_OK,
            bg=BG_PANEL,
            font=FONT_SMALL,
            anchor="e",
        )
        self.brain_lbl.pack(side=tk.LEFT, padx=(4, 0), pady=6)

        # Input FIRST with side=BOTTOM so it never collapses
        input_outer = tk.Frame(shell, bg=BORDER, height=64)
        input_outer.pack(fill=tk.X, padx=12, pady=(6, 12), side=tk.BOTTOM)
        input_outer.pack_propagate(False)

        input_frame = tk.Frame(input_outer, bg=BG_PANEL)
        input_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        tk.Label(
            input_frame,
            text="›",
            fg=FG_MAGENTA,
            bg=BG_PANEL,
            font=(FONT_MONO[0], 18, "bold"),
        ).pack(side=tk.LEFT, padx=(10, 6))

        self.entry = tk.Entry(
            input_frame,
            bg=BG_INPUT,
            fg=FG,
            insertbackground=FG_GLOW,
            font=FONT_MONO,
            relief=tk.FLAT,
            highlightbackground=BORDER_DIM,
            highlightcolor=FG_MAGENTA,
            highlightthickness=1,
        )
        self.entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=8, ipady=4)

        self.send_btn = tk.Button(
            input_frame,
            text="SEND",
            command=self._on_send,
            bg=BG_PANEL_2,
            fg=FG_GLOW,
            activebackground=FG_MAGENTA,
            activeforeground=BG,
            font=(FONT_MONO[0], 11, "bold"),
            relief=tk.FLAT,
            padx=20,
            pady=6,
            highlightbackground=BORDER,
            highlightthickness=2,
            cursor="hand2",
        )
        self.send_btn.pack(side=tk.LEFT, padx=(8, 10), pady=8)

        body = tk.Frame(shell, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=2)

        log_outer = _panel(body)
        log_outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_frame = log_outer.body  # type: ignore[attr-defined]

        log_header = tk.Frame(log_frame, bg=BG_PANEL)
        log_header.pack(fill=tk.X, padx=6, pady=(4, 0))
        tk.Label(
            log_header,
            text="›  CONVERSATION / LOG",
            fg=FG_BRIGHT,
            bg=BG_PANEL,
            font=FONT_SMALL,
            anchor="w",
        ).pack(side=tk.LEFT)
        tk.Label(
            log_header,
            text="LIVE FEED",
            fg=BORDER_DIM,
            bg=BG_PANEL,
            font=FONT_SUB,
        ).pack(side=tk.RIGHT, padx=6)

        self.log = scrolledtext.ScrolledText(
            log_frame,
            bg=BG_DEEP,
            fg=FG,
            insertbackground=FG_GLOW,
            font=FONT_MONO,
            wrap=tk.WORD,
            relief=tk.FLAT,
            borderwidth=0,
            state=tk.DISABLED,
            highlightthickness=0,
            takefocus=0,
            padx=10,
            pady=8,
        )
        self.log.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.log.bind("<Button-1>", lambda _e: self._focus_entry())
        self.log.tag_configure("user", foreground=FG_USER)
        self.log.tag_configure("jarvis", foreground=FG)
        self.log.tag_configure("system", foreground=FG_DIM)
        self.log.tag_configure("error", foreground=FG_ERR)
        self.log.tag_configure("warn", foreground=FG_WARN)
        self.log.tag_configure("ok", foreground=FG_OK)
        try:
            self.log.vbar.configure(
                troughcolor=BG_PANEL,
                background=FG_MAGENTA,
                activebackground=FG_GLOW,
                borderwidth=0,
                width=12,
            )
        except Exception:
            pass

        side_col = tk.Frame(body, bg=BG, width=300)
        side_col.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        side_col.pack_propagate(False)

        mem_outer = _panel(side_col, fill=tk.X, pady=(0, 8))
        mem = mem_outer.body  # type: ignore[attr-defined]
        tk.Label(
            mem,
            text="▣  MEMORY / STATS",
            fg=FG_BRIGHT,
            bg=BG_PANEL,
            font=FONT_SMALL,
            anchor="w",
        ).pack(fill=tk.X, padx=8, pady=(6, 0))
        self.stats_text = tk.Text(
            mem,
            bg=BG_DEEP,
            fg=FG,
            font=FONT_STAT,
            relief=tk.FLAT,
            height=14,
            wrap=tk.WORD,
            state=tk.DISABLED,
            highlightthickness=0,
            takefocus=0,
            width=28,
        )
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.stats_text.tag_configure("ok", foreground=FG_OK)
        self.stats_text.tag_configure("dim", foreground=FG_DIM)
        self.stats_text.tag_configure("hi", foreground=FG_BRIGHT)

        sk_outer = _panel(side_col, fill=tk.BOTH, expand=True)
        sk = sk_outer.body  # type: ignore[attr-defined]
        tk.Label(
            sk,
            text="☰  SKILLS REGISTRY",
            fg=FG_BRIGHT,
            bg=BG_PANEL,
            font=FONT_SMALL,
            anchor="w",
        ).pack(fill=tk.X, padx=8, pady=(6, 0))
        self.skills_text = scrolledtext.ScrolledText(
            sk,
            bg=BG_DEEP,
            fg=FG,
            font=FONT_SMALL,
            relief=tk.FLAT,
            wrap=tk.WORD,
            state=tk.DISABLED,
            highlightthickness=0,
            takefocus=0,
        )
        self.skills_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.skills_text.tag_configure("cand", foreground=FG_MAGENTA)
        self.skills_text.tag_configure("brok", foreground=FG_ERR)
        self.skills_text.tag_configure("actv", foreground=FG_OK)
        self.skills_text.tag_configure("dim", foreground=FG_DIM)

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
        if "THINKING" in status or "WORKING" in status or "REPAIR" in status:
            self._dot_phase = (self._dot_phase + 1) % 4
            self._status_dots.configure(text="." * self._dot_phase)
        else:
            self._status_dots.configure(text="")
        self.win.after(400, self._animate_status_dots)

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
        banner = (
            "[SYSTEM] JARVIS cyber-core online. Skills learn themselves.\n"
            f"[SYSTEM] Ollama multi-model — {hint}\n"
            f"[SYSTEM] FAST:{fast}  REASONING:{reason}  CODING:{coding}\n"
            + (f"[SYSTEM] {catalog}\n" if catalog else "")
            + "[SYSTEM] Type below, then Enter or SEND  ·  F11 fullscreen  ·  Esc exit FS"
        )
        self._append("system", banner)
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
            elif status in ("THINKING", "WORKING", "PLAN", "RESEARCH"):
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
        self.send_btn.configure(state=tk.DISABLED)
        self._append("user", f"[USER] {text}")
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
