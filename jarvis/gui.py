"""
JARVIS cyber-HUD GUI — visual shell only.

Magenta neon / scanline aesthetic matching the product HUD mockup.
Orchestrator / brain / skills behavior is unchanged — this module is presentation.
"""

from __future__ import annotations

import math
import random
import threading
import time
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

# ── Cyber HUD palette (reference mockup) ────────────────────────────────
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
SCAN = "#1e041f"
BIN = "#4a0f3a"

FONT_MONO = ("Consolas", 11)
FONT_TITLE = ("Consolas", 42, "bold")
FONT_SUB = ("Consolas", 9)
FONT_SMALL = ("Consolas", 9)
FONT_STAT = ("Consolas", 10)


def _mono() -> str:
    """Pick first available monospace-ish family."""
    if tk is None or tkfont is None:
        return "Consolas"
    families = {f.lower() for f in tkfont.families()}
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


class _HudFrame(tk.Frame):
    """Panel with magenta double-line border and corner ticks."""

    def __init__(self, master, **kw):
        super().__init__(master, bg=BG_PANEL, highlightthickness=0, **kw)
        self._border = tk.Canvas(
            self, bg=BG, highlightthickness=0, height=1, width=1
        )
        self._border.place(x=0, y=0, relwidth=1, relheight=1)
        self._inner = tk.Frame(self, bg=BG_PANEL, highlightthickness=0)
        self._inner.place(x=8, y=8, relwidth=1, relheight=1, width=-16, height=-16)
        self.bind("<Configure>", self._redraw)
        self._border.bind("<Configure>", self._redraw)

    @property
    def body(self) -> tk.Frame:
        return self._inner

    def _redraw(self, _evt=None) -> None:
        c = self._border
        c.delete("all")
        w = max(c.winfo_width(), 2)
        h = max(c.winfo_height(), 2)
        # Soft outer glow layers
        c.create_rectangle(1, 1, w - 2, h - 2, outline=BORDER_DIM, width=1)
        c.create_rectangle(3, 3, w - 4, h - 4, outline=BORDER, width=1)
        c.create_rectangle(6, 6, w - 7, h - 7, outline=FG_MAGENTA, width=2)
        # Corner brackets
        L = 22
        for x0, y0, dx, dy in (
            (6, 6, 1, 1),
            (w - 7, 6, -1, 1),
            (6, h - 7, 1, -1),
            (w - 7, h - 7, -1, -1),
        ):
            c.create_line(x0, y0, x0 + dx * L, y0, fill=FG_GLOW, width=2)
            c.create_line(x0, y0, x0, y0 + dy * L, fill=FG_GLOW, width=2)
        # Top accent rail
        c.create_line(30, 6, w - 30, 6, fill=FG_MAGENTA, width=1)
        # Small tech ticks mid-sides
        mid_y = h // 2
        c.create_line(6, mid_y - 8, 6, mid_y + 8, fill=FG_GLOW, width=2)
        c.create_line(w - 7, mid_y - 8, w - 7, mid_y + 8, fill=FG_GLOW, width=2)


class JarvisGUI:
    """Magenta cyber-HUD interface for JARVIS (visual shell)."""

    def __init__(self, root_dir: Path) -> None:
        if tk is None:
            raise RuntimeError("tkinter is required for GUI mode")
        self.root_dir = root_dir
        self.busy = False
        self._request_seq = 0
        self._dot_phase = 0
        self._glitch_phase = 0
        self._bin_job = None

        family = _mono()
        global FONT_MONO, FONT_TITLE, FONT_SUB, FONT_SMALL, FONT_STAT
        FONT_MONO = (family, 11)
        FONT_TITLE = (family, 42, "bold")
        FONT_SUB = (family, 9)
        FONT_SMALL = (family, 9)
        FONT_STAT = (family, 10)

        self.win = tk.Tk()
        self.win.title("JARVIS")
        self.win.configure(bg=BG)
        self.win.geometry("1280x820")
        self.win.minsize(980, 680)

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
        self.win.after(80, self._animate_status_dots)
        self.win.after(100, self._animate_binary)
        self.win.after(350, self._animate_title_glitch)
        self.win.after(4000, self._tick_stats)

    # ── UI construction ─────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Full-window atmosphere layer (binary rain + scanlines)
        self.bg_canvas = tk.Canvas(self.win, bg=BG, highlightthickness=0)
        self.bg_canvas.place(x=0, y=0, relwidth=1, relheight=1)
        self._binary_cols: list[dict[str, Any]] = []
        self.win.bind("<Configure>", self._on_root_configure)

        # Outer chrome frame
        self.chrome = tk.Canvas(self.win, bg=BG, highlightthickness=0)
        self.chrome.place(x=0, y=0, relwidth=1, relheight=1)
        self.win.bind("<Configure>", self._redraw_chrome, add="+")

        # Foreground shell inset so binary rain / scanlines show at edges
        shell = tk.Frame(self.win, bg=BG)
        shell.place(x=18, y=14, relwidth=1, relheight=1, width=-36, height=-28)
        self.shell = shell

        # ── Header: icon | centered JARVIS | map + window chrome ─────────
        header = tk.Frame(shell, bg=BG, height=88)
        header.pack(fill=tk.X, padx=14, pady=(8, 2))
        header.pack_propagate(False)

        left_h = tk.Frame(header, bg=BG)
        left_h.place(x=0, y=8, width=160, height=72)

        self.brain_icon = tk.Canvas(
            left_h, width=48, height=48, bg=BG, highlightthickness=0
        )
        self.brain_icon.pack(side=tk.LEFT, padx=(4, 8), pady=8)
        self._draw_brain_icon(self.brain_icon, 48)
        tk.Label(
            left_h,
            text="JARVIS",
            fg=FG_DIM,
            bg=BG,
            font=FONT_SMALL,
        ).pack(side=tk.LEFT, pady=18)

        # Center title block
        center_h = tk.Frame(header, bg=BG)
        center_h.place(relx=0.5, rely=0.5, anchor="center")

        self.title_canvas = tk.Canvas(
            center_h, width=360, height=52, bg=BG, highlightthickness=0
        )
        self.title_canvas.pack()
        self._draw_title_logo()

        self.subtitle = tk.Label(
            center_h,
            text="AUTONOMOUS  •  SELF-LEARNING  •  VERIFIED",
            fg=FG_DIM,
            bg=BG,
            font=FONT_SUB,
        )
        self.subtitle.pack()

        right_h = tk.Frame(header, bg=BG)
        right_h.place(relx=1.0, x=0, y=4, width=200, height=80, anchor="ne")

        win_ctrl = tk.Canvas(
            right_h, width=70, height=18, bg=BG, highlightthickness=0
        )
        win_ctrl.pack(side=tk.TOP, anchor="e", padx=4)
        self._draw_window_controls(win_ctrl)

        self.map_canvas = tk.Canvas(
            right_h, width=130, height=48, bg=BG, highlightthickness=0
        )
        self.map_canvas.pack(side=tk.TOP, anchor="e", padx=4, pady=(4, 0))
        self._draw_world_map(self.map_canvas)

        # Status strip: STATUS left · BRAIN right
        status_wrap = _HudFrame(shell)
        status_wrap.pack(fill=tk.X, padx=14, pady=(4, 8))
        sf = status_wrap.body

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
        self.status_lbl.pack(side=tk.LEFT, padx=(12, 0), pady=8)
        self._status_dots = tk.Label(
            left_st, text="", fg=FG_MAGENTA, bg=BG_PANEL, font=FONT_MONO
        )
        self._status_dots.pack(side=tk.LEFT)

        right_st = tk.Frame(sf, bg=BG_PANEL)
        right_st.pack(side=tk.RIGHT, fill=tk.Y, padx=12)
        self.brain_var = tk.StringVar(value="BRAIN: … | multi-model")
        self.brain_dot = tk.Label(
            right_st, text="●", fg=FG_OK, bg=BG_PANEL, font=FONT_MONO
        )
        self.brain_dot.pack(side=tk.LEFT, pady=8)
        self.brain_lbl = tk.Label(
            right_st,
            textvariable=self.brain_var,
            fg=FG_OK,
            bg=BG_PANEL,
            font=FONT_SMALL,
            anchor="e",
        )
        self.brain_lbl.pack(side=tk.LEFT, padx=(4, 0), pady=8)

        # ── Body ────────────────────────────────────────────────────────
        body = tk.Frame(shell, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=14, pady=2)

        # Conversation
        log_hud = _HudFrame(body)
        log_hud.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_frame = log_hud.body

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
            padx=10,
            pady=8,
        )
        self.log.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.log.tag_configure("user", foreground=FG_USER)
        self.log.tag_configure("jarvis", foreground=FG)
        self.log.tag_configure("system", foreground=FG_DIM)
        self.log.tag_configure("error", foreground=FG_ERR)
        self.log.tag_configure("warn", foreground=FG_WARN)
        self.log.tag_configure("ok", foreground=FG_OK)
        self.log.tag_configure("tag", foreground=FG_MAGENTA)
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

        # Side column
        side_col = tk.Frame(body, bg=BG, width=320)
        side_col.pack(side=tk.RIGHT, fill=tk.Y, padx=(12, 0))
        side_col.pack_propagate(False)

        # Memory / stats
        mem_hud = _HudFrame(side_col)
        mem_hud.pack(fill=tk.BOTH, expand=False, pady=(0, 10))
        mem = mem_hud.body
        tk.Label(
            mem,
            text="▣  MEMORY / STATS",
            fg=FG_BRIGHT,
            bg=BG_PANEL,
            font=FONT_SMALL,
            anchor="w",
        ).pack(fill=tk.X, padx=8, pady=(6, 0))

        mem_body = tk.Frame(mem, bg=BG_PANEL)
        mem_body.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.stats_text = tk.Text(
            mem_body,
            bg=BG_DEEP,
            fg=FG,
            font=FONT_STAT,
            relief=tk.FLAT,
            height=12,
            wrap=tk.WORD,
            state=tk.DISABLED,
            highlightthickness=0,
            width=26,
        )
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 2), pady=4)
        self.stats_text.tag_configure("ok", foreground=FG_OK)
        self.stats_text.tag_configure("dim", foreground=FG_DIM)
        self.stats_text.tag_configure("hi", foreground=FG_BRIGHT)

        self.holo_brain = tk.Canvas(
            mem_body, width=120, height=140, bg=BG_DEEP, highlightthickness=0
        )
        self.holo_brain.pack(side=tk.RIGHT, padx=(2, 4), pady=4)
        self._draw_holo_brain(self.holo_brain)

        # Skills
        sk_hud = _HudFrame(side_col)
        sk_hud.pack(fill=tk.BOTH, expand=True)
        sk = sk_hud.body
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
        )
        self.skills_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.skills_text.tag_configure("cand", foreground=FG_MAGENTA)
        self.skills_text.tag_configure("brok", foreground=FG_ERR)
        self.skills_text.tag_configure("actv", foreground=FG_OK)
        self.skills_text.tag_configure("dim", foreground=FG_DIM)
        try:
            self.skills_text.vbar.configure(
                troughcolor=BG_PANEL,
                background=FG_MAGENTA,
                activebackground=FG_GLOW,
                borderwidth=0,
                width=10,
            )
        except Exception:
            pass

        # ── Input ───────────────────────────────────────────────────────
        input_hud = _HudFrame(shell)
        input_hud.pack(fill=tk.X, padx=14, pady=(6, 12))
        input_frame = input_hud.body

        prompt = tk.Label(
            input_frame,
            text="›",
            fg=FG_MAGENTA,
            bg=BG_PANEL,
            font=(FONT_MONO[0], 18, "bold"),
        )
        prompt.pack(side=tk.LEFT, padx=(12, 6))

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
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=12, pady=8)

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
            padx=22,
            pady=10,
            highlightbackground=BORDER,
            highlightthickness=2,
            cursor="hand2",
        )
        self.send_btn.pack(side=tk.LEFT, padx=(10, 12), pady=8)

    # ── Decorative drawers ──────────────────────────────────────────────

    def _redraw_chrome(self, _evt=None) -> None:
        c = self.chrome
        c.delete("all")
        w = max(c.winfo_width(), 2)
        h = max(c.winfo_height(), 2)
        # Outer double frame
        c.create_rectangle(4, 4, w - 5, h - 5, outline=BORDER_DIM, width=1)
        c.create_rectangle(8, 8, w - 9, h - 9, outline=BORDER, width=2)
        L = 28
        for x0, y0, dx, dy in (
            (8, 8, 1, 1),
            (w - 9, 8, -1, 1),
            (8, h - 9, 1, -1),
            (w - 9, h - 9, -1, -1),
        ):
            c.create_line(x0, y0, x0 + dx * L, y0, fill=FG_GLOW, width=2)
            c.create_line(x0, y0, x0, y0 + dy * L, fill=FG_GLOW, width=2)
        try:
            self.shell.lift()
        except Exception:
            pass

    def _draw_title_logo(self, glitch: int = 0) -> None:
        c = self.title_canvas
        c.delete("all")
        text = "JARVIS"
        # Soft glow layers
        for dx, dy, col in (
            (0, 0, FG_TITLE),
            (1, 0, FG_MAGENTA),
            (-1, 0, FG_GLOW),
        ):
            c.create_text(
                180 + dx,
                26 + dy,
                text=text,
                fill=col,
                font=FONT_TITLE,
            )
        # Horizontal glitch slash through mid (reference look)
        y = 26 + (glitch % 3) - 1
        c.create_rectangle(40, y, 320, y + 2, fill=BG, outline="")
        c.create_line(40, y + 1, 320, y + 1, fill=FG_GLOW, width=1)
        if glitch % 5 == 0:
            c.create_text(
                182, 24, text=text, fill=FG_BRIGHT, font=FONT_TITLE
            )

    def _draw_window_controls(self, c: tk.Canvas) -> None:
        c.delete("all")
        # minimize / maximize / close as thin magenta glyphs
        c.create_line(8, 10, 20, 10, fill=FG_MAGENTA, width=2)
        c.create_rectangle(30, 5, 42, 15, outline=FG_MAGENTA, width=1)
        c.create_line(52, 5, 64, 15, fill=FG_MAGENTA, width=2)
        c.create_line(64, 5, 52, 15, fill=FG_MAGENTA, width=2)

    def _draw_brain_icon(self, c: tk.Canvas, size: int) -> None:
        c.delete("all")
        cx, cy = size / 2, size / 2
        c.create_oval(6, 8, size - 6, size - 4, outline=FG_MAGENTA, width=2)
        c.create_oval(10, 12, size / 2 + 2, size - 8, outline=FG_DIM, width=1)
        c.create_oval(size / 2 - 2, 12, size - 10, size - 8, outline=FG_DIM, width=1)
        for a in range(0, 360, 36):
            r1, r2 = 9, 16
            x1 = cx + r1 * math.cos(math.radians(a))
            y1 = cy + r1 * math.sin(math.radians(a))
            x2 = cx + r2 * math.cos(math.radians(a))
            y2 = cy + r2 * math.sin(math.radians(a))
            c.create_line(x1, y1, x2, y2, fill=FG_GLOW, width=1)
        c.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill=FG_OK, outline="")

    def _draw_world_map(self, c: tk.Canvas) -> None:
        c.delete("all")
        w, h = 130, 48
        c.create_rectangle(1, 1, w - 2, h - 2, outline=BORDER_DIM, width=1)
        # Dot-matrix continents
        nodes = [
            (18, 20), (28, 14), (38, 18), (48, 12), (58, 16),
            (70, 14), (82, 20), (95, 18), (108, 24),
            (22, 30), (40, 32), (55, 28), (75, 30), (92, 34), (105, 32),
        ]
        for i, (x, y) in enumerate(nodes):
            c.create_oval(x - 1.5, y - 1.5, x + 1.5, y + 1.5, fill=FG_MAGENTA, outline="")
            if i:
                x0, y0 = nodes[i - 1]
                if abs(x - x0) < 40:
                    c.create_line(x0, y0, x, y, fill=BORDER_DIM, width=1)

    def _draw_holo_brain(self, c: tk.Canvas, pulse: float = 0.0) -> None:
        c.delete("all")
        w, h = 120, 140
        cx, cy = w / 2, h / 2 - 6
        # Crosshair + rings
        c.create_oval(6, 8, w - 6, h - 20, outline=BORDER_DIM, width=1)
        c.create_oval(14, 16, w - 14, h - 28, outline=FG_MAGENTA, width=2)
        c.create_oval(24, 26, w - 24, h - 38, outline=BORDER_DIM, width=1)
        c.create_line(cx, 10, cx, h - 22, fill=BORDER_DIM, width=1)
        c.create_line(10, cy, w - 10, cy, fill=BORDER_DIM, width=1)
        for a in range(0, 360, 12):
            r1, r2 = 48, 54
            x1 = cx + r1 * math.cos(math.radians(a))
            y1 = cy + r1 * math.sin(math.radians(a))
            x2 = cx + r2 * math.cos(math.radians(a))
            y2 = cy + r2 * math.sin(math.radians(a))
            col = FG_GLOW if a % 36 == 0 else BORDER_DIM
            c.create_line(x1, y1, x2, y2, fill=col, width=1)
        # Wireframe lobes
        scale = 1.0 + 0.03 * math.sin(pulse)
        c.create_oval(
            cx - 30 * scale, cy - 24 * scale,
            cx + 2, cy + 28 * scale,
            outline=FG_BRIGHT, width=1,
        )
        c.create_oval(
            cx - 2, cy - 24 * scale,
            cx + 30 * scale, cy + 28 * scale,
            outline=FG_BRIGHT, width=1,
        )
        for dy in (-12, -2, 8, 18):
            c.create_arc(
                cx - 28, cy + dy - 8, cx + 28, cy + dy + 8,
                start=200, extent=140, style=tk.ARC, outline=FG_DIM, width=1,
            )
        # Neural nodes
        for a in range(0, 360, 45):
            r = 18
            x = cx + r * math.cos(math.radians(a + pulse * 20))
            y = cy + r * math.sin(math.radians(a + pulse * 20))
            c.create_oval(x - 2, y - 2, x + 2, y + 2, fill=FG_OK, outline="")
        c.create_text(cx, h - 8, text="NEURAL CORE", fill=FG_DIM, font=FONT_SUB)

    def _on_root_configure(self, _evt=None) -> None:
        self._seed_binary_cols(force=True)
        self._redraw_chrome()

    def _seed_binary_cols(self, force: bool = False) -> None:
        c = self.bg_canvas
        w = max(c.winfo_width(), 2)
        h = max(c.winfo_height(), 2)
        if w < 10:
            return
        c.delete("scan")
        for y in range(0, h, 3):
            c.create_line(0, y, w, y, fill=SCAN, tags="scan")
        if force or not self._binary_cols:
            n = max(14, w // 55)
            self._binary_cols = []
            for i in range(n):
                self._binary_cols.append(
                    {
                        "x": 10 + i * (w / n) + random.uniform(-8, 8),
                        "y": random.randint(0, max(h, 1)),
                        "speed": random.uniform(1.5, 4.2),
                        "chars": "".join(
                            random.choice("01") for _ in range(22)
                        ),
                    }
                )

    def _animate_binary(self) -> None:
        c = self.bg_canvas
        w = max(c.winfo_width(), 2)
        h = max(c.winfo_height(), 2)
        if not self._binary_cols:
            self._seed_binary_cols()
        c.delete("bin")
        for col in self._binary_cols:
            col["y"] = (col["y"] + col["speed"]) % (h + 50)
            y = col["y"]
            for i, ch in enumerate(col["chars"]):
                yy = (y + i * 11) % (h + 50) - 20
                fill = FG_MAGENTA if i == 0 else (BORDER if i < 3 else BIN)
                c.create_text(
                    col["x"],
                    yy,
                    text=ch,
                    fill=fill,
                    font=FONT_SUB,
                    tags="bin",
                )
        try:
            self.chrome.lift()
            self.shell.lift()
        except Exception:
            pass
        self.win.after(70, self._animate_binary)

    def _animate_title_glitch(self) -> None:
        self._glitch_phase = (self._glitch_phase + 1) % 12
        self._draw_title_logo(self._glitch_phase)
        self.win.after(280, self._animate_title_glitch)

    def _animate_status_dots(self) -> None:
        status = self.status_var.get()
        if "THINKING" in status or "WORKING" in status or "REPAIR" in status:
            self._dot_phase = (self._dot_phase + 1) % 4
            self._status_dots.configure(text="." * self._dot_phase)
        else:
            self._status_dots.configure(text="")
        self.win.after(400, self._animate_status_dots)

    def _bind_keys(self) -> None:
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
        banner = (
            "[SYSTEM] JARVIS cyber-core online. Skills learn themselves.\n"
            f"[SYSTEM] Ollama multi-model — {hint}\n"
            f"[SYSTEM] FAST:{fast}  REASONING:{reason}  CODING:{coding}\n"
            + (f"[SYSTEM] {catalog}\n" if catalog else "")
            + "[SYSTEM] Type a message, a task, or /help"
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
            short = name.title() if name != "REASONING" else "Reasoning"
            if name == "CODING":
                short = "Coding"
            return f"{short:<10}{model} {mark}{warm}"

        self.stats_text.configure(state=tk.NORMAL)
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert(tk.END, "Brain     ", "dim")
        self.stats_text.insert(
            tk.END, f"{brain} ✓\n" if brain == "ONLINE" else f"{brain}\n",
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
            try:
                self._draw_holo_brain(
                    self.holo_brain, pulse=time.time() % 6.28
                )
            except Exception:
                pass
        self.win.after(4000, self._tick_stats)

    # ── Interaction ─────────────────────────────────────────────────────

    def _on_send(self) -> None:
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
                    self.entry.focus_set()
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
        self.entry.focus_set()
        self.win.mainloop()
