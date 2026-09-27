# JARVIS

Autonomous self-learning AI agent. Brain: **Ollama** (`qwen2.5-coder:7b`).

## Run

```bash
pip install -r requirements.txt
ollama pull qwen2.5-coder:7b   # if needed
ollama serve                   # if not already running
python JARVIS.py               # Matrix GUI
python JARVIS.py --cli         # terminal
python JARVIS.py --check       # self-check
```

## Cycle

REQUEST → PLAN → check capabilities → research → learn → build skill →
install deps → **test in subprocess** → repair → **independent verify** →
save ACTIVE skill → execute → verify → save experience → **DONE**

DONE only after verifier PASS. Skill self-evidence is never enough.

## Safety rules

- Skills run in isolated Python subprocesses (timeout / stdout / stderr / kill)
- New skill versions never overwrite ACTIVE until PASS → then archive old
- `pip install --user` + mandatory re-import check (no UAC bypass)
- Brain status: `ONLINE` only if exact model `qwen2.5-coder:7b` exists; else `MODEL MISSING` / `OFFLINE`
- Research results store url, title, source/provider, timestamp, query

## Layout

- `JARVIS.py` — entry + Matrix GUI (`--cli`, `--check`)
- `jarvis/` — stable core
- `skills/` — learned Python skills (CANDIDATE → TESTING → ACTIVE; old → archive)
- `data/` — SQLite memory & logs
