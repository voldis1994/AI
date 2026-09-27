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
install deps → test → repair → verify → save ACTIVE skill → execute →
save experience → **DONE** (only after factual verification)

## Layout

- `JARVIS.py` — entry + Matrix GUI
- `jarvis/` — stable core (brain, memory, registry, research, deps, builder, tester, verifier, ledger, orchestrator)
- `skills/` — learned Python skills (CANDIDATE → TESTING → ACTIVE)
- `data/` — SQLite memory & logs
