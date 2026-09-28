# JARVIS

Autonomous self-learning AI agent. Brain: **Ollama multi-model router**
(`FAST` / `REASONING` / `CODING` — configured in `jarvis/model_config.py`).

| Tier | Preferred | Use |
|------|-----------|-----|
| FAST | `qwen3:4b` → `qwen3:8b` → … | intent, parsing, args, query gen, simple Q&A |
| REASONING | `qwen3:30b` → `qwen3:8b` → … | planning, learning, research, diagnosis, semantics |
| CODING | `qwen3-coder:30b` → `qwen2.5-coder:7b` → … | skill generation, repair, code analysis |

JARVIS auto-selects the best **installed** model per tier (Ollama `/api/tags`).
Missing primaries fall through; unknown local tags can still be picked by size/family.
Runner crashes (`llama-server` / `0xe06d7363`) mark that model skipped and retry the next.
Failed FAST results may escalate to REASONING. Failed CODING retries CODING with failure context.

## Run

```bash
pip install -r requirements.txt
# pull any models you want from the catalog (primaries preferred)
ollama pull qwen3:4b
ollama pull qwen3:30b
ollama pull qwen3-coder:30b
# or fall back to legacy:
ollama pull qwen2.5-coder:7b
ollama serve                   # if not already running
python JARVIS.py               # Matrix GUI
python JARVIS.py --cli         # terminal
python JARVIS.py --check       # self-check
```

## Cycle

REQUEST → classify work → **ModelRouter** → EXECUTE → OBSERVE →
(escalate if needed) → VERIFY → DONE

Skill path still:
REQUEST → PLAN → check capabilities → research → learn →
**BUILD → TEST → OBSERVE → DIAGNOSE → RESEARCH? → REPAIR → RETEST → VERIFY → ACTIVE**
→ execute → verify → save experience → **DONE**

DONE only after verifier PASS. Skill self-evidence / LLM self-evaluation is never enough —
deterministic Python checks remain authoritative where possible.
Failures, diagnoses, and successful solutions are stored in SQLite.
Repeated failures force a new approach (not the same code again).

## Safety rules

- Skills run in isolated Python subprocesses (timeout / stdout / stderr / kill)
- New skill versions never overwrite ACTIVE until PASS → then archive old
- `pip install --user` + mandatory re-import check (no UAC bypass)
- Brain status: `ONLINE` if any configured catalog model is installed; else `MODEL MISSING` / `OFFLINE`
- Model names live only in `jarvis/model_config.py`
- Per-request `request_id` isolation — never reuse a prior final_result as the next answer
- Research results store url, title, source/provider, timestamp, query

## Layout

- `JARVIS.py` — entry + Matrix GUI (`--cli`, `--check`)
- `jarvis/model_config.py` — central model/tier configuration
- `jarvis/model_router.py` — Multi-Model Router
- `jarvis/` — stable core
- `skills/` — learned Python skills (CANDIDATE → TESTING → ACTIVE; old → archive)
- `data/` — SQLite memory & logs
