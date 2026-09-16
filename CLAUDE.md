# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                        # install dependencies
uv run prepare.py              # one-time data prep (~2 min): downloads shards, trains tokenizer
uv run prepare.py --num-shards 8  # prep with custom shard count (default: 10)
uv run train.py                # run one training experiment (~5 min fixed wall-clock)
uv run train.py > run.log 2>&1 # run and capture output
grep "^val_bpb:" run.log       # extract primary metric from output
```

There is no test suite. Validation is purely via the `val_bpb` metric printed to stdout after each training run.

## Architecture

The project is a two-file autonomous ML research framework for GPT pretraining experiments.

**`prepare.py` — fixed infrastructure (do not modify)**
- Downloads training shards from Hugging Face (`climbmix-400b-shuffle`) to `~/.cache/autoresearch/data/`
- Trains an 8192-vocab BPE tokenizer; stores it to `~/.cache/autoresearch/tokenizer/`
- Exports to `train.py`: `Tokenizer`, `make_dataloader()`, `evaluate_bpb()`, `MAX_SEQ_LEN=2048`, `TIME_BUDGET=300`
- `evaluate_bpb()` is the fixed evaluation function — bits/byte on a pinned validation shard (vocab-size-independent, lower is better)

**`train.py` — the agent-modifiable experiment file**
- Imports `prepare` for the dataloader and eval function
- Defines `GPT` model: transformer with causal attention, RoPE, GQA, Flash Attention 3, sliding window pattern (`SSSL`), value embeddings (ResFormer-style), per-layer residual lambdas, RMSNorm, softcap=15
- `MuonAdamW` optimizer: Muon (polar-express orthogonalization) for 2D matrix params; AdamW for embeddings, scalars, and lm_head; per-group learning rates with warmup/cooldown schedule
- Training loop runs for exactly `TIME_BUDGET=300` seconds, then calls `evaluate_bpb()` and prints a metrics summary block starting with `---`
- All hyperparameters are inline constants (no CLI flags): `DEPTH`, `ASPECT_RATIO`, `HEAD_DIM`, `WINDOW_PATTERN`, `DEVICE_BATCH_SIZE`, `TOTAL_BATCH_SIZE`, `MATRIX_LR`, `EMBEDDING_LR`, etc.

**`program.md` — agent workflow instructions**
Defines the autonomous experiment loop: modify `train.py` → `git commit` → `uv run train.py` → check `val_bpb` → keep commit if improved, `git reset --hard HEAD~1` if not → repeat indefinitely.

## Key constraints

- **Only `train.py` should be modified** during experiments. `prepare.py` is read-only infrastructure.
- **No new dependencies** — the agent cannot add packages beyond what's in `pyproject.toml`.
- The time budget (300s) is a hard wall enforced in `prepare.py`; startup (torch.compile warmup, ~20–30s) is excluded from the budget.
- `results.tsv` and `run.log` are gitignored; experiment tracking is done via git commits on a per-session branch (e.g. `autoresearch/jun18`).
- Loss explosion guard: training aborts immediately if loss > 100 or NaN.
- GC is frozen after the first training step to avoid ~500ms stalls.

## Output format

After each run, stdout ends with:

```
---
val_bpb:          0.997900   # primary metric (lower = better)
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```
