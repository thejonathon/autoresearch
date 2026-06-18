# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

**autoresearch** is an autonomous AI research loop for neural network training. An AI agent reads `program.md` for research instructions, modifies `train.py`, runs a fixed 5-minute training experiment, evaluates the result, and keeps or discards the change — cycling indefinitely until interrupted. Expected throughput: ~12 experiments/hour.

## Setup

```bash
uv sync                   # Install dependencies
uv run prepare.py         # One-time: download dataset shards and train BPE tokenizer
```

Data is cached at `~/.cache/autoresearch/`. Re-run `prepare.py` only if the cache is missing.

## Running Experiments

```bash
uv run train.py                                      # Single 5-minute experiment
uv run train.py > run.log 2>&1                      # Capture output for metric extraction
grep "^val_bpb:\|^peak_vram_mb:" run.log            # Extract key metrics
```

Kill if training exceeds 10 minutes (indicates a hang).

## Architecture

Two source files with strictly defined roles:

**`prepare.py`** — Fixed evaluation harness. Never modify this file. It provides:
- Dataset downloading and tokenization (BPE, 8,192 vocab size, `rustbpe`)
- The `evaluate_bpb()` function — the canonical metric (bits per byte, lower is better)
- The pinned validation shard (shard_06542) for fair cross-experiment comparison
- Constants: `MAX_SEQ_LEN=2048`, `TIME_BUDGET=300` (seconds), `EVAL_TOKENS=40×524,288`

**`train.py`** — Agent-modifiable training script (~630 lines). Contains:
- `GPTConfig` dataclass: architecture settings (layers, heads, embedding dim, window pattern)
- `CausalSelfAttention`: multi-head attention with alternating sliding-window patterns (`S`=half context, `L`=full sequence)
- `GPT` model: transformer with RMSNorm, rotary embeddings, optional per-layer value embeddings (ResFormer), flash-attention-3
- `MuonAdamW` optimizer: Muon for 2D weight matrices, AdamW for everything else
- Clearly labeled hyperparameter block at the top: `DEPTH`, `ASPECT_RATIO`, `HEAD_DIM`, `WINDOW_PATTERN`, `TOTAL_BATCH_SIZE`, per-group learning rates, `WEIGHT_DECAY`, `ADAM_BETAS`, warmup/cooldown schedule

**`program.md`** — Human-written agent instructions. Read this before starting an experiment loop. It specifies the branch naming convention, results logging format, and loop behavior.

## The Experiment Loop (per `program.md`)

1. Create branch `autoresearch/<tag>` (e.g., `autoresearch/jun18`)
2. Initialize `results.tsv` with header: `commit`, `val_bpb`, `memory_gb`, `status`, `description`
3. Modify `train.py` → `git commit` → `uv run train.py > run.log 2>&1`
4. Extract `val_bpb` from output; log result to `results.tsv`
5. If improved: keep commit. If not: `git reset --hard HEAD~1`
6. If crash: fix simple bugs or discard; log `status=crash`
7. Repeat — never stop until manually interrupted

## Key Metric

`val_bpb` (validation bits per byte): cross-entropy in nats divided by `ln(2) × byte_count`. Vocab-size-independent, so experiments with different vocab sizes are directly comparable. **Lower is better.**

Training output also shows: `peak_vram_mb`, `mfu_percent`, `total_tokens_M`, `num_params_M`, `num_steps`.

## Constraints

- Only modify `train.py` — no changes to `prepare.py` or new files
- No new dependencies
- Single NVIDIA GPU required (optimized for H100, CUDA 12.8)
- `results.tsv` is gitignored intentionally — the agent writes it without committing

## Result Analysis

Open `analysis.ipynb` to plot `val_bpb` over experiments, view the running minimum (true progress), and annotate kept improvements.
