# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

autoresearch is a minimal framework for autonomous AI research on LLM pretraining. Agents iterate on `train.py`, run 5-minute training experiments on a single GPU (H100), and track results in a local `results.tsv`. The goal is to minimize `val_bpb` (bits per byte on a fixed validation set).

## Commands

```bash
# Install dependencies (uses uv, a Rust-based package manager)
uv sync

# One-time data preparation: downloads training shards + trains BPE tokenizer (~2 min)
uv run prepare.py

# Run a training experiment (~5 min, outputs metrics to stdout)
uv run train.py

# Run with logging (standard agent pattern)
uv run train.py > run.log 2>&1

# Extract key metrics from a run log
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

There is no test suite, linter, or build step. Verification means running `train.py` and checking that metrics appear.

## Architecture

### File Roles

| File | Role | Mutable by agent? |
|------|------|-------------------|
| `train.py` | Model, optimizer, training loop | **Yes** |
| `prepare.py` | Data download, tokenizer training, evaluation metric | **No** |
| `program.md` | Agent instructions for a research session | Human-edited |
| `analysis.ipynb` | Reads `results.tsv`, plots experiment progress | No |

`prepare.py` is the immutable ground truth. Agents must not modify it, install new packages, or change the evaluation logic.

### Data Flow

```
Parquet shards (HuggingFace climbmix-400b-shuffle)
  → BPE tokenizer (vocab=8192, GPT-4 split pattern, trained via rustbpe)
  → Dataloader (best-fit document packing, BOS-prepended, zero padding)
  → GPT model (forward pass)
  → MuonAdamW optimizer
  → val_bpb metric (on fixed shard 6542)
```

Cache lives at `~/.cache/autoresearch/` (data shards + tokenizer pickle).

### Model (`GPT` in `train.py`)

- Transformer with Flash Attention 3 (Hopper-optimized via `kernels`, fallback via `kernels-community`)
- Sliding window attention via `WINDOW_PATTERN` (e.g. `"SSSL"` = 3 half-context + 1 full-context layers, cycling)
- Multi-query attention (`n_kv_head <= n_head`)
- Value embeddings (ResFormer-style) on alternating layers
- RoPE positional embeddings
- ReLU² activation in MLP (not standard ReLU or GELU)
- Logit soft-cap: `tanh(logits / 15) * 15`
- Scaled residuals: `x = resid_lambdas[i] * x + x0_lambdas[i] * x0` (skip connection back to input embeddings)

Key hyperparameters at the top of `train.py`:
- `DEPTH`, `ASPECT_RATIO`, `HEAD_DIM`, `WINDOW_PATTERN` — architecture shape
- `TOTAL_BATCH_SIZE` (default 2¹⁹ tokens), `DEVICE_BATCH_SIZE` (default 128) — batch/grad-accum
- `EMBEDDING_LR=0.6`, `MATRIX_LR=0.04` — separate LR schedules per param type

### Optimizer (`MuonAdamW` in `train.py`)

Hybrid optimizer: **Muon** for 2D matrix parameters, **AdamW** for embeddings/scalars/lm_head.

- **Muon**: polar-express orthogonalization (second-order approximation), NorMuon variance reduction, cautious weight decay
- Dynamic Muon momentum: ramps from 0.85 → 0.95 over training
- Weight decay schedule: decays from `wd_peak` to `wd_final` over training

### Evaluation Metric

Defined and frozen in `prepare.py`:

```
BPB = (sum of per-token cross-entropy in nats / log(2)) / sum of target byte lengths
```

- Special tokens (BOS) have byte length 0 and are excluded from the denominator
- Evaluated on fixed validation shard 6542, chunked at `MAX_SEQ_LEN=2048`
- Vocab-size-independent (byte-level normalization), lower is better

### Agent Experiment Loop (per `program.md`)

1. Create a branch: `git checkout -b autoresearch/<tag>`
2. Initialize `results.tsv` with header: `commit\tval_bpb\tmemory_gb\tstatus\tdescription`
3. Modify `train.py` → `uv run train.py > run.log 2>&1`
4. Extract `val_bpb` and `peak_vram_mb` from log
5. If improved → commit and log as `keep`; if not → `git reset --hard HEAD` and log as `discard`
6. Repeat indefinitely

`results.tsv` is gitignored (local only). Commits track only code changes.

## Constraints

- Do not modify `prepare.py` or the `evaluate_bpb()` function
- Do not add new pip/uv dependencies
- `MAX_SEQ_LEN=2048` and `VOCAB_SIZE=8192` are fixed constants set in `prepare.py`; do not override them in `train.py`
- Training time budget is 300 seconds (wall clock, excluding startup); this is enforced by the training loop
