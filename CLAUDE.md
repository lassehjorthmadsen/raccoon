# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Raccoon is a backgammon AI using AlphaZero-style self-play training (ResNet policy-value network + MCTS) to beat GNUBG at money game. OpenSpiel provides game logic; all ML/search code is written from scratch in Python/PyTorch.

## Commands

```bash
make setup          # Install in editable mode with dev deps
make test           # Run all tests (pytest tests/ -v)
make smoke          # Quick sanity check: 2 iterations, 3 games, 10 sims
make train          # Full training run
make eval           # Checkpoint vs checkpoint evaluation
make eval-gnubg     # Automated benchmark against GNUBG (gnubg-nn engine; level=world → full-width 2-ply)
make play           # Interactive terminal play
make download-wildbg # Fetch wildbg-training labeled positions (CC0) into data/wildbg/
make pull-data       # Fetch data/bglab/ + data/bgsage/ from GCS — run once on a fresh machine
make push-data       # Push local changes to data/bglab/ or data/bgsage/ back to GCS
make pretrain-smoke  # 1 epoch on 2k positions — sanity check before a full pretrain
make pretrain NAME=… # Full supervised pretraining on wildbg data

# Run a single test file or test
python3 -m pytest tests/test_encoder.py -v
python3 -m pytest tests/test_mcts.py::test_name -v

# Training with custom params (--experiment-name required; outputs go to experiments/<name>/)
python3 scripts/train.py --experiment-name my-run --iterations 100 --games-per-iter 50 \
  --simulations 100 --lr 0.001 --channels 128 --num-blocks 6 --checkpoint-every 10

# Resume training from a checkpoint (architecture is read from the checkpoint)
python3 scripts/train.py --experiment-name my-run --iterations 100 \
  --resume experiments/my-run/checkpoints/iter_0200.pt

# Supervised pretraining on wildbg data, then continue with self-play
./scripts/download_wildbg.sh   # one-off; ~17 MB into data/wildbg/
python3 scripts/pretrain.py --experiment-name pretrain-wildbg-v1 --epochs 20
python3 scripts/train.py --experiment-name exp007-pretrained \
  --resume experiments/pretrain-wildbg-v1/checkpoints/pretrained.pt --simulations 800

# Stage-2 policy pretraining via 1-ply V-lookahead on bglab match positions.
# Requires the match archive at data/bglab/{lasse,Llabba}/{raw,analyzed} (plain
# files, no git tracking — see data/README.md for how to refresh from source).
# Stage 1 (synthesis) is the long step — ~8 h on iMac CPU, ~minutes on T4 GPU.
python3 scripts/synthesize_policy_dataset.py \
  --pretrained experiments/pretrain-wildbg-v1/checkpoints/pretrained.pt \
  --out data/bglab/cache/policy_cache.npz
python3 scripts/pretrain_policy.py --experiment-name pretrain-wildbg-v2 \
  --base-checkpoint experiments/pretrain-wildbg-v1/checkpoints/pretrained.pt \
  --cache data/bglab/cache/policy_cache.npz --epochs 10
```

## Architecture

The project follows a milestone-based plan (see `docs/plan.md` for full details). The seven milestones build on each other: env → model → search → training → internal eval → GNUBG benchmark → strength scaling.

### Core Pipeline

1. **`raccoon/env/`** — OpenSpiel wrapper + custom tensor encoder + action mapping
   - `game_wrapper.py`: Wraps `pyspiel.load_game("backgammon")`, handles perspective flipping so the network always sees the board from the current player's view
   - `encoder.py`: Converts board state to **(26, 2, 12)** float32 tensor (26 channels, 2 rows, 12 columns). Channels: 4 checker planes per player, bar/borne-off/dice broadcast planes, mid-doubles flag, plus handcrafted features (pip count, blots, anchors, contact pressure). `CHANNEL_NAMES` is the authoritative registry of channel meanings; `dump_tensor()` pretty-prints the planes for debugging.
   - `actions.py`: Legal action masking over OpenSpiel's 1352 action space

2. **`raccoon/model/network.py`** — `RaccoonNet`: ResNet with shared trunk → policy head (1352 logits) + value head (scalar in [-1,1] via tanh). Default: 6 residual blocks, 128 channels. `predict()` method handles masking + softmax for MCTS inference.

3. **`raccoon/search/mcts.py`** — AlphaZero MCTS with PUCT selection. Chance nodes (dice rolls) are sampled and skipped — the tree only contains decision/terminal nodes. Temperature controls exploration vs exploitation.

4. **`raccoon/train/`** — Self-play loop
   - `self_play.py`: Plays games, records (observation, MCTS policy, outcome) tuples
   - `replay_buffer.py`: Circular buffer of training positions
   - `coach.py`: Orchestrates self-play → replay buffer → SGD training → checkpoint. Logs full config (network architecture, hyperparams, system info) to JSONL.

5. **`raccoon/eval/`** — Evaluation infrastructure
   - `arena.py`: Checkpoint vs checkpoint matches
   - `gnubg_harness.py`: Automated cubeless money game matches against GNUBG's evaluation engine (the `gnubg-nn` library — GNUBG's real nets, programmatically queryable; default `level=world` → full-width 2-ply)

6. **`raccoon/protocol/rgp.py`** — Raccoon Game Protocol: text-based stdin/stdout protocol inspired by UCI for future GUI frontends

7. **`raccoon/cli/play.py`** — Terminal interface for human vs Raccoon play

### Key Design Details

- Board encoding is always from the **current player's perspective** (perspective flip applied in the wrapper)
- MCTS never evaluates the network at chance nodes — it samples dice and advances to the next decision node
- Loss = cross-entropy(policy) + MSE(value) + L2 regularization (via optimizer weight_decay)
- Training examples store a value target blended from the terminal game outcome and MCTS root Q (`--value-bootstrap-alpha` controls the mix; 1.0 = pure outcome, 0.0 = pure Q)

## Experiment Conventions

Keep experiments clean and their conclusions buildable. (Hard-won: an earlier 0-ply-distillation arc fragmented one hypothesis across four confusingly-named experiments with mixed metrics/n's and a biased proxy — days of confusion.)

- **One hypothesis per experiment**, stated up front as a single testable question (e.g. *"does TD self-play improve a net already at near-parity with GNUBG-0-ply?"*). A required benchmark is that experiment's *completion*, not a separate experiment.
- **One primary metric, fixed before the run** — with its full protocol: opponent/benchmark, ply/tier, **n**, CI where applicable (e.g. *"cubeless-money ppg vs GNUBG-0-ply, n=6000, ±0.046"* or *"PR vs BGSage money benchmark, rollout tier, n=14693"*). Report every checkpoint on it; don't switch metric or n mid-stream.
- **The primary metric must be unbiased, or scored against a validated high-quality static benchmark.** Two accepted families:
  - **Raw play results** against a real opponent (e.g. GNUBG ppg). Near-parity comparisons need **n≥6000** (n=400 → ±0.18, too coarse); **select *and* confirm** a "best" at full n (a noisy selection inflates it — winner's curse).
  - **PR or MSE/R² against the BGSage money benchmark** (`scripts/eval_benchmark_pr.py`, `data/bgsage/money_benchmark/`) — 14,693 checker decisions with cubeless-equity references at three quality tiers (rollout > 3T > 3P). PR = mean move-selection error × 500 over all tiers (n=14,693); value-head MSE/R² is reported on the **rollout** tier (n≈149k candidate positions — the strongest reference) with 3T/3P as supporting cross-checks. Discovered in exp015: static benchmarking this way gives far more power than raw ppg at zero play cost, so prefer it for near-parity comparisons whenever a checkpoint can be scored against it.
  - Any other proxy metric is supporting-only, clearly labeled, and used only if unbiased (or the bias quantified).
- **Exception: a matched-arm fit-quality proxy, when neither raw play nor the BGSage benchmark is feasible** (e.g. comparing against a bespoke target the benchmark can't isolate). A held-out R² of each arm against its own training target, on an identical split, can stand in as the primary metric — state this explicitly as a deliberate exception, not silently. Valid only when both arms train/evaluate on identical positions and an identical split, so any measurement bias (e.g. holdout leakage) hits both symmetrically and cancels in the comparison. Gives the **sign** of a strength difference, not its ppg magnitude (exp014).
- **Every reported number carries its provenance — (checkpoint, metric, n).** No bare numbers.
- **When the hypothesized effect is small on a raw-ppg metric, report power, not just the CI.** A fixed n/CI calibrated for one comparison (checkpoint vs. teacher) understates its own noise when reused for a smaller one (checkpoint vs. checkpoint has √2× the SE); before calling a result a clean null, check whether n was actually large enough to detect the realistic effect size (needed n for 80% power ≈ `2·(1.96+0.84)²·σ²/δ²`) — "no significant difference" against an underpowered test is much weaker evidence than it sounds. (BGSage PR/MSE at n=14,693/~149k is rarely the bottleneck, but still state n.)
- **One experiment = one name = one write-up**, ending in a one-line conclusion: *"[hypothesis] → [answer]: [net] = [X ± CI] (metric, n)."* A re-run that completes a truncated attempt **supersedes** it (don't co-list both).
- **Supporting metrics are welcome, just demarcated** (training loss/SSE, coarse in-loop evals, epoch curves, secondary findings) — they never drive the headline conclusion or final selection.

## Hardware

- **Local dev (Windows work PC)**: Windows PC (WSL2, Intel i7-1365U, 6 cores / 12 threads, 16 GB RAM, no GPU). Defaults are tuned small: 6 ResNet blocks, 128 channels, 100 MCTS simulations.
- **Local dev (Windows home PC)**: Windows PC (WSL2, Intel i7-8550U, 4 cores / 8 threads, 8 GB RAM, no GPU). Same small defaults; smoke runs only.
- **Local dev (iMac)**: 2013 iMac (`lasse-iMac14-2`), Ubuntu 24.04, Intel i5-4570 (4 cores), 16 GB RAM, GT 755M (no CUDA — CPU only). Same small defaults as the Windows box; expect smoke runs only.
- **Cloud training**: GCP spot VM with T4 GPU (`raccoon-gpu` in `europe-west1-b`). Auto-detects CUDA. See `docs/gcp_guide.md` for workflow.

## Key Files

- `goal.md` — Project goal, assumptions, requirements, and success criteria
- `README.md` — Design decisions, tech stack, and references
- `docs/plan.md` — Full implementation plan with per-milestone specs, interfaces, and test requirements
- `docs/gcp_guide.md` — GCP training workflow, commands, costs, and troubleshooting
- `data/README.md` — Layout of shared datasets (`data/wildbg/`, `data/bglab/`, `data/distill/`); `data/distill/` holds the GNUBG self-play distillation caches organized by label ply then generation run, with GCS provenance
- `experiments/<name>/{checkpoints,logs}/` — All training output lives here (gitignored). Same layout on VM, in GCS (`gs://raccoon-training-lhm/experiments/`), and locally. `data/distill/` mirrors the same way under `gs://raccoon-training-lhm/data/distill/`.
