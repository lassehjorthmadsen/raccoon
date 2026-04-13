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
make eval-gnubg     # Automated benchmark against GNUBG CLI
make play           # Interactive terminal play

# Run a single test file or test
python3 -m pytest tests/test_encoder.py -v
python3 -m pytest tests/test_mcts.py::test_name -v

# Training with custom params
python3 scripts/train.py --iterations 100 --games-per-iter 50 --simulations 100 --lr 0.001 \
  --channels 128 --num-blocks 6 --experiment-name my-run --checkpoint-every 10

# Resume training from a checkpoint (architecture is read from the checkpoint)
python3 scripts/train.py --iterations 100 --resume checkpoints/iter_0200.pt
```

## Architecture

The project follows a milestone-based plan (see `docs/plan.md` for full details). The seven milestones build on each other: env → model → search → training → internal eval → GNUBG benchmark → strength scaling.

### Core Pipeline

1. **`raccoon/env/`** — OpenSpiel wrapper + custom tensor encoder + action mapping
   - `game_wrapper.py`: Wraps `pyspiel.load_game("backgammon")`, handles perspective flipping so the network always sees the board from the current player's view
   - `encoder.py`: Converts board state to **(17, 2, 12)** float32 tensor (17 channels, 2 rows, 12 columns). Channels: 4 checker planes per player, bar/borne-off/dice broadcast planes, mid-doubles flag
   - `actions.py`: Legal action masking over OpenSpiel's 1352 action space

2. **`raccoon/model/network.py`** — `RaccoonNet`: ResNet with shared trunk → policy head (1352 logits) + value head (scalar in [-1,1] via tanh). Default: 6 residual blocks, 128 channels. `predict()` method handles masking + softmax for MCTS inference.

3. **`raccoon/search/mcts.py`** — AlphaZero MCTS with PUCT selection. Chance nodes (dice rolls) are sampled and skipped — the tree only contains decision/terminal nodes. Temperature controls exploration vs exploitation.

4. **`raccoon/train/`** — Self-play loop
   - `self_play.py`: Plays games, records (observation, MCTS policy, outcome) tuples
   - `replay_buffer.py`: Circular buffer of training positions
   - `coach.py`: Orchestrates self-play → replay buffer → SGD training → checkpoint. Logs full config (network architecture, hyperparams, system info) to JSONL.

5. **`raccoon/eval/`** — Evaluation infrastructure
   - `arena.py`: Checkpoint vs checkpoint matches
   - `gnubg_harness.py`: Automated money game matches against GNUBG CLI

6. **`raccoon/protocol/rgp.py`** — Raccoon Game Protocol: text-based stdin/stdout protocol inspired by UCI for future GUI frontends

7. **`raccoon/cli/play.py`** — Terminal interface for human vs Raccoon play

### Key Design Details

- Board encoding is always from the **current player's perspective** (perspective flip applied in the wrapper)
- MCTS never evaluates the network at chance nodes — it samples dice and advances to the next decision node
- Loss = cross-entropy(policy) + MSE(value) + L2 regularization (via optimizer weight_decay)
- Training examples store the game outcome from each position's player's perspective as value target

## Hardware

- **Local dev**: 2013 Intel iMac (CPU only). Defaults are tuned small: 6 ResNet blocks, 128 channels, 100 MCTS simulations.
- **Cloud training**: GCP spot VM with T4 GPU (`raccoon-gpu` in `europe-west1-b`). Auto-detects CUDA. See `docs/gcp_guide.md` for workflow.

## Key Files

- `goal.md` — Project goal, assumptions, requirements, and success criteria
- `README.md` — Design decisions, tech stack, and references
- `docs/plan.md` — Full implementation plan with per-milestone specs, interfaces, and test requirements
- `docs/gcp_guide.md` — GCP training workflow, commands, costs, and troubleshooting
- `experiments/` — Archived training results (gitignored). Each experiment gets `checkpoints/` and `logs/` subdirs. Mirrored in GCS at `gs://raccoon-training-lhm/experiments/`
- `checkpoints/`, `logs/` — Working directories for active training runs (gitignored, transient)
