# Architecture

Raccoon follows the standard AlphaZero pipeline: a ResNet policy-value network guides Monte Carlo Tree Search, with self-play generating the training data. OpenSpiel provides game logic; all ML and search code is written from scratch in Python/PyTorch.

## Board Encoding

States are encoded as **(17, 2, 12)** float32 tensors — 17 channels, 2 rows (board halves), 12 columns (points per half).

The 2D layout mirrors the physical board: points 13–24 on the top row (left to right), points 12–1 on the bottom row. Channels:

| Channel(s) | Content |
|---|---|
| 0–3 | Current player checkers: ≥1, ≥2, ≥3, overflow |
| 4–7 | Opponent checkers: ≥1, ≥2, ≥3, overflow |
| 8–9 | Bar counts (current player, opponent) broadcast |
| 10–11 | Borne-off counts (current player, opponent) broadcast |
| 12 | Side-to-move flag |
| 13–14 | Dice values broadcast across columns |
| 15 | Doubles flag |
| 16 | Mid-doubles flag (one die already used in a doubles move) |

All encoding is from the **current player's perspective** — the network always sees "my checkers" and "their checkers" in fixed channels, regardless of which physical side is moving. The wrapper applies a perspective flip when it is the second player's turn.

`CHANNEL_NAMES` in `raccoon/env/encoder.py` is the authoritative registry. `dump_tensor()` pretty-prints all planes for debugging.

## Neural Network

PyTorch ResNet in `raccoon/model/network.py` with a shared convolutional trunk and two heads:

- **Policy head**: 1352 logits (one per OpenSpiel action). Masked softmax over legal moves gives move probabilities for MCTS.
- **Value head**: scalar via tanh, in [−1, 1]. Positive means the current player is winning.

Default size: **6 residual blocks, 128 channels** — tuned for CPU training. Scale depth/width when GPU is available. Architecture is saved in checkpoints, so `--resume` automatically picks up the correct shape.

## MCTS

AlphaZero-style MCTS in `raccoon/search/mcts.py` with PUCT selection. Key design choices:

**Chance node sampling.** Dice rolls are sampled and skipped — the tree contains only decision and terminal nodes. The network is never evaluated at chance nodes. With 100+ simulations, the various dice outcomes are explored naturally through repeated sampling.

**Temperature.** Controls exploitation vs exploration when selecting a move from visit counts. High temperature (early training) selects near-uniformly; low temperature (late training / evaluation) selects the most-visited move.

**Batched inference.** Leaf positions collected during a simulation round are evaluated in a single network forward pass, amortizing the GPU overhead.

See [MCTS Explained](mcts_explained.md) for a deep dive into simulations, PUCT, Dirichlet noise, and the plateau problem.

## Training Loop

Implemented in `raccoon/train/`:

1. **Self-play** (`self_play.py`): play N games using current network + MCTS. Record `(observation, MCTS policy, outcome)` tuples.
2. **Replay buffer** (`replay_buffer.py`): circular buffer keeps recent positions. New games overwrite the oldest.
3. **SGD** (`coach.py`): sample mini-batches; minimize cross-entropy(policy) + MSE(value) + L2 (via `weight_decay`).
4. **Checkpoint**: save network weights; optionally evaluate vs previous checkpoint.

Value targets blend the terminal game outcome and MCTS root Q-value. `--value-bootstrap-alpha` controls the mix (1.0 = pure outcome, 0.0 = pure Q).

## Action Space

OpenSpiel encodes 1352 distinct backgammon actions (base-26 packed checker moves). The policy head outputs a logit for each; illegal moves are masked to −∞ before softmax in `raccoon/env/actions.py`.

## Evaluation

- **Checkpoint vs checkpoint** (`raccoon/eval/arena.py`): tracks whether new iterations improve on old ones.
- **GNUBG benchmark** (`raccoon/eval/gnubg_harness.py`): automated money game matches against the GNUBG CLI. The primary success metric.
