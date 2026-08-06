# Raccoon

A backgammon AI engine that uses AlphaZero-style reinforcement learning to (maybe someday) outperform GNUBG at money game play. The name "Raccoon" is a nod to the doubling cube term in backgammon.

**[raccoonbg.com](https://raccoonbg.com)** — project documentation and write-ups: architecture notes, training analysis, and API reference.

## Why?

GNUBG, the strongest open-source backgammon engine, uses TD reinforcement learning with value-only neural networks and shallow search — a design from the 1990s (TD-Gammon era). Modern AlphaZero-style training (policy+value network with Monte Carlo Tree Search) has beaten world-class engines in chess, shogi, and Go. This project applies that approach to backgammon, with the vain hope of eventually surpassing GNUBG in money game play. (First step is cubeless money game without Jacoby, so gammons and backgammons count).

## Where it stands

Scored on the BGSage money benchmark — 14,693 checker decisions with cubeless-equity
references from rollouts. PR is mean move-selection error × 500; lower is better.

| Engine | Compute per decision | PR (n = 14,693) |
|---|---|---|
| GNUBG, 2-ply full-width search | ~1.3 s | 0.56 |
| **Raccoon, single network evaluation** | ~10 ms | **0.93** |
| GNUBG, single network evaluation (0-ply) | ~10 ms | 2.14 |

Raccoon figure: checkpoint `exp017-ep12` — a 10-block, 256-channel ResNet trained on
40M positions distilled from GNUBG's 2-ply search. At equal cost the network is well
past GNUBG's own evaluator and has closed most of the distance to GNUBG's 2-ply
search while doing no search at all. In direct play, an earlier checkpoint scored
−0.05 ± 0.046 ppg against GNUBG at 0-ply over 6,000 cubeless money games (exp011b).

Checker play only — the doubling cube is not handled yet.

Note the arc this took: pure AlphaZero self-play from scratch plateaued well short
of GNUBG (see the self-play analysis). The results above come from the supervised
track — distilling GNUBG's search into the network, then expert iteration (DAgger)
on self-play distributions. Both tracks are written up in full, including the dead
ends.

## Design Decisions

### Game Engine
- **OpenSpiel** provides backgammon rules, legal move generation, and state management
- OpenSpiel is used only for game logic — all ML/search code is written from scratch
- OpenSpiel encodes 1352 distinct backgammon actions (base-26 packed checker moves)

### Board Encoding
Custom 2D tensor: **(26, 2, 12)** — 26 channels, 2 rows, 12 columns. `CHANNEL_NAMES` in `raccoon/env/encoder.py` is the authoritative registry.

Points 13-24 map to the top row (left to right), points 12-1 to the bottom row. Channels:
- 4 checker planes per player: (>=1, >=2, >=3, overflow)
- Broadcast planes: bar counts, borne-off counts, side to move, dice values, doubles flag, mid-doubles flag
- Handcrafted features: pip count, blots, anchors, contact pressure

Always encoded from the **current player's perspective** — the network sees "my checkers" and "their checkers" in fixed channels.

### Neural Network
PyTorch ResNet with shared trunk, policy head (1352 logits), and value head (scalar via tanh). Defaults are small — 6 residual blocks, 128 channels — for CPU development; the trained networks are 10 blocks × 256 channels.

### Search
AlphaZero-style MCTS with PUCT selection. Chance nodes (dice rolls) are sampled and skipped — the tree only contains decision/terminal nodes. This means we never evaluate the network at chance nodes; with 100+ simulations the various dice outcomes are naturally explored.

### Training
Three tracks, all implemented: AlphaZero self-play (play games, store (state, MCTS policy, outcome), train), supervised distillation from GNUBG's search, and expert iteration (DAgger) on self-play distributions. Loss: cross-entropy(policy) + MSE(value) + L2 regularization via optimizer weight_decay. The supervised track carries the current results.

### Communication Protocol
Raccoon Game Protocol (RGP) — text-based, stdin/stdout, inspired by UCI. Commands: `newgame`, `position`, `dice`, `go`, `bestmove`, `quit`. Enables future GUI frontends.

## Tech Stack

- **Python 3.10+**
- **PyTorch** — neural network
- **OpenSpiel** — backgammon game logic
- **NumPy** — tensor encoding
- **pytest** — testing
- **GNUBG** — evaluation benchmark (CLI mode)

## Quick Start

```bash
make setup          # Install in editable mode with dev deps
make test           # Run all tests
make smoke          # Quick sanity: 2 iterations, 3 games, 10 sims
make train          # Full training run
make play           # Play against Raccoon in terminal
```

## Hardware

Development on CPU (a 2013 Intel iMac running Ubuntu, and small Windows/WSL2 boxes) — defaults are tuned for that: small network, low simulation count, small replay buffer. Training runs on a preemptible T4 GPU instance on Google Cloud; see [docs/gcp_guide.md](docs/gcp_guide.md).

## Documentation

Full write-ups at [raccoonbg.com](https://raccoonbg.com) — every experiment with its hypothesis, primary metric, sample size, and conclusion, including the failures:

- [Supervised & Expert Iteration](https://raccoonbg.com/pretraining_analysis.html) — distilling GNUBG, DAgger, consolidation; the current best results
- [Self-Play Analysis](https://raccoonbg.com/training_analysis.html) — what pure self-play achieved, and why it plateaued
- [Architecture](https://raccoonbg.com/architecture.html) · [MCTS Explained](https://raccoonbg.com/mcts_explained.html) · [Training Parameters](https://raccoonbg.com/training_parameters.html)

## References

- [OpenSpiel backgammon](https://openspiel.readthedocs.io/en/latest/games.html)
- [OpenSpiel issue #774](https://github.com/google-deepmind/open_spiel/issues/774) — AlphaZero for backgammon, chance-node handling
- [OpenSpiel discussion #1089](https://github.com/google-deepmind/open_spiel/discussions/1089) — 1D observation limitation for ResNet
- [AlphaZero paper](https://www.science.org/doi/10.1126/science.aar6404)
- [TD-Gammon](https://en.wikipedia.org/wiki/TD-Gammon) — historical context, board encoding ideas
- [jacobhilton/backgammon](https://github.com/jacobhilton/backgammon) — strong NN backgammon reference
- [suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general) — clean AlphaZero template
