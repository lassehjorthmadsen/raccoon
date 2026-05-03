# Raccoon

A backgammon AI engine that uses AlphaZero-style reinforcement learning to (maybe someday) outperform GNUBG at money game play. The name "Raccoon" is a nod to the doubling cube term in backgammon.

## Why?

GNUBG, the strongest open-source backgammon engine, uses TD reinforcement learning with value-only neural networks and shallow search — a design from the 1990s (TD-Gammon era). Modern AlphaZero-style training (policy+value network with Monte Carlo Tree Search) has beaten world-class engines in chess, shogi, and Go. This project applies that approach to backgammon, with the vain hope of eventually surpassing GNUBG in money game play. (First step is cubeless money game without Jacoby, so gammons and backgammons count). 

## Design Decisions

### Game Engine
- **OpenSpiel** provides backgammon rules, legal move generation, and state management
- OpenSpiel is used only for game logic — all ML/search code is written from scratch
- OpenSpiel encodes 1352 distinct backgammon actions (base-26 packed checker moves)

### Board Encoding
Custom 2D tensor: **(16, 2, 12)** — 16 channels, 2 rows, 12 columns.

Points 13-24 map to the top row (left to right), points 12-1 to the bottom row. Channels:
- 4 checker planes per player: (>=1, >=2, >=3, overflow)
- Broadcast planes: bar counts, borne-off counts, side to move, dice values, doubles flag

Always encoded from the **current player's perspective** — the network sees "my checkers" and "their checkers" in fixed channels.

### Neural Network
PyTorch ResNet with shared trunk, policy head (1352 logits), and value head (scalar via tanh). Starting small: 6 residual blocks, 128 channels (suitable for CPU training). Scale up when GPU is available.

### Search
AlphaZero-style MCTS with PUCT selection. Chance nodes (dice rolls) are sampled and skipped — the tree only contains decision/terminal nodes. This means we never evaluate the network at chance nodes; with 100+ simulations the various dice outcomes are naturally explored.

### Training
AlphaZero self-play loop: play games, store (state, MCTS policy, outcome), train network. Loss: cross-entropy(policy) + MSE(value) + L2 regularization via optimizer weight_decay.

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

## Hardware Constraints

Initial development on a 2013 Intel iMac running Ubuntu (CPU only). Defaults are tuned for this: small network, low simulation count, small replay buffer. When GPU becomes available, scale up network depth, channel count, and simulation budget.

## References

- [OpenSpiel backgammon](https://openspiel.readthedocs.io/en/latest/games.html)
- [OpenSpiel issue #774](https://github.com/google-deepmind/open_spiel/issues/774) — AlphaZero for backgammon, chance-node handling
- [OpenSpiel discussion #1089](https://github.com/google-deepmind/open_spiel/discussions/1089) — 1D observation limitation for ResNet
- [AlphaZero paper](https://www.science.org/doi/10.1126/science.aar6404)
- [TD-Gammon](https://en.wikipedia.org/wiki/TD-Gammon) — historical context, board encoding ideas
- [jacobhilton/backgammon](https://github.com/jacobhilton/backgammon) — strong NN backgammon reference
- [suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general) — clean AlphaZero template
