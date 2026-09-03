# Architecture

Raccoon started as a straight AlphaZero pipeline: a ResNet policy-value network
guiding Monte Carlo Tree Search, trained by self-play. That machinery is all still
here and still runnable — but it is **not** what the current engine does. The net
that plays on raccoonbg.com and gets scored on the benchmark was trained by
distilling GNUBG, and it picks moves by 0-ply value lookahead with no search at
all.

Both paths are described below, because the repository supports both and the
distinction decides which parts of the network are even used. OpenSpiel provides
game logic; all ML and search code is written from scratch in Python/PyTorch.

Since exp025 the engine also has a **doubling cube**. OpenSpiel's backgammon does
not, so the cube is a layer above it, and it changes move selection as well as
adding the double/take decision.

## Board Encoding

States are encoded as **(26, 2, 12)** float32 tensors — 26 channels, 2 rows (board
halves), 12 columns (points per half).

The 2D layout mirrors the physical board: points 13–24 on the top row (left to
right), points 12–1 on the bottom row. Channels 0–16 are the lossless base planes;
17–25 are handcrafted features layered on top.

| Channel(s) | Content |
|---|---|
| 0–3 | Current player checkers: ≥1, ≥2, ≥3, overflow `(count−3)/2` |
| 4–7 | Opponent checkers: ≥1, ≥2, ≥3, overflow |
| 8 | Side-to-move flag |
| 9–10 | Bar counts / 15 (current player, opponent) |
| 11–12 | Borne-off counts / 15 (current player, opponent) |
| 13–14 | Dice values / 6 |
| 15 | Doubles flag |
| 16 | Mid-doubles flag (one die already used in a doubles move) |
| 17–19 | Pip counts / 167 (mine, opponent) and pip ratio |
| 20–21 | Blots / 15 (mine, opponent) |
| 22–23 | Anchors / 6 (mine, opponent) |
| 24–25 | Contact / 167 (mine, opponent) |

`CHANNEL_NAMES` in `raccoon/env/encoder.py` is the authoritative registry — the
divisor is baked into each name (`"my bar / 15"`), so the normalisation is visible
at the point of definition. `dump_tensor()` pretty-prints all planes for debugging.

**The handcrafted features are normalised by default, and this matters.** Raw pip
(~95) and contact (~52) values are about 100× the scale of the base planes, which
lets them dominate the input convolution and destabilises value-head training. The
divisors in `FEATURE_SCALES` bring them into the ~[0,1] range the base planes
already occupy; see Stage 6 of the pretraining analysis for the measurement. Pass
`normalize=False` only to recover raw magnitudes for feature-math tests.

`FEATURE_GROUPS` splits the channels into `base` (0–16) plus four toggleable groups
— `pip`, `blots`, `anchors`, `contact` — so ablations can train on a subset.
Checkpoints record which subset they used in `feature_channels`, and the exported
web model carries the same list in its `.meta.json`, so a net can never be fed an
encoding it was not trained on.

All encoding is from the **current player's perspective** — the network always sees
"my checkers" and "their checkers" in fixed channels, regardless of which physical
side is moving. The wrapper applies a perspective flip when it is the second
player's turn.

## Neural Network

PyTorch ResNet in `raccoon/model/network.py` with a shared convolutional trunk and
two heads:

- **Policy head**: 1352 logits (one per OpenSpiel action). Masked softmax over legal
  moves gives move probabilities. Read only by MCTS.
- **Value head**: either `"scalar"` or `"outcomes6"`, chosen at construction.

Default size is **6 residual blocks, 128 channels** — small enough to train on a
laptop CPU. The shipped net is much larger: **10 blocks, 256 channels, 11.9M
parameters**. Architecture is saved in checkpoints, so `--resume` picks up the
correct shape automatically.

### The value convention

Every value in this codebase is **money equity / 3, in [−1, 1], from the to-move
player's point of view** — win = ±1/3, gammon = ±2/3, backgammon = ±1. Multiply by
3 for money-game points. The value head is trained on **pre-roll** positions (dice
cleared).

The two head types differ only in how they get there:

- `scalar` — a single tanh output that *is* equity/3, regressed with MSE.
- `outcomes6` — six logits over `[win, win_g, win_bg, lose, lose_g, lose_bg]`,
  trained with cross-entropy against the outcome distribution. Equity is recovered
  by dotting the softmax with the outcome points `(1, 2, 3, −1, −2, −3)` and
  dividing by 3.

`value_equity()` applies whichever conversion the head needs, so **a scalar net and
an outcomes6 net are interchangeable at play time**. Everything downstream — move
selection, search, evaluation — goes through it and never branches on head type.

The shipped net is `outcomes6`, which is why the site can show a
win/gammon/backgammon breakdown and not just a number.

## Move Selection: 0-ply Lookahead

This is what actually picks moves, in `raccoon/train/lookahead.py`. Given a decision
state, enumerate the legal moves, evaluate the value head on each resulting pre-roll
child, negate when the child is the opponent's to move, and rank.

**"0-ply" is GNUBG's numbering, used throughout this project.** It means static
evaluation of the candidate afterstates with no further lookahead, which is exactly
what GNUBG does at ply 0 — so a net playing this way is directly comparable to
`gnubg` at ply 0. TD-Gammon's papers call the same operation "1-ply"; we use
GNUBG's convention because GNUBG is the benchmark.

The negation is the subtle part, and it is why these helpers live in one shared
module rather than being reimplemented per caller: TD(λ) self-play, policy-dataset
synthesis, and the browser port all reuse the same perspective logic.

Note what this does **not** use: no policy head, no tree, no simulations. The
exported web model omits the policy head entirely for that reason.

## The Doubling Cube

`raccoon/cube/` turns a cubeless outcome distribution into a cube decision, using
Rick Janowski's cube-life-index model. exp024 measured that model against 33,395
rolled-out cubeful equities and found the published constants intact; exp025 wired
it into play. The measurement is [The Doubling Cube](cube.qmd).

**The model.** `janowski.py` implements the *piecewise-linear* live-cube variant
(GNU Backgammon's `MoneyLive`), not the paper's Appendix-1 closed form — the
piecewise version is what the engines we benchmark against ship, and it is also
more accurate and satisfies the provable `[-L, W]` bound by construction. Index
`x = 0.68` for contact, `0.60` for races. `W` and `L` are floored at 1, which is
not a nicety: variance-reduced rollout labels violate it, with a worst case of
`W = -5.29`.

**The state.** `state.py` holds `CubeState(value, owner)` and the bridge between
two vocabularies that are easy to confuse. Janowski speaks in labels *relative to
the player on roll* — `CENTERED`, `PLAYER`, `OPPONENT`; a game loop speaks in seat
indices. `flip_label` is the piece that matters: **changing perspective flips the
cube owner as well as the sign of the equity.** Negating alone hands the opponent
your cube ownership, and nothing about the resulting number looks wrong.

**Cube-aware move selection.** `child_cubeful_values` in `train/lookahead.py` is
the cubeful twin of `child_values`, sharing its enumeration, doubles handling and
leaf dedup through `enumerate_leaves`. It reads `value_probs6` instead of
`value_equity` and prices each leaf with `cl2cf_money`. This is where the cube
model earns its keep: exp024 measured that it is consulted about 22 times as often
for move selection as for a take decision, and that a *perfect cubeless* player
still throws away PR 0.336 ± 0.042 when judged cubefully.

**Two scales, deliberately separate.** `value_equity` and `child_values` are
equity/3 in [−1, 1]; `cl2cf_money` and `child_cubeful_values` are money points at
cube value 1, roughly [−3, 3]. Terminal children are `returns()` undivided on the
cubeful path and divided by 3 on the cubeless one. The cube *value* never enters a
ranking — it multiplies every candidate by the same constant — so callers apply it
at the end.

**Jacoby.** `cl2cf_money` takes the flag *already resolved* against the cube
position, because the rule stops applying the moment the cube is turned; pass
`jacoby_active(label, rule)`. Raccoon's target is money **without** Jacoby, so
`eval/cube_arena.py` runs with it off. The BGSage benchmark was generated with it
on, so scoring against that benchmark turns it on. Those are the only two
settings, and each is stated where it is used.

**GNUBG's cube.** gnubg-nn cannot make a money cube decision —
`evaluate_cube_decision` raises `Not implemented for money`, covering match play
only. So `eval/gnubg_adapter.py` reconstructs it the way GNU Backgammon computes
it for money: GNUBG's own cubeless probabilities at the requested ply, through the
same Janowski code. Reading `gnubg_nn.probabilities` uncollapsed costs nothing
extra. The consequence is that a head-to-head match has **both sides sharing one
cube model**, so it compares evaluation and checker play rather than cube models.

## Search

Two independent searches exist. Neither is used by the shipped 0-ply engine.

**MCTS** (`raccoon/search/mcts.py`) — AlphaZero-style, PUCT selection, used by
self-play training. Dice rolls are sampled and skipped, so the tree contains only
decision and terminal nodes and the network is never evaluated at a chance node;
with 100+ simulations the dice outcomes are covered by repeated sampling.
Temperature controls exploitation vs exploration when converting visit counts into
a move. Leaf positions from a simulation round are evaluated in one batched forward
pass. See [MCTS Explained](mcts_explained.md).

**Filtered expectimax** (`raccoon/search/expectimax.py`, exp021) — ranks moves by
pushing evaluation *n* rolls deeper instead of asking the value head about the
afterstate directly. Chance nodes are expanded **full width** over the 21 distinct
rolls (doubles weighted 1/36, non-doubles 2/36), the opponent replies greedily, and
each tree level is one batched forward pass. `depth=0` reduces to exactly the 0-ply
static evaluation above.

Expectimax operates on **gnubg-nn's 2×25 board layout, not OpenSpiel states** — the
BGSage benchmark stores raw boards and OpenSpiel cannot construct a state from a
board, so moves come from `gnubg_nn.moves()` (whole-turn, already deduplicated to
distinct resulting positions). Two slot-orientation conventions are easy to get
backwards and are pinned by tests; see the module docstring. The design study is
[Multi-Ply Search](search.qmd).

## Training

Three routes, all writing to `experiments/<name>/`:

**Self-play + MCTS** (`raccoon/train/`, driven by `scripts/train.py`) — the original
AlphaZero loop. `self_play.py` plays games recording `(observation, MCTS policy,
outcome)`; `replay_buffer.py` keeps a circular buffer of recent positions;
`coach.py` samples mini-batches and minimises cross-entropy(policy) + MSE(value) +
L2. Value targets blend the terminal outcome with the MCTS root Q-value, mixed by
`--value-bootstrap-alpha` (1.0 = pure outcome, 0.0 = pure Q).

**TD(λ) self-play** (`raccoon/train/td_selfplay.py`, `scripts/train_td.py`) — plays
by 0-ply lookahead with the dice supplying exploration, TD-Gammon style, and
regresses the value head toward forward-view TD(λ) targets. No policy head, no tree.

**Distillation** (`scripts/train_distill.py`) — regresses the value head against
cached GNUBG evaluations. This is where the shipped net comes from. One arm per
invocation so that `scalar` vs `outcomes6` is a clean A/B on the target definition:
arm A minimises MSE against equity/3, arm B cross-entropy against the six-outcome
distribution.

Supervised pretraining on wildbg positions (`scripts/pretrain.py`) and policy
pretraining by 1-ply lookahead on match archives (`scripts/pretrain_policy.py`) can
seed any of these.

## Action Space

OpenSpiel encodes 1352 distinct backgammon actions (base-26 packed checker moves).
The policy head outputs a logit per action; illegal moves are masked to −∞ before
softmax in `raccoon/env/actions.py`.

This indexing matters only where the policy head does — MCTS and policy pretraining.
0-ply lookahead compares *resulting positions*, so two actions reaching the same
board are the same move to it, and the browser port deliberately does not implement
the action indices at all.

## Evaluation

- **BGSage money benchmark** (`scripts/eval_benchmark_pr.py`) — the primary metric.
  14,693 checker decisions with cubeless-equity references at three quality tiers.
  PR is mean move-selection error × 500 over all tiers; value-head MSE/R² is reported
  on the rollout tier. Static, so it costs no play time and has far more statistical
  power than raw ppg near parity.
- **GNUBG benchmark** (`raccoon/eval/gnubg_harness.py`) — automated cubeless money
  sessions against GNUBG's real nets via the `gnubg-nn` package.
- **Variance-reduced arena** (`raccoon/eval/vr_arena.py`) — the same net-vs-GNUBG
  games with a variance-reduced ppg estimator, using the dice-luck control variate
  in `raccoon/eval/luck.py`.
- **Cubeful arena** (`raccoon/eval/cube_arena.py`, exp025) — full money games with
  the cube live and Jacoby off. The control variate carries over unchanged in
  principle: `h` stays a deterministic function of (pre-roll state, roll), the cube
  state is part of that state, and cube decisions are not chance events so they
  contribute no luck terms. Two things adapt — each luck term is scaled by the cube
  value at its own roll, and `h` becomes a cubeful equity so it tracks what it is
  subtracted from. Both are exactly unbiased, so which one to use is settled by
  measured variance rather than argument.
- **Checkpoint vs checkpoint** (`raccoon/eval/arena.py`) — tracks whether new
  iterations improve on old ones.
- **Doubles execution cost** (`raccoon/eval/doubles.py`, exp020) — measures what
  OpenSpiel's two-step doubles decision costs in play.

Every reported number carries its provenance: checkpoint, metric, and n.
