# Understanding Raccoon's Training Parameters

This document explains the parameters used in Raccoon's training command. A typical GPU run looks like this:

```bash
python scripts/train.py \
    --experiment-name exp001-6x128-200sims \
    --iterations 500 \
    --games-per-iter 50 \
    --simulations 200 \
    --training-steps 100 \
    --batch-size 256 \
    --replay-size 500000 \
    --checkpoint-every 1
```

A scaled-down CPU sanity check on a laptop might be `--iterations 250 --games-per-iter 10 --simulations 25 --training-steps 50 --batch-size 128`. The defaults in `train.py` sit between the two (`--games-per-iter 50 --simulations 100 --training-steps 100 --batch-size 256`).

## The Big Picture

Raccoon learns by playing against itself. The training loop repeats a simple cycle:

1. **Self-play**: The network plays backgammon games against itself, using MCTS to pick moves
2. **Collect data**: Each position is recorded along with what MCTS recommended (policy target) and who eventually won (value target)
3. **Train**: The neural network is updated to better predict MCTS's recommendations and game outcomes
4. **Repeat**: The improved network generates better self-play data, which trains an even better network

This is the core AlphaZero insight: MCTS produces stronger decisions than the raw network, so training the network to mimic MCTS creates a virtuous cycle.

## Parameter-by-Parameter

### `--iterations`

**What it is**: Number of times we repeat the full cycle (self-play + training + checkpoint). Default 100.

**Intuition**: Each iteration makes the network slightly better. Early iterations show big improvements (the network learns basic tactics). Later iterations yield smaller gains. A few hundred iterations is a modest training run — the original AlphaZero used hundreds of thousands of iterations, but on 5,000 TPUs.

**What to watch**: Policy loss and value loss in the training log. If both plateau, more iterations won't help without changing other parameters.

### `--games-per-iter`

**What it is**: How many complete self-play games are played each iteration before training. Default 50.

**Intuition**: More games per iteration = more diverse training positions, but slower iterations. Each game produces roughly 80-110 positions, so 50 games is ~5,000 new positions per iteration feeding into the replay buffer.

**Trade-off**: Too few games and the network overfits to a narrow set of positions. Too many and you waste time generating data with an outdated network when you could be training and improving it.

**Reference points**:
- Original AlphaZero: 25,000 games per iteration (on 5,000 TPUs)
- alpha-zero-general (Othello): 100 games per iteration (single GPU)
- Raccoon (T4 GPU on GCP): 50 games per iteration
- Raccoon (CPU laptop, sanity-check runs): 10 games per iteration

### `--simulations`

**What it is**: Number of MCTS simulations (tree traversals) per move during self-play. Default 100.

**Intuition**: Each simulation expands the search tree by one node. More simulations = deeper/wider search = stronger move selection = higher-quality training data. This is the single most impactful parameter for training quality.

With 100 simulations, MCTS can look a few moves ahead and compare many different continuations. The network's raw policy might suggest a bad move, but MCTS with enough simulations can discover it's bad and redirect visits to better moves.

**Trade-off**: Simulations are the main speed bottleneck (each one requires a neural network forward pass). Doubling simulations roughly doubles self-play time.

**Reference points**:
- Original AlphaZero (chess/Go): 800 simulations per move
- AlphaZero.jl (Connect Four): 600 simulations
- alpha-zero-general (Othello): 25 simulations
- Raccoon (T4 GPU, production runs): 200 simulations
- Raccoon (CPU sanity-check): 25 simulations

**Diminishing returns**: Research shows that for simpler games, improvement levels off around 40-200 simulations. For backgammon with its large branching factor, more is generally better, but even 25 provides meaningful improvement over the raw network.

### `--training-steps`

**What it is**: Number of SGD (stochastic gradient descent) steps per iteration. Each step samples a batch from the replay buffer and updates the network weights once. Default 100.

**Intuition**: After generating new self-play data, we train the network to fit that data. Each training step adjusts the network weights slightly in the direction that reduces prediction error.

**Trade-off**: Too few steps and the network barely changes between iterations. Too many and it overfits to the current replay buffer contents (especially early when the buffer is small). 100 steps is a reasonable middle ground for typical replay-buffer sizes.

### `--batch-size`

**What it is**: Number of positions sampled from the replay buffer per training step. Default 256.

**Intuition**: The network doesn't train on one position at a time — it trains on a batch. The batch provides an average gradient direction. Larger batches give more stable gradient estimates; smaller batches introduce noise that can help escape bad local optima.

**Trade-off**: Batch size interacts with learning rate — larger batches generally benefit from higher learning rates. Our default learning rate (0.001 with Adam) works well with batch sizes of 64-256.

**Why pick a smaller batch?** The default is 256. If you run with very few games per iteration (~1,000 positions), the buffer starts small, and a smaller batch (e.g. 128) lets training begin from iteration 1 instead of having to wait. With 50 games/iter (~5,000 new positions), the default 256 is fine from the start.

### `--channels` and `--num-blocks`

**What they are**: The neural network architecture — how wide and deep the ResNet is. `channels` is the number of filters in each convolutional layer; `num-blocks` is the number of residual blocks in the shared trunk. Defaults: 128 channels, 6 blocks.

**Intuition**: A larger network can represent more complex patterns but is slower to evaluate (which slows MCTS) and needs more data to train well. The defaults (6 blocks, 128 channels) are a reasonable middle ground for consumer hardware.

**Trade-off**: Doubling channels roughly quadruples computation per forward pass. Adding blocks increases depth linearly. If MCTS simulations are your bottleneck, a smaller network lets you run more simulations in the same time.

**Note**: When resuming from a checkpoint with `--resume`, the architecture is read from the checkpoint — these flags are ignored.

### `--num-workers`

**What it is**: How many self-play games run in parallel within an iteration. Each worker plays games independently and pushes its results back to the main process. Default 8.

**Intuition**: A single self-play game has a lot of idle time waiting for the GPU. Running 8 games concurrently keeps the GPU busy by overlapping their MCTS searches. On CPU it lets multiple Python threads share the load.

**Trade-off**: More workers means more memory (each holds a copy of the game state) and diminishing returns once the GPU is saturated. 8 is a good fit for a T4 with our network size; bump it down on tight-memory machines.

### `--virtual-loss`

**What it is**: How many MCTS leaves a single search step queues up for batched network evaluation. The MCTS code (`_run_batched` in `raccoon/search/mcts.py`) selects this many leaves at once, applying *virtual loss* to discourage the search from immediately re-selecting the same path, and evaluates them in one `predict_batch` call. Default 8.

**Intuition**: Without virtual loss, MCTS evaluates one leaf at a time — wasted GPU bandwidth. Virtual loss temporarily marks selected leaves as "losing" so the next selection picks a different path, letting us collect 8 leaves and evaluate them in one batched forward pass. The virtual losses are unwound after evaluation.

**Trade-off**: Higher values give better GPU utilization but slightly weaker MCTS (the search wastes some simulations exploring leaves it would have skipped with full information). 8 is a reasonable middle ground. With `--virtual-loss 1` the batched path degenerates to one leaf per step (effectively unbatched).

### `--checkpoint-every`

**What it is**: Save a checkpoint every N iterations. Default 10.

**Intuition**: Checkpoints let you resume after interruption (important on spot VMs!) and keep snapshots for evaluation. Every 10 iterations means at most 10 iterations of work lost if training stops unexpectedly.

**Trade-off**: More frequent checkpoints use more disk but give finer-grained recovery points. Less frequent checkpoints save disk but risk more lost work on interruption.

**Convention on spot VMs**: production runs use `--checkpoint-every 1` because each checkpoint is only ~22 MB and spot preemption can hit at any moment. The `docs/gcp_guide.md` workflow assumes this.

### `--experiment-name`

**What it is**: A label that identifies the run *and* determines where its outputs go. **Required** — `train.py` exits with an error if it's missing.

**Intuition**: Every checkpoint, log, and metric for the run lives under `experiments/<name>/`:

```
experiments/<name>/
├── checkpoints/iter_NNNN.pt
└── logs/training_log.jsonl
```

The same name is also written into each JSONL log entry, so you can tell runs apart when comparing logs across experiments.

**Tip**: pick a self-describing name that captures what makes this run different (e.g. `exp001-6x128-200sims`, `exp002-6x128-200sims-50kbuf`). It saves digging through configs later.

### `--resume`

**What it is**: Path to a checkpoint file to resume training from, e.g. `experiments/exp001-6x128-200sims/checkpoints/iter_0282.pt`.

**Intuition**: Instead of starting from scratch, this loads the network weights, optimizer state, and iteration counter from a previous run. Training continues from where it left off. The network architecture (channels, num-blocks) is read from the checkpoint, so you don't need to specify those again.

**Note**: only the model and optimizer state are restored. The replay buffer is not — every resume starts with an empty buffer that fills back up over the next ~10 iterations (depending on `--replay-size` and `--games-per-iter`).

**Typical use**: Resuming after a spot VM preemption, or extending a completed run with more iterations. The `scripts/resume_training.sh` helper on the GCP VM resolves the latest checkpoint automatically.

## Parameters We Don't Usually Set (CLI Defaults)

| Parameter | Default | What it does |
|-----------|---------|-------------|
| `--lr` | 0.001 | Adam optimizer learning rate — how aggressively weights change per step |
| `--weight-decay` | 1e-4 | L2 regularization — penalizes large weights to prevent overfitting (applied via Adam, not as a separate term in the loss) |
| `--replay-size` | 100,000 | Max positions in the FIFO replay buffer; once full, the oldest are dropped as new ones arrive |

## Hardcoded Knobs (Not CLI Flags)

A few important constants are baked into the code rather than exposed as flags. Worth knowing about because they affect search and training:

| Where | Value | What it does |
|-------|-------|-------------|
| `MCTS.c_puct` (`raccoon/search/mcts.py`) | 1.5 | PUCT exploration constant. Higher → more exploration, lower → trust the network's prior more |
| `play_one_game.temperature` (`raccoon/train/self_play.py`) | 1.0 | Sampling temperature for self-play move selection during the early game |
| `play_one_game.temp_threshold` | 30 | After move 30, temperature drops to 0 (greedy) — concentrates training data on the strongest moves once the position is well-defined |
| Root Dirichlet noise | *not implemented* | Standard AlphaZero adds Dirichlet noise to root priors to keep self-play exploring. Raccoon doesn't currently — exploration relies on PUCT + early-game temperature only |
| Optimizer | Adam | AlphaZero papers use SGD with momentum; Raccoon uses Adam, so learning rate and weight-decay numbers don't transfer directly |
| Value target normalization | `returns / 3.0` | Backgammon's raw returns are ±1/±2/±3 (single/gammon/backgammon). Dividing by 3 puts targets in `[-1, 1]` so they fit the value head's `tanh` output. So labels actually live in `{±1/3, ±2/3, ±1}` |
| Loss | `cross_entropy(policy) + MSE(value)` | No L2 term in the loss itself — weight decay handles regularization via the optimizer |

## How Parameters Interact

```
More simulations → better self-play quality → better training targets
                   but slower per game

More games/iter  → more diverse data → more robust learning
                   but slower per iteration

More training steps → network fits data better → stronger play
                      but risk of overfitting

Larger batch size → more stable gradients → smoother learning
                    but need more data in buffer
```

The fundamental constraint is **compute time**. On CPU, the bottleneck is MCTS simulations during self-play. The art is finding the sweet spot: enough simulations for meaningful search, enough games for data diversity, enough training steps to absorb what was learned.

## How to Read the Training Log

The first line of each run is a `{"type": "config", ...}` header recording network architecture, hyperparameters, and system info (PyTorch version, GPU). Each subsequent line is one iteration's metrics:

- **`policy_loss`** — Cross-entropy between the network's predicted policy and MCTS's visit distribution. Starts around 7.2 (`ln(1352)`, random over the 1,352-action space) and should decrease. Below ~4 the network is picking up strong move preferences; values around 2 are typical for trained Raccoon networks.
- **`value_loss`** — MSE between the network's value head and the (normalized) game outcome. Targets are in `{±1/3, ±2/3, ±1}` after the `returns / 3.0` normalization, so a network predicting 0 has MSE ≈ 0.5. Empirically Raccoon starts near ~0.10–0.12 (the value head's tanh output is small at init, and many games end in gammon/bg so outcome variance is concentrated near the extremes) and may drift slightly upward or stay flat as training progresses — see `docs/training_analysis.qmd` for what this can mean.
- **`avg_game_length`** — Average moves per game. May decrease as play becomes more efficient.
- **`avg_outcome`** — Should hover near 0 in self-play (both sides use the same network). A persistent non-zero value points to a player-side bias amplified by shallow MCTS; see the analysis doc.
- **`gammons`/`backgammons`** — Counts (not rates) per iteration. Strong-vs-strong play has gammon rates around 15–20% and backgammon rates around 1%; weak play produces much higher rates of both. Watch the trend, not the absolute value.
- **`self_play_time` / `training_time` / `total_time`** — Seconds spent in each phase. Self-play dominates; SGD is cheap.

## Further Reading

Start with these, in order:

1. **[Simple Alpha Zero](https://suragnair.github.io/posts/alphazero.html)** — The best beginner-friendly walkthrough of the full AlphaZero algorithm. Written by the author of alpha-zero-general (which Raccoon's design references). Covers self-play, MCTS, and training with clear diagrams.

2. **[Alpha Zero and Monte Carlo Tree Search](https://joshvarty.github.io/AlphaZero/)** — Visual, step-by-step explanation of how MCTS simulations work: select, expand, backup. Good for building intuition about what `--simulations` actually does.

3. **[AlphaZero.jl Connect Four Tutorial](https://jonathan-laurent.github.io/AlphaZero.jl/stable/tutorial/connect_four/)** — Practical example of tuning AlphaZero parameters for a real game on consumer hardware. Shows concrete parameter choices and their effects.

4. **[Mastering Chess and Shogi by Self-Play (AlphaZero paper)](https://arxiv.org/abs/1712.01815)** — The original paper. Dense but worth skimming Section 1 (method) and the supplementary tables (training parameters). Gives context for what "full scale" looks like.

5. **[What Does Batch Size Mean in Deep Learning? (Coursera)](https://www.coursera.org/articles/what-does-batch-size-mean-in-deep-learning)** — If the SGD/batch-size/learning-rate concepts are new, this is a clear general introduction.

6. **[Relation Between Learning Rate and Batch Size (Baeldung)](https://www.baeldung.com/cs/learning-rate-batch-size)** — Deeper dive into why batch size and learning rate are coupled, with practical guidelines.
