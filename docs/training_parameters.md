# Understanding Raccoon's Training Parameters

This document explains the parameters used in Raccoon's training command:

```bash
python scripts/train.py \
    --iterations 250 \
    --games-per-iter 10 \
    --simulations 25 \
    --training-steps 50 \
    --batch-size 128
```

## The Big Picture

Raccoon learns by playing against itself. The training loop repeats a simple cycle:

1. **Self-play**: The network plays backgammon games against itself, using MCTS to pick moves
2. **Collect data**: Each position is recorded along with what MCTS recommended (policy target) and who eventually won (value target)
3. **Train**: The neural network is updated to better predict MCTS's recommendations and game outcomes
4. **Repeat**: The improved network generates better self-play data, which trains an even better network

This is the core AlphaZero insight: MCTS produces stronger decisions than the raw network, so training the network to mimic MCTS creates a virtuous cycle.

## Parameter-by-Parameter

### `--iterations 250`

**What it is**: Number of times we repeat the full cycle (self-play + training + checkpoint).

**Intuition**: Each iteration makes the network slightly better. Early iterations show big improvements (the network learns basic tactics). Later iterations yield smaller gains. 250 iterations is a modest training run — the original AlphaZero used hundreds of thousands of iterations, but on 5,000 TPUs.

**What to watch**: Policy loss and value loss in the training log. If both plateau, more iterations won't help without changing other parameters.

### `--games-per-iter 10`

**What it is**: How many complete self-play games are played each iteration before training.

**Intuition**: More games per iteration = more diverse training positions, but slower iterations. Each game produces roughly 80-110 positions. With 10 games, that's ~1,000 new positions per iteration feeding into the replay buffer.

**Trade-off**: Too few games and the network overfits to a narrow set of positions. Too many and you waste time generating data with an outdated network when you could be training and improving it.

**Reference points**:
- Original AlphaZero: 25,000 games per iteration (on 5,000 TPUs)
- alpha-zero-general (Othello): 100 games per iteration (single GPU)
- Raccoon: 10 games per iteration (single CPU — our hardware constraint)

### `--simulations 25`

**What it is**: Number of MCTS simulations (tree traversals) per move during self-play.

**Intuition**: Each simulation expands the search tree by one node. More simulations = deeper/wider search = stronger move selection = higher-quality training data. This is the single most impactful parameter for training quality.

With 25 simulations, MCTS can look a few moves ahead and compare ~25 different continuations. The network's raw policy might suggest a bad move, but MCTS with enough simulations can discover it's bad and redirect visits to better moves.

**Trade-off**: Simulations are the main speed bottleneck (each one requires a neural network forward pass). Doubling simulations roughly doubles self-play time.

**Reference points**:
- Original AlphaZero (chess/Go): 800 simulations per move
- AlphaZero.jl (Connect Four): 600 simulations
- alpha-zero-general (Othello): 25 simulations
- Raccoon: 25 simulations (CPU constraint; increase to 50-200 with GPU)

**Diminishing returns**: Research shows that for simpler games, improvement levels off around 40-200 simulations. For backgammon with its large branching factor, more is generally better, but even 25 provides meaningful improvement over the raw network.

### `--training-steps 50`

**What it is**: Number of SGD (stochastic gradient descent) steps per iteration. Each step samples a batch from the replay buffer and updates the network weights once.

**Intuition**: After generating new self-play data, we train the network to fit that data. Each training step adjusts the network weights slightly in the direction that reduces prediction error.

**Trade-off**: Too few steps and the network barely changes between iterations. Too many and it overfits to the current replay buffer contents (especially early when the buffer is small). 50 steps is conservative — enough to move the network meaningfully without overtraining.

### `--batch-size 128`

**What it is**: Number of positions sampled from the replay buffer per training step.

**Intuition**: The network doesn't train on one position at a time — it trains on a batch. The batch provides an average gradient direction. Larger batches give more stable gradient estimates; smaller batches introduce noise that can help escape bad local optima.

**Trade-off**: Batch size interacts with learning rate — larger batches generally benefit from higher learning rates. Our default learning rate (0.001 with Adam) works well with batch sizes of 64-256.

**Why 128 here**: The default is 256, but with only 10 games per iteration (~1,000 positions), the buffer starts small. Using 128 lets training begin from iteration 1 instead of having to wait.

## Parameters We Don't Set (Using Defaults)

| Parameter | Default | What it does |
|-----------|---------|-------------|
| `--lr` | 0.001 | Adam optimizer learning rate — how aggressively weights change per step |
| `--weight-decay` | 1e-4 | L2 regularization — penalizes large weights to prevent overfitting |
| `--replay-size` | 100,000 | Max positions in the replay buffer before oldest are dropped |
| `--temperature` | 1.0 (first 30 moves) | Controls exploration in self-play move selection |

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

After each iteration, `training_log.jsonl` records:

- **`policy_loss`** — How well the network predicts MCTS's move recommendations. Starts around 7 (random over 1,352 actions) and should decrease. Below 4 means the network is learning strong move preferences.
- **`value_loss`** — How well the network predicts game outcomes. Starts near 0.25 (random guess for ±1 outcomes) and should decrease. Below 0.1 means good outcome prediction.
- **`avg_game_length`** — Average moves per game. May decrease as play becomes more efficient.
- **`avg_outcome`** — Should hover near 0 in self-play (both sides use the same network).
- **`gammons`/`backgammons`** — High early on (bad play leaves checkers behind). Should decrease as play improves.

## Further Reading

Start with these, in order:

1. **[Simple Alpha Zero](https://suragnair.github.io/posts/alphazero.html)** — The best beginner-friendly walkthrough of the full AlphaZero algorithm. Written by the author of alpha-zero-general (which Raccoon's design references). Covers self-play, MCTS, and training with clear diagrams.

2. **[Alpha Zero and Monte Carlo Tree Search](https://joshvarty.github.io/AlphaZero/)** — Visual, step-by-step explanation of how MCTS simulations work: select, expand, backup. Good for building intuition about what `--simulations` actually does.

3. **[AlphaZero.jl Connect Four Tutorial](https://jonathan-laurent.github.io/AlphaZero.jl/stable/tutorial/connect_four/)** — Practical example of tuning AlphaZero parameters for a real game on consumer hardware. Shows concrete parameter choices and their effects.

4. **[Mastering Chess and Shogi by Self-Play (AlphaZero paper)](https://arxiv.org/abs/1712.01815)** — The original paper. Dense but worth skimming Section 1 (method) and the supplementary tables (training parameters). Gives context for what "full scale" looks like.

5. **[What Does Batch Size Mean in Deep Learning? (Coursera)](https://www.coursera.org/articles/what-does-batch-size-mean-in-deep-learning)** — If the SGD/batch-size/learning-rate concepts are new, this is a clear general introduction.

6. **[Relation Between Learning Rate and Batch Size (Baeldung)](https://www.baeldung.com/cs/learning-rate-batch-size)** — Deeper dive into why batch size and learning rate are coupled, with practical guidelines.
