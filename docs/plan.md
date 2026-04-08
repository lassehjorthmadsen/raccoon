# Raccoon Implementation Plan

## Overview

This plan builds a backgammon AI that outperforms GNUBG at money game, using AlphaZero-style self-play with a ResNet policy-value network and chance-node-aware MCTS. OpenSpiel provides game logic; everything else is written from scratch in Python/PyTorch.

Work proceeds through 7 milestones, each with concrete deliverables and pass conditions. No milestone should be started until the previous one passes.

---

## M1: Game Environment

**Goal**: Wrap OpenSpiel backgammon and build a custom 2D tensor encoder + action mapping.

### M1.1: OpenSpiel Game Wrapper

**File**: `raccoon/env/game_wrapper.py`

Thin wrapper around `pyspiel.load_game("backgammon")` that:
- Creates new games and clones states
- Queries the current player, legal actions, and terminal status
- Returns checker positions, bar, borne-off from current player's perspective
- Handles the perspective flip: when player 1 is to move, point `p` becomes `23 - p` so the network always sees "my home board" in the same place
- Exposes `apply_action(action)`, `is_terminal()`, `is_chance_node()`, `chance_outcomes()`, `returns()`

Key interface:
```python
class GameWrapper:
    def new_game(self) -> GameState
    
class GameState:
    def current_player(self) -> int          # 0 or 1
    def legal_actions(self) -> list[int]     # OpenSpiel action indices (0..1351)
    def is_terminal(self) -> bool
    def is_chance_node(self) -> bool
    def chance_outcomes(self) -> list[tuple[int, float]]  # (action, probability)
    def apply_action(self, action: int) -> None
    def returns(self) -> list[float]         # terminal rewards per player
    def clone(self) -> GameState
    def board_from_perspective(self) -> BoardView  # current-player-relative
```

`BoardView` is a simple dataclass:
```python
@dataclass
class BoardView:
    my_points: np.ndarray      # shape (24,), checker counts on points 1-24
    opp_points: np.ndarray     # shape (24,)
    my_bar: int
    opp_bar: int
    my_off: int
    opp_off: int
    dice: tuple[int, int] | None
```

**Tests** (`tests/test_game_wrapper.py`):
- Starting position has correct checker counts (2+5+3+5=15 per player)
- Legal actions list is non-empty for valid positions
- Perspective flip is its own inverse (flip twice = identity)
- Terminal positions return correct outcomes (+1/-1 for normal win, +2/-2 for gammon, +3/-3 for backgammon)
- Chance nodes produce valid dice distributions (21 outcomes summing to 1.0)
- Clone produces independent copies

### M1.2: Tensor Encoder

**File**: `raccoon/env/encoder.py`

Converts a `BoardView` into a `(16, 2, 12)` float32 numpy array.

Board layout mapping:
- Top row (row 0): points 13, 14, ..., 24 → columns 0, 1, ..., 11
- Bottom row (row 1): points 12, 11, ..., 1 → columns 0, 1, ..., 11

Channel layout (16 channels):
```
 0: my checkers ≥ 1  (binary)
 1: my checkers ≥ 2  (binary)
 2: my checkers ≥ 3  (binary)
 3: my checkers 4+   (count - 3) / 2
 4: opp checkers ≥ 1
 5: opp checkers ≥ 2
 6: opp checkers ≥ 3
 7: opp checkers 4+  (count - 3) / 2
 8: side to move     (all 1s — always current player)
 9: my bar / 15      (broadcast)
10: opp bar / 15     (broadcast)
11: my off / 15      (broadcast)
12: opp off / 15     (broadcast)
13: die 1 / 6        (broadcast)
14: die 2 / 6        (broadcast)
15: doubles flag     (broadcast, 1 if d1 == d2)
```

Key interface:
```python
def encode_state(board_view: BoardView) -> np.ndarray:
    """Returns tensor of shape (16, 2, 12), dtype float32."""

def encode_batch(board_views: list[BoardView]) -> np.ndarray:
    """Returns tensor of shape (N, 16, 2, 12), dtype float32."""
```

**Tests** (`tests/test_encoder.py`):
- Starting position produces correct shape (16, 2, 12)
- All values are finite and in expected ranges
- Empty board → all checker channels zero
- Known position (e.g., 5 checkers on point 6) → channels 0-2 are 1.0, channel 3 is 1.0 at correct grid cell
- Dice (3, 1) → channel 13 = 0.5, channel 14 ≈ 0.167
- Doubles (4, 4) → channel 15 = 1.0

### M1.3: Action Mapping

**File**: `raccoon/env/actions.py`

OpenSpiel uses 1352 action indices for backgammon. We use these directly as our policy head output size.

This module provides:
```python
ACTION_SPACE_SIZE = 1352

def legal_action_mask(legal_actions: list[int]) -> np.ndarray:
    """Returns a boolean mask of shape (1352,). True for legal actions."""

def action_to_string(state: GameState, action: int) -> str:
    """Human-readable move description (e.g., '24/21 13/11')."""
```

**Tests** (`tests/test_actions.py`):
- Mask shape is (1352,)
- Mask has exactly `len(legal_actions)` True entries
- Mask entries at legal action indices are True
- `action_to_string` returns a non-empty string for valid actions

### M1 Pass Condition
All M1 tests pass. Tensor shapes are correct. A full game can be played to completion through the wrapper. Encoding and decoding round-trips produce consistent results.

---

## M2: Neural Network

**Goal**: Build a ResNet policy-value model with legal action masking.

### M2.1: ResNet Architecture

**File**: `raccoon/model/network.py`

```python
class ResidualBlock(nn.Module):
    """Conv3x3 → BN → ReLU → Conv3x3 → BN → skip add → ReLU"""
    def __init__(self, channels: int): ...
    def forward(self, x: Tensor) -> Tensor: ...

class RaccoonNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        board_h: int = 2,
        board_w: int = 12,
        num_actions: int = 1352,
        channels: int = 128,
        num_blocks: int = 6,
    ): ...
    
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: (batch, 16, 2, 12)
        Returns:
            policy_logits: (batch, 1352) — raw logits, NOT masked
            value: (batch, 1) — in [-1, 1]
        """
    
    def predict(self, obs: np.ndarray, legal_actions: list[int]) -> tuple[np.ndarray, float]:
        """
        Single-position inference for MCTS.
        
        Args:
            obs: (16, 2, 12) numpy array
            legal_actions: list of valid action indices
            
        Returns:
            policy: dict mapping action -> probability (sums to 1, only legal actions)
            value: scalar float in [-1, 1]
        """
```

Architecture details:
- Input conv: `Conv2d(16, 128, 3, padding=1)` → `BN` → `ReLU`
- Trunk: 6 × `ResidualBlock(128)`
- Policy head: `Conv2d(128, 2, 1)` → `BN` → `ReLU` → `Flatten` → `Linear(2*2*12, 1352)`
- Value head: `Conv2d(128, 1, 1)` → `BN` → `ReLU` → `Flatten` → `Linear(1*2*12, 256)` → `ReLU` → `Linear(256, 1)` → `Tanh`

The `predict` method:
1. Sets model to eval mode, wraps obs in a batch
2. Forward pass to get logits and value
3. Masks illegal actions (set logits to -inf)
4. Applies softmax to get probabilities
5. Returns dict of {action: prob} for legal actions only, and the scalar value

### M2.2: Model Utilities

Same file or a small helper:

```python
def save_checkpoint(model, optimizer, step, path): ...
def load_checkpoint(path, model, optimizer=None) -> dict: ...
```

Checkpoint format: `{"model_state_dict": ..., "optimizer_state_dict": ..., "step": ..., "config": ...}`

**Tests** (`tests/test_network.py`):
- Forward pass with random input (1, 16, 2, 12) produces policy_logits shape (1, 1352) and value shape (1, 1)
- `predict()` returns probabilities summing to ~1.0
- `predict()` returns zero probability for illegal actions
- Batch forward pass works (32, 16, 2, 12)
- Save/load checkpoint round-trips without changing model output
- Model parameter count is reasonable (print it, sanity check)

### M2 Pass Condition
All M2 tests pass. Forward pass produces correct shapes. Legal action masking works correctly. Model can be saved and loaded.

---

## M3: Search (MCTS)

**Goal**: Implement AlphaZero-style MCTS that handles backgammon's chance nodes correctly.

### M3.1: MCTS Implementation

**File**: `raccoon/search/mcts.py`

Core data structures:
```python
@dataclass
class MCTSNode:
    state: GameState
    parent: MCTSNode | None
    parent_action: int | None
    children: dict[int, MCTSNode]     # action -> child node
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0                # from policy network
    
    @property
    def q_value(self) -> float:
        """Average value = value_sum / visit_count (or 0 if unvisited)."""
    
    def is_expanded(self) -> bool:
        """True if all legal actions have child nodes."""


class MCTS:
    def __init__(
        self,
        network: RaccoonNet,
        num_simulations: int = 100,
        c_puct: float = 1.5,
        temperature: float = 1.0,
    ): ...
    
    def search(self, state: GameState) -> dict[int, float]:
        """
        Run MCTS from the given state.
        
        Returns:
            action_probs: dict mapping action -> visit probability
        """
    
    def select_action(self, action_probs: dict[int, float]) -> int:
        """Sample action from visit distribution (with temperature)."""
```

MCTS algorithm per simulation:
1. **Select**: From root, walk down tree choosing actions by PUCT:
   ```
   PUCT(s, a) = Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N(s,a))
   ```
   Choose `argmax PUCT(s, a)` over children.

2. **Expand + Evaluate**: At a leaf decision node:
   - Call `network.predict(encode(state), legal_actions)` → `(policy, value)`
   - Create child nodes for all legal actions, setting `prior = policy[a]`
   
3. **Backup**: Propagate value back up the path, negating at each level (since backgammon is two-player zero-sum — the value for the opponent is negated).

**Chance node handling** (the key adaptation):

When traversing the tree and encountering a state where the next step involves a chance node (dice roll):

```python
def _advance_through_chance(self, state: GameState) -> GameState:
    """
    If state is at a chance node, sample dice and apply
    until we reach a decision node or terminal.
    """
    while state.is_chance_node():
        outcomes = state.chance_outcomes()
        actions, probs = zip(*outcomes)
        chosen = np.random.choice(actions, p=probs)
        state.apply_action(chosen)
    return state
```

This is called:
- After expanding a node and applying a player's action, if the resulting state is a chance node (opponent's dice roll), advance through it before creating the child node
- The child node in the tree always corresponds to a decision node or terminal

This means the MCTS tree contains only decision nodes and terminals. Chance is handled by sampling during expansion. With 100+ simulations, the various dice outcomes are naturally explored.

**Temperature handling**:
- `temperature = 1.0`: sample proportionally to visit counts (exploration during training)
- `temperature → 0`: pick the most-visited action (exploitation during evaluation)

**Tests** (`tests/test_mcts.py`):
- MCTS runs without crashing on starting position
- With a random network, search returns a valid probability distribution over legal actions
- Visit counts sum to `num_simulations`
- PUCT formula correctly prioritizes high-prior + low-visit actions
- Chance nodes are never stored as tree nodes (all tree nodes are decision/terminal)
- Terminal nodes are handled correctly (no expansion, value from game result)
- Temperature=0 returns deterministic argmax
- MCTS with 1 simulation just returns the network's prior (sanity check)

### M3 Pass Condition
All M3 tests pass. MCTS runs end-to-end on arbitrary positions. Chance nodes are handled correctly. No crashes on terminal states or unusual positions (e.g., all checkers on bar).

---

## M4: Training Loop

**Goal**: Implement self-play data generation, replay buffer, and network training.

### M4.1: Self-Play

**File**: `raccoon/train/self_play.py`

```python
@dataclass
class TrainingExample:
    observation: np.ndarray     # (16, 2, 12)
    policy_target: np.ndarray   # (1352,) — MCTS visit distribution
    value_target: float         # game outcome from this player's perspective

def play_one_game(
    network: RaccoonNet,
    num_simulations: int = 100,
    temperature: float = 1.0,
    temp_threshold: int = 30,   # switch to temp=0 after this many moves
) -> list[TrainingExample]:
    """
    Play a complete self-play game.
    
    Returns list of (obs, policy, pending_value) that get filled
    with the final game result.
    """
```

Logic:
1. Create a new game via `GameWrapper`
2. At each decision node:
   a. Encode state → observation tensor
   b. Run MCTS → action probabilities
   c. Record (observation, action_probs, current_player)
   d. Sample action (with temperature) and apply
   e. Advance through any chance nodes
3. At terminal: get game result, fill in `value_target` for each recorded position (from that position's player's perspective)
4. Return training examples

### M4.2: Replay Buffer

**File**: `raccoon/train/replay_buffer.py`

```python
class ReplayBuffer:
    def __init__(self, max_size: int = 100_000): ...
    def add_game(self, examples: list[TrainingExample]): ...
    def sample_batch(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (obs_batch, policy_batch, value_batch)."""
    def __len__(self) -> int: ...
```

Simple circular buffer backed by a list/deque. Stores individual positions, not games. `sample_batch` returns random samples as PyTorch tensors ready for training.

### M4.3: Training Orchestrator

**File**: `raccoon/train/coach.py`

```python
class Coach:
    def __init__(
        self,
        network: RaccoonNet,
        optimizer: torch.optim.Optimizer,
        replay_buffer: ReplayBuffer,
        num_simulations: int = 100,
        batch_size: int = 256,
        games_per_iteration: int = 50,
        training_steps_per_iteration: int = 100,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
    ): ...
    
    def run_iteration(self, iteration: int):
        """One full iteration: self-play → train → checkpoint → log."""
    
    def self_play_phase(self) -> list[TrainingExample]: ...
    def training_phase(self) -> dict[str, float]:  # loss metrics
        """Sample from replay buffer and train network."""
    def save_checkpoint(self, iteration: int): ...
    def log_metrics(self, iteration: int, metrics: dict): ...
```

Training step loss:
```python
def compute_loss(policy_logits, value, target_policy, target_value):
    policy_loss = -torch.sum(target_policy * F.log_softmax(policy_logits, dim=1), dim=1).mean()
    value_loss = F.mse_loss(value.squeeze(-1), target_value)
    return policy_loss + value_loss  # L2 reg via optimizer weight_decay
```

Metadata logged per iteration:
- Iteration number
- Number of self-play games, total positions generated
- Replay buffer size
- Policy loss, value loss, total loss
- Average game length
- Timestamp, time per phase

**File**: `scripts/train.py` — CLI entry point:
```bash
python scripts/train.py --iterations 100 --games-per-iter 50 --simulations 100 --lr 0.001
```

**Tests** (`tests/test_self_play.py`, `tests/test_coach.py`):
- `play_one_game` completes without error and returns non-empty examples
- All examples have correct tensor shapes
- Policy targets sum to ~1.0 and have non-zero entries only at legal actions
- Value targets are in [-1, 1] (or [-3, 3] for backgammon/gammon outcomes)
- Replay buffer stores and samples correctly
- After one training iteration, loss is a finite number
- Checkpoint is saved to disk and loadable
- Log file is created with expected fields

### M4 Pass Condition
All M4 tests pass. A tiny training run (5 games, 10 training steps, 10 simulations) completes without error. Loss values are finite and decrease over a few iterations. Checkpoints are saved. Logs are written.

---

## M5: Internal Evaluation

**Goal**: Build a checkpoint-vs-checkpoint arena and verify that training improves play.

### M5.1: Arena

**File**: `raccoon/eval/arena.py`

```python
class Arena:
    def __init__(
        self,
        player1: RaccoonNet,
        player2: RaccoonNet,
        num_games: int = 100,
        num_simulations: int = 50,
    ): ...
    
    def play_match(self) -> MatchResult:
        """Play num_games, alternating who goes first."""
    
@dataclass
class MatchResult:
    wins_p1: int
    wins_p2: int
    p1_points: float            # total equity (gammons count double, etc.)
    p2_points: float
    avg_game_length: float
    
    @property
    def win_rate_p1(self) -> float: ...
    def summary(self) -> str: ...
```

Each arena game:
1. Create new game, assign player 1 and player 2
2. Alternate: player 1 is always "current model" perspective for network 1
3. Each player uses MCTS with temperature=0 (greedy best play)
4. Record result including gammon/backgammon multipliers

**Tests**:
- Arena completes a 10-game match without error
- Win counts sum to total games
- Random network vs same random network → ~50% win rate (within variance)

### M5.2: Evaluation Script

**File**: `scripts/evaluate.py`

```bash
# Compare two checkpoints
python scripts/evaluate.py --checkpoint1 checkpoints/iter_100.pt --checkpoint2 checkpoints/iter_50.pt --games 200

# Compare against random play
python scripts/evaluate.py --checkpoint1 checkpoints/iter_100.pt --random --games 200
```

### M5 Pass Condition
Arena works. After ~50 training iterations (even on CPU), the latest checkpoint shows:
- Positive average equity (>+0.3 ppg) against a random (untrained) network
- Positive average equity (>+0.05 ppg) against an earlier checkpoint (e.g., iteration 5 vs iteration 50)

---

## M6: GNUBG Benchmark

**Goal**: Automated money game sessions against GNUBG CLI.

### M6.1: GNUBG Harness

**File**: `raccoon/eval/gnubg_harness.py`

GNUBG supports `gnubg --tty` mode for non-interactive play. The harness:
1. Starts GNUBG in CLI/tty mode
2. Sets money game, no cube (initially — cubeless)
3. Sends board positions and receives GNUBG's move choices
4. Plays Raccoon's engine against GNUBG's engine with shared dice rolls
5. Records game results

Alternatively, use GNUBG's "external player" socket interface if the CLI approach is too fragile. Research both options during implementation and pick the simpler one.

```python
class GnubgHarness:
    def __init__(
        self,
        raccoon_network: RaccoonNet,
        gnubg_path: str = "gnubg",
        gnubg_level: str = "world",    # gnubg evaluation settings
        num_simulations: int = 200,
    ): ...
    
    def play_match(self, num_games: int = 1000) -> BenchmarkResult:
        """Play money game sessions against GNUBG."""

@dataclass
class BenchmarkResult:
    raccoon_wins: int
    gnubg_wins: int
    raccoon_equity: float           # total money won/lost
    gnubg_equity: float
    num_games: int
    confidence_interval_95: float   # approximate 95% CI on win rate
    
    def summary(self) -> str: ...
```

**Statistical requirements**:
- At 1000 games, 95% CI on win rate is approximately ±3 percentage points
- Log per-game results for later analysis
- Compute equity rate (average equity per game) in addition to raw win rate

### M6.2: Evaluation Script

**File**: `scripts/eval_gnubg.py`

```bash
python scripts/eval_gnubg.py --checkpoint checkpoints/best.pt --games 1000 --gnubg-level world
```

Output: summary table + detailed per-game log saved to `logs/gnubg_eval_<timestamp>.json`.

### M6.3: Game Logging

**File**: `raccoon/eval/game_log.py`

Log games in a standard format (compatible with backgammon notation):
```python
@dataclass
class GameRecord:
    moves: list[MoveRecord]       # sequence of (player, dice, action, board_state)
    result: float                 # final equity
    timestamp: str
    raccoon_version: str
    opponent: str
    
def save_match_log(games: list[GameRecord], path: str): ...
```

### M6 Pass Condition
The GNUBG harness completes a 100+ game automated money game session without crashing. Results are logged. Win rate and equity in point-per-game, ppg, are computed with confidence intervals.

---

## M7: Strength Scaling

**Goal**: Iterate on training to surpass GNUBG.

This milestone is open-ended and iterative. The focus is on scaling and tuning, not new code.

### M7.1: Scaling Knobs

Increase these progressively:
- **Network size**: 6 → 10 → 20 residual blocks, 128 → 256 channels
- **MCTS simulations**: 100 → 200 → 400 per move
- **Self-play volume**: 50 → 200 → 1000 games per iteration
- **Replay buffer**: 100K → 500K → 1M positions
- **Training iterations**: run for many hundreds/thousands of iterations

### M7.2: Backgammon-Specific Enhancements

Try these if raw scaling isn't enough:
- **Richer value target**: Instead of scalar ±1, predict 5 outputs: P(win normal), P(win gammon), P(win backgammon), P(lose gammon), P(lose backgammon). This helps the network learn the difference between gammon-saving and gammon-winning plays.
- **Bearoff database**: For positions where all checkers are in the home board, use a precomputed table instead of the network. This frees network capacity for contact positions.
- **Better encoding features**: Add pip count, contact/race classification, prime detection as extra channels.
- **Training curriculum**: Start with shorter games (bearoff positions), gradually introduce full games.

### M7.3: Evaluation Cadence

- Every 10 training iterations: internal eval (checkpoint vs previous best)
- Every 50 iterations: smoke eval vs GNUBG (100 games)
- Every 200 iterations: full eval vs GNUBG (1000+ games)
- Track Elo progression over time

### M7 Pass Condition
Raccoon has positive money game equity against GNUBG at "world class" settings over a statistically significant number of games (10000+, 95% CI excluding 0).

---

## Supporting Infrastructure

### Makefile

```makefile
setup:      # Install dependencies, verify OpenSpiel + PyTorch
test:       # Run pytest
smoke:      # 5-game self-play training run (quick sanity check)
train:      # Full training run
eval:       # Checkpoint vs checkpoint evaluation
eval-gnubg: # GNUBG benchmark
play:       # Interactive terminal play
```

### pyproject.toml

```toml
[project]
name = "raccoon"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch",
    "numpy",
    "open-spiel",
]

[project.optional-dependencies]
dev = ["pytest"]
```

### Communication Protocol (RGP)

**File**: `raccoon/protocol/rgp.py`

Text-based, line-oriented, stdin/stdout. Inspired by UCI (chess).

Commands from controller to engine:
```
rgp                     # identify protocol
isready                 # engine ready check
newgame                 # start new game
position <state>        # set board position  
dice <d1> <d2>          # set dice for next move
go simulations <N>      # search with N simulations
quit                    # exit
```

Responses from engine:
```
id name Raccoon v0.1
id author <author>
rgpok                   # protocol acknowledged
readyok                 # ready
bestmove <action>       # chosen move
info score <value> pv <moves>  # search info
```

### CLI Play Interface

**File**: `raccoon/cli/play.py`

Terminal-based interface for humans to play against Raccoon:
- Display board in ASCII art (similar to GNUBG's terminal output)
- Accept moves in standard notation (e.g., `24/21 13/11`)
- Show Raccoon's evaluation and move choice
- Support `hint` command for analysis

---

## Execution Order

```
M1.1 → M1.2 → M1.3 → (M1 tests)
  ↓
M2.1 → M2.2 → (M2 tests)
  ↓
M3.1 → (M3 tests)
  ↓
M4.1 → M4.2 → M4.3 → (M4 tests + smoke train)
  ↓
M5.1 → M5.2 → (M5 tests + verify training improves play)
  ↓
M6.1 → M6.2 → M6.3 → (M6 tests + first GNUBG benchmark)
  ↓
M7 (iterate until goal is met)
```

Estimated CPU timeline on 2013 iMac:
- M1-M3: 1-2 sessions (code + tests, no heavy compute)
- M4: 1 session (code + first smoke training)
- M5: 1 session (eval infrastructure + initial training runs)
- M6: 1 session (GNUBG integration)
- M7: Weeks/months of training runs (can run overnight)

## Key Technical Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Game engine | OpenSpiel | Proven, correct rules, 1352-action encoding |
| Board encoding | (16, 2, 12) tensor | ResNet-compatible, captures spatial structure |
| Network | ResNet 6×128 (start) | Small enough for CPU, proven architecture |
| Action space | 1352 (OpenSpiel's) | No need to reinvent; mask illegal actions |
| MCTS chance nodes | Sample + skip to decision | Never evaluate at chance; statistically covers dice |
| Value target | Scalar ±1 (start) | Simple; upgrade to 5-output later for gammon awareness |
| Framework | PyTorch | Best Python ecosystem, simple code |
| Training | From scratch | Cleaner than patching OpenSpiel's AZ; full control |
