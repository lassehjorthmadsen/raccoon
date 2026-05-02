# MCTS and AlphaZero Training — Concepts and Examples

## What is a "simulation"?

The name comes from the original Monte Carlo Tree Search, where a simulation meant playing a game out to the end randomly to collect an outcome. AlphaZero replaced those random rollouts with a neural network value estimate — but kept the name.

A better mental model: each simulation is one **guided look-ahead path** carved through an enormous game tree. It follows the most promising branches according to a selection rule (PUCT), walking a few nodes deep until it reaches a position the network hasn't seen yet. The network then answers "how good is this position?" — a single number in [-1, 1] — without looking any further. The result gets backed up through all the nodes on the path.

After 100 simulations you have a small subtree selectively expanded in the directions worth exploring. It is not like a Monte Carlo simulation; it's more like a few branches a few nodes deep in an enormous tree.

The only genuine randomness is in dice rolls: when a simulation passes through a chance node, it samples one dice outcome rather than branching on all 21 possibilities. That's what keeps the tree tractable.

### The PUCT selection rule

At each interior node, MCTS picks the child with the highest score:

```
PUCT(a) = −Q(child) + c_puct · P(a) · √N(parent) / (1 + N(child))
```

- **−Q(child)**: the child's average backed-up value, negated. Q is stored from the child's own perspective (the opponent of the node to move), so the minus converts it to the current player's viewpoint.
- **P(a)**: network prior probability for move *a*.
- **N(parent), N(child)**: visit counts. When N(child)=0, the term reduces to `c_puct · P(a) · √N(parent)` — pure prior-weighted curiosity. As visits accumulate, Q dominates and the exploration bonus shrinks.
- **c_puct**: exploration constant (1.5 in Raccoon).

### Backing up the value

After evaluating a leaf, the network value is propagated back up the search path, **negated at every step**, because the two players have opposite interests:

```
leaf value:  +0.6   ← good for the player to move at the leaf
    ↑ ×(−1)
parent:      −0.6   ← same leaf is bad for whoever moved into it
    ↑ ×(−1)
grandparent: +0.6   ← same player as the leaf
```

Q values therefore always express "expected outcome for whoever is to move at this node", automatically, at every depth.

---

## Visit counts and the policy target

Each legal move at the root starts with zero visits. Every time a simulation passes through a move on its way down the tree, that move's visit count increments. After 100 simulations, the counts might look like:

```
Move A:  61 visits  ← search kept returning here
Move B:  22 visits
Move C:  11 visits
Move D:   4 visits
Move E:   2 visits
```

Dividing by 100 gives the **visit distribution**: `{A: 0.61, B: 0.22, ...}`. This is stored as the policy target for the training example — the network learns to match it.

The visit distribution is a *search-refined* belief. It incorporates information from actually looking ahead, not just the network's raw guess. A move the network initially likes but that leads to bad positions will lose visits over time within the search; a move the network underrated but that keeps producing good outcomes will gain them.

---

## Entropy

Entropy measures how spread out a probability distribution is:

```
H = -∑ p · ln(p)   (natural log; result in nats)
```

- **Low entropy** — visits pile up on one or two moves. The search is certain, or stuck. If one move gets 95% of visits, H ≈ 0.
- **High entropy** — visits spread across many moves. The search is genuinely exploring.

A useful intuition: `e^H` gives the **effective number of moves** the search is seriously considering. H = 1.35 nats means roughly 4 moves are being meaningfully explored. H = 0 means the search has collapsed to one move.

Entropy is logged as `avg_visit_entropy` in the training log — the mean over all decision moves in a game. It's a proxy for whether exploration is healthy or has collapsed.

---

## Why visit distributions beat raw network output

The raw network policy prior is just a guess — "which moves look promising before looking ahead?" When the network is newly initialised, this is nearly random.

The search-refined visit distribution incorporates **value estimates at the leaves**. Even if the policy prior is random, simulations that reach positions the value head rates highly accumulate more visits. So as long as the value head has learned *anything at all*, the visit distribution is slightly better than the prior.

The network then trains to match visit distributions — always training to match something slightly smarter than itself. That's the bootstrapping loop.

In the very first iterations, when the value head is random too, the advantage is negligible. The system really starts learning once the value head picks up genuine patterns from game outcomes.

---

## Temperature

After running all simulations, the visit distribution needs to be converted into (a) a training target and (b) an actual move choice. Temperature governs the second without affecting the first.

The **policy target** is always the raw visit distribution — visit counts divided by total visits. Temperature is never applied to the training label.

To **select a move**, visit counts are raised to the power `1/T`, then renormalised:

```
p(a) ∝ N(a)^(1/T)
```

- **T = 1.0** — sample proportionally to visit counts. The most-visited move is most likely but alternatives get real probability. Used for the first 30 moves of each training game to expose the network to a wider variety of positions.
- **T = 0** — always pick the most-visited move (argmax). Deterministic. Used for the remainder of each game, where consistent play produces cleaner value targets.

The policy target stored in each training example is always the **raw visit distribution**, regardless of which temperature was used to pick the actual move.

---

## The learning cycle

The only ground truth in the system is **game outcomes**. Everything else is the network talking to itself. The chain is:

```
game outcome → value head improves → look-ahead becomes meaningful
             → visit distributions improve → policy head improves
```

The policy head has no independent ground truth. It only trains to match visit distributions, which are only as good as the value head behind them. This is why value loss is the more important diagnostic in the training curves.

### A toy example: one-move backgammon

To make this concrete, imagine a drastically simplified game: you roll a die, you have exactly two legal moves — Left or Right — and the game ends immediately. Left wins 70% of the time, Right 30%.

**Iteration 0 — network knows nothing**

Value head returns 0.0 everywhere. Policy prior: 50/50.

MCTS runs 4 simulations. Both moves get evaluated, both return 0.0. Visit distribution: 50/50 — identical to the prior. Search added nothing.

Game is played, Left chosen, Left wins. Training example stored:
```
policy_target = 50/50,  value_target = +1
```
Value head nudges slightly toward "Left-type positions → +1". Policy head learns nothing (target was 50/50).

---

**Iteration 1 — value head has a tiny signal**

Value head now returns +0.1 for Left positions, 0.0 for Right.

MCTS runs 4 simulations. PUCT slightly favours Left (Q=+0.1) but also explores Right (unvisited). Visit counts: Left 3, Right 1. Distribution: **75/25**.

Game played, Left wins again. Training example:
```
policy_target = 75/25,  value_target = +1
```
Now the policy head has a real signal: Left should get more visits. Value head gets another +1 from a Left position.

---

**Iteration 2 — signal compounding**

Value head: Left → +0.2, Right → 0.0.

MCTS now sends most simulations to Left. PUCT's exploration bonus keeps the distribution from reaching absolute 0 on Right, but the skew is severe — say **~95/5**. The game outcome is a **loss** (it happens 30% of the time). Training example:
```
policy_target = 95/5,  value_target = -1
```
The value head gets a -1 from a Left position and revises downward. The skewed policy target is also a problem: the policy head is learning to almost never consider Right, even though Right sometimes leads to better outcomes.

---

**Stabilisation**

Over many games, the value head converges toward the true expected outcome of each move:
- Left: wins 70% → value converges to **+0.4**
- Right: wins 30% → value converges to **-0.4**

*(In Raccoon, outcomes are ±1/±2/±3 for normal/gammon/backgammon wins, divided by 3 before being stored as value targets so they fit in the tanh output range. The ±0.4 figures above assume simple ±1 outcomes.)*

Once the value head is accurate, MCTS sends essentially all visits to Left — it is always the better move, so there is no reason to keep exploring Right. The policy head learns to strongly prefer Left, and the game is essentially solved.

---

### A backgammon example: blitz or build?

Same learning loop, more realistic setting. Mid-game, you roll 6-5. Two legal moves:

- **Move A — Blitz**: Hit two of your opponent's blots, sending both to the bar. Aggressive. If the opponent rolls poorly re-entering, you could win a gammon. But your own board is left open.
- **Move B — Build**: Make two strong blocking points in your home board. Solid. Fewer fireworks, but you reduce your own gammon risk and set up a prime.

**Early training — network knows nothing**

Value head is random. MCTS explores both moves roughly equally. Some games from Move A end in gammons (both winning and losing ones). Some games from Move B end in normal wins. The value head starts picking up a weak signal: "aggressive positions correlate with large swings in outcome."

**After ~50 iterations — network has learned something**

Both sides are leaning toward Move A. The feedback loop runs as follows: the network rates blitz positions higher → PUCT sends more simulations to Move A → visit distributions tighten around Move A → the network trains to become even more confident about blitz → repeat. Build positions (Move B) appear rarely in self-play, so the value head accumulates little data on them and cannot evaluate them reliably — not because building is bad, but because it has almost stopped being explored. MCTS now sends nearly all visits to Move A.

Self-play becomes: both sides blitz. Games are chaotic and end in gammons frequently. The value head accurately models "this type of position leads to a gammon" — but it has seen very few games where someone played Move B and reached a stable mid-game. It can't evaluate those positions well because they rarely appear in training data.

**The plateau**

Both sides have learned to blitz. Gammon rates freeze at 40%+. Policy loss keeps falling — the network is confidently and consistently choosing Move A — but that just means it has confidently learned the wrong thing. The value head is accurate for the positions it has seen, useless for the positions it hasn't.

The within-experiment benchmark test confirms this: playing iter 282 against iter 260 produces no detectable improvement. Both checkpoints have the same blind spot.

**With Dirichlet noise**

On some fraction of moves, MCTS is nudged to seriously consider Move B even though the network currently rates it lower. Games now occasionally follow the build path. The value head starts seeing positions that arise from building — and crucially, seeing their outcomes. It discovers that build positions lead to stable wins with lower variance: fewer gammons conceded, fewer gammons won.

Over time the value head can discriminate: "blitz is better in a racing game where the opponent is weak; building is better when you need to contain." Visit distributions widen. Self-play games diversify. The gammon rate starts falling not because the network was told to avoid gammons, but because it explored enough of the tree to discover that some moves lead to them less often.

---

## The plateau problem and why it happens

In Raccoon's case, both sides of self-play were conceding gammons and backgammons at roughly 10x the rate of strong play. The value head learned "this type of position leads to a gammon" accurately — but *never saw the positions that arise from avoiding gammons*, because neither side in self-play ever explored those branches.

The visit distributions collapsed: the network got opinionated about which moves to play, simulations confirmed those opinions, and the search stopped looking at alternatives. The policy head trained on those collapsed distributions, the network got more opinionated, and so on. A feedback loop.

Evidence: `avg_visit_entropy` declining over training, gammon and backgammon rates frozen across hundreds of iterations despite falling policy loss.

---

## Dirichlet noise

At the root node of every self-play search, before any simulations run, the network's prior probabilities are mixed with a sample from a Dirichlet distribution:

```
p_noisy = (1 - ε) · p_network  +  ε · Dirichlet(α)
```

With ε=0.25 and α=0.3 (Raccoon's settings), 25% of the prior is replaced by a random vector that is somewhat sparse — a few moves get a meaningful bump, most get very little, and *which* moves get the bump is different every single search.

The Dirichlet distribution itself generates random probability vectors. The α parameter controls concentration:
- Low α (0.1): spiky — most weight on one item
- α = 1: roughly uniform random
- High α (10): very close to uniform

α=0.3 produces vectors that are moderately sparse, matching the typical structure of backgammon positions where a few moves are genuinely better but some alternatives are worth considering.

**Why this helps:** the noise forces MCTS to occasionally commit simulations to moves the network currently rates lowly. Some of those moves are bad — the network learns that. But some lead to positions the search has never visited, generating training examples that cover new territory. Over time the self-play distribution widens, the value head learns more of the game tree, and the feedback loop is broken.

The `avg_visit_entropy` metric in the training log is the direct observable: if entropy holds up or rises over training (rather than collapsing toward 0), the noise is doing its job.
