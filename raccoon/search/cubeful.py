"""Cubeful search: apply Janowski at every node, not only at the root.

Raccoon's cube decision (exp025) evaluates the position once and applies
Janowski's closed form to the result. GNU Backgammon instead searches, deciding
the cube at every node it visits and using the closed form only at the leaves.
exp026 measured Raccoon's static cube against that search and lost. This module
is the same recursion GNU Backgammon runs.

The recursion, in money points normalised to a cube value of 1 and always from
the point of view of the player on roll:

.. code-block:: text

    E(position, cube, depth):
        if depth == 0:
            return cl2cf_money(net probabilities, cube, x)
        nd = sum over the 21 rolls, weighted:
                 best over the mover's legal turns of -E(child, flipped cube, depth-1)
        if the mover may double:
            dt = 2 * E(position, cube now owned by the opponent, depth)
            return max(nd, min(dt, 1.0))
        return nd

Three things follow from that shape and are worth stating before reading the
code.

**Depth 0 must reproduce the closed form exactly.** At depth 0 the two branches
are ``cl2cf_money`` on one distribution under two ownership labels, which is
precisely what :func:`raccoon.cube.janowski.cube_equities` computes. That makes
the whole recursion testable against the module it generalises, in the same way
``child_cubeful_values`` at ``x = 0`` reproduces the cubeless ranking.

**The cube state never changes the board tree.** It changes only how a leaf
distribution is converted to a number. So one expansion serves every cube state
reachable at that node, and the ``dt`` branch -- the same position with the cube
handed over -- costs arithmetic rather than a second tree. This is what makes
searching the cube affordable: at depth 1 a cube decision is one node expansion,
about 420 evaluations, against roughly a thousand for a filtered checker search.
It stops being free at depth 2 and beyond, where the move chosen at an internal
node depends on the cube state and therefore decides which subtree gets searched
deeper.

**"Too good to double" needs no special case.** ``nd`` is free to exceed 1, and
``max(nd, min(dt, 1))`` then declines the double on its own.

Boards are gnubg-nn 2x25 tuples, as in :mod:`raccoon.search.expectimax`, with
slot 1 on roll. Values here are **money points at cube value 1**, roughly
[-3, 3] -- not the equity/3 that ``expectimax`` and ``value_equity`` use. The two
scales must not be mixed; see the note in
:func:`raccoon.train.lookahead.child_cubeful_values`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from raccoon.cube.janowski import (
    CENTERED, OPPONENT, PLAYER, CubeEquities, cl2cf_money, jacoby_active,
    probs6_to_cumulative5,
)
from raccoon.cube.state import flip_label
from raccoon.env.encoder import encode_state
from raccoon.eval.gnubg_adapter import board_to_view, terminal_equity_after_move
from raccoon.search.expectimax import ROLLS, Board, _children

# The mover may turn the cube when they own it or it sits in the middle.
_MAY_DOUBLE = (CENTERED, PLAYER)


@dataclass(frozen=True)
class CubefulConfig:
    """Search parameters for the cube recursion.

    ``depth`` counts roll expansions below the pre-roll node: 0 is the closed
    form, 1 expands the mover's roll once and evaluates the resulting positions
    statically, and so on.

    ``k`` caps how many of the mover's legal turns are considered per roll,
    keeping the best by static value. It only binds at ``depth >= 2``; at depth 1
    every turn's value is already known from the one batch, so filtering would
    save nothing and could only lose accuracy.

    There is deliberately no gate. Skipping decisions the closed form calls
    clearly is tempting -- most turns in a game are obvious no-doubles -- but a
    sound gate has to bound how far the search could move ``nd``, and searching
    is what changes ``nd``. An unsound gate would silently cap the effect the
    experiment is trying to measure. At about two seconds per decision and
    roughly fourteen cube decisions per game the cost is affordable without one,
    so the knob is left out until it can be justified rather than guessed.
    """

    depth: int = 1
    k: int = 8
    batch_size: int = 512

    def __post_init__(self) -> None:
        if self.depth < 0:
            raise ValueError(f"depth must be >= 0, got {self.depth}")
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")

    def tag(self) -> str:
        return f"cf_d{self.depth}_k{self.k}"


def terminal_points(board: Board) -> float | None:
    """Points to the player **on roll** if the game is over, else ``None``.

    ``board`` holds the player who just moved in slot 0, so a finished game is a
    loss for the side on roll: -1, -2 or -3. Undivided, unlike
    ``expectimax._terminal_value`` -- this module works in points.
    """
    points = terminal_equity_after_move([list(board[0]), list(board[1])])
    return None if points is None else -points


class ProbsEvaluator:
    """Batched, deduplicating six-outcome evaluation, cached per search.

    Caches the cumulative 5-vector rather than a single number, because the cube
    conversion needs the whole distribution and because one distribution serves
    every cube state at that board.
    """

    def __init__(self, network, device, channels=None, batch_size: int = 512):
        self.network = network
        self.device = device
        self.channels = channels
        self.batch_size = batch_size
        self.cache: dict[Board, tuple[float, ...]] = {}
        self.evaluated = 0

    @torch.no_grad()
    def probs(self, boards: list[Board]) -> list[tuple[float, ...]]:
        todo = [b for b in dict.fromkeys(boards) if b not in self.cache]
        for start in range(0, len(todo), self.batch_size):
            chunk = todo[start:start + self.batch_size]
            obs = np.stack([
                encode_state(board_to_view([list(b[0]), list(b[1])]),
                             channels=self.channels)
                for b in chunk
            ])
            x = torch.from_numpy(obs).float().to(self.device, non_blocking=True)
            out = self.network.value_probs6(x).cpu().numpy()
            for board, p6 in zip(chunk, out):
                self.cache[board] = probs6_to_cumulative5(p6)
        self.evaluated += len(todo)
        return [self.cache[b] for b in boards]

    def leaf_value(self, board: Board, label: str, x: float, jacoby: bool) -> float:
        """Closed-form cubeful points for a cached board, from its mover's POV."""
        return cl2cf_money(self.cache[board], label, x,
                           jacoby_active(label, jacoby))


def _node_value(board: Board, label: str, depth: int, ev: ProbsEvaluator,
                x: float, jacoby: bool, cfg: CubefulConfig,
                memo: dict) -> float:
    """Cubeful points to the player on roll at ``board``, cube seen as ``label``.

    Memoised on ``(board, label, depth)``. The cube *value* is absent by design:
    equities are normalised to a cube of 1 throughout and the caller scales, so
    two positions differing only in stake share a subtree.
    """
    key = (board, label, depth)
    hit = memo.get(key)
    if hit is not None:
        return hit

    nd = _no_double_value(board, label, depth, ev, x, jacoby, cfg, memo)
    if label in _MAY_DOUBLE:
        # After a double is taken the opponent owns the cube at twice the stake.
        # Normalised to a cube of 1 that is twice the mover's equity with the
        # cube handed over -- the same board, so the tree is shared.
        dt = 2.0 * _node_value(board, OPPONENT, depth, ev, x, jacoby, cfg, memo)
        value = max(nd, min(dt, 1.0))
    else:
        value = nd
    memo[key] = value
    return value


def _no_double_value(board: Board, label: str, depth: int, ev: ProbsEvaluator,
                     x: float, jacoby: bool, cfg: CubefulConfig,
                     memo: dict) -> float:
    """Value of playing on: the roll-weighted best turn, cube left where it is."""
    if depth == 0:
        ev.probs([board])
        return ev.leaf_value(board, label, x, jacoby)

    child_label = flip_label(label)
    total = 0.0
    for die1, die2, weight in ROLLS:
        kids = _children(board, die1, die2)
        terminals = [terminal_points(k) for k in kids]
        live = [k for k, t in zip(kids, terminals) if t is None]
        if live:
            ev.probs(live)
        if depth > 1 and len(live) > cfg.k:
            # Only at depth >= 2 does filtering save anything: the static values
            # are already in hand, so keep the k best for the mover and search
            # only those deeper.
            statics = [-ev.leaf_value(k, child_label, x, jacoby) for k in live]
            keep = set(np.argsort(statics)[::-1][:cfg.k].tolist())
            live = [k for i, k in enumerate(live) if i in keep]
        live_set = set(live)

        best = -np.inf
        for kid, term in zip(kids, terminals):
            # Both branches give the value to the player on roll AT THE CHILD,
            # which is the opponent, so both are negated to reach the mover's
            # value. Negating only the recursive branch scores every
            # game-ending move with its sign flipped, which is wrong exactly
            # where cube decisions are sharpest.
            if term is not None:
                value = -term
            elif kid in live_set:
                value = -_node_value(kid, child_label, depth - 1, ev, x,
                                     jacoby, cfg, memo)
            else:
                continue    # filtered out at depth >= 2
            if value > best:
                best = value
        total += weight * best
    return total


def cube_equities_searched(
    board: Board, label: str, network, device, cfg: CubefulConfig,
    x: float, jacoby: bool = False, channels=None,
) -> tuple[CubeEquities, ProbsEvaluator]:
    """The three cubeful equities for a pre-roll position, by search.

    Same shape and sign convention as
    :func:`raccoon.cube.janowski.cube_equities`, so the two are directly
    comparable and ``cfg.depth == 0`` must reproduce it exactly. ``label`` is
    from the point of view of the player on roll and must not be ``OPPONENT`` --
    a player cannot double a cube their opponent owns.

    Returns the equities and the evaluator, so a caller can read how many boards
    the search cost.
    """
    if label == OPPONENT:
        raise ValueError("cannot make a cube decision when the opponent owns the cube")

    ev = ProbsEvaluator(network, device, channels, cfg.batch_size)
    memo: dict = {}
    nd = _no_double_value(board, label, cfg.depth, ev, x, jacoby, cfg, memo)
    dt = 2.0 * _node_value(board, OPPONENT, cfg.depth, ev, x, jacoby, cfg, memo)
    return CubeEquities(nd=nd, dt=dt, dp=1.0), ev


def cube_action_searched(
    board: Board, label: str, network, device, cfg: CubefulConfig,
    x: float, jacoby: bool = False, channels=None,
) -> tuple[bool, bool]:
    """``(should_double, should_take)`` from the searched equities."""
    eq, _ = cube_equities_searched(board, label, network, device, cfg, x,
                                   jacoby, channels)
    return min(eq.dt, eq.dp) > eq.nd, eq.dt < eq.dp
