"""Dice-luck control variate for variance-reduced rollouts (XG/GNUBG/BGSage style).

At every **pre-roll** state ``s`` (an OpenSpiel chance node), before the dice are
sampled:

.. code-block:: text

    h(s, d) = GNUBG 0-ply equity after best play for roll d, signed to a fixed
              player's POV                      -- a pure function of (s, d)
    m(s)    = sum_d p(d) * h(s, d)              -- the 1-ply pre-roll mean
    luck    = h(s, d_actual) - m(s)             -- the luck of the roll that happened

A rollout then reports ``vr = raw_game_result - sum_t luck_t``.

**Why this is exactly unbiased.** ``E[luck | history] = sum_d p(d) h(s,d) - m(s) = 0``
by construction, so the accumulated luck is a martingale with mean zero and
subtracting it cannot move the expectation — *however bad the evaluator is*. Variance
falls because the game result is dominated by dice luck, which the accumulated luck
tracks. See David Montgomery, "Variance Reduction" (GammOnLine, Feb 2000),
https://bkgm.com/articles/GOL/Feb00/var.htm.

**The rule that makes it self-consistent.** ``h`` must be the *same function* on both
legs and must **not depend on the move actually played**. We always use GNUBG's 0-ply
best-play value for whoever is on roll — regardless of who plays, what they play, or at
what ply. A weak move by either side changes no luck term; the cost shows up in the raw
result and in the next pre-roll state, which is correct. This is what the project's
earlier "error-rate ppg" proxy got wrong: it charged the played move against a pre-roll
value that was not that quantity's own dice-average, so its implicit luck term was not
zero-mean (biased ~0.07, dropped in commit e1940a5). It is also stricter than BGSage,
which scores the *played* move on the actual leg against the 1-ply-*best* move in the
mean and polices the resulting gap with a t-test.

**Implementation notes, all verified against this machine's ``gnubg-nn`` 1.1.0a6:**

- The sweep goes through :func:`raccoon.eval.gnubg_adapter.best_move_equity`, which uses
  gnubg's native move generator and then re-evaluates the position it lands on. A 21-roll
  sweep costs **~2.6 ms** versus **~260 ms** for the equivalent loop over
  ``candidate_equities``, so variance reduction adds only ~5% to the cost of a game, and
  it agrees with the slow route **exactly** (2160/2160 rolls over 60 real pre-roll states,
  doubles and game-ending rolls included). :func:`python_roll_value` is that slow route,
  kept as the test oracle.
- **Terminal rolls need the exact value.** ``candidate_equities`` hard-codes ``+3.0`` for
  a terminal child, which is fine for ranking but would score every game-ending roll as a
  backgammon here. ``best_move_equity`` reports the true ``1.0 / 2.0 / 3.0``.
- **``ply >= 1`` segfaults** inside ``gnubg_nn.best_move``; ply 0 sweeps thousands of
  boards clean. Keep ``ply=0``. This is not a practical limitation: a 0-ply leaf gives a
  1-ply pre-roll mean, which is exactly what GNUBG and BGSage use for their own variance
  reduction.
- **No RNG is touched.** Unlike ``candidate_equities`` (which calls
  ``_advance_through_chance`` and consumes numpy draws), the native route is pure. Turning
  variance reduction on therefore does not perturb play at all — same seed, same games.
  That is what lets a variance-reduced run and a plain run be compared as two independent
  samples of one process.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from raccoon.env.game_wrapper import GameState
from raccoon.eval.gnubg_adapter import (
    best_move_equity, board_from_view, candidate_equities,
)

# OpenSpiel's backgammon chance nodes. The opening node also encodes who starts.
MIDGAME_OUTCOMES = 36
OPENING_OUTCOMES = 30

# The 15 unordered non-doubles in OpenSpiel's ordering: (1,2),(1,3),...,(5,6).
_NON_DOUBLES = [(a, b) for a in range(1, 7) for b in range(a + 1, 7)]

# The 21 distinct rolls, as unordered (lo, hi) pairs. `best_move` treats (a,b) and
# (b,a) identically, so one sweep entry serves both orderings of a non-double.
ROLL_KEYS: list[tuple[int, int]] = [
    (a, b) for a in range(1, 7) for b in range(a, 7)
]

# Chance-action index -> unordered dice. Mid-game nodes use all 36 entries; the
# opening node uses the first 30 (its doubles are excluded, and index parity picks
# the starter: even = player 0, odd = player 1). Verified against
# `action_to_string` in tests/test_luck.py.
DICE_BY_CHANCE_ACTION: list[tuple[int, int]] = (
    [_NON_DOUBLES[k // 2] for k in range(2 * len(_NON_DOUBLES))]
    + [(i, i) for i in range(1, 7)]
)


def roll_value_table(
    board: list[list[int]], ply: int = 0,
) -> dict[tuple[int, int], float]:
    """The 21-roll sweep: ``{(lo, hi): mover-POV best-play equity}``.

    ``board`` is in :func:`raccoon.eval.gnubg_adapter.board_from_view` layout. This is
    the whole cost of variance reduction — ~1.7 ms per pre-roll state.
    """
    return {key: best_move_equity(board, key[0], key[1], ply) for key in ROLL_KEYS}


def python_roll_value(state: GameState, chance_action: int, ply: int = 0) -> float:
    """Reference route for the native sweep, over OpenSpiel's own move generation.

    Applies ``chance_action`` to a clone of the chance node ``state`` and returns the
    best-play equity from the mover's POV, with **exact** terminal values — unlike
    ``candidate_equities``, which reports ``+3.0`` for every terminal child. Mirrors that
    function otherwise, including the recursion through the second half of a doubles
    turn. ~150x slower than the native route; the test oracle, not used in play.
    """
    child = state.clone()
    child.apply_action(chance_action)
    return _best_play_value(child, child.current_player(), ply)


def _best_play_value(state: GameState, mover: int, ply: int) -> float:
    """Best equity ``mover`` can reach from decision state ``state``, from their POV."""
    from raccoon.search.mcts import _advance_through_chance

    best = -np.inf
    for action in state.legal_actions():
        child = state.clone()
        child.apply_action(action)
        if child.is_terminal():
            value = child.returns()[mover]
        elif child.current_player() == mover:
            # Doubles: the same player still owes a second pair of half-moves.
            value = _best_play_value(child, mover, ply)
        else:
            # Opponent to roll. Advancing through the chance node only sets dice, which
            # are not part of a gnubg board, but it is needed for board_from_perspective.
            child = _advance_through_chance(child)
            value = -candidate_board_equity(child, ply)
        best = max(best, value)
    return float(best)


def candidate_board_equity(state: GameState, ply: int = 0) -> float:
    """GNUBG's equity of decision state ``state`` from the side-to-move's POV."""
    from raccoon.eval.gnubg_adapter import evaluate_equity

    return evaluate_equity(board_from_view(state.board_from_perspective()), ply)


def _pre_roll_board(state: GameState, chance_action: int) -> list[list[int]]:
    """The mover's gnubg board at chance node ``state``, for outcome ``chance_action``.

    Applying a chance action only sets the dice (and, at the opening node, the
    starter); the board itself is untouched, and dice are not part of a gnubg board.
    """
    probe = state.clone()
    probe.apply_action(chance_action)
    return board_from_view(probe.board_from_perspective())


@lru_cache(maxsize=1)
def _opening_values(ply: int) -> np.ndarray:
    """Player-0-POV values of the 30 opening outcomes — identical in every game.

    The opening chance node encodes the *starter* as well as the roll (even index =
    player 0 starts), so the sign alternates with the index. Computed once per process.
    """
    from raccoon.env.game_wrapper import GameWrapper

    state = GameWrapper().new_game()
    table_by_starter = {
        starter: roll_value_table(_pre_roll_board(state, starter), ply)
        for starter in (0, 1)
    }
    return np.array(
        [
            (1.0 if k % 2 == 0 else -1.0) * table_by_starter[k % 2][DICE_BY_CHANCE_ACTION[k]]
            for k in range(OPENING_OUTCOMES)
        ],
        dtype=np.float64,
    )


def pre_roll_values(
    state: GameState, perspective_player: int, ply: int = 0,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """``(actions, probs, values)`` for every outcome of the chance node ``state``.

    ``values[i]`` is ``h(s, d_i)`` — GNUBG's best-play equity after outcome
    ``actions[i]``, signed to ``perspective_player``'s POV on the ±3 scale. The
    probability-weighted mean of ``values`` is the pre-roll mean ``m(s)``.

    Raises ``ValueError`` if ``state`` is not a chance node.
    """
    if not state.is_chance_node():
        raise ValueError("pre_roll_values called on a non-chance state")

    outcomes = state.chance_outcomes()
    actions = [a for a, _ in outcomes]
    probs = np.array([p for _, p in outcomes], dtype=np.float64)

    if len(outcomes) == OPENING_OUTCOMES:
        values = _opening_values(ply).copy()
        if perspective_player != 0:
            values = -values
        return actions, probs, values

    # Mid-game: every outcome shares one board and one mover, so one sweep serves all.
    probe = state.clone()
    probe.apply_action(actions[0])
    mover = probe.current_player()
    table = roll_value_table(board_from_view(probe.board_from_perspective()), ply)
    sign = 1.0 if mover == perspective_player else -1.0
    values = np.array(
        [sign * table[DICE_BY_CHANCE_ACTION[a]] for a in actions], dtype=np.float64
    )
    return actions, probs, values


def pre_roll_luck(
    state: GameState, chance_action: int, perspective_player: int, ply: int = 0,
) -> tuple[float, float]:
    """``(luck, mean)`` for ``chance_action`` at chance node ``state``.

    ``luck = h(s, d) - m(s)`` from ``perspective_player``'s POV. Averaged over the
    node's own dice distribution this is exactly zero — that is the property the whole
    estimator rests on, and ``tests/test_luck.py`` asserts it directly.
    """
    actions, probs, values = pre_roll_values(state, perspective_player, ply)
    mean = float(probs @ values)
    return float(values[actions.index(chance_action)]) - mean, mean


# --- the cubeful control variate ---------------------------------------------
#
# With a live cube the game result is no longer +/-1/2/3 but that times the cube
# value, so the control variate has to be on the same scale or it stops tracking
# what it is subtracted from. Two things change and neither costs a gnubg call:
#
# * every luck term is multiplied by the cube value **at the roll it belongs
#   to** (applied by the caller, which owns the cube state), and
# * ``h`` becomes a cubeful equity rather than a cubeless one, so it moves with
#   the quantity being estimated.
#
# Unbiasedness is untouched, and for exactly the reason the module docstring
# gives: ``h`` stays a deterministic function of (pre-roll state, roll), and the
# cube state is part of the pre-roll state. Cube decisions are not chance
# events, so they contribute no luck terms at all. Both variants are therefore
# valid estimators of the same quantity and the choice between them is purely
# about variance -- which is why ``scripts/eval_gnubg_cube.py`` exposes it as a
# flag and settles it on measured ``sd_ratio``.

def roll_probs_table(
    board: list[list[int]], ply: int = 0,
) -> dict[tuple[int, int], tuple[float, ...]]:
    """The 21-roll sweep again, returning outcome distributions instead of equities.

    Costs what :func:`roll_value_table` costs -- ``best_move_probs`` reads the
    same ``probabilities`` call uncollapsed.
    """
    from raccoon.eval.gnubg_adapter import best_move_probs

    return {key: best_move_probs(board, key[0], key[1], ply) for key in ROLL_KEYS}


def _cubeful_roll_table(
    board: list[list[int]], label: str, x: float, jacoby: bool, ply: int,
) -> dict[tuple[int, int], float]:
    """``{(lo, hi): mover-POV cubeful equity after best play}``, cube value 1."""
    from raccoon.cube.janowski import cl2cf_money, jacoby_active

    # `jacoby` is the raw rule setting; resolve it against where the cube sits.
    jac = jacoby_active(label, jacoby)
    return {
        key: cl2cf_money(probs, label, x, jac)
        for key, probs in roll_probs_table(board, ply).items()
    }


@lru_cache(maxsize=4)
def _opening_cubeful_values(x: float, jacoby: bool, ply: int) -> np.ndarray:
    """Player-0-POV cubeful values of the 30 opening outcomes.

    The cube is centred and worth 1 at the opening of every game, and the opening
    position is contact, so this table depends on nothing that varies within a
    session and is computed once -- the cubeful twin of :func:`_opening_values`.
    """
    from raccoon.cube.janowski import CENTERED
    from raccoon.env.game_wrapper import GameWrapper

    state = GameWrapper().new_game()
    table_by_starter = {
        starter: _cubeful_roll_table(
            _pre_roll_board(state, starter), CENTERED, x, jacoby, ply,
        )
        for starter in (0, 1)
    }
    return np.array(
        [
            (1.0 if k % 2 == 0 else -1.0)
            * table_by_starter[k % 2][DICE_BY_CHANCE_ACTION[k]]
            for k in range(OPENING_OUTCOMES)
        ],
        dtype=np.float64,
    )


def pre_roll_cubeful_values(
    state: GameState, perspective_player: int, cube, x: float,
    jacoby: bool = False, ply: int = 0,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """:func:`pre_roll_values`, with ``h`` a cubeful equity at cube value 1.

    ``cube`` is a :class:`raccoon.cube.state.CubeState`; the label is taken from
    the *mover's* point of view before the values are signed to
    ``perspective_player``, so a cube we own stays ours through the negation.
    The cube **value** is left out on purpose -- the caller multiplies the luck
    term by it, keeping this function a pure function of the board.

    Raises ``ValueError`` if ``state`` is not a chance node.
    """
    if not state.is_chance_node():
        raise ValueError("pre_roll_cubeful_values called on a non-chance state")

    outcomes = state.chance_outcomes()
    actions = [a for a, _ in outcomes]
    probs = np.array([p for _, p in outcomes], dtype=np.float64)

    if len(outcomes) == OPENING_OUTCOMES:
        values = _opening_cubeful_values(x, jacoby, ply).copy()
        if perspective_player != 0:
            values = -values
        return actions, probs, values

    probe = state.clone()
    probe.apply_action(actions[0])
    mover = probe.current_player()
    table = _cubeful_roll_table(
        board_from_view(probe.board_from_perspective()),
        cube.label_for(mover), x, jacoby, ply,
    )
    sign = 1.0 if mover == perspective_player else -1.0
    values = np.array(
        [sign * table[DICE_BY_CHANCE_ACTION[a]] for a in actions], dtype=np.float64
    )
    return actions, probs, values
