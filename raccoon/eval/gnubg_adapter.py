"""Adapter between Raccoon/OpenSpiel and the `gnubg-nn` Python package.

All GNUBG FFI lives here so the rest of the codebase stays ignorant of the
native extension. Install the optional dependency with:

    pip install gnubg-nn

Board format used by gnubg-nn is ``[my_25, opp_25]`` — two lists of 25 ints.
Indices 0..23 are points 1..24 from that player's own perspective; index 24
is the bar. Checkers that have been borne off are implicit (missing from the
board — gnubg infers them from the sum).
"""

from __future__ import annotations

import numpy as np

from raccoon.env.game_wrapper import BoardView, GameState
from raccoon.search.mcts import _advance_through_chance

try:
    import gnubg_nn as _gnubg
except ImportError as e:  # pragma: no cover - import-time guard
    raise ImportError(
        "gnubg-nn is not installed. Install it with `pip install gnubg-nn` "
        "(or `pip install -e '.[gnubg]'` from the project root)."
    ) from e


def board_from_view(view: BoardView) -> list[list[int]]:
    """Translate a Raccoon ``BoardView`` into a gnubg-nn board.

    ``view`` is from the *current player's* perspective. The returned list is
    ``[opponent_25, current_player_25]`` because gnubg-nn's ``probabilities``
    treats **slot 1** as the side on roll — not slot 0 as the naive reading
    of "my, opp" would suggest. Both 25-element lists are stored from each
    player's own POV: indices 0..23 are points 1..24 (1 = innermost home,
    24 = back point), and index 24 is the bar.

    The current player's tuple is a direct copy of ``view.my_points``. The
    opponent's tuple needs to be mirrored: opponent's point (i+1) from their
    POV is the current player's point (24-i), i.e. ``view.opp_points[23-i]``.
    """
    me_25 = [int(view.my_points[i]) for i in range(24)]
    me_25.append(int(view.my_bar))

    opp_25 = [int(view.opp_points[23 - i]) for i in range(24)]
    opp_25.append(int(view.opp_bar))

    # Slot 1 is on-roll → current player goes there.
    return [opp_25, me_25]


def board_to_view(board: list[list[int]]) -> BoardView:
    """Inverse of :func:`board_from_view`: gnubg-nn board → ``BoardView``.

    ``board`` is ``[opponent_25, current_player_25]`` with **slot 1 on roll**,
    each 25-list holding that player's own points 1..24 then their bar. The
    returned view is pre-roll (``dice=None, mid_doubles=False``) from the
    side-to-move's perspective, which is what the value head was trained on.

    The opponent's 24 points are reversed back into the current player's
    numbering (their point ``i+1`` is our point ``24-i``) — the same index
    reversal :func:`board_from_view` applies in the other direction, and the
    one that scores R^2 = -0.06 instead of 0.9964 when omitted. Borne-off
    counts are implicit in the gnubg format and recovered as ``15 - points -
    bar``.
    """
    opp_25, me_25 = board

    my_points = np.array(me_25[:24], dtype=np.float32)
    opp_points = np.array(opp_25[:24][::-1], dtype=np.float32)
    my_bar = int(me_25[24])
    opp_bar = int(opp_25[24])
    my_off = 15 - int(my_points.sum()) - my_bar
    opp_off = 15 - int(opp_points.sum()) - opp_bar

    if my_off < 0 or opp_off < 0:
        raise ValueError(
            f"board has more than 15 checkers per side "
            f"(my_off={my_off}, opp_off={opp_off})"
        )

    return BoardView(
        my_points=my_points,
        opp_points=opp_points,
        my_bar=my_bar,
        opp_bar=opp_bar,
        my_off=my_off,
        opp_off=opp_off,
        dice=None,
        mid_doubles=False,
    )


def evaluate_equity(board: list[list[int]], ply: int = 0) -> float:
    """Return cubeless equity of ``board`` from the side-to-move's POV.

    ``gnubg_nn.probabilities`` returns ``(win, win_gammon, win_backgammon,
    lose_gammon, lose_backgammon)``. Equity follows the standard formula:

        eq = win + wg + wbg - (1-win) - lg - lbg
    """
    win, wg, wbg, lg, lbg = _gnubg.probabilities(board, ply)
    return win + wg + wbg - (1.0 - win) - lg - lbg


def outcome_probs(
    board: list[list[int]], ply: int = 0,
) -> tuple[float, float, float, float, float]:
    """GNUBG's cumulative outcome probabilities ``(win, wg, wbg, lg, lbg)``.

    From the side-to-move's POV, at ``ply`` lookahead. Nested: win >= wg >= wbg
    and (1 - win) >= lg >= lbg; P(lose) = 1 - win is the redundant sixth. Used to
    build both distillation targets (scalar equity and the six-outcome
    distribution) from one call — see scripts/gen_gnubg_selfplay.py.
    """
    win, wg, wbg, lg, lbg = _gnubg.probabilities(board, ply)
    return float(win), float(wg), float(wbg), float(lg), float(lbg)


def terminal_equity_after_move(board: list[list[int]]) -> float | None:
    """Win magnitude for slot 0 if ``board`` is a finished game, else ``None``.

    ``board`` is a post-move :func:`board_from_view` layout: slot 0 is the player who
    just moved, slot 1 is the player who would be on roll. Checkers borne off are
    implicit (gnubg infers them from the sum), so the mover has won iff slot 0 is empty.
    Returns ``1.0`` for a plain win, ``2.0`` for a gammon (loser borne off none) and
    ``3.0`` for a backgammon (loser also on the bar or still in the winner's home).

    The loser's points are stored from their own POV, and their point ``24 - j`` is the
    winner's point ``j + 1``, so the winner's home board (points 1-6) is the loser's
    indices 18-23.
    """
    if sum(board[0]) > 0:
        return None
    loser = board[1]
    if sum(loser) < 15:
        return 1.0
    return 3.0 if (loser[24] > 0 or any(loser[18:24])) else 2.0


def best_move_equity(
    board: list[list[int]], die1: int, die2: int, ply: int = 0,
) -> float:
    """Equity of ``board`` after GNUBG plays ``(die1, die2)`` best, from the mover's POV.

    ``board`` is in :func:`board_from_view` layout (slot 1 on roll). The result is on
    GNUBG's native *points* scale, with **exact** terminal values — a plain win is
    ``+1.0``, a gammon ``+2.0``, a backgammon ``+3.0``. That exactness is what makes this
    the right primitive for the dice-luck control variate in :mod:`raccoon.eval.luck`;
    :func:`candidate_equities` hard-codes ``+3.0`` for every terminal child, which is fine
    for ranking but would score every game-ending roll as a backgammon.

    ``gnubg_nn.best_move`` generates and ranks the whole roll — all four half-moves of a
    double included — inside C, making it ~150x faster than the equivalent loop over
    :func:`candidate_equities` (a 21-roll sweep costs ~1.7 ms against ~260 ms). We use it
    for **move generation only** and re-evaluate the position it lands on with
    :func:`evaluate_equity`, for two reasons:

    - The equity and probabilities in its detail tuple are **not trustworthy in race and
      bear-off classes** — observed reporting ``P(win) = 1.0`` and equity ``0.912`` for a
      position ``probabilities`` scores at ``0.982`` with the opponent still winning 5.7%.
      Re-evaluating keeps one evaluator for the whole project.
    - It keeps this function consistent with :func:`candidate_equities`, so the two agree
      wherever they both apply and the cheap route can be tested against the slow one.

    The detail tuple's position *key* round-trips exactly through
    ``board_from_position_key``, so the re-evaluation costs one extra ``probabilities``
    call per roll and no move-application logic of our own.

    Two behaviours of the native call are handled here:

    - A **dance** (no legal move) returns an empty move list. The board is then unchanged
      and the turn passes; we evaluate with the sides swapped and negate.
    - ``ply >= 1`` **segfaults** on real bear-off boards (``ply=0`` is solid over
      thousands of positions), so anything but 0 is refused rather than risking a core
      dump mid-run. Use :func:`candidate_equities` if a deeper best-play value is needed.
    """
    if ply != 0:
        raise ValueError(
            f"best_move_equity: ply must be 0, got {ply}. gnubg-nn's best_move "
            "segfaults at ply >= 1; use candidate_equities for deeper lookahead."
        )
    entries = _gnubg.best_move(board, die1, die2, ply, list=1)[1]
    if not entries:
        # Dance: no legal move, board unchanged, turn passes to the opponent.
        return -evaluate_equity([board[1], board[0]], ply)
    after = [list(side) for side in _gnubg.board_from_position_key(entries[0][0])]
    terminal = terminal_equity_after_move(after)
    return terminal if terminal is not None else -evaluate_equity(after, ply)


def candidate_equities(state: GameState, ply: int = 0) -> list[tuple[int, float]]:
    """Return ``[(action, my_equity)]`` for every legal action of the side to move.

    ``my_equity`` is GNUBG's cubeless equity from the **current player's** POV
    after playing ``action`` (and, for doubles, the best half-2 continuation),
    on GNUBG's native *points* scale where a terminal win counts as ``+3.0`` and
    ``my_equity = -evaluate_equity(opponent_position)`` otherwise. The move
    GNUBG plays is ``max(candidate_equities(...), key=lambda t: t[1])[0]`` (see
    ``pick_move``); the position's value under best play is the max ``my_equity``.

    Divide by 3 for the ``[-1, 1]`` money-equity convention used by the value
    head and the wildbg/GNUBG distillation targets (``equity_from_wildbg`` =
    ``evaluate_equity / 3``).

    Doubles handling: OpenSpiel splits a doubles roll into two consecutive
    half-turns by the same player with no intervening chance node. After
    applying a half-1 action we are still the current player, so we recurse to
    score that half-1 action by its best half-2 continuation.

    Raises ``ValueError`` if ``state`` has no legal actions.
    """
    legal = state.legal_actions()
    if not legal:
        raise ValueError("candidate_equities called on a state with no legal actions")

    me = state.current_player()
    out: list[tuple[int, float]] = []
    for action in legal:
        child = state.clone()
        child.apply_action(action)
        child = _advance_through_chance(child)

        if child.is_terminal():
            # The game ended with our move — we won.
            my_equity = 3.0
        elif child.current_player() == me:
            # Doubles half-1 still pending a half-2 by the same player. Our
            # equity for this half-1 action is the best over the half-2 subgame.
            my_equity = max(eq for _, eq in candidate_equities(child, ply))
        else:
            opp_board = board_from_view(child.board_from_perspective())
            my_equity = -evaluate_equity(opp_board, ply)

        out.append((action, my_equity))
    return out


def _best_action_and_opp_equity(
    state: GameState, ply: int
) -> tuple[int, float]:
    """Back-compat shim: ``(best_action, resulting_opp_equity)`` for the side to move.

    Kept for any external callers; thin wrapper over :func:`candidate_equities`.
    """
    best_action, best_my_equity = max(
        candidate_equities(state, ply), key=lambda t: t[1]
    )
    return best_action, -best_my_equity


def pick_move(state: GameState, ply: int = 0) -> int:
    """Pick a move for the current player using GNUBG's evaluation.

    Enumerates OpenSpiel legal actions, evaluates each resulting position with
    GNUBG, and returns the action that maximises our equity (= minimises the
    opponent's post-move equity). For doubles this looks ahead through both
    halves of the turn so the half-1 choice is scored by the best half-2
    continuation. See :func:`candidate_equities`.

    Raises ``ValueError`` if ``state`` has no legal actions.
    """
    return max(candidate_equities(state, ply), key=lambda t: t[1])[0]


# Friendly level-name → ply-integer mapping matching the M6 spec.
LEVEL_TO_PLY: dict[str, int] = {
    "beginner": 0,
    "casual": 0,
    "intermediate": 0,
    "advanced": 1,
    "expert": 1,
    "world": 2,
    "world class": 2,
    "supremo": 2,
    "grandmaster": 2,
}


def level_to_ply(level: str) -> int:
    """Map a friendly GNUBG strength level string to a ply integer."""
    key = level.strip().lower()
    if key not in LEVEL_TO_PLY:
        raise ValueError(
            f"Unknown gnubg level '{level}'. "
            f"Valid options: {sorted(LEVEL_TO_PLY)}"
        )
    return LEVEL_TO_PLY[key]
