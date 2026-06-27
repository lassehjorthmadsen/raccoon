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


def evaluate_equity(board: list[list[int]], ply: int = 0) -> float:
    """Return cubeless equity of ``board`` from the side-to-move's POV.

    ``gnubg_nn.probabilities`` returns ``(win, win_gammon, win_backgammon,
    lose_gammon, lose_backgammon)``. Equity follows the standard formula:

        eq = win + wg + wbg - (1-win) - lg - lbg
    """
    win, wg, wbg, lg, lbg = _gnubg.probabilities(board, ply)
    return win + wg + wbg - (1.0 - win) - lg - lbg


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
