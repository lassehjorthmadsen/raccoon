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


def _best_action_and_opp_equity(
    state: GameState, ply: int
) -> tuple[int, float]:
    """Return ``(best_action, resulting_opp_equity)`` for the side to move.

    ``resulting_opp_equity`` is the cubeless equity from the **opponent's**
    POV after our full turn is played out, so the caller picks the action
    with the minimum value.

    Doubles handling: OpenSpiel splits a doubles roll into two consecutive
    half-turns by the same player with no intervening chance node. After
    applying a half-1 action we are still the current player, and evaluating
    the board would read back our *own* equity (not the opponent's). We
    recurse into the half-2 sub-problem to find our best continuation and
    propagate that evaluation up.
    """
    legal = state.legal_actions()
    if not legal:
        raise ValueError("pick_move called on a state with no legal actions")

    me = state.current_player()
    best_action = legal[0]
    best_opp_equity = float("inf")

    for action in legal:
        child = state.clone()
        child.apply_action(action)
        child = _advance_through_chance(child)

        if child.is_terminal():
            # The game ended with our move — we won.
            opp_equity = -3.0
        elif child.current_player() == me:
            # Doubles half-1 still pending a half-2 by the same player.
            # Recurse to find our best half-2 continuation; its opp-equity
            # is what this half-1 candidate is actually worth to us.
            _, opp_equity = _best_action_and_opp_equity(child, ply)
        else:
            opp_view = child.board_from_perspective()
            opp_board = board_from_view(opp_view)
            opp_equity = evaluate_equity(opp_board, ply)

        if opp_equity < best_opp_equity:
            best_opp_equity = opp_equity
            best_action = action

    return best_action, best_opp_equity


def pick_move(state: GameState, ply: int = 0) -> int:
    """Pick a move for the current player using GNUBG's evaluation.

    Enumerates OpenSpiel legal actions, applies each to a clone, evaluates
    the resulting position with GNUBG, and returns the action that
    minimises the opponent's post-move equity (= maximises ours). For
    doubles this looks ahead through both halves of the turn so the half-1
    choice is scored by the best half-2 continuation. See
    ``_best_action_and_opp_equity`` for details.

    Raises ``ValueError`` if ``state`` has no legal actions.
    """
    return _best_action_and_opp_equity(state, ply)[0]


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
