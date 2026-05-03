"""GNUBG-style terminal rendering for Raccoon backgammon.

The board is drawn with X at the bottom and O at the top, using the standard
GNUBG CLI layout:

     +13-14-15-16-17-18------19-20-21-22-23-24-+
     | ...              |   | ...              |  (top half, depth 0..4)
      |                  |BAR|                  |  (middle row)
     | ...              |   | ...              |  (bottom half)
     +12-11-10--9--8--7-------6--5--4--3--2--1-+

Both player boards are indexed by OpenSpiel with `index = 24 - point`, so
`GameState.board(player, idx)` with `idx = 24 - point` gives the checker
count on that standard point.
"""

from __future__ import annotations

from raccoon.env.game_wrapper import GameState
from raccoon.search.mcts import Analysis

# --- Board layout constants ------------------------------------------------

_TOP_LEFT_POINTS = [13, 14, 15, 16, 17, 18]
_TOP_RIGHT_POINTS = [19, 20, 21, 22, 23, 24]
_BOTTOM_LEFT_POINTS = [12, 11, 10, 9, 8, 7]
_BOTTOM_RIGHT_POINTS = [6, 5, 4, 3, 2, 1]

_STACK_DEPTH = 5  # visible rows per half; overflow shown as a number


# --- Public helpers --------------------------------------------------------


def compute_pips(state: GameState) -> tuple[int, int]:
    """Return (pips_x, pips_o) for the given state.

    X on point P owes P pips to bear off; O on X-point P owes `25 - P`
    pips. Bar checkers count as 25 pips. Borne-off checkers contribute 0.
    """
    pips_x = 0
    pips_o = 0
    for point in range(1, 25):
        idx = 24 - point
        pips_x += state.board(0, idx) * point
        pips_o += state.board(1, idx) * (25 - point)
    bar_x, bar_o, _, _ = state.parse_bar_and_off(0)
    pips_x += bar_x * 25
    pips_o += bar_o * 25
    return pips_x, pips_o


def render_board(state: GameState, human_player: int = 0) -> str:
    """Render the board in GNUBG-style ASCII.

    `human_player` only controls the right-side labels ("You" vs "Raccoon")
    and the turn indicator; the board itself is always drawn from X's
    viewpoint (X at bottom), matching GNUBG's default orientation.
    """
    lines: list[str] = []

    # Right-side info lines (one per board row). Index 0..11:
    #   0: top border
    #   1-5: top half (depth 0..4)
    #   6: middle (BAR) row
    #   7-11: bottom half (depth 4..0)
    #   12: bottom border
    bar_x, bar_o, off_x, off_o = state.parse_bar_and_off(0)
    pips_x, pips_o = compute_pips(state)
    dice = state.parse_dice()
    current = state.current_player()

    x_label = "You" if human_player == 0 else "Raccoon"
    o_label = "Raccoon" if human_player == 0 else "You"
    # Perspective arrow in the middle row: `v` means O to move, `^` means
    # X to move, space when it's a chance node or terminal.
    if current == 0:
        turn_arrow = "^"
    elif current == 1:
        turn_arrow = "v"
    else:
        turn_arrow = " "

    info: list[str] = [""] * 13
    info[0] = f"     O: {o_label}"
    info[1] = f"     Off: {off_o}"
    if bar_o:
        info[2] = f"     Bar: {bar_o}"
    if dice is not None and current == 1:
        info[3] = f"     Rolled: {dice[0]}-{dice[1]}"
    if dice is not None and current == 0:
        info[9] = f"     Rolled: {dice[0]}-{dice[1]}"
    if bar_x:
        info[10] = f"     Bar: {bar_x}"
    info[11] = f"     Off: {off_x}"
    info[12] = f"     X: {x_label}"

    # Top border with point numbers
    lines.append(
        " +" + _number_strip(_TOP_LEFT_POINTS) + "------"
        + _number_strip(_TOP_RIGHT_POINTS) + "-+" + info[0]
    )

    # Top half: depth 0..4 from outside (top) to inside (middle)
    for depth in range(_STACK_DEPTH):
        left = "".join(_cell(state, p, depth) for p in _TOP_LEFT_POINTS)
        right = "".join(_cell(state, p, depth) for p in _TOP_RIGHT_POINTS)
        lines.append(f" |{left}|   |{right}|{info[1 + depth]}")

    # Middle BAR row (with perspective arrow indicating side-to-move)
    lines.append(
        f"{turn_arrow}|                  |BAR|                  |{info[6]}"
    )

    # Bottom half: depth 4..0 (inside to outside)
    for d in range(_STACK_DEPTH):
        depth = _STACK_DEPTH - 1 - d
        left = "".join(
            _cell(state, p, depth, bottom=True) for p in _BOTTOM_LEFT_POINTS
        )
        right = "".join(
            _cell(state, p, depth, bottom=True) for p in _BOTTOM_RIGHT_POINTS
        )
        lines.append(f" |{left}|   |{right}|{info[7 + d]}")

    # Bottom border with point numbers (single-digit points are dash-prefixed,
    # producing the characteristic 7-dash visual gap between 7 and 6)
    lines.append(
        " +" + _number_strip(_BOTTOM_LEFT_POINTS) + "------"
        + _number_strip(_BOTTOM_RIGHT_POINTS) + "-+" + info[12]
    )

    lines.append(f" Pip counts: O {pips_o}, X {pips_x}")

    return "\n".join(lines)


def format_move(state: GameState, action: int) -> str:
    """Return the standard backgammon notation for a single action."""
    return state.action_to_string(action)


def format_legal_moves(state: GameState) -> str:
    """Return a numbered listing of all legal moves."""
    legal = state.legal_actions()
    if not legal:
        return "No legal moves."
    lines = ["Legal moves:"]
    for i, a in enumerate(legal):
        lines.append(f"  [{i:2d}] {state.action_to_string(a)}")
    return "\n".join(lines)


def format_analysis(
    state: GameState,
    analysis: Analysis,
    top_n: int = 5,
) -> str:
    """Render an MCTS Analysis as a GNUBG-style candidate-move listing.

    Each row shows: rank, notation, equity (Q from the side-to-move), diff
    to best, visit count, and network prior. Best row is prefixed with `*`.
    """
    if not analysis.candidates:
        return "(no candidates)"

    dice = state.parse_dice()
    header_bits = [f"Analysis ({analysis.num_simulations} sims"]
    header_bits.append(f"root value {analysis.root_value:+.3f})")
    header = ", ".join(header_bits[:1]) + ", " + header_bits[1]
    if dice is not None:
        header = f"Rolled {dice[0]}-{dice[1]} — " + header

    lines = [header]

    best_q = analysis.candidates[0].q_value
    # Find widest notation for alignment
    notations = [
        state.action_to_string(c.action) for c in analysis.candidates[:top_n]
    ]
    width = max((len(n) for n in notations), default=0)

    for rank, (cand, notation) in enumerate(
        zip(analysis.candidates[:top_n], notations), start=1
    ):
        marker = "*" if rank == 1 else " "
        if rank == 1:
            diff_str = "        "
        else:
            diff = cand.q_value - best_q
            diff_str = f"({diff:+.3f})"
        lines.append(
            f"  {marker}{rank}. {notation:<{width}}  "
            f"Eq.: {cand.q_value:+.3f} {diff_str}  "
            f"N={cand.visits:<4d} P={cand.prior:.3f}"
        )
    return "\n".join(lines)


def format_result(state: GameState, human_player: int = 0) -> str:
    """Render the terminal result line."""
    assert state.is_terminal()
    equity, result_type = state.terminal_result()
    # equity is from player 0's perspective: +N means X wins, -N means O wins
    x_won = equity > 0
    winner_is_human = (x_won and human_player == 0) or (
        not x_won and human_player == 1
    )
    winner_label = "You win" if winner_is_human else "Raccoon wins"
    multiplier = abs(int(equity))
    return f"{winner_label} — {result_type} (×{multiplier})"


# --- Internal helpers ------------------------------------------------------


def _number_strip(points: list[int]) -> str:
    """Return the `13-14-...` number strip for a row of 6 points.

    Single-digit points are dash-prefixed (e.g. `-9` instead of ` 9`) so
    the strip blends into the surrounding border, matching GNUBG output.
    """
    return "-".join(f"{p:->2d}" for p in points)


def _cell(
    state: GameState, point: int, depth: int, *, bottom: bool = False
) -> str:
    """Render a single 3-char cell for the given point at the given depth.

    `depth` is distance from the outside of the board (0 = outermost row).
    Overflow (>5 checkers) is shown as a 2-digit number in the innermost
    row (depth 4) for the top, or in the innermost row for the bottom.
    """
    idx = 24 - point
    n_x = state.board(0, idx)
    n_o = state.board(1, idx)
    if n_x == 0 and n_o == 0:
        return "   "
    if n_x > 0:
        count, symbol = n_x, "X"
    else:
        count, symbol = n_o, "O"

    if count <= _STACK_DEPTH:
        return f" {symbol} " if depth < count else "   "

    # Overflow: show (_STACK_DEPTH - 1) symbols, then the count at the last row
    overflow_row = _STACK_DEPTH - 1
    if depth < overflow_row:
        return f" {symbol} "
    if depth == overflow_row:
        return f"{count:2d} " if count < 100 else f"{count:3d}"
    return "   "
