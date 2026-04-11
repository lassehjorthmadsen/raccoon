"""OpenSpiel backgammon wrapper with perspective-relative board views.

OpenSpiel's Python bindings for backgammon do not expose direct accessors
for bar counts, borne-off counts, or the current dice roll — only for board
points. This wrapper therefore parses those fields out of ``str(state)``,
which is a known-fragile pattern (OpenSpiel makes no stability guarantee on
its ``__str__`` format). The parsing is isolated to ``parse_bar_and_off``
and ``parse_dice``; if OpenSpiel ever changes formatting, those two
functions are the only places that need updating.
"""

from dataclasses import dataclass
import re

import numpy as np
import pyspiel


@dataclass
class BoardView:
    """Board state from the current player's perspective.

    All arrays use perspective-relative indexing where point 1 (index 0)
    is closest to the current player's bearoff.
    """
    my_points: np.ndarray      # shape (24,), checker counts on points 1-24
    opp_points: np.ndarray     # shape (24,)
    my_bar: int
    opp_bar: int
    my_off: int
    opp_off: int
    dice: tuple[int, int] | None
    mid_doubles: bool = False  # True when this is the 2nd half of a doubles turn


class GameState:
    """Wrapper around an OpenSpiel BackgammonState."""

    def __init__(self, state: pyspiel.BackgammonState):
        self._state = state
        self._prev_decision_player: int | None = None

    def current_player(self) -> int:
        return self._state.current_player()

    def legal_actions(self) -> list[int]:
        return self._state.legal_actions()

    def is_terminal(self) -> bool:
        return self._state.is_terminal()

    def is_chance_node(self) -> bool:
        return self._state.is_chance_node()

    def chance_outcomes(self) -> list[tuple[int, float]]:
        return self._state.chance_outcomes()

    def apply_action(self, action: int) -> None:
        if not self.is_chance_node():
            self._prev_decision_player = self.current_player()
        self._state.apply_action(action)

    def returns(self) -> list[float]:
        return self._state.returns()

    def clone(self) -> "GameState":
        cloned = GameState(self._state.clone())
        cloned._prev_decision_player = self._prev_decision_player
        return cloned

    def action_to_string(self, action: int) -> str:
        player = self.current_player()
        return self._state.action_to_string(player, action)

    def board(self, player: int, index: int) -> int:
        """Return the checker count for `player` at board index `index`.

        OpenSpiel indexes both players' boards with index `i` corresponding
        to standard point `24 - i` (i.e. index 0 is point 24, index 23 is
        point 1). See `raccoon/cli/display.py` for the point-number layout.
        """
        return int(self._state.board(player, index))

    def board_from_perspective(self) -> BoardView:
        """Return the board from the current player's perspective."""
        cp = self.current_player()
        op = 1 - cp
        state = self._state

        my_points = np.zeros(24, dtype=np.float32)
        opp_points = np.zeros(24, dtype=np.float32)

        for i in range(24):
            # Player 0: perspective index i -> board index (23 - i)
            # Player 1: perspective index i -> board index i
            board_idx = (23 - i) if cp == 0 else i
            my_points[i] = state.board(cp, board_idx)
            opp_points[i] = state.board(op, board_idx)

        my_bar, opp_bar, my_off, opp_off = self.parse_bar_and_off(cp)
        dice = self.parse_dice()

        mid_doubles = (
            self._prev_decision_player is not None
            and self._prev_decision_player == cp
            and dice is not None
            and dice[0] == dice[1]
        )

        return BoardView(
            my_points=my_points,
            opp_points=opp_points,
            my_bar=my_bar,
            opp_bar=opp_bar,
            my_off=my_off,
            opp_off=opp_off,
            dice=dice,
            mid_doubles=mid_doubles,
        )

    def parse_bar_and_off(self, current_player: int) -> tuple[int, int, int, int]:
        """Parse bar and borne-off counts from the state string."""
        s = str(self._state)
        # Bar line: "Bar: xxoo" (letters indicate which player's checkers)
        bar_line = ""
        scores_line = ""
        for line in s.split("\n"):
            if line.startswith("Bar:"):
                bar_line = line[4:].strip()
            elif line.startswith("Scores"):
                scores_line = line

        # Player 0 = X, Player 1 = O
        bar_x = bar_line.count("x")
        bar_o = bar_line.count("o")

        m = re.search(r"X: (\d+), O: (\d+)", scores_line)
        off_x = int(m.group(1))
        off_o = int(m.group(2))

        if current_player == 0:
            return bar_x, bar_o, off_x, off_o
        else:
            return bar_o, bar_x, off_o, off_x

    def parse_dice(self) -> tuple[int, int] | None:
        s = str(self._state)
        for line in s.split("\n"):
            if line.startswith("Dice:"):
                dice_str = line[5:].strip()
                if not dice_str:
                    return None
                d1, d2 = int(dice_str[0]), int(dice_str[1])
                if not (1 <= d1 <= 6 and 1 <= d2 <= 6):
                    raise ValueError(
                        f"parse_dice: got invalid dice {(d1, d2)} from "
                        f"OpenSpiel state line {line!r}"
                    )
                return (d1, d2)
        return None

    def terminal_result(self) -> tuple[float, str]:
        """Return (equity, result_type) for a terminal state.

        equity is from player 0's perspective: +1/+2/+3 or -1/-2/-3.
        result_type is 'normal', 'gammon', or 'backgammon'.

        Relies on OpenSpiel being loaded with ``scoring_type=full_scoring``,
        which makes ``returns()`` expose the ±1/±2/±3 multipliers natively.
        """
        if not self.is_terminal():
            raise ValueError("terminal_result() called on non-terminal state")
        equity = self._state.returns()[0]
        magnitude = int(round(abs(equity)))
        result_type = {1: "normal", 2: "gammon", 3: "backgammon"}[magnitude]
        return float(equity), result_type


class GameWrapper:
    """Factory for creating new backgammon games.

    Uses OpenSpiel's ``full_scoring`` mode so ``state.returns()`` produces
    ±1/±2/±3 for normal/gammon/backgammon wins (instead of the default
    ``winloss_scoring`` which flattens everything to ±1). This matches the
    money-game semantics Raccoon is training for.
    """

    def __init__(self):
        self._game = pyspiel.load_game("backgammon(scoring_type=full_scoring)")

    def new_game(self) -> GameState:
        return GameState(self._game.new_initial_state())

    @property
    def game(self) -> pyspiel.Game:
        return self._game
