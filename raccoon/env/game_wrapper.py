"""OpenSpiel backgammon wrapper with perspective-relative board views."""

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


class GameState:
    """Wrapper around an OpenSpiel BackgammonState."""

    def __init__(self, state: pyspiel.BackgammonState):
        self._state = state

    def current_player(self) -> int:
        return self._state.current_player()

    def legal_actions(self) -> list[int]:
        return self._state.legal_actions()

    def legal_actions_mask(self) -> list[int]:
        return self._state.legal_actions_mask()

    def is_terminal(self) -> bool:
        return self._state.is_terminal()

    def is_chance_node(self) -> bool:
        return self._state.is_chance_node()

    def chance_outcomes(self) -> list[tuple[int, float]]:
        return self._state.chance_outcomes()

    def apply_action(self, action: int) -> None:
        self._state.apply_action(action)

    def returns(self) -> list[float]:
        return self._state.returns()

    def clone(self) -> "GameState":
        return GameState(self._state.clone())

    def action_to_string(self, action: int) -> str:
        player = self.current_player()
        return self._state.action_to_string(player, action)

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

        my_bar, opp_bar, my_off, opp_off = self._parse_bar_and_off(cp)
        dice = self._parse_dice()

        return BoardView(
            my_points=my_points,
            opp_points=opp_points,
            my_bar=my_bar,
            opp_bar=opp_bar,
            my_off=my_off,
            opp_off=opp_off,
            dice=dice,
        )

    def _parse_bar_and_off(self, current_player: int) -> tuple[int, int, int, int]:
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

    def _parse_dice(self) -> tuple[int, int] | None:
        s = str(self._state)
        for line in s.split("\n"):
            if line.startswith("Dice:"):
                dice_str = line[5:].strip()
                if not dice_str:
                    return None
                return (int(dice_str[0]), int(dice_str[1]))
        return None

    def terminal_result(self) -> tuple[float, str]:
        """Return (equity, result_type) for a terminal state.

        equity is from player 0's perspective: +1/+2/+3 or -1/-2/-3.
        result_type is 'normal', 'gammon', or 'backgammon'.
        """
        assert self.is_terminal()
        state = self._state
        s = str(state)

        m = re.search(r"X: (\d+), O: (\d+)", s)
        off_x, off_o = int(m.group(1)), int(m.group(2))

        bar_line = ""
        for line in s.split("\n"):
            if line.startswith("Bar:"):
                bar_line = line[4:].strip()

        # Determine winner and loser's state
        if off_x == 15:
            winner = 0
            loser_off = off_o
            loser_bar = bar_line.count("o")
            loser_board = [state.board(1, i) for i in range(24)]
        else:
            winner = 1
            loser_off = off_x
            loser_bar = bar_line.count("x")
            loser_board = [state.board(0, i) for i in range(24)]

        if loser_off > 0:
            result_type = "normal"
            multiplier = 1
        elif loser_bar > 0 or self._has_checkers_in_home(loser_board, winner):
            result_type = "backgammon"
            multiplier = 3
        else:
            result_type = "gammon"
            multiplier = 2

        equity = multiplier if winner == 0 else -multiplier
        return equity, result_type

    @staticmethod
    def _has_checkers_in_home(loser_board: list[int], winner: int) -> bool:
        """Check if loser has checkers in the winner's home board."""
        # Winner's home board indices:
        # Player 0 home = indices 18-23 (standard points 1-6)
        # Player 1 home = indices 0-5 (standard points 24-19)
        if winner == 0:
            return any(loser_board[i] > 0 for i in range(18, 24))
        else:
            return any(loser_board[i] > 0 for i in range(0, 6))


class GameWrapper:
    """Factory for creating new backgammon games."""

    def __init__(self):
        self._game = pyspiel.load_game("backgammon")

    def new_game(self) -> GameState:
        return GameState(self._game.new_initial_state())

    @property
    def game(self) -> pyspiel.Game:
        return self._game
