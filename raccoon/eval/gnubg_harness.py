"""Automated money-game sessions against GNUBG via the `gnubg-nn` package.

This is the M6 benchmark harness. Raccoon (driven by MCTS on a RaccoonNet)
plays a cubeless money game against GNUBG's neural-net evaluator. Gammons
and backgammons are scored (2x / 3x) via OpenSpiel's terminal_result.

The GNUBG FFI is isolated in ``raccoon.eval.gnubg_adapter`` so this module
stays free of conditional imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from raccoon.env.game_wrapper import GameWrapper
from raccoon.eval.game_log import GameRecord, MoveRecord, new_game_record
from raccoon.model.network import RaccoonNet
from raccoon.search.mcts import MCTS, _advance_through_chance, select_action


@dataclass
class BenchmarkResult:
    raccoon_wins: int
    gnubg_wins: int
    raccoon_equity: float
    gnubg_equity: float
    num_games: int
    raccoon_gammons_won: int = 0
    raccoon_backgammons_won: int = 0
    gnubg_gammons_won: int = 0
    gnubg_backgammons_won: int = 0
    games: list[GameRecord] = field(default_factory=list)

    @property
    def raccoon_win_rate(self) -> float:
        return self.raccoon_wins / self.num_games if self.num_games else 0.0

    @property
    def equity_per_game(self) -> float:
        return self.raccoon_equity / self.num_games if self.num_games else 0.0

    @property
    def confidence_interval_95(self) -> float:
        """Approximate 95% CI on win rate using normal approximation."""
        if self.num_games == 0:
            return 0.0
        p = self.raccoon_win_rate
        return 1.96 * math.sqrt(p * (1 - p) / self.num_games)

    def summary(self) -> str:
        ci = self.confidence_interval_95
        return (
            f"Raccoon vs GNUBG ({self.num_games} games)\n"
            f"  Wins: {self.raccoon_wins}-{self.gnubg_wins} "
            f"({self.raccoon_win_rate:.1%} +/- {ci:.1%})\n"
            f"  Equity: Raccoon {self.raccoon_equity:+.1f}, "
            f"GNUBG {self.gnubg_equity:+.1f}\n"
            f"  Equity/game: {self.equity_per_game:+.3f}\n"
            f"  Raccoon gammons/bg: {self.raccoon_gammons_won}/"
            f"{self.raccoon_backgammons_won}, "
            f"GNUBG gammons/bg: {self.gnubg_gammons_won}/"
            f"{self.gnubg_backgammons_won}"
        )


class GnubgHarness:
    """Play a session of money games between Raccoon and GNUBG.

    ``gnubg_level`` is a friendly string (e.g. ``"world"``) mapped to a ply
    integer via ``gnubg_adapter.level_to_ply``. Pass ``ply`` explicitly to
    override (``0`` = fastest, ``2`` = strongest).
    """

    def __init__(
        self,
        raccoon_network: RaccoonNet,
        gnubg_level: str = "world",
        num_simulations: int = 200,
        ply: int | None = None,
        log_games: bool = False,
        raccoon_version: str = "unknown",
    ):
        # Import lazily so the rest of raccoon.eval.* doesn't require gnubg-nn
        from raccoon.eval import gnubg_adapter

        self._adapter = gnubg_adapter
        self.raccoon_network = raccoon_network
        self.num_simulations = num_simulations
        self.ply = ply if ply is not None else gnubg_adapter.level_to_ply(gnubg_level)
        self.gnubg_level = gnubg_level
        self.log_games = log_games
        self.raccoon_version = raccoon_version

    def play_match(self, num_games: int = 1000) -> BenchmarkResult:
        """Play ``num_games`` money games and return aggregated results."""
        wrapper = GameWrapper()
        mcts = MCTS(self.raccoon_network, num_simulations=self.num_simulations)

        raccoon_wins = 0
        gnubg_wins = 0
        raccoon_equity = 0.0
        raccoon_gammons_won = 0
        raccoon_backgammons_won = 0
        gnubg_gammons_won = 0
        gnubg_backgammons_won = 0
        game_records: list[GameRecord] = []

        for game_idx in range(num_games):
            done = game_idx + 1
            avg_eq = raccoon_equity / game_idx if game_idx else 0.0
            print(
                f"\rGame {done}/{num_games}  "
                f"Raccoon {raccoon_wins}-{gnubg_wins}  "
                f"Equity: {avg_eq:+.3f} ppg",
                end="", flush=True,
            )

            # Alternate who plays as OpenSpiel player 0
            raccoon_is_player0 = (game_idx % 2 == 0)

            state = wrapper.new_game()
            state = _advance_through_chance(state)

            record: GameRecord | None = None
            if self.log_games:
                record = new_game_record(
                    raccoon_is_player0=raccoon_is_player0,
                    raccoon_version=self.raccoon_version,
                    opponent=f"gnubg(ply={self.ply},level={self.gnubg_level})",
                )

            while not state.is_terminal():
                current = state.current_player()
                is_raccoon_turn = (current == 0) == raccoon_is_player0

                if is_raccoon_turn:
                    action_probs = mcts.search(state)
                    if not action_probs:
                        break
                    action = select_action(action_probs, temperature=0)
                else:
                    action = self._adapter.pick_move(state, ply=self.ply)

                if record is not None:
                    dice = state.board_from_perspective().dice
                    record.moves.append(
                        MoveRecord(
                            player=current,
                            dice=dice,
                            action=action,
                            action_str=state.action_to_string(action),
                        )
                    )

                state.apply_action(action)
                state = _advance_through_chance(state)

            if state.is_terminal():
                equity_p0, result_type = state.terminal_result()
                # Convert to Raccoon's perspective
                raccoon_game_equity = (
                    equity_p0 if raccoon_is_player0 else -equity_p0
                )

                raccoon_equity += raccoon_game_equity

                if raccoon_game_equity > 0:
                    raccoon_wins += 1
                    if result_type == "gammon":
                        raccoon_gammons_won += 1
                    elif result_type == "backgammon":
                        raccoon_backgammons_won += 1
                elif raccoon_game_equity < 0:
                    gnubg_wins += 1
                    if result_type == "gammon":
                        gnubg_gammons_won += 1
                    elif result_type == "backgammon":
                        gnubg_backgammons_won += 1

                if record is not None:
                    record.result = float(raccoon_game_equity)
                    record.result_type = result_type
                    game_records.append(record)

        print()  # newline after the progress line

        return BenchmarkResult(
            raccoon_wins=raccoon_wins,
            gnubg_wins=gnubg_wins,
            raccoon_equity=raccoon_equity,
            gnubg_equity=-raccoon_equity,
            num_games=num_games,
            raccoon_gammons_won=raccoon_gammons_won,
            raccoon_backgammons_won=raccoon_backgammons_won,
            gnubg_gammons_won=gnubg_gammons_won,
            gnubg_backgammons_won=gnubg_backgammons_won,
            games=game_records,
        )
