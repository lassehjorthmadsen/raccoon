"""Checkpoint vs checkpoint evaluation arena."""

from dataclasses import dataclass

from raccoon.env.encoder import encode_state
from raccoon.env.game_wrapper import GameWrapper
from raccoon.model.network import RaccoonNet
from raccoon.search.mcts import MCTS, select_action, _advance_through_chance


@dataclass
class MatchResult:
    wins_p1: int
    wins_p2: int
    p1_equity: float
    p2_equity: float
    num_games: int
    total_moves: int

    @property
    def win_rate_p1(self) -> float:
        return self.wins_p1 / self.num_games if self.num_games else 0.0

    @property
    def avg_game_length(self) -> float:
        return self.total_moves / self.num_games if self.num_games else 0.0

    def summary(self) -> str:
        return (
            f"P1 wins: {self.wins_p1}/{self.num_games} ({self.win_rate_p1:.1%}), "
            f"P1 equity: {self.p1_equity:+.1f}, "
            f"P2 equity: {self.p2_equity:+.1f}, "
            f"Avg length: {self.avg_game_length:.0f} moves"
        )


class Arena:
    """Play matches between two networks."""

    def __init__(
        self,
        player1: RaccoonNet,
        player2: RaccoonNet,
        num_games: int = 100,
        num_simulations: int = 50,
    ):
        self.player1 = player1
        self.player2 = player2
        self.num_games = num_games
        self.num_simulations = num_simulations

    def play_match(self) -> MatchResult:
        """Play num_games, alternating who goes first."""
        wrapper = GameWrapper()
        wins_p1 = 0
        wins_p2 = 0
        p1_equity = 0.0
        p2_equity = 0.0
        total_moves = 0

        mcts1 = MCTS(self.player1, num_simulations=self.num_simulations)
        mcts2 = MCTS(self.player2, num_simulations=self.num_simulations)

        for game_idx in range(self.num_games):
            # Alternate who is OpenSpiel player 0
            # Even games: p1=player0, odd games: p1=player1
            p1_is_player0 = (game_idx % 2 == 0)

            state = wrapper.new_game()
            state = _advance_through_chance(state)
            moves = 0

            while not state.is_terminal():
                current = state.current_player()

                if (current == 0) == p1_is_player0:
                    mcts = mcts1
                else:
                    mcts = mcts2

                action_probs = mcts.search(state)
                if not action_probs:
                    break
                action = select_action(action_probs, temperature=0)
                state.apply_action(action)
                state = _advance_through_chance(state)
                moves += 1

            total_moves += moves

            if state.is_terminal():
                equity, _ = state.terminal_result()
                # equity is from OpenSpiel player 0's perspective
                if p1_is_player0:
                    game_equity_p1 = equity
                else:
                    game_equity_p1 = -equity

                p1_equity += game_equity_p1
                p2_equity -= game_equity_p1

                if game_equity_p1 > 0:
                    wins_p1 += 1
                elif game_equity_p1 < 0:
                    wins_p2 += 1

        return MatchResult(
            wins_p1=wins_p1,
            wins_p2=wins_p2,
            p1_equity=p1_equity,
            p2_equity=p2_equity,
            num_games=self.num_games,
            total_moves=total_moves,
        )
