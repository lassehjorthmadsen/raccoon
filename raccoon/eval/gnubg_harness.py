"""Automated money game matches against GNUBG CLI.

This module will be implemented when GNUBG integration is ready (M6).
The approach will be determined during implementation: either GNUBG's
--tty CLI mode or its external player socket interface.
"""

from dataclasses import dataclass
import math


@dataclass
class BenchmarkResult:
    raccoon_wins: int
    gnubg_wins: int
    raccoon_equity: float
    gnubg_equity: float
    num_games: int

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
            f"  Equity/game: {self.equity_per_game:+.3f}"
        )
