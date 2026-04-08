"""Smoke test for the GNUBG harness. Skipped if gnubg-nn isn't installed."""

import pytest

pytest.importorskip("gnubg_nn")

from raccoon.eval.gnubg_harness import BenchmarkResult, GnubgHarness
from raccoon.model.network import RaccoonNet


def test_harness_plays_one_game_without_crashing():
    net = RaccoonNet(num_blocks=2, channels=32)
    net.eval()

    harness = GnubgHarness(
        raccoon_network=net,
        gnubg_level="beginner",
        num_simulations=5,
        log_games=True,
        raccoon_version="test",
    )
    result = harness.play_match(num_games=1)

    assert isinstance(result, BenchmarkResult)
    assert result.num_games == 1
    # Exactly one side must win a terminated game
    assert (result.raccoon_wins + result.gnubg_wins) == 1
    # Equity is consistent with the winner
    if result.raccoon_wins == 1:
        assert result.raccoon_equity > 0
    else:
        assert result.raccoon_equity < 0
    assert abs(result.raccoon_equity + result.gnubg_equity) < 1e-9
    # One game recorded
    assert len(result.games) == 1
    assert len(result.games[0].moves) > 0
    assert result.games[0].result_type in {"normal", "gammon", "backgammon"}
