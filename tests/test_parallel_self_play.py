"""Tests for parallel self-play."""

import pytest

from raccoon.model.network import RaccoonNet
from raccoon.train.parallel_self_play import parallel_self_play


@pytest.fixture
def network():
    return RaccoonNet()


def test_parallel_produces_valid_results(network):
    results = parallel_self_play(
        network, num_games=4, num_simulations=5,
        num_workers=2, batch_size=4,
    )
    assert len(results) == 4
    for r in results:
        assert r.num_moves > 0
        assert len(r.examples) > 0
        assert r.result_type in ("normal", "gammon", "backgammon")


def test_example_shapes(network):
    results = parallel_self_play(
        network, num_games=2, num_simulations=5,
        num_workers=2, batch_size=4,
    )
    for r in results:
        for ex in r.examples:
            assert ex.observation.shape == (17, 2, 12)
            assert ex.policy_target.shape == (1352,)
            assert -1.0 <= ex.value_target <= 1.0
