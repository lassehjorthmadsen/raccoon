"""Tests for self-play data generation."""

import numpy as np
import pytest

from raccoon.model.network import RaccoonNet
from raccoon.train.self_play import play_one_game


@pytest.fixture
def model():
    return RaccoonNet()


def test_play_one_game_completes(model):
    result = play_one_game(model, num_simulations=5)
    assert len(result.examples) > 0
    assert result.num_moves > 0


def test_example_shapes(model):
    result = play_one_game(model, num_simulations=5)
    for ex in result.examples:
        assert ex.observation.shape == (17, 2, 12)
        assert ex.policy_target.shape == (1352,)
        assert isinstance(ex.value_target, float)


def test_policy_targets_valid(model):
    result = play_one_game(model, num_simulations=5)
    for ex in result.examples:
        assert abs(ex.policy_target.sum() - 1.0) < 1e-5
        assert (ex.policy_target >= 0).all()


def test_value_targets_in_range(model):
    result = play_one_game(model, num_simulations=5)
    for ex in result.examples:
        assert -1.0 <= ex.value_target <= 1.0


def test_value_target_is_raw_return_over_three(model):
    """With alpha=1.0 (default), value targets are the terminal outcome / 3."""
    for _ in range(10):
        result = play_one_game(model, num_simulations=5, value_bootstrap_alpha=1.0)
        assert abs(result.outcome) in (1, 2, 3), (
            f"Unexpected raw outcome {result.outcome}"
        )
        expected_abs = abs(result.outcome) / 3.0
        for ex in result.examples:
            assert abs(abs(ex.value_target) - expected_abs) < 1e-6, (
                f"value_target={ex.value_target}, outcome={result.outcome}"
            )


def test_value_target_blends_with_alpha_zero(model):
    """With alpha=0.0, value targets are MCTS root Q, not the terminal outcome."""
    result = play_one_game(model, num_simulations=5, value_bootstrap_alpha=0.0)
    assert len(result.examples) > 0
    for ex in result.examples:
        assert -1.0 <= ex.value_target <= 1.0
    # Pure Q targets should differ from the pure terminal outcome for at least
    # some positions (a random network's Q is near 0, not ±1/3/2/3/1).
    outcome_value = abs(result.outcome) / 3.0
    matches = sum(
        1 for ex in result.examples if abs(abs(ex.value_target) - outcome_value) < 1e-4
    )
    assert matches < len(result.examples), (
        "Expected alpha=0.0 to produce values different from the terminal outcome"
    )


def test_game_result_fields(model):
    result = play_one_game(model, num_simulations=5)
    assert result.result_type in ("normal", "gammon", "backgammon")
    assert result.outcome in (-3, -2, -1, 1, 2, 3)
