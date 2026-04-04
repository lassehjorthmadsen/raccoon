"""Tests for MCTS search."""

import numpy as np
import pytest

from raccoon.env.game_wrapper import GameWrapper
from raccoon.model.network import RaccoonNet
from raccoon.search.mcts import MCTS, MCTSNode, select_action, _advance_through_chance


@pytest.fixture
def wrapper():
    return GameWrapper()


@pytest.fixture
def model():
    return RaccoonNet()


@pytest.fixture
def decision_state(wrapper):
    state = wrapper.new_game()
    state = _advance_through_chance(state)
    return state


def test_mcts_runs_without_crashing(model, decision_state):
    mcts = MCTS(model, num_simulations=10)
    action_probs = mcts.search(decision_state)
    assert len(action_probs) > 0


def test_mcts_returns_valid_distribution(model, decision_state):
    mcts = MCTS(model, num_simulations=20)
    action_probs = mcts.search(decision_state)

    # All actions should be legal
    legal = set(decision_state.legal_actions())
    for a in action_probs:
        assert a in legal

    # Probabilities should sum to ~1
    total = sum(action_probs.values())
    assert abs(total - 1.0) < 1e-6

    # All probabilities should be non-negative
    assert all(p >= 0 for p in action_probs.values())


def test_mcts_visit_counts(model, decision_state):
    num_sims = 30
    mcts = MCTS(model, num_simulations=num_sims)
    mcts.search(decision_state)

    # We can't directly check visit counts from the public API,
    # but the action_probs should reflect them


def test_advance_through_chance(wrapper):
    state = wrapper.new_game()
    assert state.is_chance_node()
    state = _advance_through_chance(state)
    assert not state.is_chance_node()
    assert state.current_player() in (0, 1)


def test_select_action_temperature_zero():
    probs = {0: 0.1, 1: 0.7, 2: 0.2}
    action = select_action(probs, temperature=0)
    assert action == 1  # highest probability


def test_select_action_temperature_one():
    probs = {0: 0.5, 1: 0.5}
    # With equal probs, both should be possible
    seen = set()
    for _ in range(100):
        seen.add(select_action(probs, temperature=1.0))
    assert len(seen) == 2


def test_mcts_single_simulation(model, decision_state):
    """With 1 simulation, the result should approximate the network prior."""
    mcts = MCTS(model, num_simulations=1)
    action_probs = mcts.search(decision_state)
    assert len(action_probs) > 0
    assert abs(sum(action_probs.values()) - 1.0) < 1e-6
