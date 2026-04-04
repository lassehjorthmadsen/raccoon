"""Tests for the OpenSpiel backgammon wrapper."""

import random

import numpy as np
import pytest

from raccoon.env.game_wrapper import GameWrapper, GameState


@pytest.fixture
def wrapper():
    return GameWrapper()


@pytest.fixture
def decision_state(wrapper):
    """A state where a player has dice and must move."""
    state = wrapper.new_game()
    # Advance past the opening chance node
    outcomes = state.chance_outcomes()
    state.apply_action(outcomes[0][0])
    return state


def test_starting_position_checker_counts(decision_state):
    """Each player should have 15 checkers on the board at start."""
    state = decision_state
    bv = state.board_from_perspective()
    assert bv.my_points.sum() + bv.my_bar + bv.my_off == 15
    assert bv.opp_points.sum() + bv.opp_bar + bv.opp_off == 15


def test_starting_position_no_bar_or_off(decision_state):
    bv = decision_state.board_from_perspective()
    assert bv.my_bar == 0
    assert bv.opp_bar == 0
    assert bv.my_off == 0
    assert bv.opp_off == 0


def test_starting_position_has_dice(decision_state):
    bv = decision_state.board_from_perspective()
    assert bv.dice is not None
    d1, d2 = bv.dice
    assert 1 <= d1 <= 6
    assert 1 <= d2 <= 6


def test_legal_actions_nonempty(decision_state):
    assert len(decision_state.legal_actions()) > 0


def test_initial_state_is_chance(wrapper):
    state = wrapper.new_game()
    assert state.is_chance_node()
    assert state.current_player() == -1  # chance player


def test_chance_outcomes_valid(wrapper):
    state = wrapper.new_game()
    outcomes = state.chance_outcomes()
    assert len(outcomes) == 30  # 15 non-double combos * 2 who-goes-first
    probs = [p for _, p in outcomes]
    assert abs(sum(probs) - 1.0) < 1e-6


def test_clone_is_independent(decision_state):
    clone = decision_state.clone()
    action = decision_state.legal_actions()[0]
    decision_state.apply_action(action)
    # Clone should still be at the original state
    assert len(clone.legal_actions()) > 0
    assert not clone.is_terminal()


def test_perspective_flip_consistency(wrapper):
    """Both players should see 15 checkers each from their perspective."""
    state = wrapper.new_game()
    state.apply_action(state.chance_outcomes()[0][0])  # dice roll
    state.apply_action(state.legal_actions()[0])  # player 0 moves

    # Advance to player 1's turn
    if state.is_chance_node():
        state.apply_action(state.chance_outcomes()[0][0])

    bv = state.board_from_perspective()
    assert bv.my_points.sum() + bv.my_bar + bv.my_off == 15
    assert bv.opp_points.sum() + bv.opp_bar + bv.opp_off == 15


def test_play_full_game_to_terminal(wrapper):
    """A game should eventually terminate with valid returns."""
    state = wrapper.new_game()
    max_moves = 5000
    moves = 0
    while not state.is_terminal() and moves < max_moves:
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            idx = random.choices(range(len(actions)), weights=probs)[0]
            state.apply_action(actions[idx])
        else:
            state.apply_action(random.choice(state.legal_actions()))
            moves += 1
    assert state.is_terminal()
    returns = state.returns()
    assert len(returns) == 2
    assert returns[0] == -returns[1]


def test_terminal_result_types(wrapper):
    """Play several random games and verify terminal_result returns valid types."""
    for _ in range(10):
        state = wrapper.new_game()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                actions, probs = zip(*outcomes)
                idx = random.choices(range(len(actions)), weights=probs)[0]
                state.apply_action(actions[idx])
            else:
                state.apply_action(random.choice(state.legal_actions()))
        equity, result_type = state.terminal_result()
        assert result_type in ("normal", "gammon", "backgammon")
        assert abs(equity) in (1, 2, 3)
