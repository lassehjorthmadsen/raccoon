"""Tests for action space utilities."""

import numpy as np
import pytest

from raccoon.env.actions import ACTION_SPACE_SIZE, legal_action_mask
from raccoon.env.game_wrapper import GameWrapper


@pytest.fixture
def decision_state():
    wrapper = GameWrapper()
    state = wrapper.new_game()
    state.apply_action(state.chance_outcomes()[0][0])
    return state


def test_mask_shape(decision_state):
    mask = legal_action_mask(decision_state.legal_actions())
    assert mask.shape == (ACTION_SPACE_SIZE,)
    assert mask.dtype == bool


def test_mask_count_matches(decision_state):
    legal = decision_state.legal_actions()
    mask = legal_action_mask(legal)
    assert mask.sum() == len(legal)


def test_mask_correct_indices(decision_state):
    legal = decision_state.legal_actions()
    mask = legal_action_mask(legal)
    for a in legal:
        assert mask[a] is np.True_


def test_action_to_string(decision_state):
    for a in decision_state.legal_actions()[:5]:
        s = decision_state.action_to_string(a)
        assert isinstance(s, str)
        assert len(s) > 0
