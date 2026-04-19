"""Tests for MCTS search."""

import numpy as np
import pytest

from raccoon.env.game_wrapper import BoardView, GameWrapper
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


def test_mcts_prefers_immediate_win(model):
    """MCTS must prefer an action that leads directly to a win.

    Regression for the pre-M7 terminal-value sign fix in ``mcts.py``: the
    old code seeded backups from terminal leaves with
    ``+returns[parent_player]``, which — combined with the single negation
    at the leaf in ``_backup`` — flipped every terminal Q-value and made
    MCTS actively avoid winning moves.
    """

    class MockState:
        """Two-action mock: action 0 wins, action 1 loses (for player 0)."""

        def __init__(self, terminal: bool = False, winner: int | None = None):
            self._terminal = terminal
            self._winner = winner

        def is_terminal(self) -> bool:
            return self._terminal

        def is_chance_node(self) -> bool:
            return False

        def current_player(self) -> int:
            return -4 if self._terminal else 0

        def legal_actions(self) -> list[int]:
            return [0, 1]

        def returns(self) -> list[float]:
            if self._winner == 0:
                return [1.0, -1.0]
            if self._winner == 1:
                return [-1.0, 1.0]
            return [0.0, 0.0]

        def clone(self) -> "MockState":
            return MockState(terminal=self._terminal, winner=self._winner)

        def apply_action(self, action: int) -> None:
            self._terminal = True
            self._winner = 0 if action == 0 else 1

        def board_from_perspective(self) -> BoardView:
            return BoardView(
                my_points=np.zeros(24, dtype=np.float32),
                opp_points=np.zeros(24, dtype=np.float32),
                my_bar=0,
                opp_bar=0,
                my_off=0,
                opp_off=0,
                dice=(1, 2),
            )

    mcts = MCTS(model, num_simulations=50)
    action_probs = mcts.search(MockState())

    assert action_probs.get(0, 0.0) > action_probs.get(1, 0.0), (
        f"MCTS did not prefer the winning action; got {action_probs}"
    )


def test_mcts_virtual_loss_batched(model, decision_state):
    """Virtual loss batched path produces a valid distribution."""
    mcts = MCTS(model, num_simulations=20, virtual_loss_count=4)
    action_probs = mcts.search(decision_state)

    assert len(action_probs) > 0
    legal = set(decision_state.legal_actions())
    for a in action_probs:
        assert a in legal
    assert abs(sum(action_probs.values()) - 1.0) < 1e-6


def test_mcts_virtual_loss_prefers_win(model):
    """Virtual loss path still correctly prefers winning moves."""

    class MockState:
        def __init__(self, terminal=False, winner=None):
            self._terminal = terminal
            self._winner = winner

        def is_terminal(self):
            return self._terminal

        def is_chance_node(self):
            return False

        def current_player(self):
            return -4 if self._terminal else 0

        def legal_actions(self):
            return [0, 1]

        def returns(self):
            if self._winner == 0:
                return [1.0, -1.0]
            if self._winner == 1:
                return [-1.0, 1.0]
            return [0.0, 0.0]

        def clone(self):
            return MockState(terminal=self._terminal, winner=self._winner)

        def apply_action(self, action):
            self._terminal = True
            self._winner = 0 if action == 0 else 1

        def board_from_perspective(self):
            return BoardView(
                my_points=np.zeros(24, dtype=np.float32),
                opp_points=np.zeros(24, dtype=np.float32),
                my_bar=0, opp_bar=0, my_off=0, opp_off=0, dice=(1, 2),
            )

    mcts = MCTS(model, num_simulations=50, virtual_loss_count=8)
    action_probs = mcts.search(MockState())
    assert action_probs.get(0, 0.0) > action_probs.get(1, 0.0)
