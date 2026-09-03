"""Cube-aware 0-ply move ranking.

The load-bearing test here is :func:`test_dead_cube_ranking_is_exactly_cubeless`.
At ``x = 0`` and no Jacoby, ``cl2cf_money`` reduces to ``cubeless_equity``
identically, so the cubeful path must reproduce the cubeless one to the last bit
on real positions. That ties the whole new code path -- the six-outcome batch,
the leaf dedup, the owner flip, the points scale -- to the one that has been
shipping, and no amount of averaging can hide a break in it.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from raccoon.cube.janowski import CENTERED, OPPONENT, PLAYER, X_CONTACT
from raccoon.cube.state import CubeState
from raccoon.env.game_wrapper import GameWrapper
from raccoon.model.network import RaccoonNet, load_model
from raccoon.search.mcts import _advance_through_chance
from raccoon.train.lookahead import (
    child_cubeful_values, child_values, select_move_cubeful,
)

CPU = torch.device("cpu")


def _net():
    torch.manual_seed(11)
    net = RaccoonNet(channels=16, num_blocks=1, value_head="outcomes6")
    net.eval()
    return net


def _decisions(net, n_games=4, limit=120):
    """Real decision states, reached by playing the net against itself greedily."""
    wrapper = GameWrapper()
    out = []
    np.random.seed(5)
    for _ in range(n_games):
        state = _advance_through_chance(wrapper.new_game())
        while not state.is_terminal() and len(out) < limit:
            out.append(state._state.clone())
            legal, cv, _ = child_values(state._state, net, CPU)
            state.apply_action(legal[int(np.argmax(cv))])
            _advance_through_chance(state)
        if len(out) >= limit:
            break
    return out


# --- the tie-back -------------------------------------------------------------

@pytest.mark.parametrize("label", [CENTERED, PLAYER, OPPONENT])
def test_dead_cube_ranking_is_exactly_cubeless(label):
    """x = 0, no Jacoby: the cubeful values are the cubeless ones, times 3."""
    net = _net()
    for state in _decisions(net):
        legal_a, cubeless, v_a = child_values(state, net, CPU)
        legal_b, cubeful, v_b = child_cubeful_values(
            state, net, CPU, label, x=0.0, jacoby=False,
        )
        assert legal_a == legal_b
        # child_values is equity/3; child_cubeful_values is points at cube value 1.
        np.testing.assert_allclose(cubeful, np.asarray(cubeless) * 3.0, atol=2e-5)
        assert v_b == pytest.approx(v_a * 3.0, abs=2e-5)


def test_dead_cube_picks_the_same_move_as_the_cubeless_engine():
    net = _net()
    for state in _decisions(net):
        legal, cubeless, _ = child_values(state, net, CPU)
        _, cubeful, _ = child_cubeful_values(state, net, CPU, CENTERED, 0.0, False)
        assert legal[int(np.argmax(cubeless))] == legal[int(np.argmax(cubeful))]


# --- the owner flip -----------------------------------------------------------

def test_owning_the_cube_is_never_worth_less_than_not_owning_it():
    """E_O >= E_C >= E_U on the same position -- the coherence ordering exp024
    used as its gate. Nothing in the ranking enforces it, so it is a real check."""
    net = _net()
    for state in _decisions(net, n_games=2, limit=40):
        vals = {}
        for label in (PLAYER, CENTERED, OPPONENT):
            _, cv, v = child_cubeful_values(state, net, CPU, label, X_CONTACT, False)
            vals[label] = v
        assert vals[PLAYER] >= vals[CENTERED] - 1e-6
        assert vals[CENTERED] >= vals[OPPONENT] - 1e-6


SHIPPED = Path("experiments/exp018-distill/checkpoints/ep22.pt")


@pytest.mark.skipif(not SHIPPED.exists(), reason="shipped checkpoint not on disk")
def test_a_live_cube_changes_which_move_is_best_at_least_sometimes():
    """If the owner label never changed a ranking the experiment would be pointless.

    Needs the shipped net: an untrained one produces near-degenerate outcome
    distributions whose candidates are separated by far less than the cube term,
    so it never flips a ranking and the check would pass vacuously. exp024
    measured the real rate at 6.8% of benchmark decisions
    (``experiments/exp024-cube/results/cube_blind_floor.json``); over self-play
    decisions, which include many forced or obvious moves, it is about 2%, so
    this needs a few hundred decisions to be a reliable check rather than a
    coin flip.
    """
    net = load_model(str(SHIPPED))
    net.eval()
    differs = 0
    for state in _decisions(net, n_games=8, limit=250):
        _, owned, _ = child_cubeful_values(state, net, CPU, PLAYER, X_CONTACT, False)
        _, theirs, _ = child_cubeful_values(state, net, CPU, OPPONENT, X_CONTACT, False)
        differs += int(np.argmax(owned) != np.argmax(theirs))
    assert differs > 0


# --- scale --------------------------------------------------------------------

def test_terminal_children_score_in_points_not_equity_over_three():
    """A won game is worth 1/2/3 points on the cubeful scale, where child_values
    reports 1/3, 2/3, 1. Mixing the two scales is the obvious future bug, so the
    exact ratio is pinned rather than a range."""
    net = _net()
    wrapper = GameWrapper()
    np.random.seed(3)
    for _ in range(80):
        state = _advance_through_chance(wrapper.new_game())
        while not state.is_terminal():
            parent = state._state.clone()
            legal, cubeless, _ = child_values(parent, net, CPU)
            best = int(np.argmax(cubeless))
            state.apply_action(legal[best])
            _advance_through_chance(state)
            if not state.is_terminal():
                continue
            _, cubeful, _ = child_cubeful_values(
                parent, net, CPU, CENTERED, X_CONTACT, False,
            )
            # The winning action is terminal, so both paths report an exact
            # terminal value and the cubeful one is three times the cubeless one.
            won = state.returns()[parent.current_player()]
            assert abs(won) in (1.0, 2.0, 3.0)
            assert cubeful[best] == pytest.approx(won, abs=1e-6)
            assert cubeless[best] == pytest.approx(won / 3.0, abs=1e-6)
            return
    pytest.skip("no game finished inside the budget")


# --- the wrapper --------------------------------------------------------------

def test_select_move_cubeful_returns_a_legal_action():
    net = _net()
    for state in _decisions(net, n_games=1, limit=20):
        action, e_state = select_move_cubeful(state, net, CPU, CubeState())
        assert action in state.legal_actions()
        assert -3.0 <= e_state <= 3.0


def test_select_move_cubeful_needs_the_six_outcome_head():
    torch.manual_seed(2)
    scalar_net = RaccoonNet(channels=16, num_blocks=1, value_head="scalar")
    scalar_net.eval()
    state = _advance_through_chance(GameWrapper().new_game())._state
    with pytest.raises(ValueError):
        select_move_cubeful(state, scalar_net, CPU, CubeState())
