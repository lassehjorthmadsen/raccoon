"""The cubeful money arena, and the control variate that measures it.

These tests are about the *accounting* — who wins how many points when the cube
is live, and whether the variance-reduced estimator stays the unbiased thing it
claims to be. Strength is not measured here; that is what exp025's run does.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from raccoon.cube.janowski import X_CONTACT
from raccoon.cube.state import CubeState
from raccoon.env.game_wrapper import GameWrapper
from raccoon.eval.cube_arena import CV_CUBEFUL, CV_CUBELESS, gnubg_cube_arena
from raccoon.eval.luck import pre_roll_cubeful_values
from raccoon.model.network import RaccoonNet, load_model

CPU = torch.device("cpu")
SHIPPED = Path("experiments/exp018-distill/checkpoints/ep22.pt")


def _net():
    torch.manual_seed(13)
    net = RaccoonNet(channels=16, num_blocks=1, value_head="outcomes6")
    net.eval()
    return net


def _run(**kw):
    kw.setdefault("games", 8)
    kw.setdefault("gnubg_ply", 0)
    kw.setdefault("seed", 4)
    return gnubg_cube_arena(_net(), CPU, **kw)


# --- the control variate is still exactly zero-mean ---------------------------

def test_cubeful_luck_is_zero_mean_at_every_pre_roll_state():
    """``sum_d p(d) h(s,d) - m(s) = 0`` by construction -- the whole reason the
    estimator cannot move the expectation however bad the evaluator is."""
    wrapper = GameWrapper()
    np.random.seed(1)
    state = wrapper.new_game()
    checked = 0
    for cube in (CubeState(), CubeState(2, 0), CubeState(4, 1)):
        while checked < 8 and not state.is_terminal():
            if state.is_chance_node():
                _, probs, values = pre_roll_cubeful_values(
                    state, 0, cube, X_CONTACT, False, 0,
                )
                mean = float(probs @ values)
                luck = values - mean
                assert float(probs @ luck) == pytest.approx(0.0, abs=1e-12)
                checked += 1
                outcomes = state.chance_outcomes()
                idx = int(np.random.choice(len(outcomes),
                                           p=[p for _, p in outcomes]))
                state.apply_action(outcomes[idx][0])
            else:
                state.apply_action(state.legal_actions()[0])
    assert checked == 8


def test_cubeful_control_variate_respects_the_owner_flip():
    """Signed to a fixed player, a cube that player owns must be worth at least
    as much as the same cube in the middle, at every roll."""
    state = GameWrapper().new_game()
    state.apply_action(state.chance_outcomes()[0][0])   # past the opening node
    while not state.is_chance_node():
        state.apply_action(state.legal_actions()[0])
    _, _, owned = pre_roll_cubeful_values(state, 0, CubeState(2, 0), X_CONTACT, False, 0)
    _, _, centred = pre_roll_cubeful_values(state, 0, CubeState(), X_CONTACT, False, 0)
    _, _, theirs = pre_roll_cubeful_values(state, 0, CubeState(2, 1), X_CONTACT, False, 0)
    assert np.all(owned >= centred - 1e-9)
    assert np.all(centred >= theirs - 1e-9)


# --- what a cubeful game pays -------------------------------------------------

@pytest.mark.skipif(not SHIPPED.exists(), reason="shipped checkpoint not on disk")
def test_a_drop_pays_exactly_the_stake_before_the_double():
    """Needs the shipped net: an untrained one evaluates every position at much
    the same number, so it takes every cube and no game ever ends in a pass —
    the branch under test would never run."""
    net = load_model(str(SHIPPED))
    net.eval()
    r = gnubg_cube_arena(net, CPU, games=8, gnubg_ply=0, seed=1)
    dropped = r["game_ended_by_drop"]
    assert dropped.any(), "no game ended in a pass -- the cube never got turned"
    np.testing.assert_array_equal(
        np.abs(r["game_pts"][dropped]), r["game_cube"][dropped],
    )
    # And a pass pays the stake as it stood *before* the refused double.
    assert np.all(r["game_pts"][dropped] != 0)


def test_a_played_out_game_pays_one_two_or_three_times_the_cube():
    r = _run(games=12)
    played = ~r["game_ended_by_drop"]
    ratio = np.abs(r["game_pts"][played]) / r["game_cube"][played]
    assert set(np.unique(ratio)) <= {1.0, 2.0, 3.0}


def test_the_cube_only_ever_holds_a_power_of_two():
    r = _run(games=12)
    cubes = r["game_cube"]
    assert np.all(cubes >= 1)
    assert np.all((cubes & (cubes - 1)) == 0)


def test_the_cube_actually_gets_turned():
    """A run where nobody doubles still produces a plausible ppg, so this is the
    check that the cube layer is wired in at all rather than quietly inert."""
    r = _run(games=12)
    assert (r["game_cube"] > 1).any()
    assert r["net_doubles_offered"] > 0


def test_double_and_take_counters_cannot_exceed_their_offers():
    r = _run(games=12)
    assert 0 <= r["net_doubles"] <= r["net_doubles_offered"]
    assert 0 <= r["net_takes"] <= r["net_takes_offered"]


# --- the invariance that lets two runs be compared ----------------------------

def test_variance_reduction_does_not_perturb_play():
    """vr on and vr off must play the identical games from one seed: the control
    variate reads gnubg through a pure C path and consumes no RNG. That is what
    lets a plain run and a reduced run be two samples of one process."""
    on = _run(games=6, vr=True)
    off = _run(games=6, vr=False)
    np.testing.assert_array_equal(on["game_pts"], off["game_pts"])
    np.testing.assert_array_equal(on["game_rolls"], off["game_rolls"])
    np.testing.assert_array_equal(on["game_cube"], off["game_cube"])
    assert not off["game_luck"].any()
    np.testing.assert_array_equal(on["game_vr"], on["game_pts"] - on["game_luck"])


def test_the_choice_of_control_variate_does_not_perturb_play_either():
    a = _run(games=6, cv=CV_CUBEFUL)
    b = _run(games=6, cv=CV_CUBELESS)
    np.testing.assert_array_equal(a["game_pts"], b["game_pts"])
    # ...but it does change the estimator, or there would be nothing to choose.
    assert not np.allclose(a["game_luck"], b["game_luck"])


def test_an_unknown_control_variate_is_refused():
    with pytest.raises(ValueError):
        _run(games=1, cv="magic")
