"""Tests for the dice-luck control variate. Skipped if gnubg-nn isn't installed.

The load-bearing test here is :func:`test_luck_is_zero_mean_exact`. Everything the
variance-reduced estimator claims rests on that one property: averaged over a pre-roll
state's own dice distribution, the recorded luck is exactly zero, so subtracting it
cannot move the expectation of a rollout. The others guard the ways that property can
be broken in practice — a wrong dice table, a stale terminal value, or a control variate
that quietly perturbs the game it is measuring.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("gnubg_nn")

from raccoon.env.game_wrapper import GameWrapper
from raccoon.eval import luck as L
from raccoon.eval.gnubg_adapter import pick_move, terminal_equity_after_move
from raccoon.eval.vr_arena import gnubg_arena_vr
from raccoon.model.network import RaccoonNet


def _pre_roll_states(games: int = 2, seed: int = 5, limit: int = 12):
    """Play greedy GNUBG-vs-GNUBG games, yielding the mid-game chance nodes."""
    np.random.seed(seed)
    wrapper = GameWrapper()
    out = []
    for _ in range(games):
        state = wrapper.new_game()
        steps = 0
        while not state.is_terminal() and steps < 400:
            if state.is_chance_node():
                if len(state.chance_outcomes()) == L.MIDGAME_OUTCOMES:
                    out.append(state.clone())
                actions, probs = zip(*state.chance_outcomes())
                state.apply_action(int(np.random.choice(actions, p=probs)))
            else:
                state.apply_action(pick_move(state, 0))
            steps += 1
        if len(out) >= limit:
            break
    return out[:limit]


def test_dice_table_matches_action_to_string():
    """The chance-action -> dice table must match OpenSpiel, at both node sizes."""
    wrapper = GameWrapper()
    opening = wrapper.new_game()
    assert len(opening.chance_outcomes()) == L.OPENING_OUTCOMES

    def dice_in(text):
        digits = text.split("roll:")[1].strip(" )")
        return tuple(sorted(int(c) for c in digits))

    for action, _ in opening.chance_outcomes():
        label = opening._state.action_to_string(action)
        assert dice_in(label) == L.DICE_BY_CHANCE_ACTION[action]
        # The opening node also encodes the starter: even index = player 0.
        assert label.startswith("chance outcome")
        assert ("X starts" in label) == (action % 2 == 0)

    midgame = _pre_roll_states(games=1, limit=1)[0]
    assert len(midgame.chance_outcomes()) == L.MIDGAME_OUTCOMES
    for action, _ in midgame.chance_outcomes():
        label = midgame._state.action_to_string(action)
        assert dice_in(label) == L.DICE_BY_CHANCE_ACTION[action]


def test_luck_is_zero_mean_exact():
    """THE property: E[luck] over a node's own dice distribution is exactly zero.

    This is what makes the estimator unbiased no matter how good or bad the evaluator
    is, so it is asserted through the production code path rather than reasoned about.
    """
    for state in _pre_roll_states():
        for perspective in (0, 1):
            actions, probs, _ = L.pre_roll_values(state, perspective)
            lucks = np.array(
                [L.pre_roll_luck(state, a, perspective)[0] for a in actions]
            )
            assert abs(float(probs @ lucks)) < 1e-9


def test_opening_mean_is_zero_by_symmetry():
    """The opening position is symmetric, so its pre-roll mean must vanish."""
    state = GameWrapper().new_game()
    actions, probs, values = L.pre_roll_values(state, 0)
    assert len(actions) == L.OPENING_OUTCOMES
    assert abs(float(probs @ values)) < 1e-9
    # Every roll is a real advantage to whoever wins it, so the values are not all zero.
    assert np.abs(values).max() > 0.05
    _, _, flipped = L.pre_roll_values(state, 1)
    assert np.allclose(flipped, -values)


def test_native_sweep_matches_openspiel_route():
    """The fast sweep must agree with OpenSpiel's own move generation, exactly.

    ``gnubg_nn.best_move`` is used for move generation only; the position it lands on is
    re-evaluated with ``probabilities``, because the equity in its detail tuple is not
    trustworthy in race and bear-off classes. This test is what pins that down — it
    covers doubles (whose second half OpenSpiel splits into a separate action) and
    game-ending rolls (where ``candidate_equities`` would report a flat +3.0).
    """
    for state in _pre_roll_states():
        actions, _, values = L.pre_roll_values(state, 0)
        mover = state.clone()
        mover.apply_action(actions[0])
        sign = 1.0 if mover.current_player() == 0 else -1.0
        for i, action in enumerate(actions):
            expected = sign * L.python_roll_value(state, action)
            assert float(values[i]) == pytest.approx(expected, abs=1e-6)


def test_terminal_equity_after_move_classifies_wins():
    """Plain win / gammon / backgammon from a finished post-move board."""
    empty = [0] * 25
    # Loser has borne off (sum < 15) -> plain win.
    loser_partly_off = [0] * 25
    loser_partly_off[5] = 10
    assert terminal_equity_after_move([empty, loser_partly_off]) == 1.0
    # Loser has all 15 checkers, none in the winner's home, none on the bar -> gammon.
    loser_gammoned = [0] * 25
    loser_gammoned[5] = 15
    assert terminal_equity_after_move([empty, loser_gammoned]) == 2.0
    # ... plus a checker on the bar, or in the winner's home (loser indices 18-23).
    loser_on_bar = [0] * 25
    loser_on_bar[5] = 14
    loser_on_bar[24] = 1
    assert terminal_equity_after_move([empty, loser_on_bar]) == 3.0
    loser_in_home = [0] * 25
    loser_in_home[5] = 14
    loser_in_home[20] = 1
    assert terminal_equity_after_move([empty, loser_in_home]) == 3.0
    # Not finished at all.
    assert terminal_equity_after_move([loser_gammoned, loser_gammoned]) is None


def test_vr_identity_and_no_perturbation_of_play():
    """``vr == raw - luck``, and switching the estimator on does not change the games.

    The second half is load-bearing for how the demonstration is framed: a plain run and
    a variance-reduced run are two independent samples of *one* process, not samples of
    two different processes. It holds because the control variate reads gnubg through a
    pure C path and draws its roll from the same call on the same distribution.
    """
    net = RaccoonNet()
    net.eval()
    device = torch.device("cpu")

    with_vr = gnubg_arena_vr(net, device, games=4, seed=17, vr=True)
    without = gnubg_arena_vr(net, device, games=4, seed=17, vr=False)

    assert np.allclose(with_vr["game_vr"], with_vr["game_pts"] - with_vr["game_luck"])
    assert np.array_equal(with_vr["game_pts"], without["game_pts"])
    assert np.array_equal(with_vr["game_rolls"], without["game_rolls"])
    assert np.count_nonzero(with_vr["game_luck"]) > 0
    assert not np.any(without["game_luck"])


def test_cv_ply_above_zero_is_refused():
    """gnubg-nn's best_move segfaults at ply >= 1, so it must never be reached."""
    from raccoon.eval.gnubg_adapter import best_move_equity

    board = [[0] * 25, [0] * 25]
    with pytest.raises(ValueError, match="ply must be 0"):
        best_move_equity(board, 3, 1, ply=1)
