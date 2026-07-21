"""Tests for TD(λ) self-play (raccoon/train/td_selfplay.py + lookahead reuse)."""
import numpy as np
import pytest
import torch

from raccoon.env.game_wrapper import GameWrapper
from raccoon.model.network import RaccoonNet
from raccoon.search.mcts import _advance_through_chance
from raccoon.train.lookahead import child_values, select_move
from raccoon.train.td_selfplay import lambda_returns, net_arena, play_td_game

CPU = torch.device("cpu")


def _small_net():
    net = RaccoonNet(channels=16, num_blocks=1)
    net.eval()
    return net


# --- lambda_returns: pure, hand-computed --------------------------------------

def test_lambda_returns_empty_and_singleton():
    assert lambda_returns([], [], [1.0, -1.0], 0.7) == []
    # one decision by player 0, player 0 wins a backgammon (+3)
    assert lambda_returns([0], [0.9], [3.0, -3.0], 0.7) == pytest.approx([1.0])


def test_lambda_returns_monte_carlo_lambda1():
    # alternating players, player 0 wins a plain game (+1). MC target = outcome.
    g = lambda_returns([0, 1], [0.1, -0.2], [1.0, -1.0], lam=1.0)
    assert g == pytest.approx([1.0 / 3, -1.0 / 3])


def test_lambda_returns_one_step_lambda0():
    # one-step TD: g[t] = -V(s_{t+1}) for the alternating transition.
    g = lambda_returns([0, 1], [0.1, -0.2], [1.0, -1.0], lam=0.0)
    assert g[1] == pytest.approx(-1.0 / 3)
    assert g[0] == pytest.approx(0.2)  # -values[1] = -(-0.2)


def test_lambda_returns_blended():
    g = lambda_returns([0, 1], [0.1, -0.2], [1.0, -1.0], lam=0.7)
    # g[0] = -((0.3)(-0.2) + 0.7(-1/3)) = 0.06 + 0.23333
    assert g[0] == pytest.approx(0.29333, abs=1e-4)
    assert g[1] == pytest.approx(-1.0 / 3)


def test_lambda_returns_same_player_doubles_sign():
    # players [0, 0, 1]: the 0->0 step (a doubles half-move) must NOT flip sign,
    # the 0->1 step must. Player 0 wins a gammon (+2).
    g = lambda_returns([0, 0, 1], [0.5, 0.5, 0.4], [2.0, -2.0], lam=0.0)
    assert g[2] == pytest.approx(-2.0 / 3)      # terminal, player 1 POV
    assert g[1] == pytest.approx(-0.4)          # 0->1: sign -1, -values[2]
    assert g[0] == pytest.approx(0.5)           # 0->0: sign +1, +values[1]


def test_lambda_returns_bounded():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = int(rng.integers(1, 12))
        players = [int(rng.integers(0, 2)) for _ in range(n)]
        values = list(rng.uniform(-1, 1, n))
        outcome = float(rng.choice([1, 2, 3])) * rng.choice([1.0, -1.0])
        g = lambda_returns(players, values, [outcome, -outcome], lam=0.7)
        assert all(-1.0 <= x <= 1.0 for x in g)


# --- 0-ply selection (integration, small net) ---------------------------------

def test_select_move_greedy_picks_argmax_child():
    net = _small_net()
    wrapper = GameWrapper()
    state = _advance_through_chance(wrapper.new_game())
    legal, cv, v_state = child_values(state._state, net, CPU)
    assert len(legal) == len(cv) > 0
    action, v2 = select_move(state._state, net, CPU, temperature=0.0)
    assert action == legal[int(np.argmax(cv))]
    assert v2 == pytest.approx(v_state)
    assert action in state.legal_actions()


# --- full tiny game -----------------------------------------------------------

def test_play_td_game_wellformed():
    np.random.seed(0)
    net = _small_net()
    result = play_td_game(net, CPU)
    assert result is not None
    obs, players, values, returns = result
    assert len(obs) == len(players) == len(values) > 0
    assert all(p in (0, 1) for p in players)
    assert obs[0].shape == (26, 2, 12)
    assert abs(returns[0]) in (1.0, 2.0, 3.0)
    assert returns[0] == pytest.approx(-returns[1])
    # targets align and stay in range
    g = lambda_returns(players, values, returns, lam=0.7)
    assert len(g) == len(players)
    assert all(-1.0 <= x <= 1.0 for x in g)


def test_net_arena_runs():
    net = _small_net()
    res = net_arena(net, net, CPU, games=2, seed=1)
    assert res["games"] == 2
    assert 0 <= res["net_a_wins"] <= 2
    assert -3.0 <= res["equity_per_game"] <= 3.0
