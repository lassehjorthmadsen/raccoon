"""Joint doubles enumeration in 0-ply lookahead (exp020).

OpenSpiel splits a doubles turn into two consecutive decisions by the same player,
two of the four half-moves each. ``child_values(..., joint_doubles=True)`` scores a
half-1 action by its best half-2 continuation, so the turn is optimised jointly —
the same thing ``gnubg_adapter.candidate_equities`` does for GNUBG.
``joint_doubles=False`` keeps the older behaviour (rank half-1 by the static value
of the intermediate position, then pick half-2 greedily), which exp020 measured.

The load-bearing test is :func:`test_non_doubles_decisions_are_identical`: the flag
must be a no-op everywhere except doubles, since that is what makes the exp020
non-doubles control a real check on the harness rather than a tautology.
"""
import numpy as np
import pyspiel
import torch

from raccoon.model.network import RaccoonNet
from raccoon.train.lookahead import (
    child_values, encode_pre_roll, eval_values_batch, select_move,
    state_after_apply, terminal_value,
)

CPU = torch.device("cpu")

# Chance-action indices 30..35 are the doubles 11..66 at a 36-outcome node.
DOUBLES_ACTIONS = list(range(30, 36))


def _small_net():
    torch.manual_seed(0)
    net = RaccoonNet(channels=16, num_blocks=1)
    net.eval()
    return net


class _HashNet:
    """Deterministic pseudo-random value function over encoded positions.

    An untrained ResNet is very nearly constant, so greedy and joint doubles play
    agree on almost every position and a dominance test against one passes without
    proving anything. Hashing the observation gives a value function with real
    spread across positions, in the same [-1, 1] range as the value head, and — being
    a row-wise reduction — one whose output does not depend on batch composition.
    """

    def value_equity(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.shape[0], -1).double()
        weights = torch.arange(1, flat.shape[1] + 1, dtype=torch.float64)
        return torch.sin((flat * weights).sum(dim=1) * 12.9898).float()


def _is_two_step(state) -> bool:
    """True when this decision is half 1 of a doubles turn that needs two actions."""
    me = state.current_player()
    _, dec_player, is_term = state_after_apply(state, state.legal_actions()[0])
    return (not is_term) and dec_player == me


def _two_step_states(limit: int, seed: int = 0) -> list:
    """Sample decision states that are half 1 of a two-action doubles turn.

    Forces a double at every mid-game chance node so such states turn up quickly;
    the opening node has 30 outcomes and cannot be a double, so it is sampled.
    """
    rng = np.random.default_rng(seed)
    game = pyspiel.load_game("backgammon")
    found: list = []
    for _ in range(60):
        if len(found) >= limit:
            break
        state = game.new_initial_state()
        moves = 0
        while not state.is_terminal() and moves < 300 and len(found) < limit:
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                if len(outcomes) == 36:
                    state.apply_action(int(rng.choice(DOUBLES_ACTIONS)))
                else:
                    state.apply_action(outcomes[int(rng.integers(len(outcomes)))][0])
                continue
            if _is_two_step(state):
                found.append(state.clone())
            legal = state.legal_actions()
            state.apply_action(legal[int(rng.integers(len(legal)))])
            moves += 1
    return found


def _turn_value(final, dec_player: int, is_term: bool, me: int, net) -> float:
    """Value of a completed turn's resting position, from the mover's POV."""
    if is_term:
        return terminal_value(final, me)
    value = float(eval_values_batch(net, encode_pre_roll(final, dec_player)[None], CPU)[0])
    return value if dec_player == me else -value


def _play_turn(state, net, joint_doubles: bool):
    """Play a whole turn with ``select_move``; return the resting position."""
    me = state.current_player()
    cur = state.clone()
    while True:
        action, _ = select_move(cur, net, CPU, joint_doubles=joint_doubles)
        nxt, dec_player, is_term = state_after_apply(cur, action)
        if is_term or dec_player != me:
            return nxt, dec_player, is_term
        cur = nxt


def test_non_doubles_decisions_are_identical():
    """The flag must change nothing when the turn is a single OpenSpiel action.

    This is the control the exp020 measurement leans on: any concession it reports
    on a non-doubles turn would be a harness bug, not a strength difference.
    """
    net = _small_net()
    game = pyspiel.load_game("backgammon")
    rng = np.random.default_rng(1)
    state = game.new_initial_state()
    checked = 0
    moves = 0
    while not state.is_terminal() and moves < 200 and checked < 25:
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            state.apply_action(outcomes[int(rng.integers(len(outcomes)))][0])
            continue
        if not _is_two_step(state):
            legal_j, cv_j, _ = child_values(state, net, CPU, joint_doubles=True)
            legal_g, cv_g, _ = child_values(state, net, CPU, joint_doubles=False)
            assert legal_j == legal_g
            np.testing.assert_array_equal(cv_j, cv_g)
            checked += 1
        legal = state.legal_actions()
        state.apply_action(legal[int(rng.integers(len(legal)))])
        moves += 1
    assert checked >= 10, f"only saw {checked} non-doubles decisions"


def test_joint_half1_equals_best_half2_continuation():
    """Each half-1 value is the max over its half-2 subgame — candidate_equities' rule."""
    net = _HashNet()
    states = _two_step_states(6, seed=2)
    assert states, "no two-step doubles states found"
    for state in states:
        me = state.current_player()
        legal, cv, _ = child_values(state, net, CPU, joint_doubles=True)
        for i, action in enumerate(legal):
            child, dec_player, is_term = state_after_apply(state, action)
            if is_term or dec_player != me:
                continue
            _, sub_cv, _ = child_values(child, net, CPU, joint_doubles=True)
            assert float(sub_cv.max()) == float(cv[i])


def test_dedup_fires_and_preserves_values():
    """Leaves collapse heavily, and collapsing them changes no returned value."""
    net = _HashNet()
    states = _two_step_states(4, seed=3)
    assert states
    collapsed_any = False
    for state in states:
        me = state.current_player()
        legal, cv, _ = child_values(state, net, CPU, joint_doubles=True)

        # Independent reference: evaluate every path's leaf on its own.
        n_paths = 0
        keys = set()
        for i, action in enumerate(legal):
            child, dec_player, is_term = state_after_apply(state, action)
            leaves = (
                [state_after_apply(child, a2) for a2 in child.legal_actions()]
                if (not is_term) and dec_player == me
                else [(child, dec_player, is_term)]
            )
            best = max(
                _turn_value(leaf, dp, term, me, net) for leaf, dp, term in leaves
            )
            assert best == float(cv[i])
            n_paths += len(leaves)
            for leaf, dp, term in leaves:
                if not term:
                    keys.add(encode_pre_roll(leaf, dp).tobytes())
        if keys and n_paths > len(keys):
            collapsed_any = True
    assert collapsed_any, "expected duplicate leaves across ordered half-move pairs"


def test_joint_never_does_worse_than_greedy():
    """Joint maximises over the same leaves greedy searches within, so it dominates."""
    net = _HashNet()
    states = _two_step_states(25, seed=4)
    assert states
    strict = 0
    for state in states:
        me = state.current_player()
        v_joint = _turn_value(*_play_turn(state, net, True), me, net)
        v_greedy = _turn_value(*_play_turn(state, net, False), me, net)
        assert v_joint >= v_greedy - 1e-6
        if v_joint > v_greedy + 1e-6:
            strict += 1
    assert strict > 0, "greedy two-step never lost anything — suspicious"


def test_selected_half1_value_is_realised_by_the_turn():
    """The half-1 score is the value of the position the full turn actually reaches."""
    net = _HashNet()
    states = _two_step_states(8, seed=5)
    assert states
    for state in states:
        me = state.current_player()
        _, cv, _ = child_values(state, net, CPU, joint_doubles=True)
        realised = _turn_value(*_play_turn(state, net, True), me, net)
        assert realised == float(cv.max())
