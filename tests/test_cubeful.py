"""Cubeful search: Janowski applied at every node rather than only at the root.

The test that carries the most weight here is
:func:`test_depth_zero_reproduces_the_closed_form`, which pins the recursion's
base case against the module it generalises. But note what it cannot do: it
never touches the roll expansion, so it passed throughout while a sign error in
the terminal branch made every game-ending move score as its own opposite. A
base-case test proves the base case and nothing else, which is why
:func:`test_a_winning_roll_is_worth_points_not_minus_points` exists.
"""

import numpy as np
import pytest
import torch

from raccoon.cube.janowski import (
    CENTERED, OPPONENT, PLAYER, X_CONTACT, cube_equities,
)
from raccoon.env.game_wrapper import GameWrapper
from raccoon.model.network import RaccoonNet
from raccoon.search.cubeful import (
    CubefulConfig, ProbsEvaluator, cube_action_searched, cube_equities_searched,
    terminal_points,
)
from raccoon.search.mcts import _advance_through_chance

gnubg_nn = pytest.importorskip("gnubg_nn")

from raccoon.eval.gnubg_adapter import board_from_view, pick_move

CPU = torch.device("cpu")
START_ID = "4HPwATDgc/ABMA"


def _net():
    torch.manual_seed(19)
    net = RaccoonNet(channels=16, num_blocks=1, value_head="outcomes6")
    net.eval()
    return net


def _positions(n=5, seed=11):
    np.random.seed(seed)
    wrapper = GameWrapper()
    state = _advance_through_chance(wrapper.new_game())
    out = []
    while len(out) < n and not state.is_terminal():
        b = board_from_view(state.board_from_perspective())
        out.append((tuple(b[0]), tuple(b[1])))
        state.apply_action(pick_move(state, 0))
        _advance_through_chance(state)
    return out


# --- the base case ------------------------------------------------------------

@pytest.mark.parametrize("label", [CENTERED, PLAYER])
def test_depth_zero_reproduces_the_closed_form(label):
    """At depth 0 both branches are cl2cf_money on one distribution under two
    ownership labels, which is exactly cube_equities. Anything else means the
    recursion has changed the model rather than deepened it."""
    net = _net()
    for board in _positions():
        got, ev = cube_equities_searched(
            board, label, net, CPU, CubefulConfig(depth=0), X_CONTACT, False,
        )
        want = cube_equities(ev.cache[board], label, X_CONTACT, jacoby=False)
        assert got.nd == pytest.approx(want.nd, abs=1e-9)
        assert got.dt == pytest.approx(want.dt, abs=1e-9)
        assert got.dp == 1.0


def test_a_cube_decision_the_opponent_owns_is_refused():
    net = _net()
    with pytest.raises(ValueError):
        cube_equities_searched(_positions(1)[0], OPPONENT, net, CPU,
                               CubefulConfig(depth=0), X_CONTACT, False)


# --- signs, which is where this went wrong ------------------------------------

def test_terminal_points_are_a_loss_for_the_side_on_roll():
    """A finished board holds the winner in slot 0, so the side on roll lost."""
    board = gnubg_nn.board_from_position_id(START_ID)
    won = ((0,) * 25, tuple(board[1]))          # slot 0 empty: slot 0 borne off
    value = terminal_points(won)
    assert value is not None and value < 0
    assert abs(value) in (1.0, 2.0, 3.0)
    assert terminal_points(((tuple(board[0])), tuple(board[1]))) is None


def test_a_winning_roll_is_worth_points_not_minus_points():
    """The regression this module actually shipped and had to fix.

    ``terminal_points(child)`` and ``_node_value(child, ...)`` both report to the
    player on roll AT THE CHILD -- the opponent -- so both need negating to reach
    the mover's value. Negating only the recursive branch flipped the sign of
    every game-ending move, which is invisible until a position can reach one.

    A near-borne-off position can bear off within a roll, so its value to the
    mover must be strongly positive; with the sign flipped it comes out strongly
    negative.
    """
    net = _net()
    # Mover has two checkers on the 1-point and nothing else; the opponent is far
    # away. Almost every roll ends the game in the mover's favour.
    me = (2,) + (0,) * 24
    opp = (0,) * 12 + (15,) + (0,) * 12
    board = (opp, me)
    eq, _ = cube_equities_searched(board, CENTERED, net, CPU,
                                   CubefulConfig(depth=1), X_CONTACT, False)
    assert eq.nd > 0.5, (
        f"a position that bears off next roll scored {eq.nd:+.3f}; a negated "
        "terminal branch is the usual cause"
    )


def test_search_keeps_equities_inside_the_provable_bounds():
    """No line pays more than a point when the cube is turned, and an undoubled
    game cannot pay more than the natural size of the win. The closed form
    satisfies this by construction; the recursion has to be checked."""
    net = _net()
    for board in _positions():
        for label in (CENTERED, PLAYER):
            eq, _ = cube_equities_searched(board, label, net, CPU,
                                           CubefulConfig(depth=1), X_CONTACT, False)
            assert -3.0 <= eq.nd <= 3.0
            assert -6.0 <= eq.dt <= 6.0     # dt is already doubled
            assert eq.dp == 1.0


# --- the cube ordering --------------------------------------------------------

def test_owning_the_cube_is_never_worse_than_it_being_centred():
    """E_O >= E_C at one cube value, the coherence ordering exp024 used as a gate.
    Nothing in the recursion enforces it, so it is a real check on the search."""
    net = _net()
    for board in _positions():
        owned, _ = cube_equities_searched(board, PLAYER, net, CPU,
                                          CubefulConfig(depth=1), X_CONTACT, False)
        centred, _ = cube_equities_searched(board, CENTERED, net, CPU,
                                            CubefulConfig(depth=1), X_CONTACT, False)
        assert owned.nd >= centred.nd - 1e-9


def test_deeper_search_costs_more_boards_and_changes_the_answer():
    """If depth made no difference there would be nothing to measure."""
    net = _net()
    board = _positions(1)[0]
    d0, ev0 = cube_equities_searched(board, CENTERED, net, CPU,
                                     CubefulConfig(depth=0), X_CONTACT, False)
    d1, ev1 = cube_equities_searched(board, CENTERED, net, CPU,
                                     CubefulConfig(depth=1), X_CONTACT, False)
    assert ev1.evaluated > ev0.evaluated * 50
    assert d1.nd != pytest.approx(d0.nd, abs=1e-6)


def test_cube_action_agrees_with_its_own_equities():
    net = _net()
    for board in _positions(3):
        eq, _ = cube_equities_searched(board, CENTERED, net, CPU,
                                       CubefulConfig(depth=1), X_CONTACT, False)
        double, take = cube_action_searched(board, CENTERED, net, CPU,
                                            CubefulConfig(depth=1), X_CONTACT, False)
        assert double == (min(eq.dt, eq.dp) > eq.nd)
        assert take == (eq.dt < eq.dp)


# --- caching ------------------------------------------------------------------

def test_one_expansion_serves_both_cube_states():
    """The cube changes how a leaf converts to a number, never the board tree, so
    the dt branch must not cost a second expansion. This is what makes searching
    the cube affordable at depth 1."""
    net = _net()
    board = _positions(1)[0]
    ev = ProbsEvaluator(net, CPU, None, 512)
    _, ev_full = cube_equities_searched(board, CENTERED, net, CPU,
                                        CubefulConfig(depth=1), X_CONTACT, False)
    # One roll expansion is ~21 rolls x ~20 turns before dedup; a second, separate
    # expansion for the dt branch would roughly double it.
    assert ev_full.evaluated < 1200, (
        f"{ev_full.evaluated} boards for one depth-1 cube decision suggests the "
        "dt branch is re-expanding the tree instead of reusing it"
    )


def test_config_rejects_nonsense():
    with pytest.raises(ValueError):
        CubefulConfig(depth=-1)
    with pytest.raises(ValueError):
        CubefulConfig(k=0)
