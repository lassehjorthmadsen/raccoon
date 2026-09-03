"""Cube state: the seat/label bridge, and the transitions a money game allows."""

import numpy as np
import pytest

from raccoon.cube.janowski import CENTERED, OPPONENT, PLAYER
from raccoon.cube.state import MAX_CUBE, CubeState, flip_label, is_race, x_for
from raccoon.cube.janowski import X_CONTACT, X_RACE
from raccoon.env.game_wrapper import BoardView, GameWrapper
from raccoon.search.mcts import _advance_through_chance


# --- seat <-> label -----------------------------------------------------------

def test_centred_cube_reads_as_centred_from_both_seats():
    cube = CubeState()
    assert cube.label_for(0) == CENTERED
    assert cube.label_for(1) == CENTERED


def test_owned_cube_reads_from_each_seat():
    cube = CubeState(value=2, owner=1)
    assert cube.label_for(1) == PLAYER
    assert cube.label_for(0) == OPPONENT


def test_label_for_is_exactly_flip_label_between_the_seats():
    """The seat translation and the perspective flip must agree, or a negated
    value and its cube ownership drift apart."""
    for cube in (CubeState(), CubeState(2, 0), CubeState(4, 1)):
        assert cube.label_for(0) == flip_label(cube.label_for(1))


def test_flip_label_is_an_involution():
    for label in (CENTERED, PLAYER, OPPONENT):
        assert flip_label(flip_label(label)) == label


def test_flip_label_rejects_nonsense():
    with pytest.raises(ValueError):
        flip_label("player_owns_it")


# --- transitions --------------------------------------------------------------

def test_either_seat_may_turn_a_centred_cube():
    cube = CubeState()
    assert cube.may_double(0) and cube.may_double(1)


def test_only_the_owner_may_redouble():
    cube = CubeState(value=2, owner=0)
    assert cube.may_double(0)
    assert not cube.may_double(1)


def test_double_doubles_the_value_and_hands_the_cube_over():
    after = CubeState().after_double(0)
    assert after == CubeState(value=2, owner=1)
    again = after.after_double(1)
    assert again == CubeState(value=4, owner=0)


def test_doubling_a_cube_you_do_not_own_is_refused():
    with pytest.raises(ValueError):
        CubeState(value=2, owner=0).after_double(1)


def test_the_runaway_guard_stops_doubling_at_max_cube():
    cube = CubeState(value=MAX_CUBE, owner=None)
    assert not cube.may_double(0)
    with pytest.raises(ValueError):
        cube.after_double(0)


def test_cube_state_is_immutable():
    with pytest.raises(Exception):
        CubeState().value = 2


# --- race detection and the index it picks ------------------------------------

def _view(my_points, opp_points, my_bar=0, opp_bar=0):
    mp, op = np.zeros(24, dtype=int), np.zeros(24, dtype=int)
    for p, n in my_points.items():
        mp[p - 1] = n
    for p, n in opp_points.items():
        op[p - 1] = n
    return BoardView(my_points=mp, opp_points=op, my_bar=my_bar, opp_bar=opp_bar,
                     my_off=0, opp_off=0, dice=None)


def test_the_opening_position_is_not_a_race():
    state = _advance_through_chance(GameWrapper().new_game())
    view = state.board_from_perspective()
    assert not is_race(view)
    assert x_for(view) == X_CONTACT


def test_fully_passed_sides_are_a_race():
    # Mine sit on 1-6; theirs, in their own indexing, sit on 1-6 too, which is my
    # 19-24. Nothing can hit anything.
    view = _view({1: 8, 3: 7}, {1: 8, 3: 7})
    assert is_race(view)
    assert x_for(view) == X_RACE


def test_a_checker_on_the_bar_is_never_a_race():
    view = _view({1: 8, 3: 7}, {1: 8, 3: 7}, my_bar=1)
    assert not is_race(view)


def test_overlapping_sides_are_not_a_race():
    # Their point 3 is my point 22, and my back checker is on 23 — still behind
    # it, so the two sides have not passed each other yet.
    view = _view({23: 2, 1: 13}, {3: 2, 1: 13})
    assert not is_race(view)
    # Pull that back checker in front of them and it becomes a race.
    assert is_race(_view({21: 2, 1: 13}, {3: 2, 1: 13}))
