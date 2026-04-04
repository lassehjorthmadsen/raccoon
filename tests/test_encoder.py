"""Tests for the board tensor encoder."""

import numpy as np
import pytest

from raccoon.env.encoder import encode_state, encode_batch
from raccoon.env.game_wrapper import BoardView, GameWrapper


@pytest.fixture
def wrapper():
    return GameWrapper()


@pytest.fixture
def starting_board(wrapper):
    state = wrapper.new_game()
    state.apply_action(state.chance_outcomes()[0][0])
    return state.board_from_perspective()


def test_encode_shape(starting_board):
    tensor = encode_state(starting_board)
    assert tensor.shape == (16, 2, 12)
    assert tensor.dtype == np.float32


def test_encode_finite_values(starting_board):
    tensor = encode_state(starting_board)
    assert np.all(np.isfinite(tensor))


def test_encode_empty_board():
    """An empty board should have all checker channels zero."""
    bv = BoardView(
        my_points=np.zeros(24),
        opp_points=np.zeros(24),
        my_bar=0, opp_bar=0, my_off=0, opp_off=0,
        dice=(3, 1),
    )
    tensor = encode_state(bv)
    # Checker channels (0-7) should all be zero
    assert np.all(tensor[:8] == 0)
    # Side to move should be 1
    assert np.all(tensor[8] == 1.0)


def test_encode_five_checkers_on_point():
    """5 checkers on point 6 should set channels 0-2 to 1 and channel 3 to 1.0."""
    my_points = np.zeros(24)
    my_points[5] = 5  # point 6 (0-indexed as 5)
    bv = BoardView(
        my_points=my_points,
        opp_points=np.zeros(24),
        my_bar=0, opp_bar=0, my_off=0, opp_off=0,
        dice=(3, 1),
    )
    tensor = encode_state(bv)
    # Point 6 (persp index 5): bottom row, col = 11 - 5 = 6
    row, col = 1, 6
    assert tensor[0, row, col] == 1.0  # >= 1
    assert tensor[1, row, col] == 1.0  # >= 2
    assert tensor[2, row, col] == 1.0  # >= 3
    assert tensor[3, row, col] == 1.0  # (5 - 3) / 2 = 1.0


def test_encode_dice():
    bv = BoardView(
        my_points=np.zeros(24),
        opp_points=np.zeros(24),
        my_bar=0, opp_bar=0, my_off=0, opp_off=0,
        dice=(3, 1),
    )
    tensor = encode_state(bv)
    assert np.allclose(tensor[13], 3 / 6)
    assert np.allclose(tensor[14], 1 / 6)
    assert np.all(tensor[15] == 0.0)  # not doubles


def test_encode_doubles():
    bv = BoardView(
        my_points=np.zeros(24),
        opp_points=np.zeros(24),
        my_bar=0, opp_bar=0, my_off=0, opp_off=0,
        dice=(4, 4),
    )
    tensor = encode_state(bv)
    assert np.all(tensor[15] == 1.0)  # doubles flag


def test_encode_bar_and_off():
    bv = BoardView(
        my_points=np.zeros(24),
        opp_points=np.zeros(24),
        my_bar=2, opp_bar=1, my_off=5, opp_off=3,
        dice=(1, 1),
    )
    tensor = encode_state(bv)
    assert np.allclose(tensor[9], 2 / 15)
    assert np.allclose(tensor[10], 1 / 15)
    assert np.allclose(tensor[11], 5 / 15)
    assert np.allclose(tensor[12], 3 / 15)


def test_encode_batch_shape(starting_board):
    batch = encode_batch([starting_board, starting_board])
    assert batch.shape == (2, 16, 2, 12)


def test_top_row_points_13_to_24():
    """Point 13 should map to row 0, col 0. Point 24 to row 0, col 11."""
    my_points = np.zeros(24)
    my_points[12] = 1  # point 13 (0-indexed as 12)
    my_points[23] = 1  # point 24 (0-indexed as 23)
    bv = BoardView(
        my_points=my_points, opp_points=np.zeros(24),
        my_bar=0, opp_bar=0, my_off=0, opp_off=0, dice=(1, 2),
    )
    tensor = encode_state(bv)
    assert tensor[0, 0, 0] == 1.0   # point 13 -> row 0, col 0
    assert tensor[0, 0, 11] == 1.0  # point 24 -> row 0, col 11


def test_bottom_row_points_1_to_12():
    """Point 1 should map to row 1, col 11. Point 12 to row 1, col 0."""
    my_points = np.zeros(24)
    my_points[0] = 1   # point 1 (0-indexed as 0)
    my_points[11] = 1  # point 12 (0-indexed as 11)
    bv = BoardView(
        my_points=my_points, opp_points=np.zeros(24),
        my_bar=0, opp_bar=0, my_off=0, opp_off=0, dice=(1, 2),
    )
    tensor = encode_state(bv)
    assert tensor[0, 1, 11] == 1.0  # point 1 -> row 1, col 11
    assert tensor[0, 1, 0] == 1.0   # point 12 -> row 1, col 0
