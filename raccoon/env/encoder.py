"""Encode a BoardView into a (17, 2, 12) tensor for the neural network."""

import numpy as np

from raccoon.env.game_wrapper import BoardView


def encode_state(board_view: BoardView) -> np.ndarray:
    """Encode a board position as a (17, 2, 12) float32 tensor.

    Board layout:
        Top row (row 0): perspective points 13..24 -> columns 0..11
        Bottom row (row 1): perspective points 12..1 -> columns 0..11

    Channels:
         0: my checkers >= 1
         1: my checkers >= 2
         2: my checkers >= 3
         3: my checkers overflow (count - 3) / 2
         4-7: opponent checkers (same scheme)
         8: side to move (all 1s)
         9: my bar / 15
        10: opp bar / 15
        11: my off / 15
        12: opp off / 15
        13: die 1 / 6
        14: die 2 / 6
        15: doubles flag
        16: mid-doubles flag (2nd half of a split doubles turn)
    """
    tensor = np.zeros((17, 2, 12), dtype=np.float32)

    # Map perspective points to (row, col)
    # Point 13+c -> row 0, col c  (c = 0..11)
    # Point 12-c -> row 1, col c  (c = 0..11)
    def _fill_checker_planes(points: np.ndarray, ch_offset: int):
        for pt_idx in range(24):
            count = points[pt_idx]
            if pt_idx >= 12:
                row, col = 0, pt_idx - 12  # points 13-24
            else:
                row, col = 1, 11 - pt_idx   # points 12-1
            if count >= 1:
                tensor[ch_offset, row, col] = 1.0
            if count >= 2:
                tensor[ch_offset + 1, row, col] = 1.0
            if count >= 3:
                tensor[ch_offset + 2, row, col] = 1.0
            if count > 3:
                tensor[ch_offset + 3, row, col] = (count - 3) / 2.0

    _fill_checker_planes(board_view.my_points, 0)
    _fill_checker_planes(board_view.opp_points, 4)

    # Broadcast planes (constant across all cells)
    tensor[8, :, :] = 1.0  # side to move
    tensor[9, :, :] = board_view.my_bar / 15.0
    tensor[10, :, :] = board_view.opp_bar / 15.0
    tensor[11, :, :] = board_view.my_off / 15.0
    tensor[12, :, :] = board_view.opp_off / 15.0

    if board_view.dice is not None:
        d1, d2 = board_view.dice
        tensor[13, :, :] = d1 / 6.0
        tensor[14, :, :] = d2 / 6.0
        tensor[15, :, :] = 1.0 if d1 == d2 else 0.0

    if board_view.mid_doubles:
        tensor[16, :, :] = 1.0

    return tensor


def encode_batch(board_views: list[BoardView]) -> np.ndarray:
    """Encode multiple board positions. Returns shape (N, 17, 2, 12)."""
    return np.stack([encode_state(bv) for bv in board_views])
