"""Encode a BoardView into a (17, 2, 12) tensor for the neural network."""

import numpy as np

from raccoon.env.game_wrapper import BoardView


CHANNEL_NAMES = [
    "my >= 1",
    "my >= 2",
    "my >= 3",
    "my overflow (count-3)/2",
    "opp >= 1",
    "opp >= 2",
    "opp >= 3",
    "opp overflow (count-3)/2",
    "side to move",
    "my bar / 15",
    "opp bar / 15",
    "my off / 15",
    "opp off / 15",
    "die 1 / 6",
    "die 2 / 6",
    "doubles flag",
    "mid-doubles flag",
]

NUM_CHANNELS = len(CHANNEL_NAMES)


def encode_state(board_view: BoardView) -> np.ndarray:
    """Encode a board position as a (17, 2, 12) float32 tensor.

    Board layout:
        Top row (row 0): perspective points 13..24 -> columns 0..11
        Bottom row (row 1): perspective points 12..1 -> columns 0..11

    Channel meanings live in ``CHANNEL_NAMES`` so the audit/debug tooling
    and the encoder can't drift apart.
    """
    tensor = np.zeros((NUM_CHANNELS, 2, 12), dtype=np.float32)

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


def dump_tensor(board_view: BoardView, *, precision: int = 3) -> str:
    """Render the encoder output for a position as a human-readable string.

    Pure debug helper: encodes ``board_view`` and pretty-prints each of the
    ``NUM_CHANNELS`` planes alongside a short header. Planes whose 24 cells
    are all equal are collapsed to a single scalar with a ``(broadcast)``
    tag — this is detected from the tensor itself, not assumed by index.
    """
    tensor = encode_state(board_view)

    dice_str = (
        f"({board_view.dice[0]},{board_view.dice[1]})"
        if board_view.dice is not None
        else "none"
    )
    header = [
        "=== Tensor audit ===",
        f"Dice: {dice_str}  mid_doubles: {board_view.mid_doubles}",
        f"Bar  my/opp: {board_view.my_bar}/{board_view.opp_bar}   "
        f"Off  my/opp: {board_view.my_off}/{board_view.opp_off}",
        f"Tensor shape: {tensor.shape}  dtype: {tensor.dtype}",
        "",
    ]

    fmt = f"{{:>{precision + 3}.{precision}f}}"
    name_width = max(len(n) for n in CHANNEL_NAMES)
    lines = list(header)
    for ch in range(NUM_CHANNELS):
        plane = tensor[ch]
        name = CHANNEL_NAMES[ch]
        if np.all(plane == plane.flat[0]):
            lines.append(
                f"Channel {ch:2d}  {name:<{name_width}}  = "
                f"{fmt.format(float(plane.flat[0]))}  (broadcast)"
            )
        else:
            lines.append(f"Channel {ch:2d}  {name}")
            for row in range(2):
                row_vals = " ".join(fmt.format(float(plane[row, col])) for col in range(12))
                lines.append(f"  {row_vals}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
