"""Encode a BoardView into a (26, 2, 12) tensor for the neural network."""

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
    # Handcrafted features (raw values; BatchNorm handles scaling)
    "my pip count",
    "opp pip count",
    "pip ratio",
    "my blots",
    "opp blots",
    "my anchors",
    "opp anchors",
    "my contact",
    "opp contact",
]

NUM_CHANNELS = len(CHANNEL_NAMES)


def encode_state(board_view: BoardView) -> np.ndarray:
    """Encode a board position as a (26, 2, 12) float32 tensor.

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

    # --- Handcrafted features (raw values; BatchNorm handles scaling) ---
    my_pts = board_view.my_points
    opp_pts = board_view.opp_points

    # Pip counts: distance from own bearoff for each side.
    # My checkers: index i → i+1 pips from my bearoff.
    # Opp checkers: index i (my frame) → 24-i pips from their bearoff.
    my_pips = np.arange(1, 25, dtype=np.float32)
    opp_pips = np.arange(24, 0, -1, dtype=np.float32)
    my_pip = float(np.dot(my_pts, my_pips)) + board_view.my_bar * 25.0
    opp_pip = float(np.dot(opp_pts, opp_pips)) + board_view.opp_bar * 25.0
    tensor[17, :, :] = my_pip
    tensor[18, :, :] = opp_pip
    pip_total = my_pip + opp_pip
    tensor[19, :, :] = (my_pip / pip_total) if pip_total > 0 else 0.5

    # Blot counts (exposed single checkers)
    tensor[20, :, :] = float(np.sum(my_pts == 1))
    tensor[21, :, :] = float(np.sum(opp_pts == 1))

    # Anchor counts (made points in opponent's home board)
    tensor[22, :, :] = float(np.sum(my_pts[18:24] >= 2))
    tensor[23, :, :] = float(np.sum(opp_pts[0:6] >= 2))

    # Contact pressure: pips needed to fully break contact.
    # my_contact = Σ my_pts[i] × max(0, i - min_opp + 1)
    #   + my_bar × (25 - min_opp)   (bar re-enters at index 24, must clear min_opp)
    # opp_contact = Σ opp_pts[j] × max(0, max_my - j + 1)
    #   + opp_bar × (max_my + 2)    (bar re-enters at index -1, must clear max_my)
    # min_opp: opp's least-advanced position. If opp has bar checkers, they will
    #   re-enter on my home board (indices 0-5), so min_opp = 0.
    # max_my: my most-advanced position. If I have bar checkers, they will
    #   re-enter on opp's home board (indices 18-23), so max_my = 23.
    opp_occupied = np.flatnonzero(opp_pts)
    my_occupied = np.flatnonzero(my_pts)
    have_opp = len(opp_occupied) > 0 or board_view.opp_bar > 0
    have_my = len(my_occupied) > 0 or board_view.my_bar > 0
    if have_opp and have_my:
        if board_view.opp_bar > 0:
            min_opp = 0  # bar checkers re-enter on my home board
        else:
            min_opp = int(opp_occupied[0])
        if board_view.my_bar > 0:
            max_my = 23  # bar checkers re-enter on opp's home board
        else:
            max_my = int(my_occupied[-1])
        my_contact = float(np.dot(my_pts, np.maximum(0, np.arange(24) - min_opp + 1)))
        my_contact += board_view.my_bar * (25 - min_opp)
        opp_contact = float(np.dot(opp_pts, np.maximum(0, max_my - np.arange(24) + 1)))
        opp_contact += board_view.opp_bar * (max_my + 2)
    else:
        my_contact = 0.0
        opp_contact = 0.0
    tensor[24, :, :] = my_contact
    tensor[25, :, :] = opp_contact

    return tensor


def encode_batch(board_views: list[BoardView]) -> np.ndarray:
    """Encode multiple board positions. Returns shape (N, 26, 2, 12)."""
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
