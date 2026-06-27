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
    # Handcrafted features, normalised into ~[0,1] by default (Fix-N); the
    # divisor is baked into the name like the base planes above, e.g.
    # "my bar / 15". See FEATURE_SCALES and encode_state(normalize=...).
    "my pip count / 167",
    "opp pip count / 167",
    "pip ratio",
    "my blots / 15",
    "opp blots / 15",
    "my anchors / 6",
    "opp anchors / 6",
    "my contact / 167",
    "opp contact / 167",
]

NUM_CHANNELS = len(CHANNEL_NAMES)

# Named feature groups -> the channel indices they own. "base" is every
# non-handcrafted channel (checker planes, bar/off, dice, doubles); the four
# handcrafted groups can be toggled on top of it for ablation experiments.
# Indices are the contract; CHANNEL_NAMES stays the single source of truth.
FEATURE_GROUPS: dict[str, list[int]] = {
    "base": list(range(0, 17)),
    "pip": [17, 18, 19],
    "blots": [20, 21],
    "anchors": [22, 23],
    "contact": [24, 25],
}

# Per-channel divisors that bring the handcrafted features into the ~[0, 1]
# range the base planes already live in (bar/15, off/15, die/6). The raw
# handcrafted values are ~100x larger (pip ~95, contact ~52), which lets them
# dominate the input convolution and destabilises value-head training — see
# Stage 6 of docs/pretraining_analysis.qmd. Pip and contact are pip-scale
# quantities (/167, the opening pip count); blots scale by checker count (/15);
# anchors by the six home-board points (/6); pip ratio is already in [0, 1].
# Applied by default (``normalize=True``); pass ``normalize=False`` to recover
# the raw magnitudes (e.g. for feature-math tests or reproducing Stage 6a).
FEATURE_SCALES: dict[int, float] = {
    17: 167.0,  # my pip count
    18: 167.0,  # opp pip count
    19: 1.0,    # pip ratio (already normalised)
    20: 15.0,   # my blots
    21: 15.0,   # opp blots
    22: 6.0,    # my anchors
    23: 6.0,    # opp anchors
    24: 167.0,  # my contact
    25: 167.0,  # opp contact
}


def resolve_channels(features: list[str] | None) -> list[int]:
    """Map a list of feature-group names to sorted channel indices.

    ``features=None`` (or a list containing ``"all"``) selects every channel,
    so existing callers that pass nothing are unaffected. ``"base"`` is always
    included. An empty list selects base-only. Unknown group names raise.
    """
    if features is None or "all" in features:
        return list(range(NUM_CHANNELS))
    unknown = [f for f in features if f not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(
            f"Unknown feature group(s) {unknown}; "
            f"valid groups: {sorted(FEATURE_GROUPS)} (or 'all')"
        )
    selected = set(FEATURE_GROUPS["base"])  # base is always included
    for f in features:
        selected.update(FEATURE_GROUPS[f])
    return sorted(selected)


def channels_for_network(config: dict) -> list[int] | None:
    """Channel indices a network expects at inference, from its checkpoint config.

    Newer checkpoints store ``feature_channels`` directly (the subset chosen at
    training time). Legacy checkpoints (pre Stage-6) store only ``in_channels``:
    ``NUM_CHANNELS`` (26) is the full Fix-N encoder (``None`` = all channels) and
    17 is base-only (no handcrafted features). Any other count is ambiguous and
    raises. The result is in the form ``encode_state(..., channels=...)`` wants,
    so callers can encode observations that match an arbitrary checkpoint —
    notably the 17-channel v5/iter_0447 nets now that the encoder defaults to 26.
    """
    fc = config.get("feature_channels")
    if fc is not None:
        return list(fc)
    in_ch = int(config.get("in_channels", NUM_CHANNELS))
    if in_ch == NUM_CHANNELS:
        return None
    if in_ch == len(FEATURE_GROUPS["base"]):
        return list(FEATURE_GROUPS["base"])
    raise ValueError(
        f"Cannot infer feature channels for in_channels={in_ch}: checkpoint "
        f"lacks 'feature_channels' and the count isn't {NUM_CHANNELS} (all) "
        f"or {len(FEATURE_GROUPS['base'])} (base)."
    )


def encode_state(
    board_view: BoardView,
    channels: list[int] | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Encode a board position as a (C, 2, 12) float32 tensor.

    Board layout:
        Top row (row 0): perspective points 13..24 -> columns 0..11
        Bottom row (row 1): perspective points 12..1 -> columns 0..11

    Channel meanings live in ``CHANNEL_NAMES`` so the audit/debug tooling
    and the encoder can't drift apart.

    All ``NUM_CHANNELS`` planes are always computed; ``channels`` (a list of
    channel indices, e.g. from ``resolve_channels``) optionally selects a
    subset, returning a ``(len(channels), 2, 12)`` tensor. ``None`` returns
    the full 26-channel tensor.

    ``normalize`` (default ``True``, i.e. Fix-N) divides the handcrafted
    feature channels by ``FEATURE_SCALES`` so they share the base planes'
    ~[0, 1] range; the base channels are untouched. Applied before any channel
    slicing. Pass ``normalize=False`` to recover the raw magnitudes (the
    pre-Stage-6 behaviour; needed for feature-math tests and for the Fix-B
    input-BatchNorm path, which standardises raw inputs itself).
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

    # --- Handcrafted features (computed raw here; scaled below if normalize) ---
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

    if normalize:
        for ch, scale in FEATURE_SCALES.items():
            tensor[ch] /= scale

    if channels is None:
        return tensor
    return tensor[channels]


def encode_batch(
    board_views: list[BoardView],
    channels: list[int] | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Encode multiple board positions. Returns shape (N, C, 2, 12).

    ``channels`` selects a channel subset (see ``encode_state``); ``None``
    yields the full 26-channel tensor. ``normalize`` (default ``True``)
    rescales the handcrafted channels into the base planes' range (see
    ``encode_state``).
    """
    return np.stack(
        [encode_state(bv, channels, normalize) for bv in board_views]
    )


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
