"""Tests for the board tensor encoder."""

import numpy as np
import pytest

from raccoon.env.encoder import (
    CHANNEL_NAMES,
    FEATURE_GROUPS,
    NUM_CHANNELS,
    dump_tensor,
    encode_batch,
    encode_state,
    resolve_channels,
)
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
    assert tensor.shape == (26, 2, 12)
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
    assert np.all(tensor[16] == 0.0)  # not mid-doubles


def test_encode_mid_doubles():
    bv = BoardView(
        my_points=np.zeros(24),
        opp_points=np.zeros(24),
        my_bar=0, opp_bar=0, my_off=0, opp_off=0,
        dice=(4, 4),
        mid_doubles=True,
    )
    tensor = encode_state(bv)
    assert np.all(tensor[15] == 1.0)  # doubles flag
    assert np.all(tensor[16] == 1.0)  # mid-doubles flag


def test_encode_mid_doubles_false_by_default():
    bv = BoardView(
        my_points=np.zeros(24),
        opp_points=np.zeros(24),
        my_bar=0, opp_bar=0, my_off=0, opp_off=0,
        dice=(3, 3),
    )
    tensor = encode_state(bv)
    assert np.all(tensor[16] == 0.0)


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
    assert batch.shape == (2, 26, 2, 12)


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


def test_channel_names_match_tensor_depth():
    """CHANNEL_NAMES must stay in lockstep with the tensor's first axis."""
    assert NUM_CHANNELS == 26
    assert len(CHANNEL_NAMES) == NUM_CHANNELS


def test_dump_tensor_includes_all_channels(starting_board):
    out = dump_tensor(starting_board)
    for ch in range(NUM_CHANNELS):
        assert f"Channel {ch:2d}" in out
    for name in CHANNEL_NAMES:
        assert name in out


def test_dump_tensor_reports_broadcast_for_constant_planes(starting_board):
    out = dump_tensor(starting_board)
    # Opening position: all 18 broadcast channels (8..25) are constant by
    # construction, and all 8 spatial channels have varied values, so
    # exactly 18 planes should be tagged.
    assert out.count("(broadcast)") == 18


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


# --- Handcrafted feature tests ---


def test_encode_pip_count_starting_position(starting_board):
    """Starting position has 167 pips for both sides."""
    tensor = encode_state(starting_board, normalize=False)
    assert tensor[17, 0, 0] == pytest.approx(167.0, abs=0.1)
    assert tensor[18, 0, 0] == pytest.approx(167.0, abs=0.1)
    # Ratio should be 0.5 when both sides are equal
    assert tensor[19, 0, 0] == pytest.approx(0.5, abs=0.01)


def test_encode_pip_count_bar():
    """Bar checkers contribute 25 pips each."""
    bv = BoardView(
        my_points=np.zeros(24), opp_points=np.zeros(24),
        my_bar=2, opp_bar=0, my_off=0, opp_off=0, dice=(3, 1),
    )
    tensor = encode_state(bv, normalize=False)
    assert tensor[17, 0, 0] == pytest.approx(50.0)  # 2 * 25
    assert tensor[18, 0, 0] == pytest.approx(0.0)


def test_encode_pip_ratio_all_zero():
    """When both pip counts are zero, ratio should be 0.5."""
    bv = BoardView(
        my_points=np.zeros(24), opp_points=np.zeros(24),
        my_bar=0, opp_bar=0, my_off=15, opp_off=15, dice=(3, 1),
    )
    tensor = encode_state(bv)
    assert tensor[19, 0, 0] == pytest.approx(0.5)


def test_encode_blots():
    """Count of exposed single checkers."""
    my_points = np.zeros(24)
    my_points[0] = 1   # blot
    my_points[5] = 1   # blot
    my_points[10] = 2  # not a blot
    opp_points = np.zeros(24)
    opp_points[3] = 1  # blot
    bv = BoardView(
        my_points=my_points, opp_points=opp_points,
        my_bar=0, opp_bar=0, my_off=0, opp_off=0, dice=(3, 1),
    )
    tensor = encode_state(bv, normalize=False)
    assert tensor[20, 0, 0] == pytest.approx(2.0)  # my blots
    assert tensor[21, 0, 0] == pytest.approx(1.0)  # opp blots


def test_encode_blots_starting_position(starting_board):
    """Starting position has no blots."""
    tensor = encode_state(starting_board)
    assert tensor[20, 0, 0] == pytest.approx(0.0)
    assert tensor[21, 0, 0] == pytest.approx(0.0)


def test_encode_anchors():
    """Anchors are made points (>=2) in opponent's home board."""
    my_points = np.zeros(24)
    my_points[18] = 2  # anchor in opp's home
    my_points[20] = 3  # anchor in opp's home
    my_points[5] = 2   # NOT an anchor (in my outer board)
    opp_points = np.zeros(24)
    opp_points[2] = 2  # anchor in my home
    opp_points[18] = 4  # NOT an anchor (in opp's outer board from their view)
    bv = BoardView(
        my_points=my_points, opp_points=opp_points,
        my_bar=0, opp_bar=0, my_off=0, opp_off=0, dice=(3, 1),
    )
    tensor = encode_state(bv, normalize=False)
    assert tensor[22, 0, 0] == pytest.approx(2.0)  # my anchors
    assert tensor[23, 0, 0] == pytest.approx(1.0)  # opp anchors


def test_encode_contact_pure_race():
    """Pure race: all my checkers already past all opp checkers — contact = 0."""
    my_points = np.zeros(24)
    my_points[0] = 5
    my_points[1] = 5
    my_points[2] = 5
    opp_points = np.zeros(24)
    opp_points[20] = 5
    opp_points[21] = 5
    opp_points[22] = 5
    bv = BoardView(
        my_points=my_points, opp_points=opp_points,
        my_bar=0, opp_bar=0, my_off=0, opp_off=0, dice=(3, 1),
    )
    tensor = encode_state(bv)
    assert tensor[24, 0, 0] == pytest.approx(0.0)  # my contact
    assert tensor[25, 0, 0] == pytest.approx(0.0)  # opp contact


def test_encode_contact_simple():
    """2 checkers each, 1 pip apart: 4 pips to break contact."""
    my_points = np.zeros(24)
    my_points[5] = 2
    opp_points = np.zeros(24)
    opp_points[4] = 2  # opp's least-advanced at index 4
    bv = BoardView(
        my_points=my_points, opp_points=opp_points,
        my_bar=0, opp_bar=0, my_off=0, opp_off=0, dice=(3, 1),
    )
    tensor = encode_state(bv, normalize=False)
    # my_contact: Σ my[i] × max(0, i - min_opp + 1) = 2 × (5-4+1) = 4
    assert tensor[24, 0, 0] == pytest.approx(4.0)
    # opp_contact: Σ opp[j] × max(0, max_my - j + 1) = 2 × (5-4+1) = 4
    assert tensor[25, 0, 0] == pytest.approx(4.0)


def test_encode_contact_asymmetric():
    """Asymmetric: I have one back checker, opp is all home — contact differs."""
    my_points = np.zeros(24)
    my_points[20] = 1   # one straggler deep in opp territory
    my_points[0] = 14   # rest safely home
    opp_points = np.zeros(24)
    opp_points[23] = 5  # opp fully home (high indices)
    opp_points[22] = 5
    opp_points[21] = 5
    bv = BoardView(
        my_points=my_points, opp_points=opp_points,
        my_bar=0, opp_bar=0, my_off=0, opp_off=0, dice=(3, 1),
    )
    tensor = encode_state(bv)
    # min_opp = 21, max_my = 20
    # my_contact: only my[20] has i=20 < min_opp=21, so max(0, 20-21+1)=0 → 0
    assert tensor[24, 0, 0] == pytest.approx(0.0)
    # opp_contact: max_my=20; opp at 21,22,23: max(0, 20-21+1)=0, etc → 0
    assert tensor[25, 0, 0] == pytest.approx(0.0)


def test_encode_contact_starting_position(starting_board):
    """Starting position: contact equals pip count (opp sits on ace point)."""
    tensor = encode_state(starting_board, normalize=False)
    # min_opp = 0 (opp has 2 on the ace point), so my_contact = my pip count = 167
    assert tensor[24, 0, 0] == pytest.approx(167.0)
    assert tensor[25, 0, 0] == pytest.approx(167.0)  # symmetric


def test_encode_contact_bar_increases_contact():
    """Bar checkers affect both min_opp and opp_contact contribution."""
    my_points = np.zeros(24)
    my_points[10] = 2  # max_my = 10
    opp_points = np.zeros(24)
    opp_points[5] = 2  # min_opp = 5 (on board, no bar)
    # Without bar: my_contact = 2 × (10-5+1) = 12; opp_contact = 2 × (10-5+1) = 12
    bv_no_bar = BoardView(my_points=my_points, opp_points=opp_points,
                          my_bar=0, opp_bar=0, my_off=0, opp_off=0, dice=(3, 1))
    t_no_bar = encode_state(bv_no_bar, normalize=False)
    assert t_no_bar[24, 0, 0] == pytest.approx(12.0)
    assert t_no_bar[25, 0, 0] == pytest.approx(12.0)

    # With 1 opp checker on bar:
    #   min_opp drops to 0 (bar re-enters on home board)
    #   my_contact: 2 × (10-0+1) = 22
    #   opp_contact: board 2×(10-5+1)=12, bar 1×(10+2)=12, total=24
    bv_bar = BoardView(my_points=my_points, opp_points=opp_points,
                       my_bar=0, opp_bar=1, my_off=0, opp_off=0, dice=(3, 1))
    t_bar = encode_state(bv_bar, normalize=False)
    assert t_bar[24, 0, 0] == pytest.approx(22.0)
    assert t_bar[25, 0, 0] == pytest.approx(24.0)


# --- Channel-subset selection (feature ablation) ---

def test_feature_groups_partition_all_channels():
    """The feature groups together must cover exactly the 26 channels once."""
    covered = sorted(i for idxs in FEATURE_GROUPS.values() for i in idxs)
    assert covered == list(range(NUM_CHANNELS))


def test_resolve_channels_none_and_all():
    full = list(range(NUM_CHANNELS))
    assert resolve_channels(None) == full
    assert resolve_channels(["all"]) == full


def test_resolve_channels_base_only():
    assert resolve_channels([]) == list(range(17))


def test_resolve_channels_pip_appends_to_base():
    assert resolve_channels(["pip"]) == list(range(17)) + [17, 18, 19]


def test_resolve_channels_is_sorted_and_deduped():
    # base is implied; passing it explicitly must not duplicate it
    assert resolve_channels(["contact", "base"]) == list(range(17)) + [24, 25]


def test_resolve_channels_unknown_raises():
    with pytest.raises(ValueError):
        resolve_channels(["bogus"])


def test_encode_state_subset_shape_and_values(starting_board):
    sel = resolve_channels(["pip"])  # 20 channels
    full = encode_state(starting_board)
    subset = encode_state(starting_board, sel)
    assert subset.shape == (len(sel), 2, 12)
    # Selected planes are identical to the corresponding full-encoder planes
    for out_idx, ch in enumerate(sel):
        assert np.array_equal(subset[out_idx], full[ch])


def test_encode_batch_subset_shape(starting_board):
    sel = [0, 1, 17]
    batch = encode_batch([starting_board, starting_board], sel)
    assert batch.shape == (2, 3, 2, 12)


def test_normalize_scales_handcrafted_into_unit_range(starting_board):
    """normalize divides handcrafted channels by FEATURE_SCALES; base untouched."""
    from raccoon.env.encoder import FEATURE_SCALES

    raw = encode_state(starting_board, normalize=False)
    normed = encode_state(starting_board, normalize=True)
    # Base channels (0..16) are byte-for-byte identical.
    for ch in range(17):
        assert np.array_equal(raw[ch], normed[ch])
    # Handcrafted channels are exactly raw / scale.
    for ch, scale in FEATURE_SCALES.items():
        assert np.allclose(normed[ch], raw[ch] / scale)
    # And all land in a sane ~[0, 2] range at the opening position.
    assert normed[17:].max() < 2.0


def test_normalize_composes_with_channel_subset(starting_board):
    sel = resolve_channels(["pip"])  # 20 channels incl. 17,18,19
    full_normed = encode_state(starting_board, normalize=True)
    subset_normed = encode_state(starting_board, sel, normalize=True)
    assert subset_normed.shape == (len(sel), 2, 12)
    for out_idx, ch in enumerate(sel):
        assert np.array_equal(subset_normed[out_idx], full_normed[ch])


def test_encode_batch_normalize(starting_board):
    batch = encode_batch([starting_board], normalize=True)
    single = encode_state(starting_board, normalize=True)
    assert np.array_equal(batch[0], single)
