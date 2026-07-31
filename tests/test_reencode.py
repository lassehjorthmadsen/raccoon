"""Tests for decode_base_planes / cache re-encoding (17-ch -> 26-ch)."""

import numpy as np
import pytest

from raccoon.env.encoder import (
    FEATURE_GROUPS,
    NUM_CHANNELS,
    decode_base_planes,
    encode_state,
)
from raccoon.env.game_wrapper import BoardView, GameWrapper
from raccoon.search.mcts import _advance_through_chance

BASE = FEATURE_GROUPS["base"]


def _random_game_views(n_games=3, seed=0):
    """BoardViews from every decision node of a few random games.

    Random play naturally visits bar/off/overflow states and (via doubles
    rolls) mid-doubles half-turns, which is the coverage the decoder needs.
    """
    rng = np.random.default_rng(seed)
    wrapper = GameWrapper()
    views = []
    for _ in range(n_games):
        state = _advance_through_chance(wrapper.new_game())
        while not state.is_terminal():
            views.append(state.board_from_perspective())
            legal = state.legal_actions()
            state.apply_action(legal[rng.integers(len(legal))])
            state = _advance_through_chance(state)
    return views


def _views_equal(a: BoardView, b: BoardView) -> bool:
    return (
        np.array_equal(a.my_points, b.my_points)
        and np.array_equal(a.opp_points, b.opp_points)
        and a.my_bar == b.my_bar and a.opp_bar == b.opp_bar
        and a.my_off == b.my_off and a.opp_off == b.opp_off
        and a.dice == b.dice and a.mid_doubles == b.mid_doubles
    )


def test_roundtrip_full_games():
    views = _random_game_views()
    assert len(views) > 50
    saw_mid_doubles = saw_bar = saw_overflow = False
    for view in views:
        full = encode_state(view)                # (26, 2, 12)
        base = full[BASE]                        # what a 17-ch cache stores
        decoded = decode_base_planes(base)
        assert _views_equal(view, decoded)
        # Re-encoding the decoded view must exactly reproduce both widths.
        assert np.array_equal(encode_state(decoded), full)
        assert np.array_equal(encode_state(decoded)[BASE], base)
        # Decoding the full 26-ch tensor must work too (extra planes ignored).
        assert _views_equal(view, decode_base_planes(full))
        saw_mid_doubles |= view.mid_doubles
        saw_bar |= view.my_bar > 0 or view.opp_bar > 0
        saw_overflow |= bool(np.any(view.my_points > 3) or np.any(view.opp_points > 3))
    assert saw_mid_doubles and saw_bar and saw_overflow


def test_roundtrip_no_dice():
    """Pre-roll positions (wildbg-style, dice=None) survive the round trip."""
    view = BoardView(
        my_points=np.zeros(24, dtype=np.int64),
        opp_points=np.zeros(24, dtype=np.int64),
        my_bar=2, opp_bar=0, my_off=13, opp_off=15,
        dice=None,
    )
    decoded = decode_base_planes(encode_state(view)[BASE])
    assert _views_equal(view, decoded)


def test_decode_rejects_bad_shape():
    with pytest.raises(ValueError):
        decode_base_planes(np.zeros((10, 2, 12), dtype=np.float32))
    with pytest.raises(ValueError):
        decode_base_planes(np.zeros((NUM_CHANNELS, 2, 11), dtype=np.float32))
