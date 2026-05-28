"""Tests for the wildbg Position ID decoder and CSV loader."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from raccoon.data.wildbg import (
    decode_position_id,
    equity_from_wildbg,
    load_wildbg_csv,
    load_wildbg_dir,
)
from raccoon.env.encoder import encode_state
from raccoon.env.game_wrapper import GameWrapper
from raccoon.model.network import RaccoonNet


# Known starting position from the GNUBG manual.
STARTING_POSITION_ID = "4HPwATDgc/ABMA"


def test_decode_starting_position():
    """Starting position has 5/3/5/2 on points 6/8/13/24, mirrored for opponent."""
    bv = decode_position_id(STARTING_POSITION_ID)

    expected_my = np.zeros(24, dtype=np.float32)
    expected_my[5] = 5   # point 6
    expected_my[7] = 3   # point 8
    expected_my[12] = 5  # point 13
    expected_my[23] = 2  # point 24
    assert np.array_equal(bv.my_points, expected_my)

    # Opponent's 6/8/13/24 from their perspective are my 19/17/12/1
    expected_opp = np.zeros(24, dtype=np.float32)
    expected_opp[18] = 5  # my point 19 (opp's 6 point)
    expected_opp[16] = 3  # my point 17 (opp's 8 point)
    expected_opp[11] = 5  # my point 12 (opp's 13 point, midpoint mirror)
    expected_opp[0] = 2   # my point 1  (opp's 24 point, in my home)
    assert np.array_equal(bv.opp_points, expected_opp)

    assert bv.my_bar == 0 and bv.opp_bar == 0
    assert bv.my_off == 0 and bv.opp_off == 0
    assert bv.dice is None
    assert bv.mid_doubles is False


def test_decode_matches_openspiel_starting_position():
    """Decoded starting position must match what OpenSpiel produces."""
    wrapper = GameWrapper()
    state = wrapper.new_game()
    state.apply_action(state.chance_outcomes()[0][0])
    osp = state.board_from_perspective()

    bv = decode_position_id(STARTING_POSITION_ID)
    assert np.array_equal(bv.my_points, osp.my_points)
    assert np.array_equal(bv.opp_points, osp.opp_points)
    assert bv.my_bar == osp.my_bar and bv.opp_bar == osp.opp_bar
    assert bv.my_off == osp.my_off and bv.opp_off == osp.opp_off


def test_decode_rejects_malformed_id():
    with pytest.raises(ValueError):
        decode_position_id("tooshort")


def test_decode_checker_count_invariant():
    """A real-data position must have <= 15 checkers per side total."""
    bv = decode_position_id(STARTING_POSITION_ID)
    my_total = int(bv.my_points.sum()) + bv.my_bar + bv.my_off
    opp_total = int(bv.opp_points.sum()) + bv.opp_bar + bv.opp_off
    assert my_total == 15
    assert opp_total == 15


def test_equity_extremes():
    assert equity_from_wildbg(1.0, 1.0, 1.0, 0.0, 0.0) == pytest.approx(1.0)
    assert equity_from_wildbg(0.0, 0.0, 0.0, 1.0, 1.0) == pytest.approx(-1.0)
    assert equity_from_wildbg(1.0, 0.0, 0.0, 0.0, 0.0) == pytest.approx(1 / 3)
    assert equity_from_wildbg(0.0, 0.0, 0.0, 0.0, 0.0) == pytest.approx(-1 / 3)


def test_equity_symmetric_50_50():
    """Pure 50/50 (no gammon/bg on either side) should give zero equity."""
    assert equity_from_wildbg(0.5, 0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0)


def test_csv_loader_roundtrip(tmp_path: Path):
    """Loader parses a small CSV correctly."""
    csv_path = tmp_path / "tiny.csv"
    csv_path.write_text(
        "position_id,win,win_g,win_bg,lose_g,lose_bg\n"
        f"{STARTING_POSITION_ID},0.51,0.13,0.005,0.15,0.007\n"
        f"{STARTING_POSITION_ID},1.0,0.0,0.0,0.0,0.0\n"
    )
    rows = load_wildbg_csv(csv_path)
    assert len(rows) == 2

    bv0, v0 = rows[0]
    assert bv0.my_off == 0 and bv0.opp_off == 0
    # equity = (0.51 - 0.49) + (0.13 + 0.005) - (0.15 + 0.007)
    # = 0.02 + 0.135 - 0.157 = -0.002, divided by 3 = -0.00067
    assert v0 == pytest.approx(((0.51 - 0.49) + 0.135 - 0.157) / 3.0)

    _, v1 = rows[1]
    assert v1 == pytest.approx(1 / 3)


def test_load_wildbg_dir(tmp_path: Path):
    """``load_wildbg_dir`` aggregates every CSV under a directory tree."""
    (tmp_path / "0021").mkdir()
    (tmp_path / "0022").mkdir()
    header = "position_id,win,win_g,win_bg,lose_g,lose_bg\n"
    row = f"{STARTING_POSITION_ID},0.5,0,0,0,0\n"
    (tmp_path / "0021" / "contact.csv").write_text(header + row + row)
    (tmp_path / "0022" / "race.csv").write_text(header + row)

    rows = load_wildbg_dir(tmp_path)
    assert len(rows) == 3


def test_load_wildbg_dir_no_csvs(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_wildbg_dir(tmp_path)


def test_decoded_position_encodes_to_tensor():
    bv = decode_position_id(STARTING_POSITION_ID)
    tensor = encode_state(bv)
    assert tensor.shape == (17, 2, 12)
    assert tensor.dtype == np.float32
    assert np.all(np.isfinite(tensor))
    # Dice/doubles channels (13-16) must be zero for pre-roll positions
    assert np.all(tensor[13] == 0.0)
    assert np.all(tensor[14] == 0.0)
    assert np.all(tensor[15] == 0.0)
    assert np.all(tensor[16] == 0.0)


def test_forward_pass_with_decoded_position():
    bv = decode_position_id(STARTING_POSITION_ID)
    tensor = encode_state(bv)
    x = torch.from_numpy(tensor).unsqueeze(0)
    net = RaccoonNet()
    net.eval()
    with torch.no_grad():
        logits, value = net(x)
    assert logits.shape == (1, 1352)
    assert value.shape == (1, 1)
    assert torch.isfinite(value).all()
