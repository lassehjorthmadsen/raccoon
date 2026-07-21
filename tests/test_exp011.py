"""Tests for exp011 — GNUBG-0-ply value distillation (multi-component head)."""
import glob

import numpy as np
import pytest
import torch

from raccoon.model.network import RaccoonNet, load_model
from raccoon.train.lookahead import eval_values_batch

CPU = torch.device("cpu")
WEIGHTS = torch.tensor([1.0, 2.0, 3.0, -1.0, -2.0, -3.0])


def _small(head):
    net = RaccoonNet(channels=8, num_blocks=1, value_head=head)
    net.eval()
    return net


def test_outcomes6_conversion_and_equity():
    # a plausible cumulative five-tuple (win, wg, wbg, lg, lbg)
    win, wg, wbg, lg, lbg = 0.60, 0.20, 0.05, 0.15, 0.03
    lose = 1 - win
    six = np.array([win - wg, wg - wbg, wbg, lose - lg, lg - lbg, lbg])
    assert six.min() >= 0
    assert abs(six.sum() - 1.0) < 1e-9
    equity_formula = win + wg + wbg - (1 - win) - lg - lbg
    equity_dist = six @ np.array([1, 2, 3, -1, -2, -3])
    assert abs(equity_formula - equity_dist) < 1e-9


def test_value_equity_scalar_matches_forward():
    net = _small("scalar")
    x = torch.randn(4, 26, 2, 12)
    with torch.no_grad():
        _, vout = net(x)
        ve = net.value_equity(x)
    assert vout.shape == (4, 1)
    assert torch.allclose(ve, vout.squeeze(-1))
    assert (ve.abs() <= 1.0).all()


def test_value_equity_outcomes6_derives_equity():
    net = _small("outcomes6")
    x = torch.randn(4, 26, 2, 12)
    with torch.no_grad():
        _, vout = net(x)
        ve = net.value_equity(x)
    assert vout.shape == (4, 6)
    assert ve.shape == (4,)
    assert (ve.abs() <= 1.0 + 1e-6).all()
    manual = (torch.softmax(vout, dim=-1) @ WEIGHTS) / 3.0
    assert torch.allclose(ve, manual, atol=1e-6)


def test_eval_values_batch_handles_outcomes6():
    net = _small("outcomes6")
    obs = np.random.randn(5, 26, 2, 12).astype(np.float32)
    vals = eval_values_batch(net, obs, CPU)
    assert vals.shape == (5,)
    assert np.isfinite(vals).all() and (np.abs(vals) <= 1.0 + 1e-6).all()


def test_outcomes6_config_roundtrips(tmp_path):
    net = RaccoonNet(channels=8, num_blocks=1, value_head="outcomes6")
    p = tmp_path / "n.pt"
    torch.save({"model_state_dict": net.state_dict(), "config": net.config}, p)
    net2 = load_model(str(p))
    assert net2.value_head == "outcomes6"
    assert net2.value_fc2.out_features == 6


def test_scalar_is_backward_compatible():
    # a config from before value_head existed reconstructs as scalar
    net = RaccoonNet(channels=8, num_blocks=1)
    cfg = dict(net.config)
    cfg.pop("value_head")
    net2 = RaccoonNet(**cfg)
    assert net2.value_head == "scalar"
    assert net2.value_fc2.out_features == 1


def test_generator_smoke(tmp_path):
    pytest.importorskip("gnubg_nn")
    from scripts.gen_gnubg_selfplay import _worker
    written = _worker(0, 40, 100, str(tmp_path), seed_base=0)
    assert written >= 40
    shards = glob.glob(str(tmp_path / "shard_*.npz"))
    assert shards
    z = np.load(shards[0])
    assert z["observations"].shape[1:] == (26, 2, 12)
    assert z["outcomes6"].shape[1] == 6
    assert abs(float(z["outcomes6"][0].sum()) - 1.0) < 1e-4
    assert (z["equity"] >= -1.0).all() and (z["equity"] <= 1.0).all()
