"""Tests for the ResNet policy-value network."""

import os
import tempfile

import numpy as np
import torch
import pytest

from raccoon.model.network import (
    RaccoonNet,
    save_checkpoint,
    load_checkpoint,
    load_model,
)


@pytest.fixture
def model():
    return RaccoonNet()


def test_forward_shapes(model):
    x = torch.randn(1, 26, 2, 12)
    logits, value = model(x)
    assert logits.shape == (1, 1352)
    assert value.shape == (1, 1)


def test_forward_batch(model):
    x = torch.randn(32, 26, 2, 12)
    logits, value = model(x)
    assert logits.shape == (32, 1352)
    assert value.shape == (32, 1)


def test_value_range(model):
    x = torch.randn(10, 26, 2, 12)
    _, value = model(x)
    assert (value >= -1).all()
    assert (value <= 1).all()


def test_predict_probabilities(model):
    obs = np.random.randn(26, 2, 12).astype(np.float32)
    legal_actions = [0, 10, 100, 500, 1000]
    policy, value = model.predict(obs, legal_actions)

    assert set(policy.keys()) == set(legal_actions)
    assert abs(sum(policy.values()) - 1.0) < 1e-5
    assert all(p >= 0 for p in policy.values())
    assert -1 <= value <= 1


def test_predict_only_legal_actions(model):
    obs = np.random.randn(26, 2, 12).astype(np.float32)
    legal_actions = [42, 99]
    policy, _ = model.predict(obs, legal_actions)

    assert len(policy) == 2
    for a in range(1352):
        if a not in legal_actions:
            assert a not in policy


def test_save_load_roundtrip(model):
    obs = np.random.randn(26, 2, 12).astype(np.float32)
    legal = [0, 1, 2]
    policy_before, value_before = model.predict(obs, legal)

    optimizer = torch.optim.Adam(model.parameters())

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name

    try:
        save_checkpoint(model, optimizer, step=42, path=path)

        model2 = RaccoonNet()
        checkpoint = load_checkpoint(path, model2)
        assert checkpoint["step"] == 42

        policy_after, value_after = model2.predict(obs, legal)
        for a in legal:
            assert abs(policy_before[a] - policy_after[a]) < 1e-6
        assert abs(value_before - value_after) < 1e-6
    finally:
        os.unlink(path)


def test_parameter_count(model):
    total = sum(p.numel() for p in model.parameters())
    # Should be in the hundreds of thousands to low millions
    assert 100_000 < total < 10_000_000
    print(f"Model parameter count: {total:,}")


def test_feature_channels_sets_in_channels():
    """feature_channels overrides in_channels and is stored in config."""
    sel = list(range(17)) + [17, 18, 19]  # base + pip
    net = RaccoonNet(feature_channels=sel)
    assert net.config["in_channels"] == len(sel)
    assert net.config["feature_channels"] == sel
    assert net.feature_channels == sel
    # Forward pass accepts the matching channel count
    x = torch.randn(2, len(sel), 2, 12)
    logits, value = net(x)
    assert logits.shape == (2, 1352)
    assert value.shape == (2, 1)


def test_default_model_has_no_feature_channels(model):
    """Full-26 default keeps feature_channels None (back-compat for old ckpts)."""
    assert model.config["in_channels"] == 26
    assert model.config["feature_channels"] is None


def test_load_model_roundtrips_feature_channels():
    sel = list(range(17)) + [20, 21]  # base + blots = 19ch
    net = RaccoonNet(channels=32, num_blocks=2, feature_channels=sel)
    optimizer = torch.optim.Adam(net.parameters())
    obs = np.random.randn(len(sel), 2, 12).astype(np.float32)
    legal = [0, 1, 2]
    policy_before, value_before = net.predict(obs, legal)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        save_checkpoint(net, optimizer, step=1, path=path)
        net2 = load_model(path)  # reconstructs from config alone
        assert net2.feature_channels == sel
        assert net2.config["in_channels"] == len(sel)
        policy_after, value_after = net2.predict(obs, legal)
        assert abs(value_before - value_after) < 1e-6
        for a in legal:
            assert abs(policy_before[a] - policy_after[a]) < 1e-6
    finally:
        os.unlink(path)


def test_input_bn_adds_input_norm_layer():
    """input_bn=True attaches a BatchNorm over raw inputs and is stored in config."""
    net = RaccoonNet(input_bn=True)
    assert net.config["input_bn"] is True
    assert net.input_norm is not None
    assert net.input_norm.num_features == 26
    x = torch.randn(4, 26, 2, 12)
    logits, value = net(x)
    assert logits.shape == (4, 1352)
    assert value.shape == (4, 1)


def test_default_model_has_no_input_norm(model):
    assert model.config["input_bn"] is False
    assert model.input_norm is None


def test_input_bn_with_feature_subset_roundtrips():
    sel = list(range(17)) + [17, 18, 19]  # base + pip = 20ch
    net = RaccoonNet(channels=32, num_blocks=2,
                     feature_channels=sel, input_bn=True)
    assert net.input_norm.num_features == len(sel)
    optimizer = torch.optim.Adam(net.parameters())
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        save_checkpoint(net, optimizer, step=1, path=path)
        net2 = load_model(path)
        assert net2.config["input_bn"] is True
        assert net2.input_norm is not None
        assert net2.input_norm.num_features == len(sel)
    finally:
        os.unlink(path)
