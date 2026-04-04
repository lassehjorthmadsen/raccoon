"""Tests for the ResNet policy-value network."""

import os
import tempfile

import numpy as np
import torch
import pytest

from raccoon.model.network import RaccoonNet, save_checkpoint, load_checkpoint


@pytest.fixture
def model():
    return RaccoonNet()


def test_forward_shapes(model):
    x = torch.randn(1, 16, 2, 12)
    logits, value = model(x)
    assert logits.shape == (1, 1352)
    assert value.shape == (1, 1)


def test_forward_batch(model):
    x = torch.randn(32, 16, 2, 12)
    logits, value = model(x)
    assert logits.shape == (32, 1352)
    assert value.shape == (32, 1)


def test_value_range(model):
    x = torch.randn(10, 16, 2, 12)
    _, value = model(x)
    assert (value >= -1).all()
    assert (value <= 1).all()


def test_predict_probabilities(model):
    obs = np.random.randn(16, 2, 12).astype(np.float32)
    legal_actions = [0, 10, 100, 500, 1000]
    policy, value = model.predict(obs, legal_actions)

    assert set(policy.keys()) == set(legal_actions)
    assert abs(sum(policy.values()) - 1.0) < 1e-5
    assert all(p >= 0 for p in policy.values())
    assert -1 <= value <= 1


def test_predict_only_legal_actions(model):
    obs = np.random.randn(16, 2, 12).astype(np.float32)
    legal_actions = [42, 99]
    policy, _ = model.predict(obs, legal_actions)

    assert len(policy) == 2
    for a in range(1352):
        if a not in legal_actions:
            assert a not in policy


def test_save_load_roundtrip(model):
    obs = np.random.randn(16, 2, 12).astype(np.float32)
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
