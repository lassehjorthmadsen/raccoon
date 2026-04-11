"""Tests for the training orchestrator."""

import json
import tempfile
from pathlib import Path

import torch
import pytest

from raccoon.model.network import RaccoonNet
from raccoon.train.coach import Coach
from raccoon.train.replay_buffer import ReplayBuffer


@pytest.fixture
def tmp_dirs(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    log_dir = tmp_path / "logs"
    return str(checkpoint_dir), str(log_dir)


@pytest.fixture
def coach(tmp_dirs):
    network = RaccoonNet()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001, weight_decay=1e-4)
    replay_buffer = ReplayBuffer(max_size=10_000)
    checkpoint_dir, log_dir = tmp_dirs
    return Coach(
        network=network,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        num_simulations=5,
        batch_size=32,
        games_per_iteration=2,
        training_steps_per_iteration=5,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        checkpoint_every=1,
    )


def test_run_iteration(coach):
    metrics = coach.run_iteration(0)
    assert metrics["iteration"] == 0
    assert metrics["num_games"] == 2
    assert metrics["num_positions"] > 0
    assert isinstance(metrics["total_loss"], float)


def test_checkpoint_saved(coach, tmp_dirs):
    coach.run_iteration(0)
    checkpoint_dir = Path(tmp_dirs[0])
    assert (checkpoint_dir / "iter_0000.pt").exists()


def test_log_written(coach, tmp_dirs):
    coach.run_iteration(0)
    log_path = Path(tmp_dirs[1]) / "training_log.jsonl"
    assert log_path.exists()
    with open(log_path) as f:
        lines = f.readlines()
    # First line is config header, second is iteration 0 metrics
    config = json.loads(lines[0])
    assert config["type"] == "config"
    assert "network" in config
    entry = json.loads(lines[1])
    assert entry["iteration"] == 0


def test_loss_is_finite(coach):
    # First iteration fills buffer, second iteration trains
    coach.run_iteration(0)
    metrics = coach.run_iteration(1)
    assert metrics["total_loss"] > 0
    assert metrics["policy_loss"] > 0 or metrics["value_loss"] > 0
