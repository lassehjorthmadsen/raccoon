"""Circular replay buffer for training positions."""

import random
from collections import deque

import numpy as np
import torch

from raccoon.train.self_play import TrainingExample


class ReplayBuffer:
    """Stores training positions and samples random batches."""

    def __init__(self, max_size: int = 100_000):
        self._buffer: deque[TrainingExample] = deque(maxlen=max_size)

    def add_game(self, examples: list[TrainingExample]) -> None:
        self._buffer.extend(examples)

    def sample_batch(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random batch as PyTorch tensors.

        Returns:
            obs_batch: (batch, 16, 2, 12)
            policy_batch: (batch, 1352)
            value_batch: (batch,)
        """
        samples = random.sample(list(self._buffer), min(batch_size, len(self._buffer)))

        obs = np.stack([s.observation for s in samples])
        policy = np.stack([s.policy_target for s in samples])
        value = np.array([s.value_target for s in samples], dtype=np.float32)

        return (
            torch.from_numpy(obs),
            torch.from_numpy(policy),
            torch.from_numpy(value),
        )

    def __len__(self) -> int:
        return len(self._buffer)
