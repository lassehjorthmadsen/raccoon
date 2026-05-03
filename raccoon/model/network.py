"""ResNet policy-value network for backgammon."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> skip add -> ReLU."""

    def __init__(self, channels: int):
        super().__init__()
        # Use padding=(1,1) for 3x3 convs to preserve spatial dims
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class RaccoonNet(nn.Module):
    """ResNet with policy and value heads for backgammon."""

    def __init__(
        self,
        in_channels: int = 17,
        board_h: int = 2,
        board_w: int = 12,
        num_actions: int = 1352,
        channels: int = 128,
        num_blocks: int = 6,
    ):
        super().__init__()
        self.config = {
            "channels": channels,
            "num_blocks": num_blocks,
            "in_channels": in_channels,
            "board_h": board_h,
            "board_w": board_w,
            "num_actions": num_actions,
        }
        self.board_h = board_h
        self.board_w = board_w
        self.num_actions = num_actions

        # Input convolution
        self.input_conv = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.input_bn = nn.BatchNorm2d(channels)

        # Residual trunk
        self.trunk = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(channels, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_h * board_w, num_actions)

        # Value head
        self.value_conv = nn.Conv2d(channels, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * board_h * board_w, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, 16, 2, 12)

        Returns:
            policy_logits: (batch, 1352) raw logits (not masked)
            value: (batch, 1) in [-1, 1]
        """
        # Shared trunk
        out = F.relu(self.input_bn(self.input_conv(x)))
        out = self.trunk(out)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def predict(
        self, obs: np.ndarray, legal_actions: list[int]
    ) -> tuple[dict[int, float], float]:
        """Single-position inference for MCTS.

        Args:
            obs: (17, 2, 12) numpy array
            legal_actions: list of valid action indices

        Returns:
            policy: dict mapping action -> probability (sums to ~1, only legal)
            value: scalar float in [-1, 1]
        """
        self.eval()
        x = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
        logits, value = self.forward(x)

        # Mask illegal actions and apply softmax
        logits = logits.squeeze(0).cpu()
        mask = torch.full((self.num_actions,), float("-inf"))
        mask[legal_actions] = 0.0
        probs = F.softmax(logits + mask, dim=0).numpy()

        policy = {a: float(probs[a]) for a in legal_actions}
        return policy, float(value.item())

    @torch.no_grad()
    def predict_batch(
        self, obs_list: list[np.ndarray], legal_actions_list: list[list[int]],
    ) -> list[tuple[dict[int, float], float]]:
        """Batched inference for multiple positions.

        Args:
            obs_list: list of (17, 2, 12) numpy arrays
            legal_actions_list: list of legal action lists

        Returns:
            list of (policy_dict, value) tuples
        """
        self.eval()
        obs_np = np.stack(obs_list)
        x = torch.from_numpy(obs_np).float().to(self.device)
        logits_batch, values_batch = self.forward(x)
        logits_batch = logits_batch.cpu()
        values_batch = values_batch.cpu()

        results = []
        for i, legal_actions in enumerate(legal_actions_list):
            logits = logits_batch[i]
            mask = torch.full((self.num_actions,), float("-inf"))
            mask[legal_actions] = 0.0
            probs = F.softmax(logits + mask, dim=0).numpy()
            policy = {a: float(probs[a]) for a in legal_actions}
            value = float(values_batch[i].item())
            results.append((policy, value))
        return results


def save_checkpoint(
    model: RaccoonNet,
    optimizer: torch.optim.Optimizer,
    step: int,
    path: str,
    **extra,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "config": model.config,
            **extra,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: RaccoonNet,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def load_model(path: str) -> RaccoonNet:
    """Create a RaccoonNet from a checkpoint, using its saved config.

    Falls back to default config for older checkpoints without config.
    """
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    config = checkpoint.get("config", {})
    model = RaccoonNet(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
