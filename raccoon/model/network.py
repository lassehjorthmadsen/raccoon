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
        in_channels: int = 26,
        board_h: int = 2,
        board_w: int = 12,
        num_actions: int = 1352,
        channels: int = 128,
        num_blocks: int = 6,
        feature_channels: list[int] | None = None,
        input_bn: bool = False,
        value_head: str = "scalar",
    ):
        super().__init__()
        # When a channel subset is given, the input channel count is derived
        # from it so the two can never disagree. ``None`` keeps the full 26
        # channels (and old checkpoints without this key reconstruct as None).
        if feature_channels is not None:
            in_channels = len(feature_channels)
        # value_head: "scalar" -> tanh money-equity/3 in [-1, 1] (default; every
        # existing checkpoint reconstructs unchanged). "outcomes6" -> six logits
        # for the mutually-exclusive win/gammon/bg x win/lose outcomes (softmax);
        # equity is derived via value_equity(). See raccoon/train/lookahead.py.
        if value_head not in ("scalar", "outcomes6"):
            raise ValueError(f"value_head must be scalar|outcomes6, got {value_head}")
        self.value_head = value_head
        self.config = {
            "channels": channels,
            "num_blocks": num_blocks,
            "in_channels": in_channels,
            "board_h": board_h,
            "board_w": board_w,
            "num_actions": num_actions,
            "feature_channels": feature_channels,
            "input_bn": input_bn,
            "value_head": value_head,
        }
        self.feature_channels = feature_channels
        self.board_h = board_h
        self.board_w = board_w
        self.num_actions = num_actions

        # Optional input normalisation: a BatchNorm over the *raw* input
        # channels (run before the input conv) standardises each channel to
        # ~unit scale per batch. This is the architecture-side alternative to
        # normalising the handcrafted features in the encoder — see Stage 6 of
        # docs/pretraining_analysis.qmd. Off by default (old checkpoints whose
        # config lacks this key reconstruct as False).
        self.input_norm = nn.BatchNorm2d(in_channels) if input_bn else None

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

        # Value head (1 scalar output, or 6 outcome logits)
        self.value_conv = nn.Conv2d(channels, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * board_h * board_w, 256)
        self.value_fc2 = nn.Linear(256, 6 if value_head == "outcomes6" else 1)

    # Outcome points for the six mutually-exclusive outcomes, in the target
    # order [win_single, win_gammon, win_bg, lose_single, lose_gammon, lose_bg].
    _OUTCOME_POINTS = (1.0, 2.0, 3.0, -1.0, -2.0, -3.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, in_channels, 2, 12)

        Returns:
            policy_logits: (batch, 1352) raw logits (not masked)
            value: "scalar" head -> (batch, 1) tanh in [-1, 1];
                   "outcomes6" head -> (batch, 6) raw logits (softmax applied by
                   the caller / value_equity).
        """
        # Shared trunk
        if self.input_norm is not None:
            x = self.input_norm(x)
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
        v = self.value_fc2(v)
        value = v if self.value_head == "outcomes6" else torch.tanh(v)

        return policy_logits, value

    def _equity_from_value_out(self, value_out: torch.Tensor) -> torch.Tensor:
        """Map a raw value-head output to equity/3 in [-1, 1], shape (batch,).

        For "scalar" this is the tanh output itself; for "outcomes6" it is the
        softmax distribution dotted with the outcome points (±1/±2/±3), then /3
        to match the scalar head's money-equity/3 convention.
        """
        if self.value_head == "outcomes6":
            probs = F.softmax(value_out, dim=-1)
            w = torch.tensor(self._OUTCOME_POINTS, device=probs.device,
                             dtype=probs.dtype)
            return (probs * w).sum(dim=-1) / 3.0
        return value_out.squeeze(-1)

    def value_equity(self, x: torch.Tensor) -> torch.Tensor:
        """Scalar equity/3 in [-1, 1] per position, for both head types.

        This is what 0-ply move selection reads (see lookahead.eval_values_batch),
        so a scalar net and an outcomes6 net are interchangeable at play time.
        """
        _, value_out = self.forward(x)
        return self._equity_from_value_out(value_out)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def predict(
        self, obs: np.ndarray, legal_actions: list[int]
    ) -> tuple[dict[int, float], float]:
        """Single-position inference for MCTS.

        Args:
            obs: (C, 2, 12) numpy array (C = in_channels)
            legal_actions: list of valid action indices

        Returns:
            policy: dict mapping action -> probability (sums to ~1, only legal)
            value: scalar float in [-1, 1]
        """
        self.eval()
        x = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
        logits, value_out = self.forward(x)
        value = float(self._equity_from_value_out(value_out).item())

        # Mask illegal actions and apply softmax
        logits = logits.squeeze(0).cpu()
        mask = torch.full((self.num_actions,), float("-inf"))
        mask[legal_actions] = 0.0
        probs = F.softmax(logits + mask, dim=0).numpy()

        policy = {a: float(probs[a]) for a in legal_actions}
        return policy, value

    @torch.no_grad()
    def predict_batch(
        self, obs_list: list[np.ndarray], legal_actions_list: list[list[int]],
    ) -> list[tuple[dict[int, float], float]]:
        """Batched inference for multiple positions.

        Args:
            obs_list: list of (C, 2, 12) numpy arrays (C = in_channels)
            legal_actions_list: list of legal action lists

        Returns:
            list of (policy_dict, value) tuples
        """
        self.eval()
        obs_np = np.stack(obs_list)
        x = torch.from_numpy(obs_np).float().to(self.device)
        logits_batch, value_out = self.forward(x)
        logits_batch = logits_batch.cpu()
        values_batch = self._equity_from_value_out(value_out).cpu()

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
