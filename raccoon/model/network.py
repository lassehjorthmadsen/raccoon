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
        trunk: str = "resnet",
        hidden: int | list[int] = 256,
        hidden_layers: int = 1,
        mlp_input: str = "flat",
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
        # trunk: "resnet" is the original convolutional stack and what every
        # checkpoint before exp028 used (old configs lack the key and rebuild as
        # that). "mlp" flattens the encoded board and runs `hidden_layers` fully
        # connected layers of width `hidden` -- the shape GNU Backgammon, XG and
        # BGSage all use, and roughly a thousandth of the arithmetic. See
        # docs/speed.qmd for why that ratio decides what the engine can do.
        if trunk not in ("resnet", "mlp"):
            raise ValueError(f"trunk must be resnet|mlp, got {trunk}")
        if hidden_layers < 1:
            raise ValueError(f"hidden_layers must be >= 1, got {hidden_layers}")
        # `hidden` is either one width repeated `hidden_layers` times, or the
        # explicit list of widths. The list form exists because the shapes worth
        # copying are not uniform -- PureTD's is [512, 512, 256, 256] -- and a
        # single width with a layer count cannot express them.
        # mlp_input: how the (C, 2, 12) encoding becomes a vector.
        #   "flat"         -- every channel flattened, C * 24 inputs.
        #   "debroadcast"  -- channels 0-7 (the per-point checker planes) kept in
        #                     full, every other channel reduced to the single
        #                     number it holds. Channels 8-25 are ALL constant
        #                     across the board: they are scalars a convolution
        #                     needs as planes and a flattened network receives as
        #                     24 identical copies. With all 26 channels that is
        #                     432 of 624 inputs carrying 18 numbers; base-only it
        #                     is 216 of 408 carrying 9. De-broadcasting gives
        #                     192 + 9 = 201 inputs, which is the size PureTD uses.
        if mlp_input not in ("flat", "debroadcast"):
            raise ValueError(f"mlp_input must be flat|debroadcast, got {mlp_input}")
        self.mlp_input = mlp_input
        widths = list(hidden) if isinstance(hidden, (list, tuple)) else [hidden] * hidden_layers
        if any(w < 1 for w in widths):
            raise ValueError(f"hidden widths must be >= 1, got {widths}")
        self.trunk_kind = trunk
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
            "trunk": trunk,
            "hidden": widths if trunk == "mlp" else hidden,
            "hidden_layers": len(widths),
            "mlp_input": mlp_input,
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

        if trunk == "mlp":
            # Flatten the encoded board and run plain fully connected layers.
            # The heads stay linear on the same hidden vector, so `forward`
            # returns the same pair and nothing downstream branches on the
            # trunk: lookahead, expectimax, the cube search and the web export
            # all keep working.
            # Channels 0-7 are the checker planes and are always present, since
            # `base` is always included and the channel list is sorted.
            n_in = (in_channels * board_h * board_w if mlp_input == "flat"
                    else 8 * board_h * board_w + (in_channels - 8))
            layers: list[nn.Module] = []
            prev = n_in
            for w in widths:
                layers += [nn.Linear(prev, w), nn.ReLU()]
                prev = w
            self.mlp = nn.Sequential(*layers)
            self.mlp_policy = nn.Linear(prev, num_actions)
            self.mlp_value = nn.Linear(prev, 6 if value_head == "outcomes6" else 1)
            return

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
        if self.input_norm is not None:
            x = self.input_norm(x)

        if self.trunk_kind == "mlp":
            h = self.mlp(self._mlp_features(x))
            v = self.mlp_value(h)
            return self.mlp_policy(h), (
                v if self.value_head == "outcomes6" else torch.tanh(v)
            )

        # Shared convolutional trunk
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

    def _mlp_features(self, x: torch.Tensor) -> torch.Tensor:
        """Turn a (batch, C, 2, 12) encoding into the MLP's input vector."""
        if self.mlp_input == "flat":
            return x.reshape(x.size(0), -1)
        per_point = x[:, :8].reshape(x.size(0), -1)
        scalars = x[:, 8:, 0, 0]
        return torch.cat([per_point, scalars], dim=1)

    def _value_out(self, x: torch.Tensor) -> torch.Tensor:
        """The value head's raw output, skipping the policy head entirely.

        0-ply move selection, expectimax and the cube search all read the value
        only, and for an MLP trunk the policy head is the larger half of the
        network -- a 256-wide hidden layer feeding 1352 actions is 346k
        multiply-accumulates against the trunk's 160k. Computing it to throw it
        away would more than double the cost of the configuration exp028 is
        measuring. The saving is negligible for a ResNet, where the policy head
        is a 1x1 convolution, but the path is shared so both benefit.
        """
        if self.input_norm is not None:
            x = self.input_norm(x)
        if self.trunk_kind == "mlp":
            v = self.mlp_value(self.mlp(self._mlp_features(x)))
            return v if self.value_head == "outcomes6" else torch.tanh(v)
        out = F.relu(self.input_bn(self.input_conv(x)))
        out = self.trunk(out)
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = self.value_fc2(v)
        return v if self.value_head == "outcomes6" else torch.tanh(v)

    def _equity_from_value_out(self, value_out: torch.Tensor) -> torch.Tensor:
        """Map a raw value-head output to equity/3 in [-1, 1], shape (batch,).

        For "scalar" this is the tanh output itself; for "outcomes6" it is the
        softmax distribution dotted with the outcome points (±1/±2/±3), then /3
        to match the scalar head's money-equity/3 convention.
        """
        if self.value_head == "outcomes6":
            probs = self._probs6_from_value_out(value_out)
            w = torch.tensor(self._OUTCOME_POINTS, device=probs.device,
                             dtype=probs.dtype)
            return (probs * w).sum(dim=-1) / 3.0
        return value_out.squeeze(-1)

    @staticmethod
    def _probs6_from_value_out(value_out: torch.Tensor) -> torch.Tensor:
        """Softmax over the six-outcome logits, shape (batch, 6).

        Both value_equity and value_probs6 route through here so the two views of
        the head can never disagree about what the distribution is.
        """
        return F.softmax(value_out, dim=-1)

    def value_equity(self, x: torch.Tensor) -> torch.Tensor:
        """Scalar equity/3 in [-1, 1] per position, for both head types.

        This is what 0-ply move selection reads (see lookahead.eval_values_batch),
        so a scalar net and an outcomes6 net are interchangeable at play time.
        """
        return self._equity_from_value_out(self._value_out(x))

    def value_probs6(self, x: torch.Tensor) -> torch.Tensor:
        """The full six-outcome distribution, shape (batch, 6).

        Order is [win, win_gammon, win_bg, lose, lose_gammon, lose_bg]: mutually
        exclusive, summing to 1. The doubling cube needs the whole distribution
        rather than the single number it collapses to, because a double or a take
        turns on the gammon rates that value_equity has already folded away.
        See raccoon.cube.janowski.probs6_to_cumulative5 for the conversion into
        the nested 5-vector the cube formulas expect.
        """
        if self.value_head != "outcomes6":
            raise ValueError(
                "value_probs6 needs an 'outcomes6' value head; this network has "
                f"'{self.value_head}', which never carried a distribution")
        return self._probs6_from_value_out(self._value_out(x))

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
