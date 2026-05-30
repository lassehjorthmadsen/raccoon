"""Second pretrain stage: train both heads on synthesised policy targets.

Loads a checkpoint from the first (value-only) pretrain stage and the
policy cache built by ``scripts/synthesize_policy_dataset.py``. Trains
both heads with combined cross-entropy + MSE loss:

  - ``policy_target`` = one-hot at ``argmax_a V(child(s, a))`` (synthesised
    via 1-ply lookahead with the loaded V head).
  - ``value_target`` = V(s) as evaluated by the loaded V head at dataset
    build time (knowledge distillation — anchors the value head to its
    wildbg-calibrated predictions while the shared trunk learns to use
    the dice channels, which were always zero during wildbg pretrain).

Output: ``experiments/<name>/checkpoints/pretrained_v2.pt``, a drop-in
``--resume`` target for ``scripts/train.py`` with the same checkpoint
shape as the first-stage ``pretrained.pt``.

Typical use:

    python scripts/pretrain_policy.py \\
        --experiment-name pretrain-wildbg-v2 \\
        --base-checkpoint experiments/pretrain-wildbg-v1/checkpoints/pretrained.pt \\
        --cache data/bglab/policy_cache.npz
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from raccoon.model.network import RaccoonNet, load_model


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
        ).strip()
    except Exception:
        return "unknown"


def load_cache(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load the npz produced by synthesize_policy_dataset.py."""
    data = np.load(path, allow_pickle=True)
    obs = data["observations"].astype(np.float32, copy=False)
    pol = data["policy_targets"].astype(np.int64, copy=False)
    val = data["value_targets"].astype(np.float32, copy=False)
    meta_raw = data["meta"].item() if "meta" in data.files else "{}"
    try:
        meta = json.loads(meta_raw) if isinstance(meta_raw, str) else {}
    except Exception:
        meta = {}
    return obs, pol, val, meta


def train_one_epoch(
    network: RaccoonNet,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    pol: torch.Tensor,
    val: torch.Tensor,
    batch_size: int,
    device: torch.device,
    value_weight: float,
) -> tuple[float, float, float]:
    """Train one epoch; returns (avg_total, avg_policy, avg_value) losses."""
    network.train()
    n = obs.size(0)
    perm = torch.randperm(n)
    total_p = total_v = total_t = 0.0
    n_batches = 0
    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        x = obs[idx].to(device, non_blocking=True)
        p_target = pol[idx].to(device, non_blocking=True)
        v_target = val[idx].to(device, non_blocking=True)
        policy_logits, value_pred = network(x)
        policy_loss = F.cross_entropy(policy_logits, p_target)
        value_loss = F.mse_loss(value_pred.squeeze(-1), v_target)
        loss = policy_loss + value_weight * value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_p += policy_loss.item()
        total_v += value_loss.item()
        total_t += loss.item()
        n_batches += 1
    n_batches = max(n_batches, 1)
    return total_t / n_batches, total_p / n_batches, total_v / n_batches


@torch.no_grad()
def evaluate(
    network: RaccoonNet,
    obs: torch.Tensor,
    pol: torch.Tensor,
    val: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float, float]:
    """Returns (val_policy_loss, val_value_loss, val_top1_acc)."""
    network.eval()
    n = obs.size(0)
    sum_p = sum_v = 0.0
    correct = 0
    count = 0
    for start in range(0, n, batch_size):
        end = start + batch_size
        x = obs[start:end].to(device, non_blocking=True)
        p_t = pol[start:end].to(device, non_blocking=True)
        v_t = val[start:end].to(device, non_blocking=True)
        policy_logits, value_pred = network(x)
        sum_p += F.cross_entropy(policy_logits, p_t, reduction="sum").item()
        sum_v += F.mse_loss(value_pred.squeeze(-1), v_t, reduction="sum").item()
        correct += int((policy_logits.argmax(dim=-1) == p_t).sum().item())
        count += p_t.size(0)
    count = max(count, 1)
    return sum_p / count, sum_v / count, correct / count


def save_pretrained(
    path: Path, network: RaccoonNet, pretrain_info: dict,
) -> None:
    """Save a checkpoint compatible with scripts/train.py --resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": network.state_dict(),
            "config": network.config,
            "step": -1,
            "pretrain_info": pretrain_info,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", type=str, required=True,
                        help="Outputs go to experiments/<name>/{checkpoints,logs}/")
    parser.add_argument("--base-checkpoint", type=str, required=True,
                        help="The pretrained.pt (value-only) starting point.")
    parser.add_argument("--cache", type=str, required=True,
                        help="Path to the policy_cache.npz from synthesis.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Lower than stage-1 LR — we're fine-tuning, "
                             "not training from scratch.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-weight", type=float, default=1.0,
                        help="Coefficient on the value (distillation) loss "
                             "relative to the policy cross-entropy.")
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--notes", type=str, default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    exp_root = Path(f"experiments/{args.experiment_name}")
    checkpoint_dir = exp_root / "checkpoints"
    log_dir = exp_root / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "pretrain_log.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    print(f"Loading base checkpoint {args.base_checkpoint}", flush=True)
    network = load_model(args.base_checkpoint).to(device)

    print(f"Loading policy cache {args.cache}", flush=True)
    obs_np, pol_np, val_np, cache_meta = load_cache(args.cache)
    print(f"  observations: {obs_np.shape}  "
          f"policy: range {pol_np.min()}-{pol_np.max()}  "
          f"value: mean={val_np.mean():.3f} std={val_np.std():.3f}", flush=True)

    n_total = obs_np.shape[0]
    n_val = max(1, int(n_total * args.val_split))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_total)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    obs_train = torch.from_numpy(obs_np[train_idx])
    pol_train = torch.from_numpy(pol_np[train_idx])
    val_train = torch.from_numpy(val_np[train_idx])
    obs_val = torch.from_numpy(obs_np[val_idx])
    pol_val = torch.from_numpy(pol_np[val_idx])
    val_val = torch.from_numpy(val_np[val_idx])
    del obs_np, pol_np, val_np
    print(f"Split: {len(obs_train):,} train / {len(obs_val):,} val", flush=True)

    optimizer = torch.optim.Adam(
        network.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    run_id = uuid.uuid4().hex[:8]
    config_entry = {
        "type": "config",
        "run_id": run_id,
        "experiment_name": args.experiment_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "network": network.config,
        "training": {
            "kind": "supervised-pretrain-policy",
            "base_checkpoint": args.base_checkpoint,
            "cache": args.cache,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "value_weight": args.value_weight,
            "val_split": args.val_split,
            "n_train": len(obs_train),
            "n_val": len(obs_val),
            "seed": args.seed,
        },
        "cache_meta": cache_meta,
        "system": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": (torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else None),
            "hostname": platform.node(),
            "cpu_count": os.cpu_count(),
        },
        "notes": args.notes,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(config_entry) + "\n")

    # Baseline (no training yet).
    p0, v0, acc0 = evaluate(
        network, obs_val, pol_val, val_val, args.batch_size, device,
    )
    print(f"Baseline val: policy_CE={p0:.4f}  value_MSE={v0:.4f}  "
          f"top1_acc={acc0*100:.1f}%", flush=True)

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_total, train_p, train_v = train_one_epoch(
            network, optimizer, obs_train, pol_train, val_train,
            args.batch_size, device, args.value_weight,
        )
        val_p, val_v, val_acc = evaluate(
            network, obs_val, pol_val, val_val, args.batch_size, device,
        )
        elapsed = time.time() - t0

        row = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "epoch": epoch,
            "train_policy_loss": round(train_p, 6),
            "train_value_loss": round(train_v, 6),
            "train_total_loss": round(train_total, 6),
            "val_policy_loss": round(val_p, 6),
            "val_value_loss": round(val_v, 6),
            "val_top1_acc": round(val_acc, 6),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": round(elapsed, 1),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        print(
            f"Epoch {epoch:3d}/{args.epochs}: "
            f"train pol={train_p:.4f} val={train_v:.4f} | "
            f"val pol={val_p:.4f} val={val_v:.4f} top1={val_acc*100:.1f}% | "
            f"{elapsed:.1f}s",
            flush=True,
        )

        if epoch % args.checkpoint_every == 0 and epoch < args.epochs:
            save_pretrained(
                checkpoint_dir / f"epoch_{epoch:03d}.pt",
                network,
                pretrain_info={
                    "stage": "policy-distill",
                    "base_checkpoint": args.base_checkpoint,
                    "cache": args.cache,
                    "epoch": epoch,
                    "val_policy_loss": val_p,
                    "val_value_loss": val_v,
                    "val_top1_acc": val_acc,
                    "git_sha": config_entry["git_sha"],
                },
            )

    total_time = time.time() - t_start
    final_path = checkpoint_dir / "pretrained_v2.pt"
    save_pretrained(
        final_path,
        network,
        pretrain_info={
            "stage": "policy-distill",
            "base_checkpoint": args.base_checkpoint,
            "cache": args.cache,
            "epochs": args.epochs,
            "final_val_policy_loss": val_p,
            "final_val_value_loss": val_v,
            "final_val_top1_acc": val_acc,
            "n_train": len(obs_train),
            "n_val": len(obs_val),
            "lr": args.lr,
            "value_weight": args.value_weight,
            "git_sha": config_entry["git_sha"],
            "wall_time_sec": round(total_time, 1),
        },
    )
    print(
        f"\nSaved {final_path}\n"
        f"  final val: pol={val_p:.4f} val={val_v:.4f} top1={val_acc*100:.1f}%\n"
        f"  total {total_time/60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
