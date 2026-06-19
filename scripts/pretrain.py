"""Supervised pretraining on wildbg-training rollout-labeled positions.

Cold-starts the value head + shared trunk by regressing onto wildbg's
equity labels. The policy head receives no gradient signal here — it stays
at random init and is refined later by self-play.

Usage (typical):

    ./scripts/download_wildbg.sh                                   # one-off
    python scripts/pretrain.py --experiment-name pretrain-wildbg-v1

The resulting checkpoint at
``experiments/<name>/checkpoints/pretrained.pt`` is a drop-in for
``scripts/train.py --resume <path>``: it carries the model state and
``config`` dict but intentionally omits the optimizer state, so self-play
will create a fresh Adam at its own --lr.
"""

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

from raccoon.data.wildbg import load_wildbg_dir
from raccoon.env.encoder import encode_batch
from raccoon.model.network import RaccoonNet


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
        ).strip()
    except Exception:
        return "unknown"


def load_dataset(
    data_dir: str, max_positions: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode + encode every CSV under ``data_dir`` into stacked arrays.

    Returns ``(observations, values)`` of shapes ``(N, 26, 2, 12)`` and
    ``(N,)`` respectively. ~300k positions fit in ~32 MB so we hold the
    full dataset in memory.
    """
    print(f"Loading wildbg CSVs from {data_dir} ...", flush=True)
    t0 = time.time()
    rows = load_wildbg_dir(data_dir)
    if max_positions is not None:
        rows = rows[:max_positions]
    print(f"  decoded {len(rows):,} positions in {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    boards = [bv for bv, _ in rows]
    obs = encode_batch(boards)
    values = np.array([v for _, v in rows], dtype=np.float32)
    print(f"  encoded tensor {obs.shape} ({obs.nbytes / 1e6:.1f} MB) "
          f"in {time.time() - t0:.1f}s", flush=True)
    return obs, values


def train_one_epoch(
    network: RaccoonNet,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    values: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    network.train()
    n = obs.size(0)
    perm = torch.randperm(n)
    total_loss = 0.0
    n_batches = 0
    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        x = obs[idx].to(device, non_blocking=True)
        y = values[idx].to(device, non_blocking=True)
        _, value_pred = network(x)
        loss = F.mse_loss(value_pred.squeeze(-1), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    network: RaccoonNet,
    obs: torch.Tensor,
    values: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    network.eval()
    n = obs.size(0)
    total = 0.0
    count = 0
    for start in range(0, n, batch_size):
        x = obs[start:start + batch_size].to(device, non_blocking=True)
        y = values[start:start + batch_size].to(device, non_blocking=True)
        _, value_pred = network(x)
        total += F.mse_loss(value_pred.squeeze(-1), y, reduction="sum").item()
        count += y.size(0)
    return total / max(count, 1)


def save_pretrained(
    path: Path, network: RaccoonNet, pretrain_info: dict,
) -> None:
    """Save a checkpoint compatible with ``scripts/train.py --resume``.

    We deliberately omit the optimizer state so self-play creates its own
    Adam at its own --lr (otherwise ``load_checkpoint`` would overwrite
    self-play's LR with the pretrain LR).
    """
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
    parser = argparse.ArgumentParser(
        description="Pretrain RaccoonNet on wildbg rollout-labeled positions",
    )
    parser.add_argument("--experiment-name", type=str, required=True,
                        help="Outputs go to experiments/<name>/{checkpoints,logs}/")
    parser.add_argument("--data-dir", type=str, default="data/wildbg")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=5,
                        help="Save an epoch_NN.pt every N epochs (always saves pretrained.pt at end)")
    parser.add_argument("--max-positions", type=int, default=None,
                        help="Optional cap on positions for smoke runs.")
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

    obs_np, values_np = load_dataset(args.data_dir, max_positions=args.max_positions)
    n_total = obs_np.shape[0]
    n_val = max(1, int(n_total * args.val_split))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_total)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    obs_train = torch.from_numpy(obs_np[train_idx])
    val_train = torch.from_numpy(values_np[train_idx])
    obs_val = torch.from_numpy(obs_np[val_idx])
    val_val = torch.from_numpy(values_np[val_idx])
    del obs_np, values_np
    print(f"Split: {len(obs_train):,} train / {len(obs_val):,} val", flush=True)

    network = RaccoonNet(channels=args.channels, num_blocks=args.num_blocks).to(device)
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
            "kind": "supervised-pretrain",
            "data_dir": args.data_dir,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "val_split": args.val_split,
            "n_train": len(obs_train),
            "n_val": len(obs_val),
            "seed": args.seed,
            "max_positions": args.max_positions,
        },
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

    val_mse_baseline = evaluate(network, obs_val, val_val, args.batch_size, device)
    print(f"Baseline val MSE (random init): {val_mse_baseline:.4f}", flush=True)

    best_val_mse = float("inf")
    last_val_mse = val_mse_baseline
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_mse = train_one_epoch(
            network, optimizer, obs_train, val_train, args.batch_size, device,
        )
        val_mse = evaluate(network, obs_val, val_val, args.batch_size, device)
        elapsed = time.time() - t0
        last_val_mse = val_mse
        best_val_mse = min(best_val_mse, val_mse)

        row = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "epoch": epoch,
            "train_mse": round(train_mse, 6),
            "val_mse": round(val_mse, 6),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": round(elapsed, 1),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        print(
            f"Epoch {epoch:3d}/{args.epochs}: "
            f"train_mse={train_mse:.4f}  val_mse={val_mse:.4f}  "
            f"time={elapsed:.1f}s",
            flush=True,
        )

        if epoch % args.checkpoint_every == 0 and epoch < args.epochs:
            save_pretrained(
                checkpoint_dir / f"epoch_{epoch:03d}.pt",
                network,
                pretrain_info={
                    "source": args.data_dir,
                    "epoch": epoch,
                    "val_mse": val_mse,
                    "n_train": len(obs_train),
                    "n_val": len(obs_val),
                    "lr": args.lr,
                    "git_sha": config_entry["git_sha"],
                },
            )

    total_time = time.time() - t_start
    final_path = checkpoint_dir / "pretrained.pt"
    save_pretrained(
        final_path,
        network,
        pretrain_info={
            "source": args.data_dir,
            "epochs": args.epochs,
            "final_val_mse": last_val_mse,
            "best_val_mse": best_val_mse,
            "n_train": len(obs_train),
            "n_val": len(obs_val),
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "git_sha": config_entry["git_sha"],
            "wall_time_sec": round(total_time, 1),
        },
    )
    print(
        f"\nSaved {final_path}  "
        f"(final val_mse={last_val_mse:.4f}, best={best_val_mse:.4f}, "
        f"total={total_time / 60:.1f} min)",
        flush=True,
    )
    print(
        f"\nNext step:\n"
        f"  python scripts/train.py --experiment-name exp007-... "
        f"--resume {final_path} [...]",
        flush=True,
    )


if __name__ == "__main__":
    main()
