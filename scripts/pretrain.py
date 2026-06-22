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

from raccoon.data.wildbg import load_wildbg_dir_tagged
from raccoon.env.encoder import encode_batch, resolve_channels
from raccoon.model.network import RaccoonNet


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
        ).strip()
    except Exception:
        return "unknown"


def load_dataset(
    data_dir: str,
    max_positions: int | None = None,
    channels: list[int] | None = None,
    seed: int = 42,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode + encode every CSV under ``data_dir`` into stacked arrays.

    Returns ``(observations, values, sources)`` of shapes ``(N, C, 2, 12)``,
    ``(N,)`` and ``(N,)`` respectively, where ``C = len(channels)`` (or 26
    when ``channels is None``) and ``sources[i]`` is the originating CSV stem
    (e.g. ``"race"`` / ``"contact"``). ~300k positions fit in ~32 MB so we
    hold the full dataset in memory.

    When ``max_positions`` is set, a deterministic random subsample (keyed by
    ``seed``) is drawn *across all CSVs* — not the first N rows, which would be
    a single file since the loader concatenates files in sorted order (and
    would defeat the race/contact breakdown).
    """
    print(f"Loading wildbg CSVs from {data_dir} ...", flush=True)
    t0 = time.time()
    rows, sources = load_wildbg_dir_tagged(data_dir)
    if max_positions is not None and max_positions < len(rows):
        sub = np.random.default_rng(seed).permutation(len(rows))[:max_positions]
        rows = [rows[i] for i in sub]
        sources = [sources[i] for i in sub]
    print(f"  decoded {len(rows):,} positions in {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    boards = [bv for bv, _ in rows]
    obs = encode_batch(boards, channels, normalize)
    values = np.array([v for _, v in rows], dtype=np.float32)
    sources = np.array(sources)
    print(f"  encoded tensor {obs.shape} ({obs.nbytes / 1e6:.1f} MB) "
          f"in {time.time() - t0:.1f}s", flush=True)
    return obs, values, sources


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
    sources: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    """Return ``(overall_mse, per_source_mse)`` over ``obs``.

    ``per_source_mse`` maps each distinct ``sources`` label (e.g. ``"race"``,
    ``"contact"``) to its exact MSE (sum of squared errors / count). It is
    empty when ``sources is None``.
    """
    network.eval()
    n = obs.size(0)
    total = 0.0
    count = 0
    src_sse: dict[str, float] = {}
    src_cnt: dict[str, int] = {}
    for start in range(0, n, batch_size):
        x = obs[start:start + batch_size].to(device, non_blocking=True)
        y = values[start:start + batch_size].to(device, non_blocking=True)
        _, value_pred = network(x)
        se = (value_pred.squeeze(-1) - y) ** 2
        total += se.sum().item()
        count += y.size(0)
        if sources is not None:
            batch_src = sources[start:start + batch_size]
            se_np = se.cpu().numpy()
            for s in np.unique(batch_src):
                mask = batch_src == s
                src_sse[s] = src_sse.get(s, 0.0) + float(se_np[mask].sum())
                src_cnt[s] = src_cnt.get(s, 0) + int(mask.sum())
    overall = total / max(count, 1)
    per_source = {s: src_sse[s] / max(src_cnt[s], 1) for s in src_sse}
    return overall, per_source


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
    parser.add_argument("--features", type=str, nargs="*", default=None,
                        help="Handcrafted feature groups to enable on top of "
                             "base channels: any of {pip, blots, anchors, "
                             "contact}, or 'all'. Omit the flag for the full "
                             "26-channel encoder; pass it with no values for "
                             "base-only (17ch). E.g. '--features pip' -> 20ch.")
    parser.add_argument("--raw-features", action="store_true",
                        help="Emit handcrafted channels at raw magnitude "
                             "(pre-Stage-6 behaviour). Default is normalized "
                             "(Fix-N) — the encoder default. Use this to "
                             "reproduce the Stage 6a broken baseline.")
    parser.add_argument("--feature-norm", action="store_true",
                        help="(deprecated; normalization is now the default) "
                             "accepted as a no-op for older invocations.")
    parser.add_argument("--input-bn", action="store_true",
                        help="Fix-B: add a BatchNorm over the raw input "
                             "channels before the input conv. Implies raw "
                             "features (the BN standardises scale itself).")
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

    channels = resolve_channels(args.features)
    if args.features is None:
        feat_label = "all (full 26ch)"
    elif args.features == []:
        feat_label = "base-only"
    else:
        feat_label = "base + " + ", ".join(args.features)
    # Fix-N (normalize in encoder) is the default. Fix-B (--input-bn) and the
    # explicit --raw-features baseline both feed raw-magnitude channels.
    normalize = not (args.raw_features or args.input_bn)
    fix_label = (
        "input-bn (Fix-B; raw inputs)" if args.input_bn
        else "raw (Stage 6a baseline)" if args.raw_features
        else "feature-norm (Fix-N, default)"
    )
    print(f"Feature groups: {feat_label} -> {len(channels)} channels  "
          f"| scaling: {fix_label}", flush=True)

    obs_np, values_np, sources_np = load_dataset(
        args.data_dir, max_positions=args.max_positions, channels=channels,
        seed=args.seed, normalize=normalize,
    )
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
    sources_val = sources_np[val_idx]
    del obs_np, values_np
    print(f"Split: {len(obs_train):,} train / {len(obs_val):,} val", flush=True)

    network = RaccoonNet(
        channels=args.channels, num_blocks=args.num_blocks,
        feature_channels=channels, input_bn=args.input_bn,
    ).to(device)
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
            "features": args.features,
            "feature_channels": channels,
            "in_channels": len(channels),
            "feature_norm": normalize,
            "raw_features": args.raw_features,
            "input_bn": args.input_bn,
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

    val_mse_baseline, _ = evaluate(
        network, obs_val, val_val, args.batch_size, device, sources_val,
    )
    print(f"Baseline val MSE (random init): {val_mse_baseline:.4f}", flush=True)

    best_val_mse = float("inf")
    last_val_mse = val_mse_baseline
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_mse = train_one_epoch(
            network, optimizer, obs_train, val_train, args.batch_size, device,
        )
        val_mse, val_mse_by_source = evaluate(
            network, obs_val, val_val, args.batch_size, device, sources_val,
        )
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
        for src, mse in sorted(val_mse_by_source.items()):
            row[f"val_mse_{src}"] = round(mse, 6)
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        by_src = "  ".join(
            f"{src}={mse:.4f}" for src, mse in sorted(val_mse_by_source.items())
        )
        print(
            f"Epoch {epoch:3d}/{args.epochs}: "
            f"train_mse={train_mse:.4f}  val_mse={val_mse:.4f}  "
            f"[{by_src}]  time={elapsed:.1f}s",
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
