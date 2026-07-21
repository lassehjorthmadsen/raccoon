"""Second pretrain stage: train both heads on synthesised policy targets.

Loads a checkpoint from the first (value-only) pretrain stage and the
policy cache built by ``scripts/synthesize_policy_dataset.py``. Trains
both heads with combined cross-entropy + MSE loss:

  - ``policy_target`` = one-hot at ``argmax_a V(child(s, a))`` (synthesised
    via 0-ply lookahead with the loaded V head).
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

# Idle OpenMP threads sleep instead of busy-spinning at barriers, so CPU
# contention (e.g. opening a browser mid-run) degrades gracefully rather than
# collapsing throughput (IPC ~0.09). Must run before the torch import below.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import platform
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# CPU fine-tuning drifts weights/activations into subnormal floats, which hit the
# slow FPU path and progressively collapse throughput (~10x on this iMac — see
# the exp008 denormal diagnosis; also explains exp007's "SGD crept 145->1074
# s/iter"). Flushing denormals to zero is negligible for a tanh/softmax net and
# keeps epoch time flat. Set once, as early as possible, on the main thread.
torch.set_flush_denormal(True)

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
) -> tuple[np.ndarray, object, np.ndarray, dict, bool]:
    """Load a policy-pretrain npz; returns ``(obs, policy, val, meta, is_soft)``.

    Two cache formats are supported:
      - **hard** (v1/v2/v2b, ``synthesize_policy_dataset.py``): ``policy_targets``
        (N,) int — one class index per example. ``policy`` is that array.
      - **soft** (v3, ``synthesize_gnubg_dataset.py``): ``policy_actions`` (N,K)
        and ``policy_probs`` (N,K) — a small distribution over candidate
        actions. ``policy`` is the tuple ``(actions, probs)``.
    """
    data = np.load(path, allow_pickle=True)
    obs = data["observations"].astype(np.float32, copy=False)
    val = data["value_targets"].astype(np.float32, copy=False)
    is_soft = "policy_actions" in data.files
    if is_soft:
        actions = data["policy_actions"].astype(np.int64, copy=False)
        probs = data["policy_probs"].astype(np.float32, copy=False)
        policy: object = (actions, probs)
    else:
        policy = data["policy_targets"].astype(np.int64, copy=False)
    meta_raw = data["meta"].item() if "meta" in data.files else "{}"
    try:
        meta = json.loads(meta_raw) if isinstance(meta_raw, str) else {}
    except Exception:
        meta = {}
    return obs, policy, val, meta, is_soft


def _policy_loss(policy_logits: torch.Tensor, pol, is_soft: bool) -> torch.Tensor:
    """Policy loss: soft cross-entropy over candidates, or hard cross-entropy."""
    if is_soft:
        actions, probs = pol  # (B,K) long, (B,K) float
        logp = F.log_softmax(policy_logits, dim=1)
        gathered = logp.gather(1, actions.clamp(min=0))  # pad -1 → col 0, prob 0
        return -(probs * gathered).sum(dim=1).mean()
    return F.cross_entropy(policy_logits, pol)


def _policy_correct(policy_logits: torch.Tensor, pol, is_soft: bool) -> tuple[int, int]:
    """Return (correct, n_valid): argmax matches the (money-)best target, over
    rows that actually carry a policy target. Value-only rows (recovered
    doubles) use a -1 sentinel in column 0 and are excluded so they neither
    count as wrong nor deflate top-1."""
    pred = policy_logits.argmax(dim=-1)
    target = pol[0][:, 0] if is_soft else pol  # soft: col 0 is the best move
    valid = target >= 0
    correct = int(((pred == target) & valid).sum().item())
    return correct, int(valid.sum().item())


def _index_policy(pol, idx, is_soft: bool):
    """Index into the policy tensors for a batch/split."""
    if is_soft:
        return pol[0][idx], pol[1][idx]
    return pol[idx]


def _to_device(pol, idx, is_soft, device):
    """Slice a batch of policy targets and move to device."""
    if is_soft:
        a, p = pol[0][idx], pol[1][idx]
        return a.to(device, non_blocking=True), p.to(device, non_blocking=True)
    return pol[idx].to(device, non_blocking=True)


def train_one_epoch(
    network: RaccoonNet,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    pol,
    val: torch.Tensor,
    batch_size: int,
    device: torch.device,
    value_weight: float,
    is_soft: bool,
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
        p_target = _to_device(pol, idx, is_soft, device)
        v_target = val[idx].to(device, non_blocking=True)
        policy_logits, value_pred = network(x)
        policy_loss = _policy_loss(policy_logits, p_target, is_soft)
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
    pol,
    val: torch.Tensor,
    batch_size: int,
    device: torch.device,
    is_soft: bool,
) -> tuple[float, float, float]:
    """Returns (val_policy_loss, val_value_loss, val_top1_acc)."""
    network.eval()
    n = obs.size(0)
    sum_p = sum_v = 0.0
    correct = 0
    valid_count = 0
    count = 0
    for start in range(0, n, batch_size):
        end = start + batch_size
        idx = slice(start, end)
        x = obs[idx].to(device, non_blocking=True)
        p_t = _to_device(pol, idx, is_soft, device)
        v_t = val[idx].to(device, non_blocking=True)
        policy_logits, value_pred = network(x)
        bs = x.size(0)
        sum_p += _policy_loss(policy_logits, p_t, is_soft).item() * bs
        sum_v += F.mse_loss(value_pred.squeeze(-1), v_t, reduction="sum").item()
        c, vc = _policy_correct(policy_logits, p_t, is_soft)
        correct += c
        valid_count += vc
        count += bs
    count = max(count, 1)
    valid_count = max(valid_count, 1)
    # Policy CE and top-1 over policy-labeled rows only (value-only doubles
    # rows carry a -1 sentinel and are excluded); value MSE over all rows.
    return sum_p / valid_count, sum_v / count, correct / valid_count


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
    parser.add_argument("--select-best", action="store_true",
                        help="Ship the epoch with the lowest val combined loss "
                             "as pretrained_v2.pt instead of the last epoch "
                             "(the untrained baseline counts as a candidate). "
                             "exp008 rounds overfit past epoch 1, so the last "
                             "epoch is usually the worst one.")
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
    obs_np, policy_np, val_np, cache_meta, is_soft = load_cache(args.cache)
    if is_soft:
        actions_np, probs_np = policy_np
        pol_desc = (f"SOFT targets, {actions_np.shape[1]} cols, "
                    f"actions {actions_np.max()} max")
    else:
        pol_desc = f"HARD targets, range {policy_np.min()}-{policy_np.max()}"
    print(f"  observations: {obs_np.shape}  policy: {pol_desc}  "
          f"value: mean={val_np.mean():.3f} std={val_np.std():.3f}", flush=True)

    n_total = obs_np.shape[0]
    n_val = max(1, int(n_total * args.val_split))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_total)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    obs_train = torch.from_numpy(obs_np[train_idx])
    obs_val = torch.from_numpy(obs_np[val_idx])
    val_train = torch.from_numpy(val_np[train_idx])
    val_val = torch.from_numpy(val_np[val_idx])
    if is_soft:
        pol_train = (torch.from_numpy(actions_np[train_idx]),
                     torch.from_numpy(probs_np[train_idx]))
        pol_val = (torch.from_numpy(actions_np[val_idx]),
                   torch.from_numpy(probs_np[val_idx]))
    else:
        pol_train = torch.from_numpy(policy_np[train_idx])
        pol_val = torch.from_numpy(policy_np[val_idx])
    del obs_np, val_np
    print(f"Split: {len(obs_train):,} train / {len(obs_val):,} val "
          f"({'soft' if is_soft else 'hard'} policy targets)", flush=True)

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
            "policy_target": "soft" if is_soft else "hard",
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
            "select_best": args.select_best,
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
        network, obs_val, pol_val, val_val, args.batch_size, device, is_soft,
    )
    print(f"Baseline val: policy_CE={p0:.4f}  value_MSE={v0:.4f}  "
          f"top1_acc={acc0*100:.1f}%", flush=True)

    def _combined(p: float, v: float) -> float:
        return p + args.value_weight * v

    # --select-best: the baseline is candidate epoch 0, so a round that only
    # overfits ships its input unchanged instead of a degraded net.
    best = {
        "epoch": 0, "loss": _combined(p0, v0),
        "val_policy_loss": p0, "val_value_loss": v0, "val_top1_acc": acc0,
        "state": ({k: v.detach().cpu().clone()
                   for k, v in network.state_dict().items()}
                  if args.select_best else None),
    }

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_total, train_p, train_v = train_one_epoch(
            network, optimizer, obs_train, pol_train, val_train,
            args.batch_size, device, args.value_weight, is_soft,
        )
        val_p, val_v, val_acc = evaluate(
            network, obs_val, pol_val, val_val, args.batch_size, device, is_soft,
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

        if args.select_best and _combined(val_p, val_v) < best["loss"]:
            best = {
                "epoch": epoch, "loss": _combined(val_p, val_v),
                "val_policy_loss": val_p, "val_value_loss": val_v,
                "val_top1_acc": val_acc,
                "state": {k: v.detach().cpu().clone()
                          for k, v in network.state_dict().items()},
            }

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
    selected_epoch = args.epochs
    if args.select_best:
        selected_epoch = best["epoch"]
        val_p, val_v, val_acc = (best["val_policy_loss"],
                                 best["val_value_loss"], best["val_top1_acc"])
        if selected_epoch < args.epochs:
            network.load_state_dict(best["state"])
        print(f"Selected epoch {selected_epoch} "
              f"(lowest val combined loss {best['loss']:.4f})", flush=True)
    final_path = checkpoint_dir / "pretrained_v2.pt"
    save_pretrained(
        final_path,
        network,
        pretrain_info={
            "stage": "policy-distill",
            "base_checkpoint": args.base_checkpoint,
            "cache": args.cache,
            "epochs": args.epochs,
            "selected_epoch": selected_epoch,
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
