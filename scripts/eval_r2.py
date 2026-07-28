"""exp014 — R^2 fit-quality eval for the 0-ply-vs-2-ply scalar distillation A/B.

The hypothesis: can the scalar net fit GNUBG-2-ply labels about as well as it
fits GNUBG-0-ply labels? Since 2-ply is a stronger player, fitting it equally
well implies a stronger net (a proxy for play strength, not a ppg number).

``experiments/exp011-distill/cache/`` (0-ply) and ``cache_2ply/`` (2-ply) are
shard-for-shard byte-identical in ``observations`` — only the labels differ —
so this script uses ``raccoon.data.cache_split.held_out_mask`` (the SAME split
function ``train_distill.py --holdout-frac`` uses) to pull the identical held-out
rows from either cache. Pass ``--cross-cache-dir`` to also score the checkpoint
against the *other* label set (the exp014 2x2), and ``--sample-train N`` to also
report an in-sample R^2 for the overfitting check.

The real 10x256 net is ~10,000x more compute per sample than a tiny test net, so
a full-precision pass (all 18 shards, ~2M held-out rows) can take well over an
hour on CPU — fine for the one headline (ep3) comparison, wasteful to repeat for
every epoch of a supporting curve. Use ``--max-holdout N`` to stop reading
shards once N held-out rows are collected (skips both the disk I/O and the
forward pass for the rest) — R^2's SE is already ~1e-3 at N=200k, plenty for a
supporting/monotonicity check.

    # own-label R^2 (a 2-ply-trained checkpoint scored on 2-ply labels)
    python scripts/eval_r2.py \\
        --checkpoint experiments/exp014-distill/scalar_2ply/checkpoints/ep3.pt \\
        --cache-dir experiments/exp011-distill/cache_2ply

    # fast supporting-curve pass (subsampled, for a non-headline epoch)
    python scripts/eval_r2.py \\
        --checkpoint experiments/exp014-distill/scalar_2ply/checkpoints/ep1.pt \\
        --cache-dir experiments/exp011-distill/cache_2ply --max-holdout 200000

    # full 2x2 cell + train-sample R^2 (overfitting check)
    python scripts/eval_r2.py \\
        --checkpoint experiments/exp014-distill/scalar_2ply/checkpoints/ep3.pt \\
        --cache-dir experiments/exp011-distill/cache_2ply \\
        --cross-cache-dir experiments/exp011-distill/cache \\
        --sample-train 200000
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")  # avoid CPU spin-collapse
                                                       # under contention (must
                                                       # be set before torch
                                                       # spins up its OpenMP pool)

import argparse
from pathlib import Path

import numpy as np
import torch

from raccoon.data.cache_split import held_out_mask
from raccoon.model.network import load_model


def predict(net, device, obs: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Batch-forward the scalar value head; returns equity predictions."""
    net.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(obs), batch_size):
            xb = torch.from_numpy(obs[i:i + batch_size].astype(np.float32)).to(device)
            _, v = net(xb)
            preds.append(v.squeeze(-1).cpu().numpy())
    return np.concatenate(preds) if preds else np.empty(0, dtype=np.float32)


def r2_score(pred: np.ndarray, target: np.ndarray) -> float:
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def collect(cache_dir: str, holdout_frac: float, split_seed: int,
            cross_cache_dir: str | None, sample_train: int, sample_seed: int,
            max_holdout: int = 0):
    """One pass over ``cache_dir``'s shards. Returns:

    ``ho_obs`` (held-out observations, from ``cache_dir``), ``ho_eq`` (held-out
    equity from ``cache_dir``), ``ho_eq_cross`` (held-out equity from
    ``cross_cache_dir``, aligned row-for-row, or ``None``), ``tr_obs``/``tr_eq``
    (a random ``sample_train``-row sample of the TRAIN rows, or empty arrays).

    Observations are read once from ``cache_dir`` and reused for the cross-cache
    score (the two caches are byte-identical in ``observations``) — avoids
    loading the 26-channel array twice.

    If ``max_holdout`` > 0, stops reading further shards once that many held-out
    rows are collected — later shards are never opened, so both the disk I/O and
    the (expensive, on a 10x256 net) forward pass shrink proportionally. Note
    this can also short the ``sample_train`` draw if it stops before covering
    all shards; leave ``max_holdout`` at 0 (full precision) when combining with
    ``--sample-train`` for a headline read.
    """
    shards = sorted(Path(cache_dir).glob("shard_*.npz"))
    if not shards:
        raise SystemExit(f"no shards in {cache_dir}")
    cross_shards = None
    if cross_cache_dir:
        cross_shards = {p.name: p for p in Path(cross_cache_dir).glob("shard_*.npz")}

    rng = np.random.default_rng(sample_seed)
    ho_obs, ho_eq, ho_eq_cross = [], [], [] if cross_shards is not None else None
    tr_obs, tr_eq = [], []
    remaining_train = sample_train
    ho_count = 0

    for sh in shards:
        if max_holdout and ho_count >= max_holdout:
            break
        with np.load(sh) as z:
            obs, eq = z["observations"], z["equity"]
        mask = held_out_mask(sh.name, len(obs), holdout_frac, split_seed)
        ho_obs.append(obs[mask])
        ho_eq.append(eq[mask])
        ho_count += int(mask.sum())
        if cross_shards is not None:
            cross_path = cross_shards.get(sh.name)
            if cross_path is None:
                raise SystemExit(f"cross-cache-dir is missing shard {sh.name} — "
                                 "caches must be shard-for-shard aligned")
            with np.load(cross_path) as zc:
                eq_c = zc["equity"]
            if len(eq_c) != len(eq):
                raise SystemExit(f"{sh.name}: row-count mismatch between caches "
                                 f"({len(eq)} vs {len(eq_c)}) — not aligned")
            ho_eq_cross.append(eq_c[mask])
        if remaining_train > 0:
            train_idx = np.flatnonzero(~mask)
            take = rng.choice(train_idx, size=min(remaining_train, len(train_idx)),
                              replace=False)
            tr_obs.append(obs[take])
            tr_eq.append(eq[take])
            remaining_train -= len(take)

    ho_obs = np.concatenate(ho_obs)
    ho_eq = np.concatenate(ho_eq)
    ho_eq_cross = np.concatenate(ho_eq_cross) if ho_eq_cross is not None else None
    tr_obs = np.concatenate(tr_obs) if tr_obs else np.empty((0,) + ho_obs.shape[1:],
                                                             dtype=ho_obs.dtype)
    tr_eq = np.concatenate(tr_eq) if tr_eq else np.empty(0, dtype=np.float32)
    return ho_obs, ho_eq, ho_eq_cross, tr_obs, tr_eq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache-dir", required=True,
                    help="Cache whose labels this checkpoint was trained toward "
                         "(own-label R^2).")
    ap.add_argument("--cross-cache-dir", default=None,
                    help="Optional second cache (the other label set) to also "
                         "score against on the SAME held-out rows (the 2x2 cell).")
    ap.add_argument("--holdout-frac", type=float, default=0.25,
                    help="Must match the value passed to train_distill.py's "
                         "--holdout-frac for this checkpoint.")
    ap.add_argument("--split-seed", type=int, default=14,
                    help="Must match train_distill.py's --split-seed.")
    ap.add_argument("--sample-train", type=int, default=0,
                    help="If >0, also report R^2 on this many TRAIN rows "
                         "(overfitting check: train R^2 vs held-out R^2). "
                         "Caveat: positions are game-contiguous with no game "
                         "IDs, so a random train sample shares games with the "
                         "held-out set — this understates true overfitting.")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--max-holdout", type=int, default=0,
                    help="If >0, stop after collecting this many held-out rows "
                         "(skips reading/forwarding the rest — see module "
                         "docstring). 0 (default) = full held-out set, the "
                         "precision the exp014 headline metric uses.")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    torch.set_flush_denormal(True)  # iMac CPU: avoid denormal slowdown
    device = torch.device("cpu")
    net = load_model(args.checkpoint).to(device)

    ho_obs, ho_eq, ho_eq_cross, tr_obs, tr_eq = collect(
        args.cache_dir, args.holdout_frac, args.split_seed,
        args.cross_cache_dir, args.sample_train, args.sample_seed,
        args.max_holdout)

    tag = f"[{args.label}] " if args.label else ""
    print(f"{tag}{args.checkpoint}", flush=True)

    ho_pred = predict(net, device, ho_obs)
    r2_own = r2_score(ho_pred, ho_eq)
    print(f"  held-out n={len(ho_eq)}  R^2 vs {Path(args.cache_dir).name} labels "
          f"(own) = {r2_own:.4f}", flush=True)

    if ho_eq_cross is not None:
        r2_cross = r2_score(ho_pred, ho_eq_cross)
        print(f"  held-out n={len(ho_eq_cross)}  R^2 vs "
              f"{Path(args.cross_cache_dir).name} labels (cross) = {r2_cross:.4f}",
              flush=True)

    if len(tr_eq):
        tr_pred = predict(net, device, tr_obs)
        r2_train = r2_score(tr_pred, tr_eq)
        print(f"  train-sample n={len(tr_eq)}  R^2 vs {Path(args.cache_dir).name} "
              f"labels = {r2_train:.4f}  (train - held-out = "
              f"{r2_train - r2_own:+.4f}; understated — see --sample-train help)",
              flush=True)


if __name__ == "__main__":
    main()
