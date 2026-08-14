"""Variance-reduced ppg for a checkpoint vs GNUBG, sharded across processes.

Plays net-vs-GNUBG games exactly like ``scripts/eval_gnubg0.py`` but records the
realised dice luck at every roll, so each game yields both the raw result and the
variance-reduced one (``vr = raw - luck``). See :mod:`raccoon.eval.luck` for the
estimator and why it is exactly unbiased.

Unlike ``eval_gnubg0.py`` this writes its numbers to disk rather than stdout only:
an append-only summary line to ``<exp-dir>/logs/vr_eval_log.jsonl`` and an idempotent
``<exp-dir>/results/<tag>.json`` carrying the summary **plus the per-game raw and luck
arrays**, so an analysis page can recompute blocks, histograms and diagnostics at render
time from committed data.

    # headline: one large sample
    python scripts/eval_gnubg_vr.py --checkpoint .../ep22.pt --games 6000 --workers 3 \
        --exp-dir experiments/exp019-vr --tag ep22_vr

    # demo: 20 independent matches of 50 games, variance reduction off
    python scripts/eval_gnubg_vr.py --checkpoint .../ep22.pt --blocks 20 --block-size 50 \
        --workers 3 --no-vr --exp-dir experiments/exp019-vr --tag demo_raw
"""
from __future__ import annotations

import os

# CPU-bound torch: park idle OpenMP threads instead of busy-spinning (a spin-collapse
# costs ~12x under desktop contention). Must precede the torch import.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from raccoon.eval.vr_arena import gnubg_arena_vr
from raccoon.model.network import load_model


def _shard(ckpt: str, n: int, ply: int, cv_ply: int, seed: int, vr: bool,
           joint_doubles: bool):
    torch.set_flush_denormal(True)
    torch.set_num_threads(1)
    net = load_model(ckpt)
    net.eval()
    r = gnubg_arena_vr(
        net, torch.device("cpu"), n, gnubg_ply=ply, cv_ply=cv_ply, seed=seed, vr=vr,
        joint_doubles=joint_doubles,
    )
    return r["game_pts"], r["game_luck"], r["game_rolls"], r["net_wins"]


def summarise(pts: np.ndarray, luck: np.ndarray) -> dict:
    """Point estimates, CIs and the control-variate diagnostics.

    ``beta_hat`` is the regression-optimal control-variate coefficient. We keep the
    estimator at beta = 1 (any *fixed* beta stays exactly unbiased, whereas one fitted
    on the same sample does not), so it is reported purely as a calibration read: a
    well-scaled control variate lands near 1.
    """
    n = len(pts)
    vr = pts - luck
    out = {
        "games": n,
        "raw_ppg": float(pts.mean()),
        "raw_sd": float(pts.std(ddof=1)),
        "raw_ci95": float(1.96 * pts.std(ddof=1) / n**0.5),
        "vr_ppg": float(vr.mean()),
        "vr_sd": float(vr.std(ddof=1)),
        "vr_ci95": float(1.96 * vr.std(ddof=1) / n**0.5),
        "mean_luck": float(luck.mean()),
        "luck_sd": float(luck.std(ddof=1)),
    }
    if luck.std(ddof=1) > 0:
        se_luck = luck.std(ddof=1) / n**0.5
        out["luck_t"] = float(luck.mean() / se_luck)
        out["sd_ratio"] = float(pts.std(ddof=1) / vr.std(ddof=1))
        out["var_ratio"] = out["sd_ratio"] ** 2
        out["corr_raw_luck"] = float(np.corrcoef(pts, luck)[0, 1])
        out["beta_hat"] = float(np.cov(pts, luck)[0, 1] / np.var(luck, ddof=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=400,
                    help="total games; split evenly across --workers shards")
    ap.add_argument("--blocks", type=int, default=0,
                    help="run this many independent matches of --block-size games "
                         "instead (each block is its own shard, for the demo figure)")
    ap.add_argument("--block-size", type=int, default=50)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--ply", type=int, default=0, help="GNUBG opponent ply")
    ap.add_argument("--cv-ply", type=int, default=0,
                    help="control-variate ply; must be 0 (best_move segfaults deeper)")
    ap.add_argument("--no-vr", action="store_true",
                    help="skip the control variate entirely; plays identical games")
    ap.add_argument("--greedy-doubles", action="store_true",
                    help="execute doubles by the old greedy two-step path instead of "
                         "enumerating all four half-moves jointly (the exp019 engine)")
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--exp-dir", default="", help="experiment dir for logs/ + results/")
    ap.add_argument("--tag", default="", help="results filename stem (defaults to label)")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    vr = not a.no_vr
    joint_doubles = not a.greedy_doubles
    if a.blocks:
        sizes = [a.block_size] * a.blocks
    else:
        sizes = [
            a.games // a.workers + (1 if k < a.games % a.workers else 0)
            for k in range(a.workers)
        ]
    sizes = [s for s in sizes if s > 0]
    seeds = [a.seed_base + k for k in range(len(sizes))]

    started = time.time()
    pts_parts: list[np.ndarray] = []
    luck_parts: list[np.ndarray] = []
    rolls_parts: list[np.ndarray] = []
    shard_parts: list[np.ndarray] = []
    wins = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [
            ex.submit(_shard, a.checkpoint, n, a.ply, a.cv_ply, seed, vr, joint_doubles)
            for n, seed in zip(sizes, seeds)
        ]
        for k, f in enumerate(futs):
            p, l, r, w = f.result()
            pts_parts.append(p)
            luck_parts.append(l)
            rolls_parts.append(r)
            shard_parts.append(np.full(len(p), k, dtype=np.int32))
            wins += w

    pts = np.concatenate(pts_parts)
    luck = np.concatenate(luck_parts)
    rolls = np.concatenate(rolls_parts)
    shard = np.concatenate(shard_parts)
    elapsed = time.time() - started

    tag = a.tag or a.label or Path(a.checkpoint).stem
    summary = {
        "tag": tag,
        "checkpoint": a.checkpoint,
        "gnubg_ply": a.ply,
        "cv_ply": a.cv_ply,
        "variance_reduction": vr,
        "joint_doubles": joint_doubles,
        "seed_base": a.seed_base,
        "shard_sizes": sizes,
        "net_wins": wins,
        "win_pct": 100.0 * wins / len(pts),
        "elapsed_sec": round(elapsed, 1),
        "sec_per_game_per_worker": round(elapsed * a.workers / len(pts), 3),
        "mean_rolls_per_game": round(float(rolls.mean()), 2),
        **summarise(pts, luck),
    }

    print(f"[{tag}] {a.checkpoint}", flush=True)
    print(f"  vs GNUBG-{a.ply}ply, n={summary['games']} games, "
          f"{wins} wins ({summary['win_pct']:.1f}%), "
          f"{elapsed / 60:.1f} min", flush=True)
    print(f"  raw ppg = {summary['raw_ppg']:+.4f}  (95% CI ±{summary['raw_ci95']:.4f})",
          flush=True)
    if vr:
        print(f"  VR  ppg = {summary['vr_ppg']:+.4f}  (95% CI ±{summary['vr_ci95']:.4f})"
              f"   <- {summary['sd_ratio']:.2f}x tighter "
              f"(= {summary['var_ratio']:.1f}x the games)", flush=True)
        print(f"  mean luck = {summary['mean_luck']:+.4f} (t={summary['luck_t']:+.2f}, "
              f"should be ~0)   corr(raw,luck) = {summary['corr_raw_luck']:.3f}   "
              f"beta_hat = {summary['beta_hat']:.3f}", flush=True)

    if a.exp_dir:
        exp = Path(a.exp_dir)
        (exp / "logs").mkdir(parents=True, exist_ok=True)
        (exp / "results").mkdir(parents=True, exist_ok=True)
        with (exp / "logs" / "vr_eval_log.jsonl").open("a") as fh:
            fh.write(json.dumps(summary) + "\n")
        record = dict(summary)
        record["game_pts"] = [int(x) for x in pts]
        record["game_luck"] = [round(float(x), 6) for x in luck]
        record["game_shard"] = [int(x) for x in shard]
        (exp / "results" / f"{tag}.json").write_text(json.dumps(record) + "\n")
        print(f"  wrote {exp / 'results' / f'{tag}.json'}", flush=True)


if __name__ == "__main__":
    main()
