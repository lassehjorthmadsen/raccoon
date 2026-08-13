"""Evaluate a checkpoint's 0-ply play vs GNUBG (at --ply) at large n, sharded.

Splits the games across worker processes (each loads its own net + gnubg_nn),
pools the result, and prints wins / equity-per-game with a rough 95% CI. Works
for both scalar and outcomes6 value heads (move selection reads value_equity).

    python scripts/eval_gnubg0.py --checkpoint .../best.pt --games 400 --workers 3
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from raccoon.model.network import load_model
from raccoon.train.td_selfplay import gnubg_arena


def _chunk(ckpt: str, n: int, ply: int, seed: int):
    torch.set_flush_denormal(True)
    torch.set_num_threads(1)
    net = load_model(ckpt)
    net.eval()
    r = gnubg_arena(net, torch.device("cpu"), n, gnubg_ply=ply, seed=seed)
    return r["net_wins"], r["game_pts"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--ply", type=int, default=0)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    per = [a.games // a.workers + (1 if k < a.games % a.workers else 0)
           for k in range(a.workers)]
    wins = 0
    parts = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_chunk, a.checkpoint, per[k], a.ply, 1000 + k)
                for k in range(a.workers) if per[k] > 0]
        for f in futs:
            w, p = f.result()
            wins += w; parts.append(p)

    pts = np.concatenate(parts)
    games = len(pts)
    ppg = float(pts.mean())
    # Empirical CI. This used to assume a per-game SD of 1.8, but 1.8 is the *variance*
    # (E[X^2] ~ 1.85 at near-parity, so SD ~ 1.36) — that constant overstated every
    # interval by ~32%. See docs/variance_reduction.qmd.
    sd = float(pts.std(ddof=1))
    ci = 1.96 * sd / games ** 0.5
    tag = f"[{a.label}] " if a.label else ""
    print(f"{tag}{a.checkpoint}", flush=True)
    print(f"  {wins}/{games} wins ({100 * wins / games:.1f}%), "
          f"{ppg:+.4f} ppg vs GNUBG-{a.ply}ply  (95% CI ±{ci:.4f}, per-game SD {sd:.3f})",
          flush=True)


if __name__ == "__main__":
    main()
