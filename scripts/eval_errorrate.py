"""Fine eval: raw ppg + low-variance error-rate for a checkpoint vs GNUBG.

Runs ``gnubg_arena_scored`` across worker processes and pools per-game results,
reporting from the SAME net-vs-GNUBG games:

  - **raw ppg** — mean game result. Unbiased ground truth, high variance
    (per-game SD ~1.8, so CI = ±3.53/sqrt(n)).
  - **error-rate ppg** — minus the mean per-game net error (equity the net's move
    choices concede to GNUBG, summed per game). Same quantity as raw ppg up to a
    small luck offset, but MUCH lower variance → the fine ruler. Cross-check: it
    must agree with raw ppg within the (wide) raw CI; a gap reveals the offset.
  - **error rate** — points conceded per net decision (mppg/decision), the
    standard backgammon skill metric. Exact measurement, no control-variate bias.

    python scripts/eval_errorrate.py --checkpoint .../ep3.pt --games 2000 --workers 3
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from raccoon.model.network import load_model
from raccoon.train.td_selfplay import gnubg_arena_scored


def _chunk(ckpt: str, n: int, ply: int, seed: int):
    torch.set_flush_denormal(True)
    torch.set_num_threads(1)
    net = load_model(ckpt)
    net.eval()
    r = gnubg_arena_scored(net, torch.device("cpu"), n, ref_ply=ply, seed=seed)
    return r["game_pts"], r["game_err"], r["decisions"], r["net_wins"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--ply", type=int, default=0)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    per = [a.games // a.workers + (1 if k < a.games % a.workers else 0)
           for k in range(a.workers)]
    pts_parts: list[np.ndarray] = []
    err_parts: list[np.ndarray] = []
    decisions = wins = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_chunk, a.checkpoint, per[k], a.ply, 1000 + k)
                for k in range(a.workers) if per[k] > 0]
        for f in futs:
            gp, ge, d, w = f.result()
            pts_parts.append(gp); err_parts.append(ge); decisions += d; wins += w

    pts = np.concatenate(pts_parts)
    err = np.concatenate(err_parts)
    n = len(pts)

    raw_ppg = float(pts.mean())
    raw_ci = 1.96 * float(pts.std(ddof=1)) / n ** 0.5
    proxy_ppg = -float(err.mean())
    proxy_ci = 1.96 * float(err.std(ddof=1)) / n ** 0.5
    err_rate = float(err.sum()) / decisions if decisions else 0.0

    tag = f"[{a.label}] " if a.label else ""
    print(f"{tag}{a.checkpoint}", flush=True)
    print(f"  vs GNUBG-{a.ply}ply, n={n} games, {decisions} net decisions, "
          f"{100 * wins / n:.1f}% wins", flush=True)
    print(f"  raw ppg        = {raw_ppg:+.4f}  (95% CI ±{raw_ci:.4f})", flush=True)
    print(f"  error-rate ppg = {proxy_ppg:+.4f}  (95% CI ±{proxy_ci:.4f})   "
          f"<- low variance", flush=True)
    print(f"  error rate     = {1000 * err_rate:.2f} mppg/decision", flush=True)


if __name__ == "__main__":
    main()
