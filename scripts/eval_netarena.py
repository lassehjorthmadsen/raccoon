"""Evaluate one checkpoint's 0-ply greedy play against ANOTHER checkpoint, at
large n, sharded across worker processes — the checkpoint-vs-checkpoint analog
of eval_gnubg0.py. Both sides play greedily (temperature=0, no MCTS/search),
matching how these value-only distilled nets are trained and evaluated
elsewhere (raccoon.train.td_selfplay.net_arena / gnubg_arena). Not a
substitute for raccoon.eval.arena.Arena, which is for MCTS-search checkpoints.

    python scripts/eval_netarena.py --checkpoint-a A.pt --checkpoint-b B.pt \\
        --games 3000 --workers 3
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import torch

from raccoon.model.network import load_model
from raccoon.train.td_selfplay import net_arena


def _chunk(ckpt_a: str, ckpt_b: str, n: int, seed: int):
    torch.set_flush_denormal(True)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    net_a = load_model(ckpt_a).to(device)
    net_a.eval()
    net_b = load_model(ckpt_b).to(device)
    net_b.eval()
    r = net_arena(net_a, net_b, device, n, seed=seed)
    return r["net_a_wins"], r["games"], r["equity_per_game"] * r["games"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-a", required=True)
    ap.add_argument("--checkpoint-b", required=True)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    per = [a.games // a.workers + (1 if k < a.games % a.workers else 0)
           for k in range(a.workers)]
    wins = games = 0
    eq_sum = 0.0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_chunk, a.checkpoint_a, a.checkpoint_b, per[k], 2000 + k)
                for k in range(a.workers) if per[k] > 0]
        for f in futs:
            w, g, e = f.result()
            wins += w; games += g; eq_sum += e

    ppg = eq_sum / games
    ci = 1.96 * 1.8 / (games ** 0.5)  # per-game equity SD ~1.8 (see docs)
    tag = f"[{a.label}] " if a.label else ""
    print(f"{tag}A={a.checkpoint_a}", flush=True)
    print(f"     B={a.checkpoint_b}", flush=True)
    print(f"  {wins}/{games} wins for A ({100 * wins / games:.1f}%), "
          f"{ppg:+.4f} ppg (A's POV)  (95% CI ~±{ci:.3f})", flush=True)


if __name__ == "__main__":
    main()
