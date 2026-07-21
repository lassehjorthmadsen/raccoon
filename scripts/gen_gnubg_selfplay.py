"""exp011 — build a GNUBG-0-ply value-distillation dataset by GNUBG self-play.

GNUBG plays itself (both sides ``pick_move(state, 0)``, greedy — the dice supply
coverage, so no temperature). At each start-of-turn pre-roll position we record
the 26-channel observation plus GNUBG's 0-ply five-tuple, from which both targets
follow:

  - equity (scalar, stored /3 to match the value head's [-1,1] convention)
  - the six mutually-exclusive outcome probabilities
    ``[win_single, win_gammon, win_bg, lose_single, lose_gammon, lose_bg]``.

Positions are written as sharded ``.npz`` (obs float16, equity float32,
outcomes6 float32) so a 10M-position set streams through training (scripts/
train_distill.py) without holding it all in RAM. Net-free and fast (~0.003 s per
decision); parallel across workers via a spawn pool (fresh gnubg_nn per process).

    python scripts/gen_gnubg_selfplay.py --out-dir experiments/exp011-distill/cache \\
        --positions 10000000 --shard-size 500000 --workers 3
"""
from __future__ import annotations

import argparse
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np


def _outcomes6_and_equity(probs):
    """Cumulative (win, wg, wbg, lg, lbg) -> (six-outcome dist, equity in ppg)."""
    win, wg, wbg, lg, lbg = probs
    lose = 1.0 - win
    six = np.array([win - wg, wg - wbg, wbg, lose - lg, lg - lbg, lbg],
                   dtype=np.float32)
    six = np.clip(six, 0.0, None)          # scrub tiny negatives from rounding
    s = float(six.sum())
    if s > 0.0:
        six /= s
    equity = win + wg + wbg - lose - lg - lbg
    return six, float(equity)


def _worker(worker_id: int, n_positions: int, shard_size: int,
            out_dir: str, seed_base: int) -> int:
    # Imported inside the worker so a spawn pool gets a fresh gnubg_nn per process.
    from raccoon.env.encoder import encode_state
    from raccoon.env.game_wrapper import GameWrapper
    from raccoon.eval.gnubg_adapter import board_from_view, outcome_probs, pick_move
    from raccoon.search.mcts import _advance_through_chance

    np.random.seed(seed_base)
    wrapper = GameWrapper()
    obs_buf, eq_buf, six_buf = [], [], []
    written = 0
    shard_k = 0

    def flush():
        nonlocal shard_k, obs_buf, eq_buf, six_buf
        if not obs_buf:
            return
        path = Path(out_dir) / f"shard_w{worker_id}_{shard_k:04d}.npz"
        np.savez(path,
                 observations=np.stack(obs_buf).astype(np.float16),
                 equity=np.array(eq_buf, dtype=np.float32),
                 outcomes6=np.stack(six_buf).astype(np.float32))
        shard_k += 1
        obs_buf, eq_buf, six_buf = [], [], []

    while written < n_positions:
        state = _advance_through_chance(wrapper.new_game())
        moves = 0
        while not state.is_terminal() and moves < 2000:
            bv = state.board_from_perspective()
            if not bv.mid_doubles:  # one record per turn (start-of-turn pre-roll)
                pre = replace(bv, dice=None, mid_doubles=False)
                six, equity = _outcomes6_and_equity(
                    outcome_probs(board_from_view(pre), 0))
                obs_buf.append(encode_state(pre))
                eq_buf.append(np.float32(equity / 3.0))
                six_buf.append(six)
                written += 1
                if len(obs_buf) >= shard_size:
                    flush()
                if written >= n_positions:
                    break
            state.apply_action(pick_move(state, 0))
            state = _advance_through_chance(state)
            moves += 1
    flush()
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--positions", type=int, default=10_000_000)
    p.add_argument("--shard-size", type=int, default=500_000)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per = args.positions // args.workers
    print(f"generating ~{args.positions:,} positions, {args.workers} workers, "
          f"~{per:,} each, shard {args.shard_size:,}", flush=True)

    t0 = time.time()
    ctx = multiprocessing.get_context("spawn")
    total = 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futs = [ex.submit(_worker, w, per, args.shard_size, str(out),
                          args.seed + w * 10_007)
                for w in range(args.workers)]
        for f in futs:
            total += f.result()

    n_shards = len(list(out.glob("shard_*.npz")))
    dt = time.time() - t0
    print(f"wrote {total:,} positions in {n_shards} shards "
          f"({dt/60:.1f} min, {total/max(dt,1):.0f} pos/s)", flush=True)


if __name__ == "__main__":
    main()
