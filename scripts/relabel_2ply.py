"""Re-label existing 0-ply shards at a higher GNUBG ply.

Reads shard_*.npz from --in-dir, decodes each stored observation back to a
board position, calls gnubg_nn at the requested ply, and writes new shards to
--out-dir with the same observations but updated equity and outcomes6.

Existing output shards are skipped (safe to resume after interruption).

Speed (benchmarked on 16-core Windows work PC via WSL):
  ply=2: ~0.025 s/pos → ~145k pos/h/core → 14 workers ≈ 2M pos/h.
  All 18 shards (9M positions) ≈ 4.5 h.

    python scripts/relabel_2ply.py \\
        --in-dir experiments/exp011-distill/cache \\
        --out-dir experiments/exp011-distill/cache_2ply \\
        --ply 2 --workers 14
"""
from __future__ import annotations

import argparse
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def _outcomes6_and_equity(probs):
    """Cumulative (win, wg, wbg, lg, lbg) -> (six-outcome dist, equity in ppg).

    Identical to the helper in gen_gnubg_selfplay.py — same column order, same
    normalisation.
    """
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


def _relabel_worker(in_path: str, out_path: str, ply: int) -> int:
    """Re-evaluate all positions in one shard at `ply` and write a new shard."""
    # Imported inside the worker so a spawn pool gets a fresh gnubg_nn per process.
    from raccoon.env.encoder import decode_base_planes
    from raccoon.eval.gnubg_adapter import board_from_view, outcome_probs

    data = np.load(in_path)
    obs = data["observations"]   # float16, (N, 26, 2, 12) — pass through unchanged
    n = len(obs)

    eq_buf = np.empty(n, dtype=np.float32)
    six_buf = np.empty((n, 6), dtype=np.float32)

    for i in range(n):
        view = decode_base_planes(obs[i])
        board = board_from_view(view)
        probs = outcome_probs(board, ply)
        six, equity = _outcomes6_and_equity(probs)
        eq_buf[i] = np.float32(equity / 3.0)
        six_buf[i] = six

    np.savez(out_path,
             observations=obs,
             equity=eq_buf,
             outcomes6=six_buf)
    data.close()
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", required=True,
                   help="Directory containing 0-ply shard_*.npz files")
    p.add_argument("--out-dir", required=True,
                   help="Output directory for relabelled shards (same filenames)")
    p.add_argument("--ply", type=int, default=2,
                   help="GNUBG ply to relabel at (default: 2)")
    p.add_argument("--workers", type=int, default=14,
                   help="Number of parallel worker processes")
    p.add_argument("--max-shards", type=int, default=None,
                   help="Stop after processing this many shards (for staged runs)")
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(in_dir.glob("shard_*.npz"))
    if args.max_shards is not None:
        shards = shards[:args.max_shards]

    todo = [(s, out_dir / s.name) for s in shards if not (out_dir / s.name).exists()]
    skipped = len(shards) - len(todo)

    print(f"{len(shards)} shards found, {skipped} already done, "
          f"{len(todo)} to relabel at ply={args.ply}", flush=True)
    if not todo:
        print("nothing to do", flush=True)
        return

    t0 = time.time()
    total = 0
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futs = {ex.submit(_relabel_worker, str(s), str(o), args.ply): s
                for s, o in todo}
        for fut in as_completed(futs):
            n = fut.result()
            total += n
            dt = time.time() - t0
            rate = total / max(dt, 1)
            remaining = sum(1 for f in futs if not f.done())
            print(f"  done {futs[fut].name}: {n:,} pos "
                  f"({total:,} total, {rate:.0f} pos/s, "
                  f"{remaining} shards remaining)", flush=True)

    elapsed_h = (time.time() - t0) / 3600
    print(f"\nrelabelled {total:,} positions in {elapsed_h:.2f}h "
          f"({total/max(elapsed_h,0.001)/1e6:.1f}M pos/h)", flush=True)


if __name__ == "__main__":
    main()
