"""What the two-step doubles execution costs, sharded across processes (exp020).

Plays net-vs-GNUBG on the exp019 distribution and, at every net turn, compares the
position the net's *greedy two-step* doubles path comes to rest on against the one
*joint enumeration* over all four half-moves reaches. Disagreements are judged by
GNUBG at ``--judge-ply``; agreements are exactly zero and cost nothing. See
:mod:`raccoon.eval.doubles` for the estimator and the non-doubles control.

Writes an append-only summary line to ``<exp-dir>/logs/doubles_log.jsonl`` and an
idempotent ``<exp-dir>/results/<tag>.json`` carrying the summary **plus the per-turn
arrays**, so the analysis page recomputes its figures at render time from committed
data.

    # gate run
    python scripts/eval_doubles.py --checkpoint .../ep22.pt --games 1500 --workers 3 \
        --exp-dir experiments/exp020-doubles --tag ep22_concession
"""
from __future__ import annotations

import os

# CPU-bound torch: park idle OpenMP threads instead of busy-spinning. Precedes torch.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from raccoon.eval.doubles import doubles_concession_arena
from raccoon.model.network import load_model


def _shard(ckpt: str, n: int, ply: int, judge_ply: int, seed: int, play_joint: bool):
    torch.set_flush_denormal(True)
    torch.set_num_threads(1)
    net = load_model(ckpt)
    net.eval()
    r = doubles_concession_arena(
        net, torch.device("cpu"), n, gnubg_ply=ply, judge_ply=judge_ply,
        seed=seed, play_joint=play_joint,
    )
    return (r["game_pts"], r["two_step"], r["disagreed"], r["concession"],
            r["concession_0ply"], r["n_paths"])


def summarise(
    game_pts: np.ndarray, two_step: np.ndarray, disagreed: np.ndarray,
    concession: np.ndarray, concession_0ply: np.ndarray, n_paths: np.ndarray,
) -> dict:
    """Concession per doubles turn, its CI, and the ppg cost that implies."""
    games = len(game_pts)
    n_turns = len(concession)
    dbl = concession[two_step]
    n_dbl = len(dbl)

    out = {
        "games": games,
        "net_turns": n_turns,
        "two_step_turns": n_dbl,
        "two_step_per_game": round(n_dbl / games, 3) if games else 0.0,
        "raw_ppg": float(game_pts.mean()) if games else 0.0,
        # Control: a single-action turn enumerates one leaf per legal action in both
        # arms, so any disagreement here is a harness bug, not a strength difference.
        "non_doubles_turns": int((~two_step).sum()),
        "non_doubles_disagreements": int(disagreed[~two_step].sum()),
        "non_doubles_max_abs_concession": float(
            np.abs(concession[~two_step]).max()
        ) if (~two_step).any() else 0.0,
    }
    if n_dbl:
        sd = float(dbl.std(ddof=1))
        mean = float(dbl.mean())
        ci = 1.96 * sd / n_dbl**0.5
        nonzero = dbl[dbl != 0.0]
        out.update({
            "concession_mean": mean,
            "concession_sd": sd,
            "concession_ci95": float(ci),
            "disagreement_rate": float(disagreed[two_step].mean()),
            "concession_when_disagreed": float(nonzero.mean()) if len(nonzero) else 0.0,
            "ppg_cost": mean * n_dbl / games,
            "ppg_cost_ci95": float(ci * n_dbl / games),
            "mean_paths_per_two_step_turn": float(n_paths[two_step].mean()),
        })
        dbl0 = concession_0ply[two_step]
        out["concession_mean_0ply"] = float(dbl0.mean())
        out["concession_ci95_0ply"] = float(1.96 * dbl0.std(ddof=1) / n_dbl**0.5)
        out["ppg_cost_0ply"] = float(dbl0.mean() * n_dbl / games)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--ply", type=int, default=0, help="GNUBG opponent ply")
    ap.add_argument("--judge-ply", type=int, default=2,
                    help="GNUBG ply used to price a disagreement")
    ap.add_argument("--play-joint", action="store_true",
                    help="play the joint arm in the main line instead of the greedy "
                         "one (default: greedy, i.e. the exp019 distribution)")
    ap.add_argument("--seed-base", type=int, default=5000)
    ap.add_argument("--exp-dir", default="")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    sizes = [
        a.games // a.workers + (1 if k < a.games % a.workers else 0)
        for k in range(a.workers)
    ]
    sizes = [s for s in sizes if s > 0]
    seeds = [a.seed_base + k for k in range(len(sizes))]

    started = time.time()
    parts: list[tuple] = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [
            ex.submit(_shard, a.checkpoint, n, a.ply, a.judge_ply, seed, a.play_joint)
            for n, seed in zip(sizes, seeds)
        ]
        parts = [f.result() for f in futs]

    game_pts = np.concatenate([p[0] for p in parts])
    two_step = np.concatenate([p[1] for p in parts])
    disagreed = np.concatenate([p[2] for p in parts])
    concession = np.concatenate([p[3] for p in parts])
    concession_0ply = np.concatenate([p[4] for p in parts])
    n_paths = np.concatenate([p[5] for p in parts])
    elapsed = time.time() - started

    tag = a.tag or Path(a.checkpoint).stem
    summary = {
        "tag": tag,
        "checkpoint": a.checkpoint,
        "gnubg_ply": a.ply,
        "judge_ply": a.judge_ply,
        "main_line": "joint" if a.play_joint else "greedy",
        "seed_base": a.seed_base,
        "shard_sizes": sizes,
        "elapsed_sec": round(elapsed, 1),
        "sec_per_game_per_worker": round(elapsed * a.workers / len(game_pts), 3),
        **summarise(game_pts, two_step, disagreed, concession, concession_0ply,
                    n_paths),
    }

    print(f"[{tag}] {a.checkpoint}", flush=True)
    print(f"  {summary['games']} games vs GNUBG-{a.ply}ply, judged at "
          f"{a.judge_ply}-ply, {elapsed / 60:.1f} min", flush=True)
    print(f"  two-step doubles turns: {summary['two_step_turns']} "
          f"({summary['two_step_per_game']}/game)", flush=True)
    print(f"  CONTROL non-doubles turns: {summary['non_doubles_turns']}, "
          f"disagreements {summary['non_doubles_disagreements']} (must be 0), "
          f"max |concession| {summary['non_doubles_max_abs_concession']:.1e}", flush=True)
    if summary.get("two_step_turns"):
        print(f"  greedy != joint: {100 * summary['disagreement_rate']:.1f}% of them; "
              f"when it differs, mean {summary['concession_when_disagreed']:+.4f}",
              flush=True)
        print(f"  concession/doubles turn = {summary['concession_mean']:+.5f} "
              f"(95% CI ±{summary['concession_ci95']:.5f})", flush=True)
        print(f"  => ppg cost = {summary['ppg_cost']:+.4f} "
              f"(95% CI ±{summary['ppg_cost_ci95']:.4f})", flush=True)
        print(f"     cross-check, judged at 0-ply: ppg cost = "
              f"{summary['ppg_cost_0ply']:+.4f}", flush=True)
    print(f"  raw ppg on these games = {summary['raw_ppg']:+.4f} "
          f"(exp019 cross-check)", flush=True)

    if a.exp_dir:
        exp = Path(a.exp_dir)
        (exp / "logs").mkdir(parents=True, exist_ok=True)
        (exp / "results").mkdir(parents=True, exist_ok=True)
        with (exp / "logs" / "doubles_log.jsonl").open("a") as fh:
            fh.write(json.dumps(summary) + "\n")
        record = dict(summary)
        record["two_step"] = [bool(x) for x in two_step]
        record["disagreed"] = [bool(x) for x in disagreed]
        record["concession"] = [round(float(x), 6) for x in concession]
        record["concession_0ply"] = [round(float(x), 6) for x in concession_0ply]
        record["n_paths"] = [int(x) for x in n_paths]
        record["game_pts"] = [int(x) for x in game_pts]
        (exp / "results" / f"{tag}.json").write_text(json.dumps(record) + "\n")
        print(f"  wrote {exp / 'results' / f'{tag}.json'}", flush=True)


if __name__ == "__main__":
    main()
