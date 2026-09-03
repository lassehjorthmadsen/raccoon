#!/usr/bin/env python3
"""exp025 — what cube-aware checker ranking is worth, paired on identical decisions.

Reads the per-decision error file written by

    scripts/eval_benchmark_pr.py --metric cubeful --cubeful-rank --paired-dump ...

and reports the PR difference between ranking candidates by cubeless equity and
ranking them by Janowski cubeful equity, on the same 14,693 BGSage checker
decisions, with the same net, scored against the same cubeful references.

**Why paired.** The two arms agree on the great majority of decisions — exp024
measured cubeless-best and cubeful-best differing on 6.8% of them — so their
per-decision errors are identical almost everywhere. Comparing two independently
computed PR numbers throws that away and leaves an interval far wider than the
effect. The dump solves it at the source: both errors come from the *same*
six-outcome forward pass, since for an outcomes6 head the value head's equity is
exactly ``cubeless_equity`` of its own distribution. (Checked: the derived
cubeless arm reproduces a standalone ``--metric cubeful`` run to rounding.)

**Why clustered.** Decisions are drawn from 500 games and positions within a game
are the same game played on, so they are not independent. Intervals are
bootstrapped over *games*, which is what exp024 did for the same reason and
roughly doubles them against treating decisions as independent.

**What it cannot tell you.** This is move selection only. The double/take
decision is not in it, and neither is play against an opponent — exp025 does not
contain a head-to-head, because gnubg-nn cannot make a money cube decision and
so cannot supply an honest cubeful opponent. That is exp026's job; see the
"WHY THE HEAD-TO-HEAD IS NOT IN THIS EXPERIMENT" section of
experiments/pipeline_exp025.sh.

    python scripts/exp025_paired_pr.py \\
        --dump experiments/exp025-cube-ranking/dumps/ep22_paired.npz \\
        --output experiments/exp025-cube-ranking/results
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

PR_MULTIPLIER = 500
BOOTSTRAP = 10_000


def clustered_ci(delta: np.ndarray, groups: np.ndarray, rng, n_boot=BOOTSTRAP):
    """95% interval on ``mean(delta)``, resampling whole groups with replacement."""
    keys, inverse = np.unique(groups, return_inverse=True)
    by_group = [np.flatnonzero(inverse == g) for g in range(len(keys))]
    sums = np.array([delta[idx].sum() for idx in by_group])
    counts = np.array([len(idx) for idx in by_group], dtype=np.float64)
    draws = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi), float(means.std(ddof=1))


def summarise(cubeless: np.ndarray, cubeful: np.ndarray, groups: np.ndarray,
              rng) -> dict:
    delta = cubeless - cubeful          # positive = cubeful ranking is better
    lo, hi, sd = clustered_ci(delta, groups, rng)
    return {
        "n": int(len(delta)),
        "n_games": int(len(np.unique(groups))),
        "pr_cubeless_rank": float(PR_MULTIPLIER * cubeless.mean()),
        "pr_cubeful_rank": float(PR_MULTIPLIER * cubeful.mean()),
        "pr_gain": float(PR_MULTIPLIER * delta.mean()),
        "pr_gain_ci95": [float(PR_MULTIPLIER * lo), float(PR_MULTIPLIER * hi)],
        "pr_gain_se": float(PR_MULTIPLIER * sd),
        "decisions_where_ranking_differs": int((cubeless != cubeful).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True, help="npz from --paired-dump")
    ap.add_argument("--output", help="directory to write exp025_paired_pr.json into")
    ap.add_argument("--seed", type=int, default=25)
    a = ap.parse_args()

    d = np.load(a.dump, allow_pickle=True)
    cubeless = d["error_cubeless_rank"]
    cubeful = d["error_cubeful_rank"]
    games = d["game_seed"]
    rng = np.random.default_rng(a.seed)

    out = {
        "measurement": "cubeful PR, cubeless ranking vs cubeful ranking, paired",
        "benchmark": "BGSage Money Benchmark (checker decisions, cubeful reference)",
        "metric": str(d["metric"]),
        "cube_jacoby": bool(d["cube_jacoby"]),
        "picks_differ": int(d["picks_differ"].sum()),
        "picks_differ_fraction": float(d["picks_differ"].mean()),
        "overall": summarise(cubeless, cubeful, games, rng),
    }
    for key in ("cube_owner", "game_plan", "tier"):
        labels = d[key]
        out["by_" + key] = {
            str(v): summarise(cubeless[labels == v], cubeful[labels == v],
                              games[labels == v], rng)
            for v in np.unique(labels)
        }

    o = out["overall"]
    print(f"n = {o['n']:,} decisions over {o['n_games']} games")
    print(f"  cubeless ranking   PR {o['pr_cubeless_rank']:.3f}")
    print(f"  cubeful  ranking   PR {o['pr_cubeful_rank']:.3f}")
    print(f"  gain               {o['pr_gain']:+.3f} "
          f"(95% CI {o['pr_gain_ci95'][0]:+.3f} to {o['pr_gain_ci95'][1]:+.3f}, "
          f"clustered on game)")
    print(f"  rankings differ on {out['picks_differ']:,} decisions "
          f"({out['picks_differ_fraction']:.1%})")
    print()
    print(f"  {'cube position':<24} {'cubeless':>9} {'cubeful':>9} {'gain':>9}  95% CI")
    for owner, v in out["by_cube_owner"].items():
        print(f"  {owner:<24} {v['pr_cubeless_rank']:>9.3f} "
              f"{v['pr_cubeful_rank']:>9.3f} {v['pr_gain']:>+9.3f}  "
              f"[{v['pr_gain_ci95'][0]:+.3f}, {v['pr_gain_ci95'][1]:+.3f}]")

    if a.output:
        os.makedirs(a.output, exist_ok=True)
        path = os.path.join(a.output, "exp025_paired_pr.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
