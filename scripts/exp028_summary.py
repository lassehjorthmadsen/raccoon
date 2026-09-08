#!/usr/bin/env python3
"""exp028 — consolidate the arms into one artifact, with every comparison paired.

``docs/speed.qmd``'s exp028 section computes its tables from the file this script
writes. Nothing in the write-up recomputes a number; if a figure there needs to
change, it changes here.

**Why paired, and why clustered on games.** Every epoch of every arm is scored on
the same 14,693 BGSage checker decisions, and ``eval_benchmark_pr.py
--error-dump`` records the per-decision error for each. Two arms therefore differ
only on the decisions where they actually pick different moves, and comparing
their PR numbers independently discards that. The decisions come from 500 games
and positions within a game are the same game played on, so intervals resample
whole games, as exp024 and exp025 did for the same reason.

The resulting resolution is about 0.09 PR. Differences smaller than that are
reported as *not resolved*, which is a statement about the instrument and not a
null result.

    python scripts/exp028_summary.py --exp-dir experiments/exp028-mlp
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

PR_MULTIPLIER = 500
BOOTSTRAP = 10_000
ARMS = ("ref20", "scalar20", "cosine20", "combined20")
EPOCHS = 20

# Everything below is measured elsewhere and quoted here so the write-up has a
# single source. Provenance is in the value's "source" field.
REFERENCE_POINTS = {
    "ep22_resnet_10x256": {
        "pr": 0.950, "macs": 285_000_000,
        "source": "exp021 ladder; MACs from exp027-speed-audit/results/eval_speed.json",
    },
    "gnubg_1ply": {"pr": 1.206, "source": "exp021 ladder, same benchmark"},
    "gnubg_0ply": {"pr": 2.14, "macs": 33_000, "source": "exp021 ladder"},
    "puretd": {
        "pr": 0.94, "macs": 561_000, "parameters": 562_000,
        "source": "Strehl, arXiv:2608.15146",
        "note": "0.94 is XG++ Performance Rating on CUBEFUL money, not this "
                "benchmark's cubeless checker PR. The scales agree on the GNUBG "
                "anchors they share (0-ply 2.18 there against 2.14 here, 1-ply "
                "1.21 against 1.206), which is a calibration argument, not "
                "identity.",
    },
}

OUR_TRAINING_COST = {
    "core_hours_per_arm": 38,
    "gpu": False,
    "puretd_core_hours": 4200,
    "puretd_note": "200M self-play games, ~65 h on a 64-core workstation with a "
                   "GPU. Capacity and inference cost are matched with this MLP; "
                   "training compute is not, by about two orders of magnitude.",
}

ARCHITECTURE_CURVE_NOTE = (
    "These arms ran under the constant learning rate later shown to leave about "
    "0.2 PR unclaimed, so they order the configurations correctly but none of "
    "them is converged."
)


def clustered_ci(delta: np.ndarray, groups: np.ndarray, rng, n_boot=BOOTSTRAP):
    """95% interval on ``mean(delta)``, resampling whole games with replacement."""
    keys, inverse = np.unique(groups, return_inverse=True)
    by_group = [np.flatnonzero(inverse == g) for g in range(len(keys))]
    sums = np.array([delta[idx].sum() for idx in by_group])
    counts = np.array([len(idx) for idx in by_group], dtype=np.float64)
    draws = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def load(dumps: str, arm: str, epoch: int):
    d = np.load(os.path.join(dumps, f"{arm}_e{epoch}.npz"), allow_pickle=True)
    return d["error"], d["game_seed"]


def compare(dumps: str, a: tuple[str, int], b: tuple[str, int], rng) -> dict:
    """Paired PR gain of ``b`` over ``a``: positive means b makes smaller errors."""
    err_a, games = load(dumps, *a)
    err_b, games_b = load(dumps, *b)
    assert np.array_equal(games, games_b), "dumps are not the same decisions"
    delta = err_a - err_b
    lo, hi = clustered_ci(delta, games, rng)
    gain = PR_MULTIPLIER * float(delta.mean())
    ci = [PR_MULTIPLIER * lo, PR_MULTIPLIER * hi]
    return {
        "baseline": f"{a[0]}_e{a[1]}", "arm": f"{b[0]}_e{b[1]}",
        "pr_baseline": PR_MULTIPLIER * float(err_a.mean()),
        "pr_arm": PR_MULTIPLIER * float(err_b.mean()),
        "gain": gain, "ci95": ci,
        "verdict": "real" if ci[0] > 0 else "not resolved",
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", default="experiments/exp028-mlp")
    ap.add_argument("--seed", type=int, default=28)
    a = ap.parse_args()
    dumps = os.path.join(a.exp_dir, "dumps")
    rng = np.random.default_rng(a.seed)

    err, games = load(dumps, "ref20", EPOCHS)
    out = {
        "protocol": {
            "benchmark": "BGSage Money Benchmark, checker decisions, all tiers",
            "n": int(len(err)),
            "n_games": int(len(np.unique(games))),
            "subsampling": "none",
            "pairing": "per-decision, identical decisions across every arm and epoch",
            "bootstrap": f"{BOOTSTRAP} resamples of whole games",
            "resolution_pr": 0.09,
            "resolution_note": "differences below this are reported as 'not "
                               "resolved', which is not a null result",
            "training_data": "43M GNUBG 2-ply labels, data/distill/2ply",
            "architecture": "MLP [512,512,256,256], 201 de-broadcast inputs",
            "epochs": EPOCHS,
        },
        "reference_points": REFERENCE_POINTS,
        "our_training_cost": OUR_TRAINING_COST,
        "architecture_curve_note": ARCHITECTURE_CURVE_NOTE,
        "architecture_curve": [
            {"name": "MLP [256], 624 inputs", "pr": 5.40},
            {"name": "MLP [256,256], 624 inputs", "pr": 2.99},
            {"name": "MLP [512,512], 624 inputs", "pr": 2.31},
            {"name": "MLP [512,512,256,256], 624 inputs", "pr": 1.93},
            {"name": "...dropping the handcrafted channels, 408 inputs", "pr": 1.74},
            {"name": "...de-broadcast to 201 inputs", "pr": 1.68},
        ],
        "arms": {}, "paired": {},
    }

    for arm in ARMS:
        curve = {}
        for ep in range(1, EPOCHS + 1):
            path = os.path.join(dumps, f"{arm}_e{ep}.npz")
            if os.path.exists(path):
                e, _ = load(dumps, arm, ep)
                curve[str(ep)] = PR_MULTIPLIER * float(e.mean())
        out["arms"][arm] = {"pr_by_epoch": curve,
                            "pr_final": curve[str(EPOCHS)]}

    p = out["paired"]
    p["scalar_vs_outcomes6"] = compare(dumps, ("ref20", 20), ("scalar20", 20), rng)
    p["cosine_vs_constant_lr"] = compare(dumps, ("ref20", 20), ("cosine20", 20), rng)
    p["combined_vs_reference"] = compare(dumps, ("ref20", 20), ("combined20", 20), rng)
    p["combined_vs_cosine"] = compare(dumps, ("cosine20", 20), ("combined20", 20), rng)
    p["combined_vs_scalar"] = compare(dumps, ("scalar20", 20), ("combined20", 20), rng)
    for arm in ARMS:
        p[f"{arm}_e10_to_e20"] = compare(dumps, (arm, 10), (arm, 20), rng)
        p[f"{arm}_e15_to_e20"] = compare(dumps, (arm, 15), (arm, 20), rng)

    ref = out["arms"]["ref20"]["pr_final"]
    out["additivity"] = {
        "predicted_if_independent": ref - p["scalar_vs_outcomes6"]["gain"]
                                        - p["cosine_vs_constant_lr"]["gain"],
        "observed": out["arms"]["combined20"]["pr_final"],
        "note": "The prediction was written into the run script before the "
                "combined arm started. The effects overlap: the schedule does "
                "most of the work.",
    }

    results = os.path.join(a.exp_dir, "results")
    os.makedirs(results, exist_ok=True)
    path = os.path.join(results, "exp028_summary.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)

    for arm in ARMS:
        print(f"{arm:<12} PR e20 = {out['arms'][arm]['pr_final']:.3f}")
    print()
    for key, v in p.items():
        print(f"{key:<28} {v['gain']:+.3f} "
              f"[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  {v['verdict']}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
