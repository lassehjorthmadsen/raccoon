#!/usr/bin/env python3
"""exp029 — consolidate the long run into one artifact, paired against exp028.

``docs/speed.qmd``'s exp029 section computes its tables from the file this script
writes. Nothing in the write-up recomputes a number; if a figure there needs to
change, it changes here.

The estimator is exp028's, imported rather than reimplemented: the same
per-decision pairing on identical BGSage decisions and the same interval
bootstrapped over whole games. What differs is only that the baseline dump lives
in another experiment's directory, so the loader here takes paths.

    python scripts/exp029_summary.py --exp-dir experiments/exp029-longrun
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp028_summary import BOOTSTRAP, PR_MULTIPLIER, clustered_ci  # noqa: E402

ARM = "combined100"
BASELINE_DUMP = "experiments/exp028-mlp/dumps/combined20_e20.npz"
BASELINE_LABEL = "combined20_e20"
# The baseline's own supporting metric, so the write-up can put the two fits side
# by side from one file. Re-scoring combined20/ep20.pt is deterministic and
# reproduces exp028's PR of 1.3683519644073494 exactly, which is also the check
# that the two arms are being compared on identical decisions.
BASELINE_ROLLOUT = {"mse": 0.0021854185588796775, "r2": 0.996355277928583,
                    "n_positions": 149113,
                    "source": "scripts/eval_benchmark_pr.py on "
                              "experiments/exp028-mlp/combined20/checkpoints/ep20.pt"}

# exp028's pre-registered reading of the result, written before this run started.
DECISION_RULE = [
    {"gain_at_least": 0.27, "reading": "scale is the dominant explanation",
     "next": "keep scaling, and get more labels, before any RL work"},
    {"gain_at_least": 0.09, "reading": "partial",
     "next": "report the extrapolation of epochs needed for the rest"},
    {"gain_at_least": None, "reading": "training scale on fixed labels is exhausted",
     "next": "the remaining difference is method or the 43M-label ceiling; "
             "PureTD-style TD becomes the next candidate, not the next assumption"},
]


def load(path: str):
    d = np.load(path, allow_pickle=True)
    return d["error"], d["game_seed"]


def compare(path_a: str, label_a: str, path_b: str, label_b: str, rng) -> dict:
    """Paired PR gain of ``b`` over ``a``: positive means b makes smaller errors."""
    err_a, games = load(path_a)
    err_b, games_b = load(path_b)
    assert np.array_equal(games, games_b), "dumps are not the same decisions"
    delta = err_a - err_b
    lo, hi = clustered_ci(delta, games, rng)
    ci = [PR_MULTIPLIER * lo, PR_MULTIPLIER * hi]
    return {
        "baseline": label_a, "arm": label_b,
        "pr_baseline": PR_MULTIPLIER * float(err_a.mean()),
        "pr_arm": PR_MULTIPLIER * float(err_b.mean()),
        "gain": PR_MULTIPLIER * float(delta.mean()),
        "ci95": ci,
        "verdict": "real" if ci[0] > 0 else "not resolved",
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", default="experiments/exp029-longrun")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=29)
    a = ap.parse_args()
    dumps = os.path.join(a.exp_dir, "dumps")
    scratch = os.path.join(a.exp_dir, "scratch")
    rng = np.random.default_rng(a.seed)

    def dump_path(ep: int) -> str:
        return os.path.join(dumps, f"{ARM}_e{ep}.npz")

    scored = [ep for ep in range(1, a.epochs + 1) if os.path.exists(dump_path(ep))]
    if not scored:
        raise SystemExit(f"no dumps in {dumps} — run the scoring pass first")
    last = scored[-1]

    err, games = load(dump_path(last))
    exp028 = json.load(open(
        "experiments/exp028-mlp/results/exp028_summary.json"))

    out = {
        "protocol": {
            "benchmark": "BGSage Money Benchmark, checker decisions, all tiers",
            "n": int(len(err)),
            "n_games": int(len(np.unique(games))),
            "subsampling": "none",
            "pairing": "per-decision, identical decisions across every epoch and "
                       "against exp028's combined20",
            "bootstrap": f"{BOOTSTRAP} resamples of whole games",
            "resolution_pr": 0.09,
            "resolution_note": "differences below this are reported 'not resolved', "
                               "which is not a null result",
            "training_data": "43M GNUBG 2-ply labels, data/distill/2ply — the same "
                             "86 shards exp028 read, so the only scale axis varied "
                             "here is passes over fixed data, not fresh data",
            "architecture": "MLP [512,512,256,256], 201 de-broadcast inputs, scalar "
                            "head, cosine LR — exp028's combined arm unchanged",
            "epochs": last,
            "baseline": BASELINE_LABEL,
            "supporting_metric": "rollout-tier MSE/R^2 on ~149k candidate positions "
                                 "never trained on, from the same scoring pass",
        },
        "decision_rule": DECISION_RULE,
        "reference_points": exp028["reference_points"],
        "combined20_pr_by_epoch": exp028["arms"]["combined20"]["pr_by_epoch"],
        "baseline_rollout": BASELINE_ROLLOUT,
        "pr_by_epoch": {}, "rollout_by_epoch": {}, "paired": {},
    }

    for ep in scored:
        e, _ = load(dump_path(ep))
        out["pr_by_epoch"][str(ep)] = PR_MULTIPLIER * float(e.mean())
        js = os.path.join(scratch, f"{ARM}_e{ep}.json")
        if os.path.exists(js):
            acc = json.load(open(js))["eval_accuracy"]["rollout"]
            out["rollout_by_epoch"][str(ep)] = {"mse": acc["mse"], "r2": acc["r2"],
                                                "n_positions": acc["n_positions"]}

    out["pr_final"] = out["pr_by_epoch"][str(last)]

    p = out["paired"]
    p["vs_combined20"] = compare(BASELINE_DUMP, BASELINE_LABEL,
                                 dump_path(last), f"{ARM}_e{last}", rng)
    # Long intervals only: exp028 showed adjacent-epoch steps read "not resolved"
    # three times in a row on a curve that was still descending.
    for lo, hi in ((20, 50), (50, last), (max(1, last - 10), last)):
        if lo in scored and hi in scored and lo != hi:
            p[f"e{lo}_to_e{hi}"] = compare(dump_path(lo), f"{ARM}_e{lo}",
                                           dump_path(hi), f"{ARM}_e{hi}", rng)

    gain = p["vs_combined20"]["gain"]
    resolved = p["vs_combined20"]["ci95"][0] > 0
    rule = (DECISION_RULE[0] if resolved and gain >= 0.27 else
            DECISION_RULE[1] if resolved and gain >= 0.09 else DECISION_RULE[2])
    out["conclusion"] = {
        "gain_over_combined20": gain,
        "ci95": p["vs_combined20"]["ci95"],
        "reading": rule["reading"], "next": rule["next"],
        "pr_final": out["pr_final"],
        "ep22_pr": exp028["reference_points"]["ep22_resnet_10x256"]["pr"],
    }

    results = os.path.join(a.exp_dir, "results")
    os.makedirs(results, exist_ok=True)
    path = os.path.join(results, "exp029_summary.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)

    print(f"{ARM} PR e{last} = {out['pr_final']:.3f} "
          f"(combined20 e20 = {p['vs_combined20']['pr_baseline']:.3f})")
    for key, v in p.items():
        print(f"{key:<20} {v['gain']:+.3f} "
              f"[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  {v['verdict']}")
    print(f"\nreading: {out['conclusion']['reading']}")
    print(f"next:    {out['conclusion']['next']}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
