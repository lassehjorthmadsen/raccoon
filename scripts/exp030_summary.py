#!/usr/bin/env python3
"""exp030 — consolidate the capacity arm, paired against both 1x references.

Same estimator as exp028 and exp029, imported rather than reimplemented: paired
on identical BGSage decisions, interval bootstrapped over whole games.

Two baselines, both fixed before the run. The primary is exp028's combined20 at
epoch 20 — the same recipe at the same budget, differing only in width. The
secondary is exp029's combined100 at epoch 100, the 1x network at its best under
any budget, so a win cannot be dismissed as beating an unfinished baseline.

    python scripts/exp030_summary.py --exp-dir experiments/exp030-capacity
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp028_summary import BOOTSTRAP, PR_MULTIPLIER  # noqa: E402
from exp029_summary import compare, load  # noqa: E402

ARM = "width2x"
BASELINES = {
    "vs_1x_same_budget": ("experiments/exp028-mlp/dumps/combined20_e20.npz",
                          "combined20_e20"),
    "vs_1x_best": ("experiments/exp029-longrun/dumps/combined100_e100.npz",
                   "combined100_e100"),
}

# Multiply-accumulates in the value-only forward path 0-ply play actually uses,
# from the layer shapes alone: 201 inputs, four hidden layers, scalar head.
# Counted from the layer shapes of the constructed networks, not by hand.
MACS = {"1x [512,512,256,256]": 561_664, "2x [1024,1024,512,512]": 2_040_832,
        "4x [2048,2048,1024,1024]": 7_751_680}

DECISION_RULE = [
    {"gain_at_least": 0.27, "reading": "capacity binds hard",
     "next": "the 562k-parameter class is the limit; the 4x arm is worth a GPU "
             "VM, and PureTD's 0.94 at this capacity is the claim to scrutinise"},
    {"gain_at_least": 0.09, "reading": "capacity binds partially",
     "next": "report the slope, and what width the extrapolation says 0.95 needs"},
    {"gain_at_least": None,
     "reading": "capacity is not the limit at 0.563 M MACs under this recipe",
     "next": "method or position distribution is; RL becomes worth its weeks "
             "rather than an assumption"},
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", default="experiments/exp030-capacity")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=30)
    a = ap.parse_args()
    dumps, scratch = (os.path.join(a.exp_dir, d) for d in ("dumps", "scratch"))
    rng = np.random.default_rng(a.seed)

    def dump_path(ep):
        return os.path.join(dumps, f"{ARM}_e{ep}.npz")

    scored = [e for e in range(1, a.epochs + 1) if os.path.exists(dump_path(e))]
    if not scored:
        raise SystemExit(f"no dumps in {dumps} — run the scoring pass first")
    last = scored[-1]
    err, games = load(dump_path(last))
    exp028 = json.load(open("experiments/exp028-mlp/results/exp028_summary.json"))

    out = {
        "protocol": {
            "benchmark": "BGSage Money Benchmark, checker decisions, all tiers",
            "n": int(len(err)), "n_games": int(len(np.unique(games))),
            "subsampling": "none",
            "pairing": "per-decision, identical decisions across every epoch and "
                       "against both 1x baselines",
            "bootstrap": f"{BOOTSTRAP} resamples of whole games",
            "resolution_pr": 0.09,
            "training_data": "43M GNUBG 2-ply labels, data/distill/2ply — "
                             "unchanged from exp028 and exp029",
            "architecture": "MLP [1024,1024,512,512], 201 de-broadcast inputs, "
                            "scalar head, cosine LR — exp028's combined recipe "
                            "with width doubled and nothing else varied",
            "epochs": last,
            "primary_baseline": BASELINES["vs_1x_same_budget"][1],
            "secondary_baseline": BASELINES["vs_1x_best"][1],
        },
        "decision_rule": DECISION_RULE,
        "macs": MACS,
        "reference_points": exp028["reference_points"],
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

    for key, (path, label) in BASELINES.items():
        if os.path.exists(path):
            out["paired"][key] = compare(path, label, dump_path(last),
                                         f"{ARM}_e{last}", rng)

    speed = os.path.join(a.exp_dir, "results", "eval_speed.json")
    if os.path.exists(speed):
        out["eval_speed"] = json.load(open(speed))

    d = out["paired"]["vs_1x_same_budget"]
    gain, resolved = d["gain"], d["ci95"][0] > 0
    rule = (DECISION_RULE[0] if resolved and gain >= 0.27 else
            DECISION_RULE[1] if resolved and gain >= 0.09 else DECISION_RULE[2])
    out["conclusion"] = {
        "gain_over_1x_same_budget": gain, "ci95": d["ci95"],
        "reading": rule["reading"], "next": rule["next"],
        "pr_final": out["pr_final"],
        "macs_ratio_to_1x": MACS["2x [1024,1024,512,512]"] / MACS["1x [512,512,256,256]"],
        "ep22_pr": exp028["reference_points"]["ep22_resnet_10x256"]["pr"],
    }

    results = os.path.join(a.exp_dir, "results")
    os.makedirs(results, exist_ok=True)
    path = os.path.join(results, "exp030_summary.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)

    print(f"{ARM} PR e{last} = {out['pr_final']:.3f}")
    for key, v in out["paired"].items():
        print(f"{key:<22} {v['gain']:+.3f} "
              f"[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  {v['verdict']}")
    print(f"\nreading: {out['conclusion']['reading']}")
    print(f"next:    {out['conclusion']['next']}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
