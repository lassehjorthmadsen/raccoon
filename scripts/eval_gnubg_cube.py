"""Variance-reduced **cubeful** ppg for a checkpoint vs GNUBG, sharded across processes.

The cube-aware sibling of ``scripts/eval_gnubg_vr.py``. Plays full money games
with the doubling cube live and the Jacoby rule **off** (``goal.md``), records
the realised dice luck at every roll scaled by the cube value at that roll, and
reports both the raw and the variance-reduced ppg. See
:mod:`raccoon.eval.cube_arena` for the game loop and :mod:`raccoon.eval.luck`
for why subtracting luck cannot move the expectation.

**Pick the opponent deliberately.** ``--opponent gnubg-cli`` (the default) plays
the real ``/usr/games/gnubg`` binary, whose cube decisions come out of a cubeful
search that decides the cube at every node. That is the honest opponent for
``goal.md`` and the one exp026 measures against.

``--opponent gnubg-nn`` is the exp025 **stand-in**, kept only so that
experiment's estimator measurements reproduce. The package cannot make a money
cube decision (``evaluate_cube_decision`` raises ``Not implemented for money``)
and ranks checker plays cubelessly, so its cubeful game is our own Janowski
layer bolted onto its probabilities — leaving both sides sharing one cube model,
which is weaker than real GNUBG on exactly the dimension a cube experiment
tests. **A ppg against the stand-in is not a strength result.**

The two engines were measured against each other and are equally strong: -0.0026
+/- 0.0047 ppg cubeless at ply 0 over 6,000 games
(``experiments/exp026-gnubg-cli/results/binary_vs_package_0ply.json``), so
switching opponents does not shift the scale a result is quoted on.

``--cv`` selects the control variate. Both choices are exactly unbiased, so the
pilot picks whichever measures a larger ``sd_ratio`` and the headline run is
fixed to it.

    # the SD-only pilot: read sd_raw and sd_vr, nothing else
    python scripts/eval_gnubg_cube.py --checkpoint .../ep22.pt --games 600 --ply 2 \
        --workers 3 --cv cubeful --exp-dir experiments/exp025-cube-ranking --tag pilot_cvcubeful

    # the headline run, at the n the pilot implies
    python scripts/eval_gnubg_cube.py --checkpoint .../ep22.pt --games 8400 --ply 2 \
        --workers 3 --exp-dir experiments/exp025-cube-ranking --tag ep22_cube_2ply
"""
from __future__ import annotations

import os

# CPU-bound torch: park idle OpenMP threads instead of busy-spinning (a spin-collapse
# costs ~12x under desktop contention). Must precede the torch import.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from raccoon.eval.cube_arena import CV_CUBEFUL, CV_CUBELESS
from raccoon.model.network import load_model

COUNTERS = ("net_doubles", "net_doubles_offered", "net_takes", "net_takes_offered")


OPP_CLI, OPP_NN = "gnubg-cli", "gnubg-nn"


def _shard(ckpt: str, n: int, ply: int, cv_ply: int, seed: int, vr: bool,
           cv: str, joint_doubles: bool, opponent: str):
    from raccoon.eval.cube_arena import cubeful_match
    from raccoon.eval.opponents import (
        GnubgCliOpponent, GnubgNnOpponent, NetOpponent,
    )

    torch.set_flush_denormal(True)
    torch.set_num_threads(1)
    net = load_model(ckpt)
    net.eval()
    player = NetOpponent(net, torch.device("cpu"), joint_doubles=joint_doubles)
    opp = (GnubgCliOpponent(ply=ply) if opponent == OPP_CLI
           else GnubgNnOpponent(ply=ply))
    try:
        result = cubeful_match(
            player, opp, n, cv_ply=cv_ply, seed=seed, vr=vr, cv=cv,
        )
    finally:
        if hasattr(opp, "close"):
            opp.close()
    # Each worker owns its own gnubg subprocess; surface how often it had to be
    # restarted, because a run that quietly restarted hundreds of times is a run
    # to distrust.
    result["gnubg_restarts"] = getattr(getattr(opp, "cli", None), "restarts", 0)
    return result


def summarise(pts: np.ndarray, luck: np.ndarray) -> dict:
    """Point estimates, CIs and the control-variate diagnostics.

    Identical in shape to ``eval_gnubg_vr.summarise`` so the two experiments'
    numbers line up in a table. ``beta_hat`` is the regression-optimal control
    variate coefficient; the estimator stays at beta = 1 (any *fixed* beta is
    exactly unbiased, one fitted on the same sample is not), so it is reported
    purely as a calibration read — a well-scaled control variate lands near 1.
    """
    n = len(pts)
    vr = pts - luck
    out = {
        "games": n,
        "raw_ppg": float(pts.mean()),
        "raw_sd": float(pts.std(ddof=1)),
        "raw_ci95": float(1.96 * pts.std(ddof=1) / n**0.5),
        "vr_ppg": float(vr.mean()),
        "vr_sd": float(vr.std(ddof=1)),
        "vr_ci95": float(1.96 * vr.std(ddof=1) / n**0.5),
        "mean_luck": float(luck.mean()),
        "luck_sd": float(luck.std(ddof=1)),
    }
    if luck.std(ddof=1) > 0:
        se_luck = luck.std(ddof=1) / n**0.5
        out["luck_t"] = float(luck.mean() / se_luck)
        out["sd_ratio"] = float(pts.std(ddof=1) / vr.std(ddof=1))
        out["var_ratio"] = out["sd_ratio"] ** 2
        out["corr_raw_luck"] = float(np.corrcoef(pts, luck)[0, 1])
        out["beta_hat"] = float(np.cov(pts, luck)[0, 1] / np.var(luck, ddof=1))
    return out


def cube_diagnostics(cube: np.ndarray, dropped: np.ndarray, counts: dict) -> dict:
    """What the cube actually did over the run.

    Not colour: a wiring bug that leaves the cube inert produces a perfectly
    plausible ppg with a perfectly plausible CI, and the only thing that shows it
    is a double rate of zero or a cube distribution of all ones.
    """
    values, freq = np.unique(cube, return_counts=True)
    out = {
        "mean_terminal_cube": float(cube.mean()),
        "cube_distribution": {int(v): int(c) for v, c in zip(values, freq)},
        "drop_rate": float(dropped.mean()),
        "games_with_cube_turned": int((cube > 1).sum()),
        **{k: int(counts[k]) for k in COUNTERS},
    }
    if counts["net_doubles_offered"]:
        out["net_double_rate"] = counts["net_doubles"] / counts["net_doubles_offered"]
    if counts["net_takes_offered"]:
        out["net_take_rate"] = counts["net_takes"] / counts["net_takes_offered"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=400,
                    help="total games; split evenly across --workers shards")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--ply", type=int, default=2, help="GNUBG opponent ply")
    ap.add_argument("--opponent", choices=[OPP_CLI, OPP_NN], default=OPP_CLI,
                    help="gnubg-cli: the real binary, with its own cubeful cube "
                         "engine (default). gnubg-nn: the exp025 stand-in, which "
                         "cannot make a money cube decision -- its ppg is not a "
                         "strength result")
    ap.add_argument("--cv-ply", type=int, default=0,
                    help="control-variate ply; must be 0 (best_move segfaults deeper)")
    ap.add_argument("--cv", choices=[CV_CUBEFUL, CV_CUBELESS], default=CV_CUBEFUL,
                    help="control variate: cubeful equity (default) or the cubeless "
                         "one scaled by the cube value. Both unbiased; pick on sd_ratio")
    ap.add_argument("--no-vr", action="store_true",
                    help="skip the control variate entirely; plays identical games")
    ap.add_argument("--greedy-doubles", action="store_true",
                    help="execute doubles by the old greedy two-step path instead of "
                         "enumerating all four half-moves jointly (the exp019 engine)")
    ap.add_argument("--seed-base", type=int, default=2000)
    ap.add_argument("--exp-dir", default="", help="experiment dir for logs/ + results/")
    ap.add_argument("--tag", default="", help="results filename stem (defaults to label)")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    vr = not a.no_vr
    joint_doubles = not a.greedy_doubles
    sizes = [
        a.games // a.workers + (1 if k < a.games % a.workers else 0)
        for k in range(a.workers)
    ]
    sizes = [s for s in sizes if s > 0]
    seeds = [a.seed_base + k for k in range(len(sizes))]

    started = time.time()
    parts: list[dict] = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [
            ex.submit(_shard, a.checkpoint, n, a.ply, a.cv_ply, seed, vr, a.cv,
                      joint_doubles, a.opponent)
            for n, seed in zip(sizes, seeds)
        ]
        parts = [f.result() for f in futs]

    def cat(key):
        return np.concatenate([p[key] for p in parts])

    pts, luck = cat("game_pts"), cat("game_luck")
    rolls, cube, dropped = cat("game_rolls"), cat("game_cube"), cat("game_ended_by_drop")
    shard = np.concatenate(
        [np.full(len(p["game_pts"]), k, dtype=np.int32) for k, p in enumerate(parts)]
    )
    counts = {k: sum(p[k] for p in parts) for k in COUNTERS}
    restarts = sum(p.get("gnubg_restarts", 0) for p in parts)
    wins = sum(p["net_wins"] for p in parts)
    elapsed = time.time() - started

    tag = a.tag or a.label or Path(a.checkpoint).stem
    summary = {
        "tag": tag,
        "checkpoint": a.checkpoint,
        "opponent": a.opponent,
        "gnubg_ply": a.ply,
        "cv_ply": a.cv_ply,
        "control_variate": a.cv if vr else None,
        "variance_reduction": vr,
        "joint_doubles": joint_doubles,
        "jacoby": False,
        "cube": True,
        "seed_base": a.seed_base,
        "shard_sizes": sizes,
        "net_wins": wins,
        "win_pct": 100.0 * wins / len(pts),
        "elapsed_sec": round(elapsed, 1),
        "sec_per_game_per_worker": round(elapsed * a.workers / len(pts), 3),
        "mean_rolls_per_game": round(float(rolls.mean()), 2),
        "gnubg_restarts": restarts,
        **summarise(pts, luck),
        **cube_diagnostics(cube, dropped, counts),
    }

    print(f"[{tag}] {a.checkpoint}", flush=True)
    print(f"  cubeful money (no Jacoby) vs {a.opponent}-{a.ply}ply, "
          f"n={summary['games']} games, "
          f"{wins} wins ({summary['win_pct']:.1f}%), {elapsed / 60:.1f} min", flush=True)
    print(f"  raw ppg = {summary['raw_ppg']:+.4f}  (95% CI ±{summary['raw_ci95']:.4f})",
          flush=True)
    if vr:
        print(f"  VR  ppg = {summary['vr_ppg']:+.4f}  (95% CI ±{summary['vr_ci95']:.4f})"
              f"   <- {summary['sd_ratio']:.2f}x tighter "
              f"(= {summary['var_ratio']:.1f}x the games), cv={a.cv}", flush=True)
        print(f"  mean luck = {summary['mean_luck']:+.4f} (t={summary['luck_t']:+.2f}, "
              f"should be ~0)   corr(raw,luck) = {summary['corr_raw_luck']:.3f}   "
              f"beta_hat = {summary['beta_hat']:.3f}", flush=True)
    print(f"  cube: mean terminal {summary['mean_terminal_cube']:.2f}, "
          f"turned in {summary['games_with_cube_turned']}/{summary['games']} games, "
          f"drop rate {summary['drop_rate']:.1%}, "
          f"distribution {summary['cube_distribution']}", flush=True)
    print(f"  net doubled {counts['net_doubles']}/{counts['net_doubles_offered']} "
          f"opportunities, took {counts['net_takes']}/{counts['net_takes_offered']} "
          f"offers", flush=True)
    if a.opponent == OPP_CLI:
        # A run that quietly restarted gnubg hundreds of times is a run to
        # distrust: each restart is a position it could not digest.
        print(f"  gnubg subprocess restarts: {restarts}", flush=True)

    if a.exp_dir:
        exp = Path(a.exp_dir)
        (exp / "logs").mkdir(parents=True, exist_ok=True)
        (exp / "results").mkdir(parents=True, exist_ok=True)
        with (exp / "logs" / "cube_eval_log.jsonl").open("a") as fh:
            fh.write(json.dumps(summary) + "\n")
        record = dict(summary)
        record["game_pts"] = [float(x) for x in pts]
        record["game_luck"] = [round(float(x), 6) for x in luck]
        record["game_cube"] = [int(x) for x in cube]
        record["game_ended_by_drop"] = [bool(x) for x in dropped]
        record["game_shard"] = [int(x) for x in shard]
        (exp / "results" / f"{tag}.json").write_text(json.dumps(record) + "\n")
        print(f"  wrote {exp / 'results' / f'{tag}.json'}", flush=True)


if __name__ == "__main__":
    main()
