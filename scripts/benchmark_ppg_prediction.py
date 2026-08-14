"""Turn a BGSage PR difference into a ppg prediction, with a proper CI (exp020).

PR is an error *rate* in points per checker decision, so the difference between two
engines makes a falsifiable prediction about points per game:

    predicted ppg = (PR_ref - PR_engine) / 500 * checker decisions per player per game

exp019 made that comparison with no error bar on the prediction, which made it
impossible to say whether a shortfall against measured play was real. This script
supplies one. Both engines are scored on the **same** decisions, so the difference is
computed **paired** — per-decision error differences, not a difference of two
independently-estimated means. That is much tighter, and it is the correct standard
error for the comparison being made.

Two caveats the number cannot carry on its own:

- The conversion assumes per-decision equity errors accumulate linearly into the final
  result. That is a model, not an identity. exp020 calibrates it: summing per-decision
  errors over real games predicted +0.0345 ppg where the outcome difference measured
  +0.0331, so the assumption is good to ~4% over that range.
- PR is measured on BGSage's position set (Sage 3-ply self-play). Nothing guarantees an
  error-rate advantage there transfers to the distribution a different matchup generates.

Writes ``<exp-dir>/results/<tag>.json`` with the summary and the paired per-decision
error arrays, so an analysis page can recompute the interval at render time.

    python scripts/benchmark_ppg_prediction.py \
        --checkpoint experiments/exp018-distill/checkpoints/ep22.pt \
        --rolls-per-game 56.09 --exp-dir experiments/exp020-doubles --tag benchmark_prediction
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")  # avoid CPU spin-collapse

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_benchmark_pr import (  # noqa: E402
    DEFAULT_BENCHMARK, PR_MULTIPLIER, _eval_gnubg_decision, flip_board_to_opp,
    flipped_to_board_view, load_benchmark,
)


def net_decision_errors(decisions: list[dict], checkpoint: str) -> np.ndarray:
    """Per-decision move-selection error for a Raccoon checkpoint, in equity points."""
    from raccoon.env.encoder import channels_for_network, encode_state
    from raccoon.model.network import load_model

    torch.set_flush_denormal(True)
    net = load_model(checkpoint)
    net.eval()
    channels = channels_for_network(net.config)

    errors = np.empty(len(decisions))
    for i, decision in enumerate(decisions):
        obs = np.stack([
            encode_state(
                flipped_to_board_view(flip_board_to_opp(m["board"])), channels=channels,
            )
            for m in decision["moves"]
        ])
        with torch.no_grad():
            values = net.value_equity(torch.from_numpy(obs).float()).numpy()
        refs = [m["cubeless_equity"] for m in decision["moves"]]
        # values are from the opponent's POV on the [-1,1] scale; negate for the mover.
        chosen = refs[int(np.argmax(-values * 3.0))]
        errors[i] = max(0.0, max(refs) - chosen)
    return errors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    ap.add_argument("--ref-ply", type=int, default=0, help="GNUBG reference ply")
    ap.add_argument("--rolls-per-game", type=float, required=True,
                    help="measured rolls per game; halved to get decisions per player")
    ap.add_argument("--exp-dir", default="")
    ap.add_argument("--tag", default="benchmark_prediction")
    a = ap.parse_args()

    decisions, _ = load_benchmark(os.path.realpath(a.benchmark))
    print(f"{len(decisions)} checker decisions", flush=True)

    err_ref = np.array([_eval_gnubg_decision((d, a.ref_ply))["error"] for d in decisions])
    print(f"GNUBG {a.ref_ply}-ply scored", flush=True)
    err_net = net_decision_errors(decisions, a.checkpoint)
    print("checkpoint scored", flush=True)

    paired = err_ref - err_net
    n = len(paired)
    se = float(paired.std(ddof=1) / n**0.5)
    per_player = a.rolls_per_game / 2.0

    summary = {
        "tag": a.tag,
        "checkpoint": a.checkpoint,
        "ref_ply": a.ref_ply,
        "n_decisions": n,
        "rolls_per_game": a.rolls_per_game,
        "decisions_per_player_per_game": per_player,
        "pr_ref": float(err_ref.mean() * PR_MULTIPLIER),
        "pr_net": float(err_net.mean() * PR_MULTIPLIER),
        "edge_per_decision": float(paired.mean()),
        "edge_per_decision_ci95": float(1.96 * se),
        "predicted_ppg": float(paired.mean() * per_player),
        "predicted_ppg_ci95": float(1.96 * se * per_player),
    }

    print(f"  PR ref = {summary['pr_ref']:.3f}   PR net = {summary['pr_net']:.3f}")
    print(f"  predicted ppg = {summary['predicted_ppg']:+.4f} "
          f"± {summary['predicted_ppg_ci95']:.4f}  (paired, n={n})", flush=True)

    if a.exp_dir:
        out = Path(a.exp_dir) / "results"
        out.mkdir(parents=True, exist_ok=True)
        record = dict(summary)
        record["err_ref"] = [round(float(x), 6) for x in err_ref]
        record["err_net"] = [round(float(x), 6) for x in err_net]
        (out / f"{a.tag}.json").write_text(json.dumps(record) + "\n")
        print(f"  wrote {out / f'{a.tag}.json'}", flush=True)


if __name__ == "__main__":
    main()
