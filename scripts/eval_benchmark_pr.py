#!/usr/bin/env python3
"""exp015 - PR benchmark against BGSage money-game positions.

Scores checker-play skill on the BGSage money-game benchmark (14,693 checker
decisions from 500 games) using cubeless equity as the reference.

Two complementary metrics:

1. **PR** (move selection): for each decision, does the engine pick the best
   move?  PR = mean(error) * 500 where error = best_cl_eq - chosen_cl_eq.
   Uses all 14,693 decisions across all reference tiers.

2. **Eval accuracy** (R^2 / MSE on equity): how well does the engine's raw
   value estimate track the reference cubeless equity across all ~327k
   candidate post-move positions? Split by reference tier (rollout = primary,
   3T = secondary, 3P = teacher-agreement only).

Only checker decisions are scored; cube positions are skipped.

Usage:
    # Score GNUBG at 0-ply (fast, ~5s):
    python scripts/eval_benchmark_pr.py --gnubg-ply 0

    # Score GNUBG at 0-ply and 2-ply:
    python scripts/eval_benchmark_pr.py --gnubg-ply 0 --gnubg-ply 2 --workers 8

    # Score a raccoon checkpoint:
    python scripts/eval_benchmark_pr.py \\
        --checkpoint experiments/exp011b-distill/outcomes6/checkpoints/ep3.pt \\
        --engine-label "exp011b (0-ply dist)"

    # Quick smoke test (first 200 decisions):
    python scripts/eval_benchmark_pr.py --gnubg-ply 0 --max-positions 200

    # Full exp015 comparison:
    python scripts/eval_benchmark_pr.py \\
        --gnubg-ply 0 --gnubg-ply 2 \\
        --checkpoint path/to/exp011b.pt --engine-label "exp011b (0-ply dist)" \\
        --checkpoint path/to/exp014.pt --engine-label "exp014 (2-ply dist)" \\
        --output experiments/exp015-benchmark/results/
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GAME_PLANS = ["purerace", "racing", "attacking", "priming", "anchoring"]
BLUNDER_THRESHOLD = 0.08  # equity error > 0.08 is a blunder (BGSage convention)
PR_MULTIPLIER = 500
TIERS = ["rollout", "3T", "3P"]

DEFAULT_BENCHMARK = str(
    Path(__file__).resolve().parent.parent / "data" / "bgsage"
    / "money_benchmark" / "benchmark.json.gz"
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_benchmark(path: str) -> tuple[list[dict], dict]:
    """Load benchmark.json.gz, return (checker_decisions, meta)."""
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    meta = data["meta"]
    decisions = [d for d in data["decisions"] if d["kind"] == "checker"]
    return decisions, meta


# ---------------------------------------------------------------------------
# Board conversion utilities
# ---------------------------------------------------------------------------

def flip_board_to_opp(board26: list[int]) -> list[int]:
    """Flip a BGSage board from mover's POV to opponent's (next-to-move) POV.

    After a checker play, the opponent is next to move. This function converts
    the post-move board (still in mover's perspective) to the opponent's
    perspective so we can evaluate V(position) from their side.

    BGSage convention:
      board[0]  = opponent's bar count (positive int)
      board[k]  = checkers on point k, k=1..24 (pos=mover, neg=opponent)
      board[25] = mover's bar count (positive int)
    """
    flipped = [0] * 26
    flipped[0] = board26[25]   # mover's bar -> new opponent's bar
    flipped[25] = board26[0]   # old opponent bar -> new player's bar
    for k in range(1, 25):
        flipped[k] = -board26[25 - k]  # mirror points AND negate signs
    return flipped


def flipped_to_board_view(flipped26: list[int]):
    """Convert a flipped BGSage board (next-to-move's POV) to raccoon BoardView.

    Returns a BoardView with dice=None (pre-roll evaluation for value head).
    Computes borne-off counts from the remaining checkers.
    """
    from raccoon.env.game_wrapper import BoardView

    my_points = np.zeros(24, dtype=np.float32)
    opp_points = np.zeros(24, dtype=np.float32)

    for k in range(1, 25):
        if flipped26[k] > 0:
            my_points[k - 1] = flipped26[k]
        elif flipped26[k] < 0:
            opp_points[k - 1] = abs(flipped26[k])

    my_bar = flipped26[25]
    opp_bar = flipped26[0]
    my_off = 15 - int(my_points.sum()) - my_bar
    opp_off = 15 - int(opp_points.sum()) - opp_bar

    return BoardView(
        my_points=my_points,
        opp_points=opp_points,
        my_bar=my_bar,
        opp_bar=opp_bar,
        my_off=my_off,
        opp_off=opp_off,
        dice=None,
        mid_doubles=False,
    )


def flipped_to_gnubg(flipped26: list[int]) -> list[list[int]]:
    """Convert a flipped BGSage board to gnubg-nn [opp_25, me_25] format.

    gnubg-nn convention: board = [slot0, slot1] where slot1 is the side on roll.
    Each slot is 25 ints: indices 0..23 = points 1..24 from that player's POV,
    index 24 = bar.

    The 'flipped' board is from the next-to-move (on-roll) player's perspective:
      flipped[k] > 0 -> on-roll player's checkers on point k
      flipped[k] < 0 -> opponent's checkers on point k
      flipped[25] = on-roll player's bar
      flipped[0] = opponent's bar
    """
    # On-roll player (slot 1): points from their own POV
    me_25 = [max(0, flipped26[i + 1]) for i in range(24)]
    me_25.append(flipped26[25])  # on-roll player's bar

    # Opponent (slot 0): points from THEIR OWN POV
    # Their point j from their POV = on-roll's point (25-j) from on-roll's POV
    # = flipped26[25-j], but with negated sign (opponent's checkers are negative)
    opp_25 = [abs(min(0, flipped26[24 - i])) for i in range(24)]
    opp_25.append(flipped26[0])  # opponent's bar

    return [opp_25, me_25]


# ---------------------------------------------------------------------------
# GNUBG scoring
# ---------------------------------------------------------------------------

def _eval_gnubg_decision(args: tuple) -> dict:
    """Score one checker decision with gnubg-nn. Worker for ProcessPoolExecutor.

    Returns dict with 'error', 'game_plan', 'tier', and per-candidate
    'predicted'/'reference' equity pairs for eval-accuracy metrics.
    """
    decision, ply = args
    import gnubg_nn

    # Reference best cubeless equity
    best_cl = max(m["cubeless_equity"] for m in decision["moves"])

    # Evaluate each candidate from opponent's perspective
    predicted_eqs: list[float] = []
    for move in decision["moves"]:
        flipped = flip_board_to_opp(move["board"])
        gnubg_board = flipped_to_gnubg(flipped)
        win, wg, wbg, lg, lbg = gnubg_nn.probabilities(gnubg_board, ply)
        # Cubeless equity from on-roll (opponent) perspective
        opp_eq = win + wg + wbg - (1.0 - win) - lg - lbg
        # Convert to mover's equity for comparison with cubeless_equity reference
        mover_eq = -opp_eq
        predicted_eqs.append(mover_eq)

    # Engine picks the move with highest predicted mover equity
    best_idx = int(np.argmax(predicted_eqs))
    chosen_cl = decision["moves"][best_idx]["cubeless_equity"]
    error = max(0.0, best_cl - chosen_cl)

    # Collect per-candidate pairs for eval accuracy
    reference_eqs = [m["cubeless_equity"] for m in decision["moves"]]

    return {
        "error": error,
        "game_plan": decision["game_plan"],
        "tier": decision["tier"],
        "predicted": predicted_eqs,
        "reference": reference_eqs,
    }


def score_gnubg(
    decisions: list[dict],
    ply: int,
    workers: int = 1,
    max_positions: int | None = None,
) -> dict:
    """Score all checker decisions with gnubg-nn at given ply."""
    if max_positions is not None:
        decisions = decisions[:max_positions]

    n_total = len(decisions)
    label = f"GNUBG {ply}-ply"
    t0 = time.perf_counter()

    all_decision_results: list[dict] = []

    if ply == 0 or workers <= 1:
        # Sequential -- fast enough for 0-ply
        for i, dec in enumerate(decisions):
            result = _eval_gnubg_decision((dec, ply))
            all_decision_results.append(result)
            if (i + 1) % 500 == 0:
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / elapsed
                eta = (n_total - i - 1) / rate
                print(
                    f"  [{label}] {i+1}/{n_total} "
                    f"({elapsed:.0f}s, ~{eta:.0f}s remaining)",
                    flush=True,
                )
    else:
        # Parallel for expensive plies
        args_list = [(dec, ply) for dec in decisions]
        done = 0
        results_by_idx: dict[int, dict] = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_eval_gnubg_decision, a): idx
                for idx, a in enumerate(args_list)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results_by_idx[idx] = future.result()
                done += 1
                if done % 500 == 0:
                    elapsed = time.perf_counter() - t0
                    rate = done / elapsed
                    eta = (n_total - done) / rate
                    print(
                        f"  [{label}] {done}/{n_total} "
                        f"({elapsed:.0f}s, ~{eta:.0f}s remaining)",
                        flush=True,
                    )
        # Reassemble in order
        for idx in range(n_total):
            all_decision_results.append(results_by_idx[idx])

    elapsed = time.perf_counter() - t0
    print(f"  [{label}] Done: {n_total} decisions in {elapsed:.1f}s", flush=True)
    return aggregate(all_decision_results, label)


# ---------------------------------------------------------------------------
# Raccoon scoring
# ---------------------------------------------------------------------------

def score_raccoon(
    decisions: list[dict],
    checkpoint_path: str,
    engine_label: str,
    device_str: str = "cpu",
    max_positions: int | None = None,
) -> dict:
    """Score all checker decisions with a raccoon network checkpoint."""
    import torch
    from raccoon.model.network import load_model
    from raccoon.env.encoder import encode_state, channels_for_network

    if max_positions is not None:
        decisions = decisions[:max_positions]

    # Handle GCS paths
    local_path = _maybe_download_gcs(checkpoint_path)

    # Load model
    device = torch.device(device_str)
    network = load_model(local_path)
    network.to(device)
    network.eval()
    channels = channels_for_network(network.config)

    n_total = len(decisions)
    all_decision_results: list[dict] = []

    t0 = time.perf_counter()

    for i, dec in enumerate(decisions):
        # Reference best cubeless equity
        best_cl = max(m["cubeless_equity"] for m in dec["moves"])

        # Encode all candidates (flipped to opponent POV)
        obs_list = []
        for move in dec["moves"]:
            flipped = flip_board_to_opp(move["board"])
            bv = flipped_to_board_view(flipped)
            obs = encode_state(bv, channels=channels)
            obs_list.append(obs)

        # Batch forward pass
        obs_batch = np.stack(obs_list)
        with torch.no_grad():
            x = torch.from_numpy(obs_batch).float().to(device, non_blocking=True)
            values = network.value_equity(x).cpu().numpy()

        # values[j] = V from opponent's perspective (in [-1,1], equity/3 scale)
        # Convert to mover's equity on the standard [-3,3] scale for comparison:
        # mover_equity = -(opponent_value * 3)
        predicted_eqs = [float(-v * 3.0) for v in values]

        # Engine picks the move with highest predicted mover equity
        best_idx = int(np.argmax(predicted_eqs))
        chosen_cl = dec["moves"][best_idx]["cubeless_equity"]
        error = max(0.0, best_cl - chosen_cl)

        reference_eqs = [m["cubeless_equity"] for m in dec["moves"]]

        all_decision_results.append({
            "error": error,
            "game_plan": dec["game_plan"],
            "tier": dec["tier"],
            "predicted": predicted_eqs,
            "reference": reference_eqs,
        })

        if (i + 1) % 500 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            eta = (n_total - i - 1) / rate
            print(
                f"  [{engine_label}] {i+1}/{n_total} "
                f"({elapsed:.0f}s, ~{eta:.0f}s remaining)",
                flush=True,
            )

    elapsed = time.perf_counter() - t0
    print(
        f"  [{engine_label}] Done: {n_total} decisions in {elapsed:.1f}s",
        flush=True,
    )
    return aggregate(all_decision_results, engine_label)


# ---------------------------------------------------------------------------
# Result aggregation and display
# ---------------------------------------------------------------------------

def aggregate(decision_results: list[dict], label: str) -> dict:
    """Compute PR, per-plan breakdown, and eval-accuracy metrics from results."""
    errors = [r["error"] for r in decision_results]
    plans = [r["game_plan"] for r in decision_results]
    n = len(errors)
    total_error = sum(errors)
    pr = (total_error / n) * PR_MULTIPLIER if n > 0 else 0.0
    blunders = sum(1 for e in errors if e > BLUNDER_THRESHOLD)

    # Per game-plan PR
    by_plan: dict[str, dict] = {}
    for plan in GAME_PLANS:
        plan_errors = [e for e, p in zip(errors, plans) if p == plan]
        if plan_errors:
            plan_total = sum(plan_errors)
            by_plan[plan] = {
                "pr": (plan_total / len(plan_errors)) * PR_MULTIPLIER,
                "n": len(plan_errors),
                "total_error": plan_total,
                "blunders": sum(1 for e in plan_errors if e > BLUNDER_THRESHOLD),
            }
        else:
            by_plan[plan] = {"pr": 0.0, "n": 0, "total_error": 0.0, "blunders": 0}

    # Eval accuracy: R^2 and MSE on predicted vs reference equity, split by tier
    eval_accuracy = {}
    for tier in TIERS + ["all"]:
        if tier == "all":
            tier_results = decision_results
        else:
            tier_results = [r for r in decision_results if r["tier"] == tier]
        if not tier_results:
            eval_accuracy[tier] = {"r2": None, "mse": None, "n_positions": 0}
            continue

        pred_all = []
        ref_all = []
        for r in tier_results:
            pred_all.extend(r["predicted"])
            ref_all.extend(r["reference"])

        pred = np.array(pred_all, dtype=np.float64)
        ref = np.array(ref_all, dtype=np.float64)

        mse = float(np.mean((pred - ref) ** 2))
        ss_res = float(np.sum((pred - ref) ** 2))
        ss_tot = float(np.sum((ref - ref.mean()) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        eval_accuracy[tier] = {
            "r2": r2,
            "mse": mse,
            "n_positions": len(pred),
        }

    return {
        "engine_label": label,
        "pr": pr,
        "n": n,
        "total_error": total_error,
        "blunders": blunders,
        "by_plan": by_plan,
        "eval_accuracy": eval_accuracy,
    }


def print_table(results: list[dict]) -> None:
    """Print formatted results table."""
    print()
    print("=" * 90)
    print("BGSage Money Benchmark - Checker-only, cubeless reference")
    print(f"Positions: {results[0]['n']} checker decisions from 500 games")
    print("Reference: cubeless_equity (Sage 3P / 3T / Rollout)")
    print("=" * 90)

    # --- Move-selection PR ---
    print()
    print("MOVE SELECTION (PR = mean error x 500, lower is better)")
    print()
    plan_short = {
        "purerace": "prace", "racing": "race",
        "attacking": "atk", "priming": "prime", "anchoring": "anch",
    }
    header = f"  {'Engine':<32} {'PR':>6} {'N':>6} {'Blndr':>6}"
    for plan in GAME_PLANS:
        header += f"  {plan_short[plan]:>6}"
    print(header)
    print("  " + "-" * 86)

    for r in results:
        row = f"  {r['engine_label']:<32} {r['pr']:>6.2f} {r['n']:>6} {r['blunders']:>6}"
        for plan in GAME_PLANS:
            bp = r["by_plan"].get(plan, {})
            if bp.get("n", 0) > 0:
                row += f"  {bp['pr']:>6.2f}"
            else:
                row += f"  {'---':>6}"
        print(row)

    # Decision counts per plan
    print()
    r0 = results[0]
    counts = f"  {'(N per plan)':<32} {'':>6} {'':>6} {'':>6}"
    for plan in GAME_PLANS:
        n = r0["by_plan"].get(plan, {}).get("n", 0)
        counts += f"  {n:>6}"
    print(counts)

    # --- Eval accuracy ---
    print()
    print("EVAL ACCURACY (R^2 / MSE on predicted vs reference cubeless equity)")
    print()
    header2 = f"  {'Engine':<32}"
    for tier in TIERS + ["all"]:
        header2 += f"  {'R2(' + tier + ')':>12} {'MSE':>8} {'N':>8}"
    print(header2)
    print("  " + "-" * 110)

    for r in results:
        row = f"  {r['engine_label']:<32}"
        ea = r.get("eval_accuracy", {})
        for tier in TIERS + ["all"]:
            ta = ea.get(tier, {})
            r2 = ta.get("r2")
            mse = ta.get("mse")
            n_pos = ta.get("n_positions", 0)
            if r2 is not None:
                row += f"  {r2:>12.4f} {mse:>8.4f} {n_pos:>8}"
            else:
                row += f"  {'---':>12} {'---':>8} {n_pos:>8}"
        print(row)

    print()


def save_results(results: list[dict], output_dir: str) -> None:
    """Save per-engine JSON files and summary.json."""
    os.makedirs(output_dir, exist_ok=True)

    for r in results:
        # Sanitize label for filename
        fname = r["engine_label"].lower()
        for ch in " ()/-.,":
            fname = fname.replace(ch, "_")
        fname = fname.strip("_") + ".json"
        path = os.path.join(output_dir, fname)
        with open(path, "w") as f:
            json.dump(r, f, indent=2)
        print(f"  Saved: {path}")

    # Summary
    summary = {
        "benchmark": "BGSage Money Benchmark (checker-only, cubeless)",
        "n_decisions": results[0]["n"] if results else 0,
        "results": [
            {
                "engine": r["engine_label"],
                "pr": r["pr"],
                "blunders": r["blunders"],
                "eval_accuracy": r.get("eval_accuracy", {}),
            }
            for r in results
        ],
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _maybe_download_gcs(path: str) -> str:
    """If path starts with gs://, download to a temp file and return local path."""
    if not path.startswith("gs://"):
        return path
    local = os.path.join(tempfile.gettempdir(), "raccoon_checkpoint.pt")
    print(f"  Downloading {path} -> {local}...")
    result = subprocess.run(
        ["gsutil", "cp", path, local],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR: gsutil failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return local


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score engines against BGSage money-game benchmark (checker PR).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--benchmark", type=str, default=DEFAULT_BENCHMARK,
        help="Path to benchmark.json.gz",
    )
    parser.add_argument(
        "--gnubg-ply", type=int, action="append", dest="gnubg_plies",
        help="Score GNUBG at this ply (repeatable)",
    )
    parser.add_argument(
        "--checkpoint", type=str, action="append", dest="checkpoints",
        help="Path to a raccoon .pt checkpoint (repeatable)",
    )
    parser.add_argument(
        "--engine-label", type=str, action="append", dest="labels",
        help="Human label for each checkpoint (must match --checkpoint count)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device for raccoon evaluation (default: cpu)",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Processes for GNUBG 2-ply (0 = cpu_count)",
    )
    parser.add_argument(
        "--max-positions", type=int, default=None,
        help="Limit number of decisions scored (for quick testing)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Directory to save JSON results",
    )
    args = parser.parse_args()

    if args.workers <= 0:
        import multiprocessing
        args.workers = multiprocessing.cpu_count()

    # Validate checkpoint/label pairing
    checkpoints = args.checkpoints or []
    labels = args.labels or []
    if len(labels) < len(checkpoints):
        # Auto-generate labels from checkpoint paths
        for cp in checkpoints[len(labels):]:
            labels.append(Path(cp).stem)

    # Load benchmark
    benchmark_path = os.path.realpath(args.benchmark)
    print(f"Loading benchmark: {benchmark_path}")
    decisions, meta = load_benchmark(benchmark_path)
    print(f"  {len(decisions)} checker decisions from {meta['n_games']} games")
    if args.max_positions:
        print(f"  (limiting to first {args.max_positions} decisions)")
    print()

    all_results: list[dict] = []

    # Score GNUBG engines
    if args.gnubg_plies:
        for ply in args.gnubg_plies:
            print(f"Scoring GNUBG {ply}-ply...")
            result = score_gnubg(
                decisions, ply,
                workers=args.workers if ply >= 2 else 1,
                max_positions=args.max_positions,
            )
            all_results.append(result)
            print()

    # Score raccoon checkpoints
    for cp, label in zip(checkpoints, labels):
        print(f"Scoring raccoon: {label} ({cp})...")
        result = score_raccoon(
            decisions, cp, label,
            device_str=args.device,
            max_positions=args.max_positions,
        )
        all_results.append(result)
        print()

    # Print results
    if all_results:
        print_table(all_results)
        if args.output:
            print(f"Saving results to {args.output}/")
            save_results(all_results, args.output)


if __name__ == "__main__":
    main()
