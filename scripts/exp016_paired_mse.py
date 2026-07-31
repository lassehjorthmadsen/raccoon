#!/usr/bin/env python3
"""exp016 - paired cluster bootstrap of two checkpoints on the BGSage benchmark.

exp015 showed that scoring against the BGSage money benchmark (rollout tier)
is a much higher-power static substitute for raw GNUBG ppg. exp016 asks: does
that power resolve a playing-strength difference between clean A/B pairs that
underpowered play metrics (GNUBG ppg) had called ties? This script does the
paired statistics for one such pair, given two per-candidate prediction dumps
produced by `scripts/eval_benchmark_pr.py --dump-dir ...`.

Two co-headline metrics, both as a delta (arm A minus arm B) with a cluster
bootstrap CI:

1. **Rollout-tier value MSE** (candidate-weighted, i.e. the same statistic
   `eval_benchmark_pr.py` reports) plus a **decision-mean** variant that
   weights every decision equally regardless of how many candidate moves it
   has (2-773 in this benchmark) -- candidates within a decision are not
   independent draws, and the candidate-weighted number over-represents
   heavily-branching (doubles) decisions.
2. **PR** (move-selection error x 500, all three tiers) -- objective-neutral,
   since rollout-MSE is literally the scalar value head's own training loss
   and would structurally favour a scalar-headed arm even at equal strength.

Both metrics are bootstrapped at two clustering levels: **decision** (the
natural unit -- all candidates of one decision share a board) and **game**
(decisions within a game are sequential, so this is the more conservative,
coarser check). Report both; don't just take the tighter one.

A **per-game-plan breakdown** (purerace/racing/attacking/priming/anchoring) is
also computed, decision-level only, as a supporting metric -- e.g. to check
whether a handcrafted feature's expected specialty (pip count for races,
contact channels for contact-heavy plans) actually shows up where expected.
Per-plan n is a fraction of the headline n, so these CIs are wider; they never
override the headline conclusion.

Usage:
    python scripts/exp016_paired_mse.py \\
        --dump-a experiments/exp016-benchmark-revisit/dumps/scalar.npz --label-a scalar \\
        --dump-b experiments/exp016-benchmark-revisit/dumps/outcomes6.npz --label-b outcomes6 \\
        --panel "exp011b value head" \\
        --output experiments/exp016-benchmark-revisit/results/panel_a.json
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")  # avoid CPU spin-collapse

import numpy as np

B_DEFAULT = 10_000
GAME_PLANS = ["purerace", "racing", "attacking", "priming", "anchoring"]


# ---------------------------------------------------------------------------
# Loading & grouping
# ---------------------------------------------------------------------------

def load_dump(path: str) -> dict:
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def group_by_decision(d: dict) -> dict:
    """Sort rows by decision_key and locate each decision's row-slice.

    Returns per-decision aggregates: `decision_key` (sorted unique keys),
    `sum_sq`/`count`/`mean_sq` (for MSE), `error` (for PR: max(ref) minus the
    ref of the candidate the arm's own prediction ranks best -- mirrors the
    same formula `eval_benchmark_pr.py` uses), `game_seed`, and `game_plan`.
    """
    order = np.argsort(d["decision_key"], kind="stable")
    keys = d["decision_key"][order]
    pred = d["pred_eq"][order].astype(np.float64)
    ref = d["ref_eq"][order].astype(np.float64)
    game_seed = d["game_seed"][order]
    game_plan = d["game_plan"][order]

    uniq_keys, start_idx, counts = np.unique(
        keys, return_index=True, return_counts=True
    )
    sq = (pred - ref) ** 2
    sum_sq = np.add.reduceat(sq, start_idx)

    n_dec = len(uniq_keys)
    error = np.empty(n_dec, dtype=np.float64)
    boundaries = np.append(start_idx, len(keys))
    for i in range(n_dec):
        s, e = boundaries[i], boundaries[i + 1]
        error[i] = max(0.0, ref[s:e].max() - ref[s + int(np.argmax(pred[s:e]))])

    return {
        "decision_key": uniq_keys,
        "sum_sq": sum_sq,
        "count": counts,
        "mean_sq": sum_sq / counts,
        "error": error,
        "game_seed": game_seed[start_idx],
        "game_plan": game_plan[start_idx],
    }


def restrict_tier(d: dict, tier: str) -> dict:
    mask = d["tier"] == tier
    return {k: v[mask] for k, v in d.items()}


def assert_aligned(a: dict, b: dict, label_a: str, label_b: str) -> None:
    if a["decision_key"].shape != b["decision_key"].shape or not np.array_equal(
        a["decision_key"], b["decision_key"]
    ):
        raise ValueError(
            f"{label_a} and {label_b} dumps don't cover the same decisions "
            "(different --max-positions, or scored against different "
            "benchmark files?). Re-score both against the identical, "
            "untruncated benchmark."
        )
    if not np.array_equal(a["count"], b["count"]):
        raise ValueError(
            f"{label_a} and {label_b} disagree on candidate count per "
            "decision -- shouldn't happen if both scored the same benchmark."
        )
    if not np.array_equal(a["game_plan"], b["game_plan"]):
        raise ValueError(
            f"{label_a} and {label_b} disagree on game_plan per decision -- "
            "shouldn't happen if both scored the same benchmark."
        )


# ---------------------------------------------------------------------------
# Cluster bootstrap
# ---------------------------------------------------------------------------

def cluster_bootstrap(rng: np.random.Generator, n_clusters: int, B: int, statistic_fn):
    """Resample `n_clusters` cluster indices with replacement B times."""
    out = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n_clusters, size=n_clusters)
        out[b] = statistic_fn(idx)
    return out


def ci95(samples: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def summarize_delta(point: float, boot: np.ndarray) -> dict:
    lo, hi = ci95(boot)
    return {"point": point, "ci95": [lo, hi], "excludes_zero": bool(lo > 0 or hi < 0)}


# ---------------------------------------------------------------------------
# Per-comparison statistics
# ---------------------------------------------------------------------------

def to_game_level(per_dec: dict, value_keys: list[str]) -> dict:
    """Aggregate per-decision arrays up to per-game arrays (sum over decisions
    in the game). For rate-like fields (mean_sq, error) this sums the
    decision-level values; the caller divides by the per-game decision count
    to get an equally-weighted per-game mean, matching how each game-level
    cluster should count once regardless of how many decisions it contains.
    """
    order = np.argsort(per_dec["game_seed"], kind="stable")
    seeds = per_dec["game_seed"][order]
    uniq_seeds, start_idx, dec_counts = np.unique(
        seeds, return_index=True, return_counts=True
    )
    out = {"game_seed": uniq_seeds, "n_decisions": dec_counts}
    for k in value_keys:
        v = per_dec[k][order]
        out[k] = np.add.reduceat(v, start_idx)
    return out


def run_comparison(
    dec_a: dict, dec_b: dict, rng: np.random.Generator, B: int, label: str
) -> dict:
    n_dec = len(dec_a["decision_key"])

    # --- Decision-level (natural unit: candidates share this decision's board) ---
    def stat_mse_weighted(idx):
        c = dec_a["count"][idx].sum()
        return (dec_a["sum_sq"][idx].sum() - dec_b["sum_sq"][idx].sum()) / c

    def stat_mse_decmean(idx):
        return (dec_a["mean_sq"][idx] - dec_b["mean_sq"][idx]).mean()

    def stat_pr(idx):
        return (dec_a["error"][idx] - dec_b["error"][idx]).mean() * 500.0

    boot_mse_w_dec = cluster_bootstrap(rng, n_dec, B, stat_mse_weighted)
    boot_mse_m_dec = cluster_bootstrap(rng, n_dec, B, stat_mse_decmean)
    boot_pr_dec = cluster_bootstrap(rng, n_dec, B, stat_pr)

    point_mse_w = dec_a["sum_sq"].sum() / dec_a["count"].sum() - dec_b["sum_sq"].sum() / dec_b["count"].sum()
    point_mse_m = dec_a["mean_sq"].mean() - dec_b["mean_sq"].mean()
    point_pr = (dec_a["error"].mean() - dec_b["error"].mean()) * 500.0

    # --- Game-level (coarser, conservative: decisions within a game are sequential) ---
    game_a = to_game_level(dec_a, ["sum_sq", "count", "mean_sq", "error"])
    game_b = to_game_level(dec_b, ["sum_sq", "count", "mean_sq", "error"])
    if not np.array_equal(game_a["game_seed"], game_b["game_seed"]):
        raise ValueError("arm A/B games don't align -- unexpected")
    n_games = len(game_a["game_seed"])

    def stat_mse_weighted_game(idx):
        c = game_a["count"][idx].sum()
        return (game_a["sum_sq"][idx].sum() - game_b["sum_sq"][idx].sum()) / c

    def stat_mse_decmean_game(idx):
        # equal-game weight of each game's own decision-mean MSE
        a = game_a["mean_sq"][idx].sum() / game_a["n_decisions"][idx].sum()
        b = game_b["mean_sq"][idx].sum() / game_b["n_decisions"][idx].sum()
        return a - b

    def stat_pr_game(idx):
        a = game_a["error"][idx].sum() / game_a["n_decisions"][idx].sum()
        b = game_b["error"][idx].sum() / game_b["n_decisions"][idx].sum()
        return (a - b) * 500.0

    boot_mse_w_game = cluster_bootstrap(rng, n_games, B, stat_mse_weighted_game)
    boot_mse_m_game = cluster_bootstrap(rng, n_games, B, stat_mse_decmean_game)
    boot_pr_game = cluster_bootstrap(rng, n_games, B, stat_pr_game)

    return {
        "panel": label,
        "n_decisions": int(n_dec),
        "n_games": int(n_games),
        "n_candidates": int(dec_a["count"].sum()),
        "bootstrap_B": B,
        "mse_candidate_weighted": {
            "decision_level": summarize_delta(point_mse_w, boot_mse_w_dec),
            "game_level": summarize_delta(point_mse_w, boot_mse_w_game),
        },
        "mse_decision_mean": {
            "decision_level": summarize_delta(point_mse_m, boot_mse_m_dec),
            "game_level": summarize_delta(point_mse_m, boot_mse_m_game),
        },
        "pr": {
            "decision_level": summarize_delta(point_pr, boot_pr_dec),
            "game_level": summarize_delta(point_pr, boot_pr_game),
        },
        "arm_a": {
            "mse_candidate_weighted": float(dec_a["sum_sq"].sum() / dec_a["count"].sum()),
            "mse_decision_mean": float(dec_a["mean_sq"].mean()),
            "pr": float(dec_a["error"].mean() * 500.0),
        },
        "arm_b": {
            "mse_candidate_weighted": float(dec_b["sum_sq"].sum() / dec_b["count"].sum()),
            "mse_decision_mean": float(dec_b["mean_sq"].mean()),
            "pr": float(dec_b["error"].mean() * 500.0),
        },
    }


def run_by_plan(dec_a: dict, dec_b: dict, rng: np.random.Generator, B: int) -> dict:
    """Decision-level-only breakdown per game plan (purerace/racing/attacking/
    priming/anchoring) -- a supporting metric, not the panel's headline. Per
    plan n is a fraction of the full decision count, so CIs are correspondingly
    wider; there's no separate game-level variant here since restricting to
    one plan leaves too few decisions per game to cluster meaningfully.
    """
    out = {}
    for plan in GAME_PLANS:
        mask = dec_a["game_plan"] == plan
        n = int(mask.sum())
        if n == 0:
            out[plan] = None
            continue
        sub_a = {k: v[mask] for k, v in dec_a.items()}
        sub_b = {k: v[mask] for k, v in dec_b.items()}

        def stat_mse(idx, sub_a=sub_a, sub_b=sub_b):
            c = sub_a["count"][idx].sum()
            return (sub_a["sum_sq"][idx].sum() - sub_b["sum_sq"][idx].sum()) / c

        def stat_pr(idx, sub_a=sub_a, sub_b=sub_b):
            return (sub_a["error"][idx] - sub_b["error"][idx]).mean() * 500.0

        point_mse = (
            sub_a["sum_sq"].sum() / sub_a["count"].sum()
            - sub_b["sum_sq"].sum() / sub_b["count"].sum()
        )
        point_pr = (sub_a["error"].mean() - sub_b["error"].mean()) * 500.0

        out[plan] = {
            "n_decisions": n,
            "mse_candidate_weighted": summarize_delta(
                point_mse, cluster_bootstrap(rng, n, B, stat_mse)
            ),
            "pr": summarize_delta(point_pr, cluster_bootstrap(rng, n, B, stat_pr)),
            "arm_a": {
                "mse_candidate_weighted": float(sub_a["sum_sq"].sum() / sub_a["count"].sum()),
                "pr": float(sub_a["error"].mean() * 500.0),
            },
            "arm_b": {
                "mse_candidate_weighted": float(sub_b["sum_sq"].sum() / sub_b["count"].sum()),
                "pr": float(sub_b["error"].mean() * 500.0),
            },
        }
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(result: dict, label_a: str, label_b: str) -> None:
    print()
    print("=" * 78)
    print(f"exp016 paired comparison: {result['panel']}")
    print(
        f"rollout tier: {result['n_decisions']} decisions / "
        f"{result['n_candidates']} candidates / {result['n_games']} games "
        f"(bootstrap B={result['bootstrap_B']})"
    )
    print("=" * 78)

    print(f"\n{'metric':<26}{label_a:>14}{label_b:>14}{'Delta (A-B)':>16}"
          f"{'decision CI95':>22}{'game CI95':>22}")

    def row(key, mse_key):
        a = result["arm_a"][mse_key]
        b = result["arm_b"][mse_key]
        dl = result[key]["decision_level"]
        gl = result[key]["game_level"]
        print(
            f"{mse_key:<26}{a:>14.4f}{b:>14.4f}{dl['point']:>16.4f}"
            f"  [{dl['ci95'][0]:+.4f},{dl['ci95'][1]:+.4f}]"
            f"  [{gl['ci95'][0]:+.4f},{gl['ci95'][1]:+.4f}]"
        )

    row("mse_candidate_weighted", "mse_candidate_weighted")
    row("mse_decision_mean", "mse_decision_mean")
    row("pr", "pr")

    mse_excl = result["mse_candidate_weighted"]["decision_level"]["excludes_zero"]
    pr_excl = result["pr"]["decision_level"]["excludes_zero"]
    same_sign = (
        (result["mse_candidate_weighted"]["decision_level"]["point"] > 0)
        == (result["pr"]["decision_level"]["point"] > 0)
    )
    print(
        f"\nMSE CI excludes 0: {mse_excl}   PR CI excludes 0: {pr_excl}   "
        f"MSE/PR same sign: {same_sign}"
    )
    if mse_excl and pr_excl and same_sign:
        print("-> resolved: MSE and PR agree on direction, both significant.")
    else:
        print("-> not jointly resolved (see methodology flags before claiming a winner).")

    by_plan = result.get("by_plan")
    if by_plan:
        print(
            f"\nBy game plan (supporting, decision-level bootstrap only; "
            f"n = rollout dec / all-tier dec):"
        )
        print(
            f"  {'plan':<12}{'n(roll/all)':>14}{'MSE Delta':>12}{'MSE CI95':>20}"
            f"{'PR Delta':>10}{'PR CI95':>18}"
        )
        for plan in GAME_PLANS:
            row = by_plan.get(plan)
            if row is None:
                print(f"  {plan:<12}  (no decisions)")
                continue
            m = row["mse_candidate_weighted"]
            p = row["pr"]
            print(
                f"  {plan:<12}"
                f"{row['n_decisions_rollout']:>7}/{row['n_decisions_all']:<6}"
                f"{m['point']:>12.4f}"
                f"  [{m['ci95'][0]:+.4f},{m['ci95'][1]:+.4f}]"
                f"{p['point']:>10.2f}"
                f"  [{p['ci95'][0]:+.2f},{p['ci95'][1]:+.2f}]"
            )
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump-a", required=True, help="Path to arm A's --dump-dir .npz")
    parser.add_argument("--dump-b", required=True, help="Path to arm B's --dump-dir .npz")
    parser.add_argument("--label-a", required=True)
    parser.add_argument("--label-b", required=True)
    parser.add_argument("--panel", required=True, help="Human label for this comparison")
    parser.add_argument("--tier", default="rollout", help="Reference tier for the MSE metrics (default: rollout)")
    parser.add_argument("--bootstrap", type=int, default=B_DEFAULT, dest="B")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed")
    parser.add_argument("--output", default=None, help="Path to save result JSON")
    args = parser.parse_args()

    raw_a = load_dump(args.dump_a)
    raw_b = load_dump(args.dump_b)

    # PR uses all tiers (the benchmark's own convention); MSE uses --tier only.
    dec_a_all = group_by_decision(raw_a)
    dec_b_all = group_by_decision(raw_b)
    assert_aligned(dec_a_all, dec_b_all, args.label_a, args.label_b)

    tier_a = restrict_tier(raw_a, args.tier)
    tier_b = restrict_tier(raw_b, args.tier)
    dec_a_tier = group_by_decision(tier_a)
    dec_b_tier = group_by_decision(tier_b)
    assert_aligned(dec_a_tier, dec_b_tier, args.label_a, args.label_b)

    # Splice: MSE fields from the tier-restricted grouping, PR ("error") from
    # the all-tier grouping, keyed on the tier-restricted decision set for MSE
    # and the full decision set for PR. Two separate comparisons, sharing the
    # same bootstrap machinery.
    rng = np.random.default_rng(args.seed)

    mse_result = run_comparison(dec_a_tier, dec_b_tier, rng, args.B, args.panel)
    pr_result = run_comparison(dec_a_all, dec_b_all, rng, args.B, args.panel)

    result = dict(mse_result)
    result["pr"] = pr_result["pr"]
    result["arm_a"]["pr"] = pr_result["arm_a"]["pr"]
    result["arm_b"]["pr"] = pr_result["arm_b"]["pr"]
    result["n_decisions_pr"] = pr_result["n_decisions"]
    result["n_games_pr"] = pr_result["n_games"]
    result["tier"] = args.tier

    # Per-game-plan breakdown (supporting metric): MSE by-plan restricted to
    # the rollout tier, PR by-plan over all tiers, matching the headline split.
    by_plan_mse = run_by_plan(dec_a_tier, dec_b_tier, rng, args.B)
    by_plan_pr = run_by_plan(dec_a_all, dec_b_all, rng, args.B)
    by_plan = {}
    for plan in GAME_PLANS:
        m = by_plan_mse.get(plan)
        p = by_plan_pr.get(plan)
        by_plan[plan] = {
            "n_decisions_rollout": m["n_decisions"] if m else 0,
            "n_decisions_all": p["n_decisions"] if p else 0,
            "mse_candidate_weighted": m["mse_candidate_weighted"] if m else None,
            "pr": p["pr"] if p else None,
            "arm_a": {
                "mse_candidate_weighted": m["arm_a"]["mse_candidate_weighted"] if m else None,
                "pr": p["arm_a"]["pr"] if p else None,
            },
            "arm_b": {
                "mse_candidate_weighted": m["arm_b"]["mse_candidate_weighted"] if m else None,
                "pr": p["arm_b"]["pr"] if p else None,
            },
        }
    result["by_plan"] = by_plan

    print_report(result, args.label_a, args.label_b)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
