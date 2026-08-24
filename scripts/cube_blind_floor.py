#!/usr/bin/env python3
"""What cube-blind checker play costs, measured on the BGSage money benchmark.

Raccoon ranks candidate moves by cubeless equity. The benchmark scores every
candidate both ways, so we can ask what a *perfect* cubeless player would still
throw away when its moves are judged cubefully -- the floor no amount of extra
cubeless accuracy can get below.

This is a motivating measurement for the cube work, not an experiment: it sizes
a prize without claiming it. The experiment that goes after it (ranking moves by
cubeful equity instead) is pre-registered separately.

Writes a results JSON so the write-up can quote the number without reading the
benchmark archive, which is not in git.
"""

import argparse
import json
import os

import numpy as np

from raccoon.eval.cube_benchmark import (
    DEFAULT_BENCHMARK, GAME_PLANS, PR_MULTIPLIER, TIERS,
)


def load_checker_decisions(path: str) -> list[dict]:
    import gzip
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    return [d for d in data["decisions"] if d["kind"] == "checker"]


def cube_blind_errors(decisions: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Cubeful equity lost per decision by picking the cubeless-best move.

    Returns (errors, differs) where ``differs`` marks the decisions on which the
    cubeless-best and cubeful-best moves are not the same play.
    """
    errors, differs = [], []
    for entry in decisions:
        moves = entry["moves"]
        if len(moves) < 2:
            errors.append(0.0)
            differs.append(False)
            continue
        # The move a perfect cubeless player picks...
        chosen = max(moves, key=lambda m: m["cubeless_equity"])
        # ...judged against the best move by cubeful equity.
        best_cubeful = max(m["equity"] for m in moves)
        errors.append(max(0.0, best_cubeful - chosen["equity"]))
        differs.append(abs(chosen["equity"] - best_cubeful) > 1e-9)
    return np.array(errors), np.array(differs, dtype=bool)


def summarise(errors: np.ndarray, decisions: list[dict]) -> dict:
    se = errors.std(ddof=1) / np.sqrt(errors.size)
    return {
        "n": int(errors.size),
        "pr": float(PR_MULTIPLIER * errors.mean()),
        "pr_ci95": float(PR_MULTIPLIER * 1.96 * se),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    p.add_argument("--output", help="directory to write cube_blind_floor.json into")
    args = p.parse_args()

    decisions = load_checker_decisions(args.benchmark)
    errors, differs = cube_blind_errors(decisions)

    result = {
        "measurement": "PR of perfect cubeless checker play, scored cubefully",
        "benchmark": "BGSage Money Benchmark (checker decisions)",
        **summarise(errors, decisions),
        "decisions_where_best_move_differs": int(differs.sum()),
        "fraction_differs": float(differs.mean()),
        "by_game_plan": {},
        "by_cube_owner": {},
        "by_tier": {},
    }
    for plan in GAME_PLANS:
        sel = np.array([d.get("game_plan") == plan for d in decisions], dtype=bool)
        result["by_game_plan"][plan] = summarise(errors[sel], decisions)
    for owner in ("centered", "player", "opponent"):
        sel = np.array([d["cube_owner"] == owner for d in decisions], dtype=bool)
        result["by_cube_owner"][owner] = summarise(errors[sel], decisions)
    for tier in TIERS:
        sel = np.array([d["tier"] == tier for d in decisions], dtype=bool)
        result["by_tier"][tier] = summarise(errors[sel], decisions)

    print(f"n = {result['n']:,} checker decisions")
    print(f"cube-blind floor PR = {result['pr']:.3f} +- {result['pr_ci95']:.3f}")
    print(f"best move differs on {result['decisions_where_best_move_differs']:,} "
          f"({100 * result['fraction_differs']:.1f}%)")
    for owner, v in result["by_cube_owner"].items():
        print(f"  {owner:9s} n={v['n']:5d}  PR={v['pr']:.3f}")

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        path = os.path.join(args.output, "cube_blind_floor.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
