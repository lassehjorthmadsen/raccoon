"""What the search's equity window is actually worth, per position type.

exp022 compared one contact-scaled window against one fixed window and found no
difference. This script explains why, and it costs nothing to run: at depth 1 a
candidate's searched value does not depend on which *other* candidates were
searched, so any window rule no wider than an already-measured run can be
replayed exactly from that run's dump. Candidates inside the narrower window keep
their searched value, candidates outside revert to their static one -- which is
precisely what ``search_values`` does. The replay reproduces both measured arms'
PR and evaluation counts to the digit, so these are measurements, not estimates.

Writes ``results/window_sweep.json``:

* a fixed-window sweep, overall and per position type -- where each type stops
  improving;
* contact-scaled rules at several magnitudes, for the like-for-like comparison
  exp022's single pair could not make;
* an oracle that gives each position type its own saturation point. The
  benchmark's position-type labels are annotations in the data file, not
  something an engine has over the board, so this is a ceiling on any
  position-aware rule -- not a playable setting.

    python3 scripts/exp022_window_sweep.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_benchmark_pr import load_benchmark, subsample  # noqa: E402
from raccoon.search.expectimax import (  # noqa: E402
    ROLLS,
    _children,
    _terminal_value,
    board26_to_slots,
    contact_fraction,
    gate_skips,
    pass_turn,
)

STATIC_DUMP = Path("experiments/exp018-benchmark/dumps/exp018_ep22.npz")
SEARCH_DUMP = Path("experiments/exp022-contact/dumps/ep22_fixed_w018_k16.npz")
BENCH = Path("data/bgsage/money_benchmark/benchmark.json.gz")
OUT = Path("experiments/exp022-contact/results/window_sweep.json")

REPLAY_LIMIT = 0.18  # the measured run being replayed; no rule may exceed it
CAP, GATE = 16, 0.08
PLANS = ["purerace", "racing", "attacking", "priming", "anchoring"]


def _spans(key: np.ndarray) -> dict:
    _, starts = np.unique(key, return_index=True)
    starts = np.sort(starts)
    return dict(zip(key[starts], zip(starts, np.append(starts[1:], len(key)))))


def load_decisions():
    """Per decision: static values, searched values, references, contact, gate, plan."""
    z_static = np.load(STATIC_DUMP, allow_pickle=True)
    z_search = np.load(SEARCH_DUMP, allow_pickle=True)
    s_static, s_search = _spans(z_static["decision_key"]), _spans(z_search["decision_key"])

    decisions, _ = load_benchmark(str(BENCH))
    sample = subsample(decisions, 2000, 21)
    ordered = sorted(s_search.items(), key=lambda kv: kv[1][0])
    if len(ordered) != len(sample):
        raise SystemExit("search dump and subsample disagree on size")

    out = []
    for (key, (a, b)), dec in zip(ordered, sample):
        a0, b0 = s_static[key]
        static = z_static["pred_eq"][a0:b0] / 3.0
        out.append({
            "static": static,
            "searched": z_search["pred_eq"][a:b] / 3.0,
            "ref": z_search["ref_eq"][a:b],
            "contact": contact_fraction(board26_to_slots(dec["board"])),
            "gated": gate_skips(static, GATE),
            "plan": dec["game_plan"],
            "candidates": [pass_turn(board26_to_slots(m["board"])) for m in dec["moves"]],
        })
    return out


def evaluate(decisions, window_of):
    """(PR overall, PR per plan, evaluations per decision) under a window rule."""
    errors = {p: [] for p in PLANS}
    evals = 0
    for d in decisions:
        static, ref = d["static"], d["ref"]
        window = min(window_of(d["contact"], d["plan"]), REPLAY_LIMIT)

        terminal = [_terminal_value(c) for c in d["candidates"]]
        seen = {c for c, t in zip(d["candidates"], terminal) if t is None}
        evals += len(seen)

        if d["gated"]:
            errors[d["plan"]].append(ref.max() - ref[int(np.argmax(static))])
            continue

        order = np.argsort(-static)
        best = static[order[0]]
        keep = [i for i in order[:CAP] if best - static[i] <= window / 3.0]
        values = np.array([
            d["searched"][i] if i in set(keep) else static[i] for i in range(len(static))
        ])
        errors[d["plan"]].append(ref.max() - ref[int(np.argmax(values))])

        for i in [j for j in keep if terminal[j] is None]:
            for die1, die2, _ in ROLLS:
                for kid in _children(d["candidates"][i], die1, die2):
                    if kid not in seen and _terminal_value(kid) is None:
                        seen.add(kid)
                        evals += 1

    flat = [e for p in PLANS for e in errors[p]]
    return (
        float(np.mean(flat) * 500),
        {p: float(np.mean(errors[p]) * 500) for p in PLANS},
        evals / len(decisions),
    )


def main() -> None:
    decisions = load_decisions()
    mean_contact = {
        p: float(np.mean([d["contact"] for d in decisions if d["plan"] == p])) for p in PLANS
    }

    result = {
        "note": (
            "Window rules replayed exactly from the measured fixed-0.18 run: at depth 1 a "
            "candidate's searched value is independent of which other candidates were "
            "searched, so any rule no wider than 0.18 can be scored without running the "
            "network. Validated against the two measured arms, which it reproduces to the "
            "digit."
        ),
        "subsample": {"n": len(decisions), "seed": 21},
        "cap": CAP,
        "gate": GATE,
        "mean_contact_by_plan": mean_contact,
        "fixed_sweep": [],
        "contact_scaled": [],
    }

    print(f"{'rule':<44} {'PR':>7} {'evals/dec':>10}")
    for w in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]:
        pr, by_plan, cost = evaluate(decisions, lambda c, p, w=w: w)
        result["fixed_sweep"].append(
            {"window": w, "pr": pr, "by_plan": by_plan, "evals_per_decision": cost}
        )
        print(f"{'fixed ' + format(w, '.2f'):<44} {pr:>7.3f} {cost:>10,.0f}")

    # The saturation point of each type: the narrowest window within 0.005 PR of
    # that type's best. This is what the oracle hands out.
    need = {}
    for p in PLANS:
        best = min(row["by_plan"][p] for row in result["fixed_sweep"])
        need[p] = min(
            row["window"] for row in result["fixed_sweep"]
            if row["by_plan"][p] <= best + 0.005
        )
    result["saturation_window_by_plan"] = need

    for w0, w1 in [(0.02, 0.36), (0.02, 0.10), (0.02, 0.12), (0.00, 0.10)]:
        pr, by_plan, cost = evaluate(decisions, lambda c, p, a=w0, b=w1: a + (b - a) * c)
        result["contact_scaled"].append(
            {"w0": w0, "w1": w1, "pr": pr, "by_plan": by_plan, "evals_per_decision": cost}
        )
        print(f"{f'contact-scaled {w0:.2f}-{w1:.2f}':<44} {pr:>7.3f} {cost:>10,.0f}")

    pr, by_plan, cost = evaluate(decisions, lambda c, p: need[p])
    result["plan_oracle"] = {
        "window_by_plan": need, "pr": pr, "by_plan": by_plan, "evals_per_decision": cost,
        "caveat": (
            "Uses the benchmark's position-type labels, which an engine does not have "
            "over the board. A ceiling on position-aware allocation, not a setting."
        ),
    }
    print(f"{'ORACLE: each type its own saturation point':<44} {pr:>7.3f} {cost:>10,.0f}")

    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
