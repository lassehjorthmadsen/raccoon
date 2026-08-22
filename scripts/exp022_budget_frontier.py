"""What the search's two filters are worth at a given compute budget.

A candidate move must be inside the equity window AND inside the top-k cap. The
first exp022 analysis swept the window with the cap pinned at 16 and concluded
that width barely matters -- true, but only because it measured at a budget where
both filters are saturated. This script sweeps both filters together and reports
the best achievable PR at each budget, which is the question that actually
matters for play.

Everything is replayed offline from the measured fixed-0.18/k=16 run: at depth 1
a move's searched value does not depend on which other moves were searched, so
any rule inside that run's window and cap is exact. Evaluation counts are
reconstructed from move generation, and reproduce the measured run to the digit.

Writes ``results/budget_frontier.json``:

* the (window, cap) grid -- PR and cost for each combination;
* the best PR reachable at each budget under three allocation schemes: one rule
  everywhere, three contact buckets, and one rule per position type. The last is
  an ORACLE -- the benchmark's type labels are annotations in the data file, not
  something an engine has over the board -- so it bounds what any position-aware
  scheme could reach.

Selection caveat, recorded in the output: every scheme is chosen on the same
2,000 positions it is scored on, so the margins are optimistic. Confirming one
needs a run on positions this sample does not contain.

    python3 scripts/exp022_budget_frontier.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_benchmark_pr import load_benchmark, subsample  # noqa: E402
from raccoon.search.expectimax import (  # noqa: E402
    ROLLS, _children, _terminal_value, board26_to_slots, contact_fraction,
    gate_skips, pass_turn,
)

STATIC_DUMP = Path("experiments/exp018-benchmark/dumps/exp018_ep22.npz")
SEARCH_DUMP = Path("experiments/exp022-contact/dumps/ep22_fixed_w018_k16.npz")
BENCH = Path("data/bgsage/money_benchmark/benchmark.json.gz")
OUT = Path("experiments/exp022-contact/results/budget_frontier.json")

WINDOWS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.18]
CAPS = [2, 3, 4, 6, 8, 10, 12, 14, 16]
CONFIGS = [(w, k) for w in WINDOWS for k in CAPS]
CAP_MAX, GATE = 16, 0.08
PLANS = ["purerace", "racing", "attacking", "priming", "anchoring"]
COMPLEX_CONTACT = 0.6
BINS = 3000


def _spans(key):
    _, starts = np.unique(key, return_index=True)
    starts = np.sort(starts)
    return dict(zip(key[starts], zip(starts, np.append(starts[1:], len(key)))))


def build():
    """Per decision: values, and every top-16 candidate's distinct child positions.

    Boards are interned to small integers within each decision, so a rule's cost
    is the size of a set union rather than a fresh walk of the move tree.
    """
    z_static = np.load(STATIC_DUMP, allow_pickle=True)
    z_search = np.load(SEARCH_DUMP, allow_pickle=True)
    s_static, s_search = _spans(z_static["decision_key"]), _spans(z_search["decision_key"])
    decisions, _ = load_benchmark(str(BENCH))
    sample = subsample(decisions, 2000, 21)

    out = []
    for (key, (a, b)), dec in zip(sorted(s_search.items(), key=lambda kv: kv[1][0]), sample):
        a0, b0 = s_static[key]
        static = z_static["pred_eq"][a0:b0] / 3.0
        cands = [pass_turn(board26_to_slots(m["board"])) for m in dec["moves"]]
        terminal = [_terminal_value(c) for c in cands]
        order = np.argsort(-static)
        gated = gate_skips(static, GATE)

        ids, roots = {}, []
        for c, t in zip(cands, terminal):
            if t is None:
                roots.append(ids.setdefault(c, len(ids)))
        kids = {}
        if not gated:
            for i in order[:CAP_MAX]:
                i = int(i)
                if terminal[i] is not None:
                    kids[i] = np.empty(0, np.int32)
                    continue
                seen = set()
                for die1, die2, _ in ROLLS:
                    for kid in _children(cands[i], die1, die2):
                        if _terminal_value(kid) is None:
                            seen.add(ids.setdefault(kid, len(ids)))
                kids[i] = np.array(sorted(seen), np.int32)

        out.append({
            "static": static, "searched": z_search["pred_eq"][a:b] / 3.0,
            "ref": z_search["ref_eq"][a:b], "order": order, "gated": gated,
            "plan": dec["game_plan"], "contact": contact_fraction(board26_to_slots(dec["board"])),
            "roots": np.array(sorted(set(roots)), np.int32), "kids": kids,
        })
    return out


def score(decisions, configs=None):
    """(error, evaluations) of every decision under every (window, cap)."""
    configs = CONFIGS if configs is None else configs
    n, m = len(decisions), len(configs)
    err, evals = np.zeros((n, m)), np.zeros((n, m))
    for j, (window, cap) in enumerate(configs):
        for i, d in enumerate(decisions):
            static, ref = d["static"], d["ref"]
            evals[i, j] = len(d["roots"])
            if d["gated"]:
                err[i, j] = ref.max() - ref[int(np.argmax(static))]
                continue
            best = static[d["order"][0]]
            keep = [int(x) for x in d["order"][:cap] if best - static[x] <= window / 3.0]
            chosen = set(keep)
            values = np.array([
                d["searched"][x] if x in chosen else static[x] for x in range(len(static))
            ])
            err[i, j] = ref.max() - ref[int(np.argmax(values))]
            parts = [d["kids"][x] for x in keep if x in d["kids"]]
            if parts:
                evals[i, j] += len(np.union1d(d["roots"], np.concatenate(parts))) - len(d["roots"])
    return err, evals


def frontier(err, evals, masks, n):
    """Best PR at or under each budget. Buckets are independent, so this is a
    plain knapsack over per-decision average cost -- exact, not a hull."""
    dp = np.full(BINS, np.inf)
    dp[0] = 0.0
    for mask in masks:
        contrib_e = err[mask].sum(0) / n * 500
        contrib_v = np.rint(evals[mask].sum(0) / n).astype(int)
        nxt = np.full(BINS, np.inf)
        for j in range(len(CONFIGS)):
            if contrib_v[j] >= BINS:
                continue
            shifted = np.full(BINS, np.inf)
            shifted[contrib_v[j]:] = dp[:BINS - contrib_v[j]] + contrib_e[j]
            nxt = np.minimum(nxt, shifted)
        dp = nxt
    return np.minimum.accumulate(dp)


def main() -> None:
    decisions = build()
    err, evals = score(decisions)
    n = len(decisions)
    contact = np.array([d["contact"] for d in decisions])
    plan = np.array([d["plan"] for d in decisions])

    schemes = {
        "one_rule": [np.ones(n, bool)],
        "three_contact_buckets": [
            contact == 0, (contact > 0) & (contact < COMPLEX_CONTACT),
            contact >= COMPLEX_CONTACT,
        ],
        "position_type_oracle": [plan == p for p in PLANS],
    }
    fronts = {k: frontier(err, evals, v, n) for k, v in schemes.items()}

    budgets = [800, 1000, 1200, 1400, 1600, 1800, 2000, 2400]
    result = {
        "note": (
            "Replayed offline from the measured fixed-0.18/k=16 run; exact for any rule "
            "inside that run's window and cap. Evaluation counts reproduce the measured "
            "run to the digit."
        ),
        "selection_caveat": (
            "Every scheme is chosen on the same 2,000 positions it is scored on, so its "
            "margin over a simpler scheme is optimistic. The three-bucket rule is picked "
            f"from {len(CONFIGS)**3:,} combinations. Confirming one needs a run on "
            "positions outside this sample."
        ),
        "subsample": {"n": n, "seed": 21},
        "complex_contact_threshold": COMPLEX_CONTACT,
        "grid": [
            {"window": w, "cap": k, "pr": float(err[:, j].mean() * 500),
             "evals_per_decision": float(evals[:, j].mean())}
            for j, (w, k) in enumerate(CONFIGS)
        ],
        "frontier": {
            "budgets": budgets,
            **{name: [float(f[b]) for b in budgets] for name, f in fronts.items()},
        },
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")

    print(f"{'budget':>8} " + "".join(f"{k:>24}" for k in fronts))
    for b in budgets:
        print(f"{b:>8} " + "".join(f"{f[b]:>24.3f}" for f in fronts.values()))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
