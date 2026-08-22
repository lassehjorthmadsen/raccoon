"""Where exp022's search compute actually goes, broken down by position type.

The write-up originally explained the cost difference between the two window
rules by asserting that races are cheap to search -- few legal moves, few close
contenders. Both halves are wrong, and this script is what showed it: races have
*more* near-ties than tactical positions, and a race with contact is the most
expensive position type per move searched.

At depth 1 the evaluation count depends only on move generation, the root filter
and deduplication -- never on the network's values -- so it can be reconstructed
exactly from the committed 0-ply dump without running the net. The reconstruction
reproduces both arms' measured evals-per-decision to the digit, which is what
makes it trustworthy as an explanation rather than a plausible story.

    python3 scripts/exp022_cost_breakdown.py
"""
from __future__ import annotations

import json
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

DUMP = Path("experiments/exp018-benchmark/dumps/exp018_ep22.npz")
BENCH = Path("data/bgsage/money_benchmark/benchmark.json.gz")
OUT = Path("experiments/exp022-contact/results/cost_by_plan.json")
PLANS = ["purerace", "racing", "attacking", "priming", "anchoring"]

FIXED_WINDOW = 0.18
SMOOTH = (0.02, 0.36)
CAP = 16
GATE = 0.08


def evals_for(candidates, static, window_eq):
    """(evaluations, root moves searched) for one decision at depth 1.

    Mirrors ``search_values`` with ``depth=1``: evaluate the live candidates, and
    unless the decision is gated, expand every candidate that is inside the cap
    and the window over all 21 rolls, counting each distinct non-terminal
    position once. ``static`` is in equity/3, ``window_eq`` in equity.
    """
    terminal = [_terminal_value(c) for c in candidates]
    seen = {c for c, t in zip(candidates, terminal) if t is None}
    total = len(seen)
    if gate_skips(static, GATE):
        return total, 0

    order = np.argsort(-static)
    best = static[order[0]]
    keep = [
        i for i in order[:CAP]
        if best - static[i] <= window_eq / 3.0 and terminal[i] is None
    ]
    for i in keep:
        for die1, die2, _ in ROLLS:
            for kid in _children(candidates[i], die1, die2):
                if kid not in seen and _terminal_value(kid) is None:
                    seen.add(kid)
                    total += 1
    return total, len(keep)


def main() -> None:
    dump = np.load(DUMP, allow_pickle=True)
    pred, key = dump["pred_eq"], dump["decision_key"]
    _, starts = np.unique(key, return_index=True)
    starts = np.sort(starts)
    spans = list(zip(starts, np.append(starts[1:], len(key))))

    decisions, _ = load_benchmark(str(BENCH))
    if len(decisions) != len(spans):
        raise SystemExit(f"dump has {len(spans)} decisions, benchmark {len(decisions)}")
    wanted = {id(d) for d in subsample(decisions, 2000, 21)}

    w0, w1 = SMOOTH
    per_plan: dict[str, list] = {}
    for dec, (s, e) in zip(decisions, spans):
        if id(dec) not in wanted:
            continue
        candidates = [pass_turn(board26_to_slots(m["board"])) for m in dec["moves"]]
        static = pred[s:e] / 3.0
        contact = contact_fraction(board26_to_slots(dec["board"]))

        fixed = evals_for(candidates, static, FIXED_WINDOW)
        smooth = evals_for(candidates, static, w0 + (w1 - w0) * contact)
        per_plan.setdefault(dec["game_plan"], []).append(
            (len(candidates), contact, *fixed, *smooth)
        )

    n_total = sum(len(v) for v in per_plan.values())
    out = {
        "note": (
            "Depth-1 evaluation counts reconstructed from the committed 0-ply dump. "
            "At depth 1 the count depends only on move generation, the root filter "
            "and deduplication, never on the network's values, so this is exact "
            "rather than a model -- it reproduces both arms' measured "
            "evals_per_decision to the digit."
        ),
        "n_decisions": n_total,
        "subsample": {"n": 2000, "seed": 21},
        "fixed_window": FIXED_WINDOW,
        "contact_scaled_window": {"w0": w0, "w1": w1},
        "cap": CAP,
        "gate": GATE,
        "by_plan": {},
    }
    totals = np.zeros(4)
    for plan in PLANS:
        a = np.array(per_plan[plan])
        totals += [a[:, 2].sum(), a[:, 3].sum(), a[:, 4].sum(), a[:, 5].sum()]
        out["by_plan"][plan] = {
            "n": len(a),
            "mean_contact": float(a[:, 1].mean()),
            "mean_legal_moves": float(a[:, 0].mean()),
            "fixed": {
                "evals_per_decision": float(a[:, 2].mean()),
                "moves_searched_per_decision": float(a[:, 3].mean()),
                "evals_per_searched_move": float(a[:, 2].sum() / max(a[:, 3].sum(), 1)),
            },
            "contact_scaled": {
                "evals_per_decision": float(a[:, 4].mean()),
                "moves_searched_per_decision": float(a[:, 5].mean()),
                "evals_per_searched_move": float(a[:, 4].sum() / max(a[:, 5].sum(), 1)),
            },
        }
    out["overall"] = {
        "fixed": {
            "evals_per_decision": totals[0] / n_total,
            "moves_searched": int(totals[1]),
        },
        "contact_scaled": {
            "evals_per_decision": totals[2] / n_total,
            "moves_searched": int(totals[3]),
        },
        "evals_ratio": totals[2] / totals[0],
        "moves_searched_ratio": totals[3] / totals[1],
    }

    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["overall"], indent=2))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
