"""Pick the contact-dependent window endpoints for exp022, offline.

The filter's cost is driven by how many candidates clear the equity window, so a
smooth window can be made to cost the same as the fixed one *by construction*
rather than by tuning against PR. This script reads the committed 0-ply dump
(per-candidate predicted equities) plus the benchmark boards, computes each
decision's contact, and reports mean survivors under each candidate ``(w0, w1)``.

No network and no search runs: it is arithmetic over data already on disk.

    python3 scripts/exp022_calibrate_window.py
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from raccoon.env.encoder import contact_pips
from raccoon.eval.gnubg_adapter import board_to_view
from raccoon.search.expectimax import board26_to_slots

DUMP = Path("experiments/exp018-benchmark/dumps/exp018_ep22.npz")
BENCH = Path("data/bgsage/money_benchmark/benchmark.json.gz")
FIXED_WINDOW = 0.16  # the exp021 baseline the smooth rows must match on cost


def decision_slices(key: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, end) row ranges, one per decision."""
    _, starts = np.unique(key, return_index=True)
    starts = np.sort(starts)
    return list(zip(starts, np.append(starts[1:], len(key))))


def main() -> None:
    dump = np.load(DUMP, allow_pickle=True)
    pred, key = dump["pred_eq"], dump["decision_key"]
    spans = decision_slices(key)

    with gzip.open(BENCH, "rt") as f:
        decisions = [d for d in json.load(f)["decisions"] if d["kind"] == "checker"]
    if len(decisions) != len(spans):
        raise SystemExit(f"dump has {len(spans)} decisions, benchmark {len(decisions)}")

    contact = np.empty(len(decisions))
    gaps: list[np.ndarray] = []
    for i, (dec, (s, e)) in enumerate(zip(decisions, spans)):
        slots = board26_to_slots(dec["board"])
        my_c, opp_c = contact_pips(board_to_view([list(slots[0]), list(slots[1])]))
        contact[i] = min(1.0, (my_c + opp_c) / 2.0 / 167.0)
        p = pred[s:e]
        gaps.append(p.max() - p)

    def mean_survivors(window: np.ndarray) -> float:
        return float(np.mean([np.sum(g <= w) for g, w in zip(gaps, window)]))

    baseline = mean_survivors(np.full(len(gaps), FIXED_WINDOW))
    print(f"decisions {len(gaps)}   contact: mean {contact.mean():.3f}, "
          f"median {np.median(contact):.3f}, "
          f"pct<0.35 {100 * np.mean(contact < 0.35):.0f}%")
    print(f"\nfixed window {FIXED_WINDOW} -> mean survivors {baseline:.2f}  (the bar to match)\n")

    print(f"{'w0':>6} {'w1':>6} {'mean surv':>10} {'vs fixed':>9}")
    best = None
    for w0 in (0.00, 0.02, 0.04, 0.06, 0.08):
        for w1 in (0.24, 0.28, 0.32, 0.36, 0.40, 0.48):
            window = w0 + (w1 - w0) * contact
            surv = mean_survivors(window)
            ratio = surv / baseline
            flag = ""
            if 0.97 <= ratio <= 1.03:
                flag = " <- iso-cost"
                if best is None or abs(ratio - 1) < abs(best[2] / baseline - 1):
                    best = (w0, w1, surv)
            print(f"{w0:>6.2f} {w1:>6.2f} {surv:>10.2f} {ratio:>8.2f}x{flag}")

    if best:
        print(f"\nRow A (iso-cost): w0={best[0]}, w1={best[1]} -> {best[2]:.2f} survivors "
              f"({best[2] / baseline:.3f}x fixed)")
    cheap = min(
        ((w0, w1, mean_survivors(w0 + (w1 - w0) * contact))
         for w0 in (0.0, 0.02) for w1 in (0.24, 0.32)),
        key=lambda r: r[2],
    )
    print(f"Row B (cheaper):  w0={cheap[0]}, w1={cheap[1]} -> {cheap[2]:.2f} survivors "
          f"({cheap[2] / baseline:.3f}x fixed)")


if __name__ == "__main__":
    main()
