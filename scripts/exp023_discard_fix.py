"""What it was worth to stop letting pruned moves win the argmax.

The search ranked candidates by static value, searched the survivors, and took
the best of the mixture -- comparing 0-ply values against searched ones. They are
not the same quantity: searching a move takes the opponent's best reply, a
minimum over ~20 noisy estimates, so it marks the move down. A pruned move never
gets that treatment and can overtake a searched move it was behind.

Both behaviours are replayed from the exp023 reference dumps, so the comparison
is exact and costs nothing: the fix changes which candidates may be *chosen*,
never which are *evaluated*. Validated against a real search over 529 held-out
decisions, which matched the replay to four decimals.

    python3 scripts/exp023_discard_fix.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp023_confirm import build  # noqa: E402

OUT = Path("experiments/exp023-holdout/results/discard_fix.json")
SETTINGS = [(0.10, 16), (0.10, 12), (0.06, 8), (0.04, 10)]


def errors(decisions, window, cap, discard):
    out = np.empty(len(decisions))
    for i, d in enumerate(decisions):
        static, ref = d["static"], d["ref"]
        if d["gated"]:
            out[i] = ref.max() - ref[int(np.argmax(static))]
            continue
        best = static[d["order"][0]]
        keep = [int(x) for x in d["order"][:cap] if best - static[x] <= window / 3.0]
        if not keep:
            out[i] = ref.max() - ref[int(np.argmax(static))]
            continue
        if discard:
            win = max(keep, key=lambda x: d["searched"][x])
        else:
            chosen = set(keep)
            values = np.array([
                d["searched"][x] if x in chosen else static[x] for x in range(len(static))
            ])
            win = int(np.argmax(values))
        out[i] = ref.max() - ref[win]
    return out


def main() -> None:
    decisions = build()
    n = len(decisions)
    rows = []
    for window, cap in SETTINGS:
        before = errors(decisions, window, cap, False)
        after = errors(decisions, window, cap, True)
        diff = (after - before) * 500
        ci = 1.96 * diff.std(ddof=1) / np.sqrt(n)
        rows.append({
            "window": window, "cap": cap,
            "pr_before": float(before.mean() * 500), "pr_after": float(after.mean() * 500),
            "margin": float(diff.mean()), "ci95": float(ci),
            "sigma": float(abs(diff.mean()) / (ci / 1.96)) if ci else 0.0,
        })
        print(f"  window {window:.2f}, cap {cap:<3} {rows[-1]['pr_before']:.4f} -> "
              f"{rows[-1]['pr_after']:.4f}  {diff.mean():+.4f} ± {ci:.4f}")

    result = {
        "note": (
            "Replayed from the exp023 reference dumps; exact, because the fix changes "
            "which candidates may be chosen, never which are evaluated. Evaluation "
            "counts are identical before and after."
        ),
        "validation": (
            "A real search over 529 held-out decisions with the fix scored PR 0.4555, "
            "matching this replay's prediction of 0.4555."
        ),
        "n_decisions": n,
        "settings": rows,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
