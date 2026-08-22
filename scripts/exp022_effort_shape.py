"""What shape a position-dependent search rule should have.

exp022 scaled the equity window as a straight line rising with contact. Two
things were wrong with that, and this script measures both.

First, a move must clear the window *and* the top-k cap, and the cap is the
stronger filter, so a position-dependent rule has to move both together. Here a
single "effort" number in [0, 1] drives both: window 0.02 to 0.10, cap 2 to 16.

Second, effort should not rise with contact. It is worth almost nothing at
*either* extreme -- a pure race has no interaction left to search, and a
near-opening position is one the network already knows well -- and worth the
most in between, where the game is actually being decided. So the shape is a
bump, not a ramp.

The two ends are different in kind, and the shape reflects it. Contact reaching
exactly zero is a real discontinuity: contact can never come back, and 12% of
decisions sit exactly there. A hard test is right at that end. The taper at the
top is genuinely gradual, so it gets a soft shoulder:

    effort(c) = tanh(c / 0.03) * tanh((1 - c) / 0.25)

Everything is replayed offline from the measured fixed-0.18/k=16 run, exact for
any rule inside that run's window and cap. Each shape is swept over an overall
multiplier so shapes are compared at matched compute, not at matched settings.

    python3 scripts/exp022_effort_shape.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp022_budget_frontier import build, score  # noqa: E402

# A finer grid than the budget sweep's: a smooth rule lands on every integer cap,
# and snapping to a coarse ladder would be mistaken for the shape's own behaviour.
WINDOWS = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.14, 0.18]
CAPS = list(range(2, 17))
CONFIGS = [(w, k) for w in WINDOWS for k in CAPS]

OUT = Path("experiments/exp022-contact/results/effort_shape.json")
BUDGETS = [900, 1000, 1100, 1200, 1300, 1400, 1600, 1800]
BANDS = [(-0.001, 0.001, "exactly 0"), (0.001, 0.25, "0–0.25"), (0.25, 0.45, "0.25–0.45"),
         (0.45, 0.65, "0.45–0.65"), (0.65, 0.85, "0.65–0.85"), (0.85, 9.0, "above 0.85")]

# The ladder every shape moves along: effort 0 is the leanest search, 1 the richest.
WINDOW_LO, WINDOW_HI = 0.02, 0.10
CAP_LO, CAP_HI = 2, 16


def shapes(contact):
    return {
        "flat": np.ones(len(contact)),
        "rising_ramp": contact.copy(),
        "bump": np.tanh(contact / 0.03) * np.tanh(np.clip(1 - contact, 0, None) / 0.25),
    }


def main() -> None:
    decisions = build()
    err, evals = score(decisions, CONFIGS)
    n = len(decisions)
    rows = np.arange(n)
    contact = np.array([d["contact"] for d in decisions])
    index = {c: i for i, c in enumerate(CONFIGS)}
    grid_w = np.array(WINDOWS)

    def configs_for(effort):
        """Effort per decision -> the grid config each one gets."""
        out = np.empty(n, int)
        for i, b in enumerate(np.clip(effort, 0, 1)):
            w = grid_w[np.argmin(np.abs(grid_w - (WINDOW_LO + (WINDOW_HI - WINDOW_LO) * b)))]
            k = int(np.clip(round(CAP_LO + (CAP_HI - CAP_LO) * b), CAPS[0], CAPS[-1]))
            out[i] = index[(float(w), k)]
        return out

    curves, best_at = {}, {}
    for name, shape in shapes(contact).items():
        pts: dict[int, float] = {}
        picks: dict[int, np.ndarray] = {}
        for mult in np.linspace(0.02, 2.0, 400):
            js = configs_for(mult * shape)
            cost = round(float(evals[rows, js].mean()))
            pr = float(err[rows, js].mean() * 500)
            if cost not in pts or pr < pts[cost]:
                pts[cost], picks[cost] = pr, js
        curves[name] = pts
        best_at[name] = {}
        for budget in BUDGETS:
            ok = [c for c in pts if c <= budget]
            if ok:
                c = min(ok, key=lambda x: pts[x])
                best_at[name][budget] = {"pr": pts[c], "evals_per_decision": c, "picks": picks[c]}

    # How much each contact band gains from being given more search.
    lean, rich = index[(0.06, 6)], index[(0.10, 12)]
    band_rows = []
    for lo, hi, label in BANDS:
        m = (contact > lo) & (contact <= hi)
        gain = float((err[m, lean].mean() - err[m, rich].mean()) * 500)
        extra = float(evals[m, rich].mean() - evals[m, lean].mean())
        band_rows.append({
            "band": label, "n": int(m.sum()),
            "pr_lean": float(err[m, lean].mean() * 500),
            "pr_rich": float(err[m, rich].mean() * 500),
            "gain": gain, "extra_evals": extra,
            "gain_per_1000_evals": 1000 * gain / extra if extra else 0.0,
        })

    result = {
        "note": (
            "Replayed offline from the measured fixed-0.18/k=16 run. One effort number "
            f"drives both filters: window {WINDOW_LO}-{WINDOW_HI}, cap {CAP_LO}-{CAP_HI}. "
            "Each shape is swept over an overall multiplier, so shapes are compared at "
            "matched compute rather than matched settings."
        ),
        "selection_caveat": (
            "The bump's two time constants were chosen after looking at these same 2,000 "
            "positions. Its margin over the flat rule is 1.1-1.2 standard errors, so it is "
            "suggestive, not established; confirming needs positions outside this sample."
        ),
        "subsample": {"n": n, "seed": 21},
        "effort_to_settings": {
            "window": [WINDOW_LO, WINDOW_HI], "cap": [CAP_LO, CAP_HI],
            "bump": "tanh(contact / 0.03) * tanh((1 - contact) / 0.25)",
        },
        "value_of_search_by_contact_band": band_rows,
        "budgets": BUDGETS,
        "by_shape": {
            name: [round(best_at[name][b]["pr"], 4) if b in best_at[name] else None
                   for b in BUDGETS]
            for name in curves
        },
    }

    # Paired comparison at the budget where the shapes differ most.
    b = 1000
    jf, jb = best_at["flat"][b]["picks"], best_at["bump"][b]["picks"]
    diff = (err[rows, jb] - err[rows, jf]) * 500
    result["paired_at_1000"] = {
        "pr_flat": best_at["flat"][b]["pr"], "pr_bump": best_at["bump"][b]["pr"],
        "margin": float(diff.mean()),
        "ci95_same_positions": float(1.96 * diff.std(ddof=1) / np.sqrt(n)),
    }

    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{'budget':>8} " + "".join(f"{k:>16}" for k in curves))
    for i, b in enumerate(BUDGETS):
        print(f"{b:>8} " + "".join(
            f"{(result['by_shape'][k][i] if result['by_shape'][k][i] else float('nan')):>16.3f}"
            for k in curves))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
