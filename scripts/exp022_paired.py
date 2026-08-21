"""exp022 — the paired difference between a fixed and a contact-scaled window.

The two arms search the same 2,000 positions and agree on nearly all of them, so
comparing their PR figures as if they were separate samples buries the difference
under between-position variance: the interval comes out ±0.24 on a difference of
0.004. Comparing decision by decision — which is legitimate here, because the
positions are identical — is what makes the comparison readable at all.

Reads the per-candidate dumps (gitignored) and writes the conclusion to
``results/`` (committed), which is what the write-up renders from.

    python3 scripts/exp022_paired.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EXP = Path("experiments/exp022-contact")
FIXED = "ep22_fixed_w018_k16"
SMOOTH = "ep22_smooth_isocost"


def per_decision(label: str) -> tuple[list, np.ndarray, np.ndarray]:
    """(decision keys, equity error of the chosen move, index chosen) per decision."""
    z = np.load(EXP / "dumps" / f"{label}.npz", allow_pickle=True)
    pred, ref, key = z["pred_eq"], z["ref_eq"], z["decision_key"]
    _, starts = np.unique(key, return_index=True)
    starts = np.sort(starts)

    keys, errors, chosen = [], [], []
    for s, e in zip(starts, np.append(starts[1:], len(key))):
        p, r = pred[s:e], ref[s:e]
        i = int(np.argmax(p))
        keys.append(key[s])
        errors.append(float(r.max() - r[i]))
        chosen.append(i)
    return keys, np.array(errors), np.array(chosen)


def main() -> None:
    k_fixed, e_fixed, i_fixed = per_decision(FIXED)
    k_smooth, e_smooth, i_smooth = per_decision(SMOOTH)
    if k_fixed != k_smooth:
        raise SystemExit("the two arms scored different decisions — not comparable")

    n = len(e_fixed)
    diff = e_smooth - e_fixed          # positive = the contact-scaled window is worse
    ci_paired = 1.96 * diff.std(ddof=1) / np.sqrt(n) * 500
    ci_separate = 1.96 * np.sqrt(e_fixed.var(ddof=1) / n + e_smooth.var(ddof=1) / n) * 500

    cost = {}
    for label in (FIXED, SMOOTH):
        r = json.loads((EXP / "results" / f"{label}.json").read_text())
        cost[label] = r["search"]["evals_per_decision"]

    out = {
        "comparison": (
            "ep22 1-ply search, contact-scaled equity window (0.02->0.36) vs a fixed "
            "window (0.18), same k=16 cap, same positions"
        ),
        "n_decisions": n,
        "note": (
            "Both arms scored on identical positions, so the margin is measured "
            "decision by decision. Computed from the per-move dumps, which are not "
            "committed; this file is the committed conclusion."
        ),
        "pr_fixed_window": float(e_fixed.mean() * 500),
        "pr_contact_scaled_window": float(e_smooth.mean() * 500),
        "margin_pr": float(diff.mean() * 500),
        "margin_sign": "positive means the contact-scaled window is worse",
        "ci95_same_positions": float(ci_paired),
        "ci95_if_treated_as_separate_samples": float(ci_separate),
        "same_move_fraction": float(np.mean(i_fixed == i_smooth)),
        "decisions_where_they_differ": int((i_fixed != i_smooth).sum()),
        "evals_per_decision": {
            "fixed_window": cost[FIXED],
            "contact_scaled_window": cost[SMOOTH],
            "ratio": cost[SMOOTH] / cost[FIXED],
        },
    }

    path = EXP / "results" / "paired_window_rule.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
