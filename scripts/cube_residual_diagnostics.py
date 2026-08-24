#!/usr/bin/env python3
"""Where Janowski's error actually lives, and whether the Jacoby rule explains it.

The model misses the centred no-double line badly while fitting the take/pass
line well. The Jacoby rule is the obvious suspect, since it is the one piece of
the model that applies only to a centred cube. This script runs the three tests
that settle it and writes them to a results JSON, so the write-up quotes a
committed measurement rather than a remembered one.

The tests:

1. *Where the residual sits.* Jacoby only modifies the two outer segments of the
   centred-cube curve; the doubling window between the take point and the cash
   point is identical either way. If the error is inside the window, Jacoby
   cannot be causing it.
2. *Whether it scales with gammons.* Jacoby's entire mechanism is suppressing
   gammons, so a Jacoby artifact should grow with the gammon rate.
3. *Whether turning Jacoby off helps.* Each variant is given its own best-fit
   index, so the comparison is between the two models at their best rather than
   at one arbitrary constant.

Best-fit indices here are fitted by *equity RMSE*, which makes them diagnostics
and not the experiment: exp024's arms are fitted on cube PR, the metric it
reports. The two disagree, and that disagreement is itself worth seeing.
"""

import argparse
import json
import os

import numpy as np

from raccoon.cube import janowski as J
from raccoon.eval.cube_benchmark import (
    DEFAULT_BENCHMARK, load_cube_decisions, measured_only,
)
from raccoon.search.expectimax import board26_to_slots, contact_fraction

X_GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.005), 4)


def nd_residuals(entries, x: float, jacoby: bool) -> np.ndarray:
    """Modelled minus reference no-double equity, one per entry."""
    return np.array([
        J.cl2cf_money(e["probs"], e["cube_owner"], x, jacoby) - e["equity_nd"]
        for e in entries
    ])


def best_x_by_rmse(entries, jacoby: bool) -> tuple[float, float]:
    """The index minimising ND equity RMSE, and that RMSE."""
    rmses = [float(np.sqrt((nd_residuals(entries, x, jacoby) ** 2).mean()))
             for x in X_GRID]
    i = int(np.argmin(rmses))
    return float(X_GRID[i]), rmses[i]


def describe(entries, jacoby: bool, x: float = J.X_CONTACT) -> dict:
    r = nd_residuals(entries, x, jacoby)
    best_x, best_rmse = best_x_by_rmse(entries, jacoby)
    return {
        "n": len(entries),
        "bias_at_published_x": float(r.mean()),
        "rmse_at_published_x": float(np.sqrt((r ** 2).mean())),
        "best_x_by_rmse": best_x,
        "rmse_at_best_x": best_rmse,
    }


def _best_dt_x(entries) -> dict:
    """The index minimising double/take equity RMSE, fitted freely."""
    if not entries:
        return {"n": 0}
    def rmse(x: float) -> float:
        r = np.array([2.0 * J.cl2cf_money(e["probs"], J.OPPONENT, x, False)
                      - e["equity_dt"] for e in entries])
        return float(np.sqrt((r ** 2).mean()))
    vals = [rmse(x) for x in X_GRID]
    i = int(np.argmin(vals))
    return {"n": len(entries), "best_x_by_rmse": float(X_GRID[i]),
            "rmse_at_best_x": vals[i], "rmse_at_published_x": rmse(J.X_CONTACT)}


def _E_C_janowski(probs, x1: float, x2: float) -> float:
    """Janowski's own cube-centred equity, refined model equation (11).

    Collapses to the Appendix-1 closed form 2(E_O + E_U)/(4 - x) when x1 == x2.
    Note the paper states this is not applicable under the Jacoby rule -- there
    is no Jacoby-valid centred formula in it, which is the gap being measured.
    """
    W, L = J.compute_wl(probs)
    p = probs[0]
    e_o = p * (W + L + 0.5 * x1) - L
    e_u = p * (W + L + 0.5 * x2) - L - 0.5 * x2
    denom = 4.0 * (x1 + x2) - 2.0 * x1 * x2
    return 4.0 * (x1 * e_o + x2 * e_u) / denom


def _best_fit(entries, predict) -> dict:
    """Best single index by ND equity RMSE, over a formulation."""
    def rmse(x):
        r = np.array([predict(e, x) - e["equity_nd"] for e in entries])
        return float(np.sqrt((r ** 2).mean()))
    vals = [(rmse(x), x) for x in X_GRID if x > 0.0]
    best_rmse, best_x = min(vals)
    return {"n": len(entries), "best_x": best_x, "rmse_at_best_x": best_rmse,
            "rmse_at_published_x": rmse(J.X_CONTACT)}


def region_of(entry: dict) -> str:
    """Which segment of the live-cube curve this position sits on.

    The Jacoby clamp reaches only the two outer segments.
    """
    W, L = J.compute_wl(entry["probs"])
    p = entry["probs"][0]
    if p < J.take_point(W, L, 1.0):
        return "below_take_point"
    if p < J.cash_point(W, L, 1.0):
        return "doubling_window"
    return "above_cash_point"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    ap.add_argument("--output", help="directory for residual_diagnostics.json")
    args = ap.parse_args()

    entries, _ = load_cube_decisions(args.benchmark)
    # Only the rolled-out references can judge a Janowski variant; the rest were
    # produced by a search that uses Janowski at its leaves.
    entries = measured_only(entries)
    centered = [e for e in entries if e["cube_owner"] == J.CENTERED]

    out = {
        "measurement": "where the Janowski no-double residual lives (centred cube)",
        "published_x": J.X_CONTACT,
        "note": ("best-fit indices here minimise equity RMSE and are diagnostics; "
                 "exp024's arms are fitted on cube PR, the metric it reports"),
        "overall": {
            "jacoby_on": describe(centered, True),
            "jacoby_off": describe(centered, False),
        },
        "by_region": {},
        "by_gammon_quartile": [],
        "correlations": {},
        "by_contact_quintile": [],
    }

    for region in ("below_take_point", "doubling_window", "above_cash_point"):
        sel = [e for e in centered if region_of(e) == region]
        if not sel:
            continue
        out["by_region"][region] = {
            "jacoby_on": describe(sel, True),
            "jacoby_off": describe(sel, False),
        }

    # Does the residual scale with gammons, as a Jacoby artifact would?
    gammon = np.array([e["probs"][1] + e["probs"][3] for e in centered])
    resid = nd_residuals(centered, J.X_CONTACT, True)
    edges = np.quantile(gammon, [0.0, 0.25, 0.5, 0.75, 1.0])
    for i in range(4):
        hi = gammon <= edges[i + 1] if i == 3 else gammon < edges[i + 1]
        sel = (gammon >= edges[i]) & hi
        out["by_gammon_quartile"].append({
            "gammon_rate_lo": float(edges[i]),
            "gammon_rate_hi": float(edges[i + 1]),
            "n": int(sel.sum()),
            "bias": float(resid[sel].mean()),
        })

    # The take/pass line, fitted freely. If the method is sound this should
    # rediscover GNUBG's published constants without being told them.
    out["dt_validation"] = {
        "all": _best_dt_x(entries),
        "by_game_plan": {
            plan: _best_dt_x([e for e in entries if e.get("game_plan") == plan])
            for plan in sorted({e.get("game_plan") for e in entries} - {None})
        },
    }

    # Which formulation of the cube-centred equity fits best, each given its own
    # freely fitted index. This is the comparison that identifies E_C -- not the
    # index -- as the broken component.
    out["centred_formulations"] = {
        "janowski_closed_form": _best_fit(
            centered, lambda e, x: _E_C_janowski(e["probs"], x, x)),
        "piecewise_linear_jacoby_on": _best_fit(
            centered, lambda e, x: J.cl2cf_money(e["probs"], J.CENTERED, x, True)),
        "piecewise_linear_jacoby_off": _best_fit(
            centered, lambda e, x: J.cl2cf_money(e["probs"], J.CENTERED, x, False)),
    }
    # For scale: the take/pass line, which the same machinery fits cleanly.
    out["centred_formulations"]["reference_dt_line"] = {
        "n": len(entries),
        "rmse_at_published_x": float(np.sqrt(np.mean([
            (2.0 * J.cl2cf_money(e["probs"], J.OPPONENT, J.X_CONTACT, False)
             - e["equity_dt"]) ** 2 for e in entries]))),
    }

    contact = np.array([contact_fraction(board26_to_slots(e["board"]))
                        for e in centered])
    out["correlations"] = {
        "residual_vs_gammon_rate": float(np.corrcoef(gammon, resid)[0, 1]),
        "residual_vs_contact": float(np.corrcoef(contact, resid)[0, 1]),
        "residual_vs_p_win": float(np.corrcoef(
            [e["probs"][0] for e in centered], resid)[0, 1]),
    }

    # Would an index that varies with contact help? (Angle (c) in the write-up.)
    cedges = np.quantile(contact, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    for i in range(5):
        hi = contact <= cedges[i + 1] if i == 4 else contact < cedges[i + 1]
        sel = (contact >= cedges[i]) & hi
        group = [e for e, k in zip(centered, sel) if k]
        bx, brmse = best_x_by_rmse(group, True)
        out["by_contact_quintile"].append({
            "contact_lo": float(cedges[i]), "contact_hi": float(cedges[i + 1]),
            "n": len(group), "best_x_by_rmse": bx, "rmse_at_best_x": brmse,
        })

    on, off = out["overall"]["jacoby_on"], out["overall"]["jacoby_off"]
    print(f"centred ND, n={on['n']}")
    print(f"  jacoby on : bias {on['bias_at_published_x']:+.4f} "
          f"rmse {on['rmse_at_published_x']:.4f} | best x {on['best_x_by_rmse']:.3f} "
          f"rmse {on['rmse_at_best_x']:.4f}")
    print(f"  jacoby off: bias {off['bias_at_published_x']:+.4f} "
          f"rmse {off['rmse_at_published_x']:.4f} | best x {off['best_x_by_rmse']:.3f} "
          f"rmse {off['rmse_at_best_x']:.4f}")
    for region, v in out["by_region"].items():
        print(f"  {region:18s} n={v['jacoby_on']['n']:5d} "
              f"bias {v['jacoby_on']['bias_at_published_x']:+.4f}")
    print(f"  corr(residual, gammon rate) = "
          f"{out['correlations']['residual_vs_gammon_rate']:+.3f}")
    print("\ncentred-equity formulations, each at its own best index:")
    for k, v in out["centred_formulations"].items():
        if "best_x" in v:
            print(f"  {k:30s} n={v['n']:4d} best x={v['best_x']:.3f} "
                  f"rmse {v['rmse_at_best_x']:.4f}")
        else:
            print(f"  {k:30s} n={v['n']:4d} (DT line at published x) "
                  f"rmse {v['rmse_at_published_x']:.4f}")
    dv = out["dt_validation"]
    print(f"take/pass line, fitted freely: x = {dv['all']['best_x_by_rmse']:.3f} "
          f"(published {J.X_CONTACT})")
    for plan, v in dv["by_game_plan"].items():
        print(f"  {plan:10s} n={v['n']:5d} x = {v['best_x_by_rmse']:.3f}")

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        path = os.path.join(args.output, "residual_diagnostics.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
