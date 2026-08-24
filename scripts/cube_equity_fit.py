#!/usr/bin/env python3
"""Test each Janowski cubeful-equity formula against measured rollout equities.

The BGSage benchmark's *checker* decisions carry far more usable evidence about
the cube than its cube decisions do. Every candidate move rolled out to
completion has a **measured cubeful equity** alongside the cubeless probability
vector for the same position, and the parent decision records where the cube
sat. That is a direct observation of the map Janowski's model is trying to be --
42,636 of them, against 700 rolled-out cube positions.

Better still, the cube location varies across them, so each of the three
formulas can be tested on its own:

    E_O = C_V [ p(W + L + 0.5x) - L ]              we own the cube
    E_C = 4C_V/(4-x) [ p(W + L + 0.5x) - L - 0.25x ]   cube centred
    E_U = C_V [ p(W + L + 0.5x) - L - 0.5x ]       opponent owns it

(this module scores whichever centred form the engines actually ship, the
piecewise-linear one in :mod:`raccoon.cube.janowski`, not the closed form above).

Two caveats on what these positions are, both worth carrying into any
conclusion:

* They are **post-move** positions, so the opponent is on roll, whereas the
  benchmark's cube entries are pre-roll decision points. Janowski's formulas do
  not model whose turn it is -- the probabilities are supposed to carry that --
  so applying one index to both is itself an approximation, and the two datasets
  are related but not identical measurements.
* The rollout's payoff is empirical, but the cube decisions made during it come
  from a bot that also uses Janowski. So these measure equity under
  Janowski-quality cube play rather than optimal cube play.
"""

import argparse
import gzip
import json
import os

import numpy as np

from raccoon.cube import janowski as J
from raccoon.eval.cube_benchmark import DEFAULT_BENCHMARK

X_GRID = np.round(np.arange(0.05, 1.0001, 0.005), 4)

OWNERS = [(J.PLAYER, "E_O", "we own the cube"),
          (J.CENTERED, "E_C", "cube centred"),
          (J.OPPONENT, "E_U", "opponent owns it")]


def load_measured(path: str) -> list[tuple]:
    """(cube_owner, probs, equity, game_plan, key, cube_value) per rolled-out candidate.

    Both the probabilities and the equity are from the mover's point of view, and
    a checker play does not move the cube, so the parent decision's cube_owner
    and cube_value apply unchanged to each candidate. The equity is in units of
    the current cube, so a gammon is 2.0 whatever the cube reads.
    """
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    return [(e["cube_owner"], m["probs"], m["equity"], e.get("game_plan"), e["key"],
             e.get("cube_value"))
            for e in data["decisions"] if e["kind"] == "checker"
            for m in e["moves"] if m["eval_level"] == "Rollout"]


def residuals(rows, owner: str, x: float) -> np.ndarray:
    jac = J.jacoby_active(owner, True)
    return np.array([J.cl2cf_money(p, owner, x, jac) - eq for _, p, eq, *_ in rows])


def with_cube_action(probs, owner: str, x: float) -> float:
    """The position's value once the cube *action* is taken, not just held.

    ``cl2cf_money`` gives the NO-DOUBLE equity -- what a position is worth if the
    cube stays where it is. A rollout reference always includes the option to
    double, so comparing the two directly charges the formula for an option it
    was never asked to price. This applies the action so both sides mean the same
    thing.

    **Whose option it is matters, and getting it backwards inverts the result.**
    These are post-move positions, so the *opponent* is on roll:

    * we own the cube -> they cannot double, so there is no action to take;
    * cube centred or theirs -> they may double, and they will choose whatever
      is worst for us. Our value is therefore ``min(ND, max(DT, -1))``, taking a
      minimum over their choice and a maximum over ours (take or pass), not the
      ``max(ND, min(DT, DP))`` that applies when the decision is ours.

    Verified against GNU Backgammon on four positions: its "No double" line
    matches ``cl2cf_money`` closely, and its optimal line matches the rollout, so
    the distinction is real and not an artefact of this implementation.
    """
    nd = J.cl2cf_money(probs, owner, x, J.jacoby_active(owner, True))
    if owner == J.PLAYER:
        return nd
    # They double; we take (owning twice the stake) or pass at -1.
    dt = 2.0 * J.cl2cf_money(probs, J.PLAYER, x, False)
    return min(nd, max(dt, -1.0))


def residuals_action(rows, owner: str, x: float) -> np.ndarray:
    """Residuals of the cube-action value -- the like-for-like comparison."""
    return np.array([with_cube_action(p, owner, x) - eq for _, p, eq, *_ in rows])


def fit_one(rows, owner: str) -> dict:
    """Bias/RMSE at the published index, and at the freely fitted one."""
    def stats(x):
        r = residuals_action(rows, owner, x)
        return {"bias": float(r.mean()), "rmse": float(np.sqrt((r ** 2).mean()))}
    scored = [(stats(x)["rmse"], x) for x in X_GRID]
    best_rmse, best_x = min(scored)
    published = stats(J.X_CONTACT)
    nd_r = residuals(rows, owner, J.X_CONTACT)
    var = float(np.var([eq for _, _, eq, *_ in rows]))
    return {
        "n": len(rows),
        "published_x": J.X_CONTACT,
        "bias_at_published_x": published["bias"],
        "rmse_at_published_x": published["rmse"],
        "best_x": float(best_x),
        "rmse_at_best_x": float(best_rmse),
        "bias_no_double_only": float(nd_r.mean()),
        "rmse_no_double_only": float(np.sqrt((nd_r ** 2).mean())),
        "r2_at_best_x": float(1.0 - best_rmse ** 2 / var) if var > 0 else float("nan"),
    }


def _dump_scatter(rows, best_x: float, args) -> None:
    """A committed sample of (observed, modelled) pairs for the write-up's plots.

    Stored as parallel arrays with the labels factored out, which keeps a
    12,000-row sample well under a megabyte. A plain uniform sample, so the
    natural mix of cube locations and game plans is preserved.
    """
    rng = np.random.default_rng(args.scatter_seed)
    idx = rng.choice(len(rows), size=min(args.scatter_sample, len(rows)),
                     replace=False)
    sample = [rows[i] for i in sorted(idx)]

    owners = sorted({r[0] for r in sample})
    plans = sorted({r[3] for r in sample if r[3]})
    out = {
        "n": len(sample),
        "seed": args.scatter_seed,
        "published_x": J.X_CONTACT,
        "best_x_pooled": float(best_x),
        "owners": owners,
        "plans": plans,
        "owner_idx": [owners.index(r[0]) for r in sample],
        "plan_idx": [plans.index(r[3]) if r[3] in plans else -1 for r in sample],
        "p_win": [round(r[1][0], 4) for r in sample],
        "observed": [round(r[2], 4) for r in sample],
        "modelled_published": [
            round(with_cube_action(r[1], r[0], J.X_CONTACT), 4) for r in sample],
        "modelled_best": [
            round(with_cube_action(r[1], r[0], best_x), 4) for r in sample],
    }
    path = os.path.join(args.output, "cube_equity_scatter.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"  Saved: {path}  ({len(sample):,} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    ap.add_argument("--output", help="directory for cube_equity_fit.json")
    ap.add_argument("--scatter-sample", type=int, default=12000,
                    help="rows of (observed, modelled) to dump for plotting; the "
                         "benchmark archive is not in git, so the write-up plots "
                         "this committed sample rather than reading it directly")
    ap.add_argument("--scatter-seed", type=int, default=24)
    args = ap.parse_args()

    rows = load_measured(args.benchmark)
    out = {
        "measurement": ("Janowski cubeful-equity formulas vs measured rollout "
                        "equities, one formula per cube location"),
        "source": "BGSage money benchmark, checker candidates with eval_level=Rollout",
        "n_total": len(rows),
        "by_cube_location": {},
        "by_game_plan": {},
        "by_cash_point_distance": {},
        "equity_profile": {},
        "equity_profile_by_observed": {},
        "jacoby_dead_branch": {},
        "branch_divergence": {},
        "pooled": {},
    }

    print(f"measured cubeful equities: n={len(rows):,}\n")
    print("| formula | cube location | n | bias @0.68 | RMSE @0.68 | best x | RMSE there |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for owner, sym, label in OWNERS:
        sub = [r for r in rows if r[0] == owner]
        res = fit_one(sub, owner)
        out["by_cube_location"][owner] = {"formula": sym, "label": label, **res}
        print(f"| {sym} | {label} | {res['n']:,} | {res['bias_at_published_x']:+.4f} "
              f"| {res['rmse_at_published_x']:.4f} | {res['best_x']:.3f} "
              f"| {res['rmse_at_best_x']:.4f} |")

    # Does the centred formula fail everywhere, or only in some position types?
    plans = sorted({r[3] for r in rows if r[3]})
    for plan in plans:
        out["by_game_plan"][plan] = {}
        for owner, sym, _ in OWNERS:
            sub = [r for r in rows if r[3] == plan and r[0] == owner]
            if not sub:
                continue
            r = residuals(sub, owner, J.X_CONTACT)
            out["by_game_plan"][plan][owner] = {
                "formula": sym, "n": len(sub),
                "rmse_at_published_x": float(np.sqrt((r ** 2).mean())),
                "bias_at_published_x": float(r.mean()),
            }

    # Where in the range does each formula fail? Distance from the model's own
    # cash point is the natural axis: it is where the live branch hits its +1
    # ceiling, and where a blend between a capped branch and an uncapped one is
    # least likely to behave.
    EDGES = [-1.0, -0.30, -0.15, -0.05, 0.0, 0.05, 0.15, 0.30, 1.0]
    for owner, sym, _ in OWNERS:
        jac = J.jacoby_active(owner, True)
        sub = [r for r in rows if r[0] == owner]
        dist = np.array([r[1][0] - J.cash_point(*J.compute_wl(r[1]), J.X_CONTACT)
                         for r in sub])
        res = np.array([with_cube_action(r[1], owner, J.X_CONTACT) - r[2]
                        for r in sub])
        bins = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            m = (dist >= lo) & (dist < hi)
            if m.sum() < 30:
                continue
            bins.append({"lo": lo, "hi": hi, "n": int(m.sum()),
                         "mean_residual": float(res[m].mean()),
                         "rmse": float(np.sqrt((res[m] ** 2).mean()))})
        out["by_cash_point_distance"][owner] = {"formula": sym, "bins": bins}

    # What the model says against what happened, as a function of winning
    # chance -- the axis a backgammon player already thinks in. Both series come
    # from the same positions in each bin, so the gap between them IS the error.
    P_EDGES = np.round(np.arange(0.05, 1.0001, 0.05), 3)
    for owner, sym, label in OWNERS:
        jac = J.jacoby_active(owner, True)
        sub = [r for r in rows if r[0] == owner]
        pw = np.array([r[1][0] for r in sub])
        obs = np.array([r[2] for r in sub])
        mod = np.array([with_cube_action(r[1], owner, J.X_CONTACT) for r in sub])
        act = np.array([J.cl2cf_money(r[1], owner, J.X_CONTACT, jac) for r in sub])
        bins = []
        for lo, hi in zip(P_EDGES[:-1], P_EDGES[1:]):
            m = (pw >= lo) & (pw < hi)
            if m.sum() < 40:
                continue
            bins.append({"p_lo": float(lo), "p_hi": float(hi), "n": int(m.sum()),
                         "p_mean": float(pw[m].mean()),
                         "observed": float(obs[m].mean()),
                         "modelled": float(mod[m].mean()),
                         "modelled_with_action": float(act[m].mean())})
        r_nd = mod - obs
        r_act = act - obs
        out["equity_profile"][owner] = {
            "formula": sym, "label": label, "bins": bins,
            "rmse_nd": float(np.sqrt((r_nd ** 2).mean())),
            "rmse_with_action": float(np.sqrt((r_act ** 2).mean())),
            "bias_nd": float(r_nd.mean()),
            "bias_with_action": float(r_act.mean()),
        }

    # The same profile binned on observed equity, so the write-up's residual
    # figures can all share one x-axis with the scatters instead of asking the
    # reader to switch between equity and winning chance.
    E_EDGES = np.round(np.arange(-1.6, 1.601, 0.2), 2)
    for owner, sym, label in OWNERS:
        sub = [r for r in rows if r[0] == owner]
        obs_o = np.array([r[2] for r in sub])
        mod_o = np.array([with_cube_action(r[1], owner, J.X_CONTACT) for r in sub])
        keys = np.array([r[4] for r in sub])
        resid_o = mod_o - obs_o
        bins = []
        for lo, hi in zip(E_EDGES[:-1], E_EDGES[1:]):
            m = (obs_o >= lo) & (obs_o < hi)
            if m.sum() < 40:
                continue
            # Candidates inside one checker decision are the same position played
            # different ways, so they are anything but independent. Cluster on the
            # decision: average within it first, then take the interval across
            # decisions. The naive interval is reported alongside to show the gap.
            r_in = resid_o[m]
            per_decision = {}
            for key, val in zip(keys[m], r_in):
                per_decision.setdefault(key, []).append(val)
            means = np.array([np.mean(v) for v in per_decision.values()])
            n_dec = means.size
            se_clu = (means.std(ddof=1) / np.sqrt(n_dec)) if n_dec > 1 else float("nan")
            se_naive = (r_in.std(ddof=1) / np.sqrt(r_in.size)) if r_in.size > 1 else float("nan")
            bins.append({"eq_lo": float(lo), "eq_hi": float(hi), "n": int(m.sum()),
                         "n_decisions": int(n_dec),
                         "eq_mean": float(obs_o[m].mean()),
                         "observed": float(obs_o[m].mean()),
                         "modelled": float(mod_o[m].mean()),
                         "ci95_clustered": float(1.96 * se_clu),
                         "ci95_naive": float(1.96 * se_naive)})
        out["equity_profile_by_observed"][owner] = {
            "formula": sym, "label": label, "bins": bins}

    # Why the owned-cube sag sits where it does. The blend can only hurt where the
    # dead and live branches disagree, so tabulate that disagreement against the
    # measured error. If the sag is about cashing, the LIVE branch should fit it
    # well and the positions should be gammon-poor; if it were about playing on
    # for a gammon, the reverse.
    own = [r for r in rows if r[0] == J.PLAYER]
    div_bins = []
    for lo in (0.4, 0.6, 0.8, 1.0, 1.2):
        sub = [r for r in own if lo <= r[2] < lo + 0.2]
        if len(sub) < 40:
            continue
        obs_v = float(np.mean([r[2] for r in sub]))
        blend = float(np.mean([J.cl2cf_money(r[1], J.PLAYER, J.X_CONTACT, False)
                               for r in sub]))
        live = float(np.mean([J.cl2cf_money(r[1], J.PLAYER, 1.0, False) for r in sub]))
        dead = float(np.mean([J.cubeless_equity(r[1]) for r in sub]))
        div_bins.append({
            "eq_lo": lo, "eq_hi": lo + 0.2, "n": len(sub),
            "observed": obs_v, "blend_x068": blend, "live_x1": live, "dead": dead,
            "branch_divergence": live - dead,
            "gap_blend": blend - obs_v, "gap_live": live - obs_v,
            "gammon_win_rate": float(np.mean([r[1][1] for r in sub])),
            "frac_above_cash_point": float(np.mean(
                [r[1][0] > J.cash_point(*J.compute_wl(r[1]), J.X_CONTACT) for r in sub])),
        })
    out["branch_divergence"] = {"cube": "player", "bins": div_bins}

    # Before asking whether the Jacoby handling is wrong, establish that the
    # references were generated with Jacoby ON at all -- the whole centred
    # analysis assumes it. The rule is self-evidencing in the payoffs: with the
    # cube centred it suppresses gammons, so a near-certain gammon can only be
    # cashed for 1 point. With the cube owned the rule is inactive by definition,
    # which makes owned positions the matched control -- same rollout machinery,
    # same units, gammons paying double.
    GAMMON_BANDS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 1.01)]
    P_CERTAIN = 0.85
    jac_rows = {}
    for owner, _, _ in OWNERS:
        sub = [r for r in rows if r[0] == owner]
        eq = np.array([r[2] for r in sub])
        pw = np.array([r[1][0] for r in sub])
        pg = np.array([r[1][1] for r in sub])
        cl = np.array([J.cubeless_equity(r[1]) for r in sub])
        certain = pw > P_CERTAIN
        bands = []
        for lo, hi in GAMMON_BANDS:
            m = certain & (pg >= lo) & (pg < hi)
            if m.sum() < 5:
                continue
            bands.append({"g_lo": lo, "g_hi": min(hi, 1.0), "n": int(m.sum()),
                          "mean_p_win": float(pw[m].mean()),
                          "mean_equity": float(eq[m].mean()),
                          "mean_cubeless_equity": float(cl[m].mean())})
        heavy = pg > 0.5
        jac_rows[owner] = {
            "n": len(sub),
            "cube_values": sorted({r[5] for r in sub if r[5] is not None}),
            "max_equity": float(eq.max()),
            "n_gammon_favourite": int(heavy.sum()),
            "mean_equity_gammon_favourite":
                float(eq[heavy].mean()) if heavy.any() else None,
            "max_equity_gammon_favourite":
                float(eq[heavy].max()) if heavy.any() else None,
            "gammon_slope": bands,
        }

    # Jacoby caps the *undoubled* payoff at 1, not the position: doubling and
    # being taken pays 2. So a handful of centred positions do exceed +1, and
    # what they are matters -- if they were gammonish the Jacoby story would be
    # in trouble.
    cen_hi = [r for r in rows if r[0] == J.CENTERED and r[2] > 1.0]
    eqh = np.array([r[2] for r in cen_hi])
    out["jacoby_in_force"] = {
        "by_cube_location": jac_rows,
        "p_certain": P_CERTAIN,
        "centred_above_one": {
            "n": len(cen_hi),
            "mean_p_win": float(np.mean([r[1][0] for r in cen_hi])),
            "max_p_win_gammon": float(max(r[1][1] for r in cen_hi)),
            "mean_p_win_gammon": float(np.mean([r[1][1] for r in cen_hi])),
            "mean_cubeless_equity": float(np.mean([J.cubeless_equity(r[1]) for r in cen_hi])),
            "mean_equity": float(eqh.mean()),
            "max_equity": float(eqh.max()),
        },
    }

    # Is the Jacoby treatment implicated in the centred formula's error?
    #
    # Under Jacoby the dead-cube branch becomes 2p-1: if the cube is never turned
    # gammons never count. That is right *for that branch*, but it carries weight
    # (1-x) = 0.32 in positions where the cube will almost certainly be turned --
    # so a position with heavy gammon LOSSES gets a third of its value from a
    # branch pretending those gammons are free.
    #
    # The signal is therefore in the gammon *asymmetry*, P(lose gammon) minus
    # P(win gammon), not in the total gammon rate: symmetric gammons cancel.
    # Testing the total is what made an earlier version of this analysis
    # wrongly exonerate the Jacoby handling.
    cen = [r for r in rows if r[0] == J.CENTERED]
    def _cen_action(probs, jac):
        nd = J.cl2cf_money(probs, J.CENTERED, J.X_CONTACT, jac)
        dt = 2.0 * J.cl2cf_money(probs, J.PLAYER, J.X_CONTACT, False)
        return min(nd, max(dt, -1.0))
    res_on = np.array([_cen_action(r[1], True) - r[2] for r in cen])
    res_off = np.array([_cen_action(r[1], False) - r[2] for r in cen])
    asym = np.array([r[1][3] - r[1][1] for r in cen])
    total = np.array([r[1][3] + r[1][1] for r in cen])
    pw_c = np.array([r[1][0] for r in cen])

    bands = []
    for lo in np.arange(0.25, 0.75, 0.05):
        m = (pw_c >= lo) & (pw_c < lo + 0.05)
        if m.sum() < 200:
            continue
        bands.append({"p_lo": float(round(lo, 2)), "p_hi": float(round(lo + 0.05, 2)),
                      "n": int(m.sum()),
                      "corr_resid_asymmetry": float(np.corrcoef(asym[m], res_on[m])[0, 1])})
    hi_q = asym > np.quantile(asym, 0.75)
    out["jacoby_dead_branch"] = {
        "n": len(cen),
        "corr_total_gammon_rate": float(np.corrcoef(total, res_on)[0, 1]),
        "corr_gammon_asymmetry": float(np.corrcoef(asym, res_on)[0, 1]),
        "corr_winning_probability": float(np.corrcoef(pw_c, res_on)[0, 1]),
        "within_p_band": bands,
        "as_shipped": {"rmse": float(np.sqrt((res_on ** 2).mean())),
                       "bias_worst_asymmetry_quartile": float(res_on[hi_q].mean())},
        "dead_branch_keeps_gammons": {
            "rmse": float(np.sqrt((res_off ** 2).mean())),
            "bias_worst_asymmetry_quartile": float(res_off[hi_q].mean())},
    }

    # One index applied everywhere, which is what the engines actually ship.
    def pooled_rmse(x):
        r = np.concatenate([residuals_action([q for q in rows if q[0] == o], o, x)
                            for o, _, _ in OWNERS])
        return float(np.sqrt((r ** 2).mean())), float(r.mean())
    scored = [(pooled_rmse(x)[0], x) for x in X_GRID]
    best_rmse, best_x = min(scored)
    r68, b68 = pooled_rmse(J.X_CONTACT)
    out["pooled"] = {"n": len(rows), "bias_at_published_x": b68,
                     "rmse_at_published_x": r68,
                     "best_x": float(best_x), "rmse_at_best_x": float(best_rmse)}
    print("\nresidual by distance from the cash point (negative = model undershoots):")
    for owner, v in out["by_cash_point_distance"].items():
        worst = min(v["bins"], key=lambda b: b["mean_residual"])
        print(f"  {owner:9s} worst bin p-CP [{worst['lo']:+.2f},{worst['hi']:+.2f}) "
              f"n={worst['n']:,} mean residual {worst['mean_residual']:+.4f}")

    print(f"\nsingle index everywhere: x=0.68 -> rmse {r68:.4f}; "
          f"best x={best_x:.3f} -> rmse {best_rmse:.4f}")

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        _dump_scatter(rows, best_x, args)
        path = os.path.join(args.output, "cube_equity_fit.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
