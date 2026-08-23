"""exp023 — does the bump-shaped effort rule beat one fixed setting on held-out positions?

exp022 produced two hypotheses by replaying its own 2,000-decision sample: move
both filters together rather than the window alone, and make effort a bump in
contact rather than a ramp. Those positions cannot test what they generated, so
this scores both on the 12,693 benchmark decisions exp022's sample excludes.

Everything below replays the committed reference run (window 0.10, cap 16, gate
0.08) rather than searching again. At depth 1 a move's searched value does not
depend on which other moves were searched, so any rule inside that window and cap
is exact -- the same replay that reproduced both exp022 arms to the digit.

FROZEN BEFORE THE RUN, and not to be retuned here (see experiments/pipeline_exp023.sh):

    effort(c) = tanh(c / 0.03) * tanh((1 - c) / 0.25)
    window(c) = 0.02 + 0.08 * effort   cap(c) = round(2 + 14 * effort)
    multiplier chosen so the bump lands near TARGET_BUDGET evaluations/decision
    flat comparator = the single (window, cap) whose cost is CLOSEST to the
                      bump's -- not the best flat setting found afterwards

    python3 scripts/exp023_confirm.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_benchmark_pr import load_benchmark, subsample_complement  # noqa: E402
from raccoon.search.expectimax import (  # noqa: E402
    ROLLS, _children, _terminal_value, board26_to_slots, contact_fraction,
    gate_skips, pass_turn,
)

STATIC_DUMP = Path("experiments/exp018-benchmark/dumps/exp018_ep22.npz")
SHARD_DUMPS = sorted(Path("experiments/exp023-holdout/dumps").glob("ep22_ref_w010_k16_shard*.npz"))
BENCH = Path("data/bgsage/money_benchmark/benchmark.json.gz")
OUT = Path("experiments/exp023-holdout/results/paired_holdout.json")

SAMPLE_N, SAMPLE_SEED = 2000, 21          # the exp022 sample this holdout excludes
REF_WINDOW, REF_CAP, GATE = 0.10, 16, 0.08
WINDOW_LO, WINDOW_HI, CAP_LO, CAP_HI = 0.02, 0.10, 2, 16
TAU_LO, TAU_HI = 0.03, 0.25
TARGET_BUDGET = 1000

WINDOWS = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
CAPS = list(range(2, 17))


def _spans(key):
    _, starts = np.unique(key, return_index=True)
    starts = np.sort(starts)
    return dict(zip(key[starts], zip(starts, np.append(starts[1:], len(key)))))


def build():
    z_static = np.load(STATIC_DUMP, allow_pickle=True)
    s_static = _spans(z_static["decision_key"])

    searched, refs = {}, {}
    for path in SHARD_DUMPS:
        z = np.load(path, allow_pickle=True)
        for key, (a, b) in _spans(z["decision_key"]).items():
            searched[key] = z["pred_eq"][a:b] / 3.0
            refs[key] = z["ref_eq"][a:b]
    if not searched:
        raise SystemExit("no shard dumps found — has the reference run finished?")

    decisions, _ = load_benchmark(str(BENCH))
    holdout = subsample_complement(decisions, SAMPLE_N, SAMPLE_SEED)

    out = []
    for dec in holdout:
        key = dec["key"]
        if key not in searched:
            continue                      # shard incomplete; skip rather than guess
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
            for i in order[:REF_CAP]:
                i = int(i)
                if terminal[i] is not None:
                    kids[i] = np.empty(0, np.int32)
                    continue
                seen = set()
                for d1, d2, _ in ROLLS:
                    for kid in _children(cands[i], d1, d2):
                        if _terminal_value(kid) is None:
                            seen.add(ids.setdefault(kid, len(ids)))
                kids[i] = np.array(sorted(seen), np.int32)

        out.append({
            "static": static, "searched": searched[key], "ref": refs[key],
            "order": order, "gated": gated, "plan": dec["game_plan"],
            "contact": contact_fraction(board26_to_slots(dec["board"])),
            "roots": np.array(sorted(set(roots)), np.int32), "kids": kids,
        })
    return out


def run(decisions, window_of, cap_of):
    """(per-decision error, per-decision evaluations) under a rule."""
    err = np.empty(len(decisions))
    evals = np.empty(len(decisions))
    for i, d in enumerate(decisions):
        static, ref = d["static"], d["ref"]
        evals[i] = len(d["roots"])
        if d["gated"]:
            err[i] = ref.max() - ref[int(np.argmax(static))]
            continue
        window = min(window_of(d["contact"]), REF_WINDOW)
        cap = min(int(cap_of(d["contact"])), REF_CAP)
        best = static[d["order"][0]]
        keep = [int(x) for x in d["order"][:cap] if best - static[x] <= window / 3.0]
        chosen = set(keep)
        values = np.array([
            d["searched"][x] if x in chosen else static[x] for x in range(len(static))
        ])
        err[i] = ref.max() - ref[int(np.argmax(values))]
        parts = [d["kids"][x] for x in keep if x in d["kids"]]
        if parts:
            evals[i] += len(np.union1d(d["roots"], np.concatenate(parts))) - len(d["roots"])
    return err, evals


def main() -> None:
    decisions = build()
    n = len(decisions)
    print(f"replaying {n:,} held-out decisions from {len(SHARD_DUMPS)} shards")

    def bump_effort(c, mult):
        e = np.tanh(c / TAU_LO) * np.tanh(max(0.0, 1 - c) / TAU_HI)
        return float(np.clip(mult * e, 0, 1))

    def bump_rule(mult):
        return (lambda c: WINDOW_LO + (WINDOW_HI - WINDOW_LO) * bump_effort(c, mult),
                lambda c: round(CAP_LO + (CAP_HI - CAP_LO) * bump_effort(c, mult)))

    # The multiplier is set by cost alone -- the pre-registered target -- never by PR.
    best_mult, best_gap = None, None
    for mult in np.linspace(0.05, 2.0, 40):
        _, v = run(decisions, *bump_rule(mult))
        gap = abs(v.mean() - TARGET_BUDGET)
        if best_gap is None or gap < best_gap:
            best_mult, best_gap = float(mult), gap
    err_bump, ev_bump = run(decisions, *bump_rule(best_mult))
    cost_bump = float(ev_bump.mean())
    print(f"bump: multiplier {best_mult:.3f} -> {cost_bump:,.0f} evals/decision "
          f"(target {TARGET_BUDGET})")

    # The flat comparator is whichever fixed setting costs closest to that, by rule.
    flat = []
    for w in WINDOWS:
        for k in CAPS:
            e, v = run(decisions, lambda c, w=w: w, lambda c, k=k: k)
            flat.append({"window": w, "cap": k, "pr": float(e.mean() * 500),
                         "evals_per_decision": float(v.mean()), "err": e})
    comparator = min(flat, key=lambda r: abs(r["evals_per_decision"] - cost_bump))
    err_flat = comparator["err"]
    print(f"flat comparator (closest cost): window {comparator['window']:.2f}, "
          f"cap {comparator['cap']} -> {comparator['evals_per_decision']:,.0f} evals/decision")

    diff = (err_bump - err_flat) * 500
    ci = 1.96 * diff.std(ddof=1) / np.sqrt(n)
    pr_bump, pr_flat = float(err_bump.mean() * 500), comparator["pr"]

    # What the pre-registered n was actually powered to detect, given the spread here.
    detectable = (1.96 + 0.84) * diff.std(ddof=1) / np.sqrt(n)

    result = {
        "hypothesis": (
            "At matched compute, a bump-shaped position-dependent effort rule scores a "
            "lower PR than the best single fixed setting."
        ),
        "positions": "BGSage money benchmark, the decisions NOT in exp022's "
                     f"{SAMPLE_N}-decision seed-{SAMPLE_SEED} sample",
        "n_decisions": n,
        "frozen_before_run": {
            "effort": f"tanh(c/{TAU_LO}) * tanh((1-c)/{TAU_HI})",
            "window": [WINDOW_LO, WINDOW_HI], "cap": [CAP_LO, CAP_HI],
            "target_budget": TARGET_BUDGET,
            "comparator_rule": "fixed setting whose cost is closest to the bump's",
        },
        "bump": {"multiplier": best_mult, "pr": pr_bump, "evals_per_decision": cost_bump},
        "flat": {"window": comparator["window"], "cap": comparator["cap"], "pr": pr_flat,
                 "evals_per_decision": comparator["evals_per_decision"]},
        "margin_pr": float(diff.mean()),
        "margin_sign": "negative means the bump is better",
        "ci95_same_positions": float(ci),
        "sigma": float(abs(diff.mean()) / (ci / 1.96)) if ci else 0.0,
        "smallest_detectable_at_80pct_power": float(detectable),
        "exp022_in_sample_margin": -0.061,
        "flat_grid": [{k: r[k] for k in ("window", "cap", "pr", "evals_per_decision")}
                      for r in flat],
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")

    print(f"\n  bump  PR {pr_bump:.4f} at {cost_bump:,.0f} evals")
    print(f"  flat  PR {pr_flat:.4f} at {comparator['evals_per_decision']:,.0f} evals")
    print(f"  margin {diff.mean():+.4f} ± {ci:.4f}  ({result['sigma']:.1f} sigma)")
    print(f"  smallest effect this n could detect at 80% power: {detectable:.4f}")
    print(f"  exp022 measured {result['exp022_in_sample_margin']:+.3f} in-sample")
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
