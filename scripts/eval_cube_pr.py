#!/usr/bin/env python3
"""Score cube decisions on the BGSage money benchmark (exp024).

The sibling of ``eval_benchmark_pr.py``: that script scores checker play against
the benchmark's cubeless references, this one scores *cube* decisions against its
cubeful ones. Both report PR -- average equity thrown away per decision, times
500 -- so the two halves of a bot's error budget are on the same scale.

A cube model here is Janowski's, parameterised by where the cube life index
applies. The variants below vary that parameterisation; everything else -- the
probabilities fed in, the scoring formulas, the decisions scored -- is held
fixed, so a difference between two variants is a difference in cube modelling.

Two stages, selected by ``--probs``:

* ``reference`` isolates the cube model. The benchmark's own reference
  probabilities go in, so the network's evaluation error is out of the picture
  entirely and the variants differ only in how they convert probabilities to a cube
  action. This is what exp024's hypothesis is tested on.
* ``checkpoint`` / ``onnx`` / ``gnubg`` put a real engine's probabilities in, and
  give the number that engine would actually score.

Examples::

    python3 scripts/eval_cube_pr.py --probs reference --variants all \\
        --output experiments/exp024-cube/results

    python3 scripts/eval_cube_pr.py --probs checkpoint \\
        --checkpoint experiments/exp018-distill/checkpoints/ep22.pt \\
        --variants baseline --engine-label "ep22 x=0.68" \\
        --output experiments/exp024-cube/results
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_benchmark_pr import (  # noqa: E402
    check_no_clobber, flipped_to_board_view, flipped_to_gnubg, sanitize_label,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raccoon.cube import janowski as J  # noqa: E402
from raccoon.eval.cube_benchmark import (  # noqa: E402
    DEFAULT_BENCHMARK, TIERS, aggregate, load_cube_decisions, measured_only,
    ordering_violations, paired_delta, score_cube_decisions, split_by_game,
)

# BGSage's game-plan labels that mean "no contact left worth speaking of".
RACE_PLANS = ("purerace", "racing")

# What makes two cube results the same measurement. Re-scoring the same variant on
# the same decisions with the same probabilities is idempotent; anything else
# under the same label is a mistake worth refusing.
IDENTITY_FIELDS = ("n", "variant", "probs_source", "references")

# Grids for the fitted variants. The index grid is fine enough that its rounding is
# far below the resolution of the metric; the offset grid spans well past the
# residual we are trying to absorb.
X_GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.005), 4)
OFFSET_GRID = np.round(np.arange(-0.20, 0.20 + 1e-9, 0.002), 4)


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CubeVariant:
    """A cube model: which life index applies where, plus an optional shift.

    Janowski's basic model is a single index everywhere. The extra knobs exist
    because the exploratory pass found the model fits the take/pass line well at
    the published 0.68 while missing the centred no-double line badly, and the
    experiment needs to say whether that is a mis-set constant or a wrong shape.
    """

    label: str
    x_nd_centered: float = J.X_CONTACT
    x_nd_player: float = J.X_CONTACT
    x_dt: float = J.X_CONTACT
    nd_offset: float = 0.0
    x_race: float | None = None      # replaces all three in race positions

    def indices_for(self, entry: dict) -> tuple[float, float]:
        """(index for no-double, index for double/take) for one position."""
        if self.x_race is not None and entry.get("game_plan") in RACE_PLANS:
            return self.x_race, self.x_race
        x_nd = (self.x_nd_centered if entry["cube_owner"] == J.CENTERED
                else self.x_nd_player)
        return x_nd, self.x_dt

    def all_indices_for(self, entry: dict) -> tuple[float, float, float]:
        """The indices this variant would use for (E_O, E_C, E_U) at one position.

        Scoring only ever needs two of the three, because a position has one
        cube location. The coherence audit needs all three for the *same*
        position, which is exactly the comparison that lets a per-state fit go
        incoherent unnoticed.
        """
        if self.x_race is not None and entry.get("game_plan") in RACE_PLANS:
            return self.x_race, self.x_race, self.x_race
        return self.x_nd_player, self.x_nd_centered, self.x_dt

    def as_dict(self) -> dict:
        return asdict(self)


# The fields a fitted variant is allowed to move, per variant family.
FIT_FIELDS = {
    "refit_global": ("x_all",),
    "refit_per_state": ("x_nd_centered", "x_nd_player", "x_dt"),
    "offset": ("nd_offset",),
}


def base_variants() -> dict[str, CubeVariant]:
    """The six pre-registered variants, before any fitting."""
    return {
        # 1. GNU Backgammon's published contact constant.
        "baseline": CubeVariant("x=0.68 (published)"),
        # 2. GNUBG's documented contact/race split.
        "race_split": CubeVariant("x=0.68 contact / 0.60 race", x_race=J.X_RACE),
        # 3-4. Fitted variants; the values here are only starting points.
        "refit_global": CubeVariant("refit global x"),
        "refit_per_state": CubeVariant("refit x per cube state"),
        # 5. The envelope Janowski's model interpolates between.
        "dead": CubeVariant("x=0 (dead cube)", 0.0, 0.0, 0.0),
        "live": CubeVariant("x=1 (live cube)", 1.0, 1.0, 1.0),
        # 6. Is the residual just a flat shift the index could have absorbed?
        "offset": CubeVariant("x=0.68 + fitted ND offset"),
    }


def custom_variant(args) -> CubeVariant | None:
    """Build a fixed-parameter variant from the --x-* overrides, if any were given.

    A fixed variant is never fitted, so it carries exactly the model it was handed.
    """
    overrides = {
        "x_nd_centered": args.x_nd_centered,
        "x_nd_player": args.x_nd_player,
        "x_dt": args.x_dt,
        "nd_offset": args.nd_offset,
        "x_race": args.x_race,
    }
    given = {k: v for k, v in overrides.items() if v is not None}
    if not given:
        return None
    described = " ".join(f"{k}={v}" for k, v in given.items())
    return CubeVariant(label=f"{args.custom_name} ({described})", **given)


def make_decider(variant: CubeVariant, probs_by_key: dict):
    """A cube policy: entry -> (should_double, should_take)."""
    def decide(entry: dict) -> tuple[bool, bool]:
        x_nd, x_dt = variant.indices_for(entry)
        return J.cube_action(
            probs_by_key[entry["key"]], entry["cube_owner"], x_nd,
            jacoby=True, nd_offset=variant.nd_offset, x_dt=x_dt,
        )
    return decide


def coherence(variant: CubeVariant, entries, probs_by_key) -> dict:
    """Does this variant respect E_O >= E_C >= E_U on each position?"""
    def equities_for(entry):
        x_o, x_c, x_u = variant.all_indices_for(entry)
        p = probs_by_key[entry["key"]]
        return (J.cl2cf_money(p, J.PLAYER, x_o, False) + variant.nd_offset,
                J.cl2cf_money(p, J.CENTERED, x_c, True) + variant.nd_offset,
                J.cl2cf_money(p, J.OPPONENT, x_u, False))
    return ordering_violations(entries, equities_for)


def variant_pr(variant: CubeVariant, entries, probs_by_key) -> float:
    """Cube PR for one variant over one set of entries."""
    decisions = score_cube_decisions(entries, make_decider(variant, probs_by_key))
    return float(np.mean([d.error for d in decisions]) * 500.0)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_variant(variant: CubeVariant, family: str, entries, probs_by_key,
            rounds: int = 4, verbose: bool = True) -> CubeVariant:
    """Fit an variant's free parameters by minimising cube PR on ``entries``.

    Fitting on the metric we report, rather than on equity RMSE, is deliberate.
    Only the no-double / double-take gap *near the decision boundary* moves a
    decision, so an index that fits the equities better can easily choose worse;
    selecting on one estimator and reporting another is a bias this project has
    been bitten by before. PR is piecewise constant in the parameters, so the
    search is a grid rather than a gradient, and multi-parameter variants use
    coordinate descent -- with the caveat that this finds a coordinate-wise
    minimum, not a certified global one.
    """
    fields = FIT_FIELDS[family]
    best = variant
    best_pr = variant_pr(best, entries, probs_by_key)

    for round_i in range(rounds):
        improved = False
        for field in fields:
            grid = OFFSET_GRID if field == "nd_offset" else X_GRID
            for value in grid:
                if field == "x_all":
                    cand = replace(best, x_nd_centered=float(value),
                                   x_nd_player=float(value), x_dt=float(value))
                else:
                    cand = replace(best, **{field: float(value)})
                pr = variant_pr(cand, entries, probs_by_key)
                if pr < best_pr - 1e-12:
                    best, best_pr, improved = cand, pr, True
        if verbose:
            print(f"    round {round_i + 1}: fit PR {best_pr:.4f}")
        if not improved:
            break
    return best


# ---------------------------------------------------------------------------
# Probability sources
# ---------------------------------------------------------------------------

def reference_probs(entries) -> dict:
    """The benchmark's own reference probabilities, keyed by decision."""
    return {e["key"]: tuple(e["probs"]) for e in entries}


def gnubg_probs(entries, ply: int) -> dict:
    """GNUBG's own evaluation at ``ply``, in the cumulative 5-vector it returns."""
    from raccoon.eval.gnubg_adapter import outcome_probs

    out = {}
    for i, e in enumerate(entries):
        # A cube decision is pre-roll and the board is already the on-roll
        # player's view, so no perspective flip is needed here -- unlike the
        # checker path, where the board is a post-move position.
        out[e["key"]] = tuple(outcome_probs(flipped_to_gnubg(e["board"]), ply=ply))
        if (i + 1) % 500 == 0:
            print(f"    gnubg {i + 1}/{len(entries)}")
    return out


def network_probs(entries, checkpoint: str | None, onnx: str | None,
                  device_str: str = "cpu", batch_size: int = 512) -> dict:
    """Raccoon's six-outcome head, converted to the cumulative 5-vector."""
    import torch
    from raccoon.env.encoder import channels_for_network, encode_state
    from raccoon.model.network import load_model

    torch.set_flush_denormal(True)  # iMac CPU: avoid the denormal slowdown
    if onnx is not None:
        return _onnx_probs(entries, onnx, batch_size)

    device = torch.device(device_str)
    net = load_model(checkpoint).to(device)
    net.eval()
    if net.value_head != "outcomes6":
        raise SystemExit(
            f"{checkpoint} has a '{net.value_head}' value head; the cube needs "
            "the six-outcome distribution, so only an outcomes6 net can be scored")
    channels = channels_for_network(net.config)

    obs = np.stack([encode_state(flipped_to_board_view(e["board"]),
                                 channels=channels) for e in entries])
    out = {}
    with torch.no_grad():
        for start in range(0, len(entries), batch_size):
            batch = torch.from_numpy(obs[start:start + batch_size]).to(device)
            probs6 = net.value_probs6(batch).cpu().numpy()
            for e, p6 in zip(entries[start:start + batch_size], probs6):
                out[e["key"]] = J.probs6_to_cumulative5([float(v) for v in p6])
    return out


def _onnx_probs(entries, onnx_path: str, batch_size: int) -> dict:
    """Same, from the exported graph the browser engine actually downloads."""
    import onnxruntime as ort
    import re
    from raccoon.env.encoder import encode_state

    stem = re.sub(r"-(fp32|int8)$", "", Path(onnx_path).stem)
    with open(Path(onnx_path).with_name(stem + ".meta.json")) as f:
        meta = json.load(f)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    obs = np.stack([encode_state(flipped_to_board_view(e["board"]),
                                 channels=list(meta["channels"]))
                    for e in entries]).astype(np.float32)
    out = {}
    for start in range(0, len(entries), batch_size):
        probs6 = sess.run(["probs"], {"obs": obs[start:start + batch_size]})[0]
        for e, p6 in zip(entries[start:start + batch_size], probs6):
            out[e["key"]] = J.probs6_to_cumulative5([float(v) for v in p6])
    return out


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def equity_fit(variant: CubeVariant, entries, probs_by_key) -> dict:
    """Bias and RMSE of the variant's ND and DT equities against the references.

    Supporting only. The rollout tier is the least circular of the three -- its
    references are observed rollout outcomes rather than a deeper application of
    the very model being tested -- so it is the one to read for accuracy, even
    though PR must be pooled across all tiers.
    """
    out = {}
    for tier in TIERS + ["all"]:
        sel = [e for e in entries if tier == "all" or e["tier"] == tier]
        if not sel:
            continue
        nd_res, dt_res = [], []
        for e in sel:
            x_nd, x_dt = variant.indices_for(e)
            eq = J.cube_equities(probs_by_key[e["key"]], e["cube_owner"], x_nd,
                                 jacoby=True, nd_offset=variant.nd_offset, x_dt=x_dt)
            nd_res.append(eq.nd - e["equity_nd"])
            dt_res.append(eq.dt - e["equity_dt"])
        nd_res, dt_res = np.array(nd_res), np.array(dt_res)
        out[tier] = {
            "n": len(sel),
            "nd_bias": float(nd_res.mean()),
            "nd_rmse": float(np.sqrt((nd_res ** 2).mean())),
            "dt_bias": float(dt_res.mean()),
            "dt_rmse": float(np.sqrt((dt_res ** 2).mean())),
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    p.add_argument("--references", default="measured",
                   choices=["measured", "all"],
                   help="'measured' keeps only entries whose reference equities "
                        "were rolled out; 'all' also includes the ones a cubeful "
                        "search produced using Janowski at its leaves, which is "
                        "close to circular and reported for contrast only")
    p.add_argument("--probs", default="reference",
                   choices=["reference", "checkpoint", "onnx", "gnubg"],
                   help="where the cubeless probabilities come from")
    p.add_argument("--checkpoint", help="torch checkpoint for --probs checkpoint")
    p.add_argument("--onnx", help="exported ONNX for --probs onnx")
    p.add_argument("--gnubg-ply", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--variants", default="all",
                   help="comma-separated variant names, or 'all'")
    # Explicit index overrides build one extra variant with fixed parameters, which
    # is how a model selected in one run gets applied unchanged in the next --
    # re-fitting per engine would confound a cube-model comparison with the
    # engine's own evaluation error.
    p.add_argument("--custom-name", default="custom",
                   help="name for the variant built from the --x-* overrides")
    p.add_argument("--x-nd-centered", type=float)
    p.add_argument("--x-nd-player", type=float)
    p.add_argument("--x-dt", type=float)
    p.add_argument("--nd-offset", type=float)
    p.add_argument("--x-race", type=float)
    p.add_argument("--engine-label", default=None,
                   help="prefix for result labels (default: the probs source)")
    p.add_argument("--fit-split", type=float, default=0.5,
                   help="fraction of games used to fit the fitted variants")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--output", help="directory for per-variant JSON + summary.json")
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()

    entries, meta = load_cube_decisions(args.benchmark)
    n_all = len(entries)
    if args.references == "measured":
        entries = measured_only(entries)
        print(f"References: measured only -- {len(entries)} of {n_all} positions "
              f"have rolled-out reference equities")
    else:
        print(f"References: ALL {n_all} positions (includes search-derived "
              f"references; near-circular, for contrast only)")
    keys = [e["key"] for e in entries]
    if len(set(keys)) != len(keys):
        raise SystemExit("benchmark cube entries are not uniquely keyed")
    print(f"Loaded {len(entries)} cube positions "
          f"({meta['n_cube_double']} doubler + {meta['n_cube_take']} receiver "
          f"sub-decisions)")

    print(f"Probabilities: {args.probs}")
    if args.probs == "reference":
        probs_by_key = reference_probs(entries)
    elif args.probs == "gnubg":
        probs_by_key = gnubg_probs(entries, args.gnubg_ply)
    else:
        probs_by_key = network_probs(entries, args.checkpoint, args.onnx,
                                     args.device)

    all_variants = base_variants()
    custom = custom_variant(args)
    if custom is not None:
        all_variants[args.custom_name] = custom
        print(f"Custom variant {args.custom_name!r}: {custom.as_dict()}")
    wanted = list(base_variants()) if args.variants == "all" else args.variants.split(",")
    unknown = [a for a in wanted if a not in all_variants]
    if unknown:
        raise SystemExit(f"unknown variant(s): {unknown}. Known: {list(all_variants)}")

    fit_entries, confirm_entries = split_by_game(entries, args.fit_split,
                                                 args.split_seed)
    print(f"Fit/confirm split by game: {len(fit_entries)} / "
          f"{len(confirm_entries)} positions")

    baseline_decisions = score_cube_decisions(
        entries, make_decider(all_variants["baseline"], probs_by_key))
    prefix = args.engine_label or args.probs

    results = []
    for name in wanted:
        variant = all_variants[name]
        print(f"\n{name}: {variant.label}")

        fit_info = None
        if name in FIT_FIELDS and name in base_variants():
            print("  fitting on the fit split (minimising cube PR)...")
            variant = fit_variant(variant, name, fit_entries, probs_by_key)
            fit_info = {
                "family": name,
                "fields": list(FIT_FIELDS[name]),
                "n_fit_positions": len(fit_entries),
                "split_seed": args.split_seed,
                "fit_split": args.fit_split,
                "fit_pr": variant_pr(variant, fit_entries, probs_by_key),
            }
            print(f"  fitted: {variant.as_dict()}")

        decisions = score_cube_decisions(entries, make_decider(variant, probs_by_key))
        result = aggregate(decisions, entries)
        result["engine_label"] = f"{prefix} :: {name}"
        result["variant_name"] = name
        result["variant"] = variant.as_dict()
        result["probs_source"] = args.probs
        result["references"] = args.references
        result["checkpoint"] = args.checkpoint or args.onnx
        result["fit"] = fit_info
        result["equity_fit"] = equity_fit(variant, entries, probs_by_key)
        result["coherence"] = coherence(variant, entries, probs_by_key)
        # The per-sub-decision errors, in scoring order. Kept so that any later
        # paired comparison -- including across probability sources, which a
        # single run cannot do -- works off the committed results rather than
        # needing every variant re-run.
        result["errors"] = [round(d.error, 6) for d in decisions]
        result["roles"] = [d.role for d in decisions]
        # Which of those sub-decisions fall in the held-out half, so an
        # out-of-sample comparison between two variants can be made from the
        # committed results without re-deriving the split from the benchmark
        # archive (which is not in git).
        confirm_keys = {e["key"] for e in confirm_entries}
        result["in_confirm"] = [
            int(entries[d.entry_index]["key"] in confirm_keys) for d in decisions
        ]
        result["paired_vs_baseline"] = paired_delta(baseline_decisions, decisions)

        # The held-out half: for a fitted variant this is the number that counts,
        # since its full-n figure has seen its own training data.
        confirm_decisions = score_cube_decisions(
            confirm_entries, make_decider(variant, probs_by_key))
        confirm_baseline = score_cube_decisions(
            confirm_entries, make_decider(all_variants["baseline"], probs_by_key))
        result["confirm"] = aggregate(confirm_decisions, confirm_entries)
        result["confirm"]["paired_vs_baseline"] = paired_delta(
            confirm_baseline, confirm_decisions)

        print(f"  full n={result['n']}  cube PR = {result['pr']:.3f} "
              f"+- {result['pr_ci95']:.3f}")
        print(f"  paired vs baseline: {result['paired_vs_baseline']['delta_pr']:+.3f} "
              f"+- {result['paired_vs_baseline']['delta_ci95']:.3f} "
              f"({result['paired_vs_baseline']['decisions_changed']} decisions changed)")
        print(f"  held out n={result['confirm']['n']}  cube PR = "
              f"{result['confirm']['pr']:.3f} +- {result['confirm']['pr_ci95']:.3f}")
        coh = result["coherence"]
        verdict = "ok" if coh["coherent"] else "INCOHERENT - disqualified"
        print(f"  ordering E_O>=E_C>=E_U: {100 * coh['violation_rate']:.1f}% "
              f"violations ({coh['centred_above_owned']} centred-above-owned, "
              f"{coh['opponent_above_centred']} opponent-above-centred) -> {verdict}")
        results.append(result)

    if args.output:
        save(results, args.output, args.overwrite)


def save(results: list[dict], output_dir: str, overwrite: bool) -> None:
    os.makedirs(output_dir, exist_ok=True)
    paths = [(r, os.path.join(output_dir,
                              sanitize_label(r["engine_label"]) + ".json"))
             for r in results]
    if not overwrite:
        # Check everything before writing anything, so a clash cannot leave the
        # directory half-updated.
        for r, path in paths:
            check_no_clobber(r, path, fields=IDENTITY_FIELDS)
    for r, path in paths:
        with open(path, "w") as f:
            json.dump(r, f, indent=2)
        print(f"  Saved: {path}")

    # Rebuild the summary from every result file in the directory, not just the
    # ones this invocation wrote. Stage A and stage B are separate runs into the
    # same directory, and a summary showing only the last of them would be worse
    # than none at all.
    on_disk = []
    for path in sorted(Path(output_dir).glob("*.json")):
        if path.name == "summary.json":
            continue
        with open(path) as f:
            candidate = json.load(f)
        # The directory also holds measurements that are not cube variants (the
        # cube-blind checker floor, for one), so key off the variant marker rather
        # than assuming every JSON here has this shape.
        if "variant_name" in candidate:
            on_disk.append(candidate)

    summary = {
        "benchmark": "BGSage Money Benchmark (cube decisions, cubeful)",
        # No single n: the directory can hold both the measured-reference results
        # and the search-derived contrast, which score different decision sets.
        "results": [
            {
                "engine": r["engine_label"],
                "references": r.get("references", "unknown"),
                "n": r["n"],
                "variant": r["variant"],
                "pr": r["pr"],
                "pr_ci95": r["pr_ci95"],
                "paired_vs_baseline": r["paired_vs_baseline"],
                "confirm_pr": r["confirm"]["pr"],
                "confirm_paired": r["confirm"]["paired_vs_baseline"],
                "coherent": r["coherence"]["coherent"],
                "ordering_violation_rate": r["coherence"]["violation_rate"],
            }
            for r in on_disk
        ],
    }
    path = os.path.join(output_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
