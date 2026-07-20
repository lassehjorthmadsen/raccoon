#!/usr/bin/env python3
"""Cost projection + clean round-boundary stopping for round-based pipelines.

The exp008/exp009 DAgger pipelines overran their cost estimate because nothing
projected the total from the first round's measured cost, and nothing stopped a
run once it plateaued. This helper adds both, enforced *between* rounds so a stop
always leaves a complete, resumable checkpoint — never a half-trained orphan.

State lives under a --state directory (created on demand):

  budget.json    {"consumed_hours": float, "timed_rounds": int}
                 cumulative VM-hours and count of rounds actually run (skipped
                 resume rounds are not counted), so the average is accurate even
                 across watchdog relaunches.
  gnubg_history  lines "<round> <pooled_equity_ppg>", one per GNUBG eval; higher
                 (less negative) equity is better.

Subcommands:

  record-round --state DIR --seconds N
      Add one completed round's wall-time to the accumulator.

  record-eval --state DIR --round R --equity E
      Append a GNUBG pooled-equity data point (for plateau detection).

  check --state DIR --completed C --total T [--rate DKK_PER_HR]
        [--max-budget DKK] [--max-wall HOURS] [--patience N] [--min-delta PPG]
        [--calib-hours-per-round H]
      Print a one-line projection. If a stop condition is met, print a STOP
      line and exit 10; otherwise exit 0. Call this at the TOP of each round,
      before doing any expensive work, and stop the loop cleanly on exit 10.

All money is DKK; --rate is the effective spot price per VM-hour to assume for
projection (default deliberately conservative — spot discounts evaporate exactly
when a zone is busy, as this project learned the expensive way).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(state: Path) -> dict:
    f = state / "budget.json"
    if f.exists():
        return json.loads(f.read_text())
    return {"consumed_hours": 0.0, "timed_rounds": 0}


def _save(state: Path, data: dict) -> None:
    state.mkdir(parents=True, exist_ok=True)
    (state / "budget.json").write_text(json.dumps(data))


def _equities(state: Path) -> list[float]:
    f = state / "gnubg_history"
    if not f.exists():
        return []
    out = []
    for line in f.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            out.append(float(parts[1]))
    return out


def cmd_record_round(args) -> int:
    state = Path(args.state)
    data = _load(state)
    data["consumed_hours"] = round(data["consumed_hours"] + args.seconds / 3600.0, 4)
    data["timed_rounds"] = data["timed_rounds"] + 1
    _save(state, data)
    return 0


def cmd_record_eval(args) -> int:
    state = Path(args.state)
    state.mkdir(parents=True, exist_ok=True)
    with (state / "gnubg_history").open("a") as f:
        f.write(f"{args.round} {args.equity:.4f}\n")
    return 0


def _plateaued(eqs: list[float], patience: int, min_delta: float) -> bool:
    """True if the last `patience` evals never beat the prior best by min_delta.

    Needs at least one eval before the window to have a baseline, so plateau is
    never declared until patience+1 evals exist.
    """
    if patience <= 0 or len(eqs) < patience + 1:
        return False
    window = eqs[-patience:]
    before = eqs[:-patience]
    return max(window) <= max(before) + min_delta


def cmd_check(args) -> int:
    state = Path(args.state)
    data = _load(state)
    consumed = data["consumed_hours"]
    timed = data["timed_rounds"]
    rate = args.rate

    avg = consumed / timed if timed > 0 else args.calib_hours_per_round
    remaining = max(args.total - args.completed, 0)
    projected = consumed + avg * remaining

    projected_cost = projected * rate
    print(
        f"[budget] consumed {consumed:.1f}h (~kr.{consumed * rate:.0f}); "
        f"projecting {projected:.1f}h for all {args.total} rounds "
        f"(~kr.{projected_cost:.0f}) at kr.{rate:.1f}/h "
        f"[avg {avg:.1f}h/round over {timed} timed]",
        flush=True,
    )

    # Early heads-up: the FULL run is projected over budget. Non-blocking — the
    # run continues and stops cleanly at the cap (or earlier on plateau); this is
    # just the warning that was missing when exp009 quietly ran 2-4x its estimate.
    if args.max_budget > 0 and projected_cost > args.max_budget:
        short = args.total - args.completed
        print(
            f"[budget] NOTE: projected total ~kr.{projected_cost:.0f} exceeds "
            f"--max-budget kr.{args.max_budget:.0f}; expect a clean budget stop "
            f"before all {short} remaining rounds finish unless the score "
            f"plateaus first. Raise MAX_BUDGET_DKK or lower ROUNDS to change this.",
            flush=True,
        )

    # Enforce BEFORE the next round: would running one more round cross a cap?
    next_hours = consumed + avg
    next_cost = next_hours * rate

    if args.max_budget > 0 and next_cost > args.max_budget:
        print(
            f"[budget] STOP: next round would reach ~kr.{next_cost:.0f} > "
            f"--max-budget kr.{args.max_budget:.0f}. Stopping cleanly at a round "
            f"boundary; raise MAX_BUDGET_DKK and re-run to resume (completed "
            f"rounds are skipped).",
            flush=True,
        )
        return 10

    if args.max_wall > 0 and next_hours > args.max_wall:
        print(
            f"[budget] STOP: next round would reach ~{next_hours:.1f}h > "
            f"--max-wall {args.max_wall:.1f}h. Stopping cleanly.",
            flush=True,
        )
        return 10

    eqs = _equities(state)
    if _plateaued(eqs, args.patience, args.min_delta):
        best = max(eqs)
        recent = ", ".join(f"{e:+.2f}" for e in eqs[-args.patience:])
        print(
            f"[budget] STOP: no GNUBG improvement over best ({best:+.2f} ppg) "
            f"in the last {args.patience} evals [{recent}] (--min-delta "
            f"{args.min_delta}). Plateau reached; stopping cleanly.",
            flush=True,
        )
        return 10

    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record-round")
    r.add_argument("--state", required=True)
    r.add_argument("--seconds", type=float, required=True)
    r.set_defaults(func=cmd_record_round)

    e = sub.add_parser("record-eval")
    e.add_argument("--state", required=True)
    e.add_argument("--round", type=int, required=True)
    e.add_argument("--equity", type=float, required=True)
    e.set_defaults(func=cmd_record_eval)

    c = sub.add_parser("check")
    c.add_argument("--state", required=True)
    c.add_argument("--completed", type=int, required=True)
    c.add_argument("--total", type=int, required=True)
    c.add_argument("--rate", type=float, default=5.0)
    c.add_argument("--max-budget", type=float, default=0.0)
    c.add_argument("--max-wall", type=float, default=0.0)
    c.add_argument("--patience", type=int, default=0)
    c.add_argument("--min-delta", type=float, default=0.05)
    c.add_argument("--calib-hours-per-round", type=float, default=9.0)
    c.set_defaults(func=cmd_check)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
