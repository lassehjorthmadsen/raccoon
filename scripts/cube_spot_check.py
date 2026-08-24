#!/usr/bin/env python3
"""Worst-fitting centred-cube positions, as a spot-check sheet for GNU Backgammon.

The measured claim is that Janowski's *centred* cubeful-equity formula misses
rolled-out reality by up to ~0.23 points. That is a large enough error that it
ought to have been noticed, so it deserves independent checking against a
shipping engine rather than trusting this repository's own arithmetic.

That check cannot be automated here: the ``gnubg_nn`` binding refuses money play
outright (``evaluate_cube_decision`` raises "Not implemented for money"), and its
``cubeful_rollout`` returns undocumented fields. So this script emits the
positions with GNU Position IDs, for pasting into GNU Backgammon's own GUI where
money play works.

**The prediction being tested.** Janowski's closed form is a *leaf* evaluator;
GNU Backgammon applies it at the bottom of a cubeful search. So:

* at **0-ply**, GNUBG's cubeful equity should land near the Janowski column, and
  therefore far from the rollout column;
* at **2-ply**, it should move most of the way toward the rollout column.

If 0-ply does *not* match Janowski, then GNUBG's 0-ply cubeful evaluation is
doing something more than the closed form, and this page's framing needs
revising. If 2-ply does not close the gap, the error survives into shipped play,
which would be the more surprising outcome.
"""

import argparse
import gzip
import json

import gnubg_nn

from raccoon.cube import janowski as J
from raccoon.eval.cube_benchmark import DEFAULT_BENCHMARK


def flip26(board26: list[int]) -> list[int]:
    """A BGSage 26-array from the other side's point of view."""
    out = [0] * 26
    out[0] = board26[25]
    out[25] = board26[0]
    for k in range(1, 25):
        out[k] = -board26[25 - k]
    return out


def slots(board26: list[int]):
    """26-array (its own side's POV, that side on roll) -> gnubg [opp, me]."""
    me = [max(0, board26[k]) for k in range(1, 25)] + [board26[25]]
    opp = [max(0, -board26[25 - k]) for k in range(1, 25)] + [board26[0]]
    return [opp, me]


def pips(side: list[int]) -> int:
    return sum((i + 1) * side[i] for i in range(24)) + 25 * side[24]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    ap.add_argument("--n", type=int, default=3, help="examples per direction")
    ap.add_argument("--output", help="directory for cube_spot_check.json")
    args = ap.parse_args()

    with gzip.open(args.benchmark, "rt") as f:
        data = json.load(f)

    scored = []
    for e in data["decisions"]:
        if e["kind"] != "checker" or e["cube_owner"] != "centered":
            continue
        for m in e["moves"]:
            if m["eval_level"] != "Rollout":
                continue
            model = J.cl2cf_money(m["probs"], J.CENTERED, J.X_CONTACT, True)
            scored.append((model - m["equity"], e, m, model))
    scored.sort(key=lambda r: -r[0])

    groups = [("Janowski too OPTIMISTIC (model above rollout)", scored[:args.n]),
              ("Janowski too PESSIMISTIC (model below rollout)", scored[-args.n:])]

    print("Money play, cube CENTRED at 1, Jacoby on, beavers on.")
    print("After the play shown, the OPPONENT is on roll -- the Position ID below")
    print("is written from that on-roll player's side, so equities in GNU")
    print("Backgammon will appear with the sign of the 'on roll' column.\n")

    for title, rows in groups:
        print(f"=== {title} ===\n")
        for resid, e, m, model in rows:
            on_roll = flip26(m["board"])
            b = slots(on_roll)
            pid = gnubg_nn.position_id(b)
            print(f"  Position ID: {pid}")
            print(f"    game plan {e['game_plan']}, pips {pips(b[1])} (on roll) "
                  f"vs {pips(b[0])}")
            print(f"    cubeless: P(win)={m['probs'][0]:.4f} for the player who "
                  f"just moved; gammons {m['probs'][1]:.3f} won / "
                  f"{m['probs'][3]:.3f} lost")
            se = f", rollout SE {m['std_error']:.4f}" if "std_error" in m else ""
            print(f"    mover's side  — rollout {m['equity']:+.4f}{se}   "
                  f"Janowski {model:+.4f}   miss {resid:+.4f}")
            print(f"    on-roll side  — rollout {-m['equity']:+.4f}   "
                  f"Janowski {-model:+.4f}")
            print()

    if args.output:
        import os
        os.makedirs(args.output, exist_ok=True)
        payload = {"note": ("worst-fitting centred-cube positions, for independent "
                            "checking in GNU Backgammon's own money play"),
                   "groups": []}
        for title, rows in groups:
            payload["groups"].append({
                "title": title,
                "positions": [{
                    "position_id": gnubg_nn.position_id(slots(flip26(m["board"]))),
                    "game_plan": e["game_plan"],
                    "p_win_mover": m["probs"][0],
                    "gammon_win_mover": m["probs"][1],
                    "gammon_loss_mover": m["probs"][3],
                    "rollout_mover": m["equity"],
                    "janowski_mover": model,
                    "miss": resid,
                } for resid, e, m, model in rows],
            })
        path = os.path.join(args.output, "cube_spot_check.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  Saved: {path}\n")

    print("What to compare in GNU Backgammon:")
    print("  Settings > Options: money play, Jacoby on, beavers on, cube centred.")
    print("  Evaluate the position at 0-ply and at 2-ply, and read the cubeful")
    print("  'no double' equity. Compare against the two columns above.")


if __name__ == "__main__":
    main()
