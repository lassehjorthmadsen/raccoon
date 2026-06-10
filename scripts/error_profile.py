"""Profile per-decision equity loss in logged GNUBG benchmark games.

Replays a ``gnubg_eval_*.json`` game log (written by ``scripts/eval_gnubg.py``
when game logging is enabled) move-by-move through OpenSpiel and scores every
recorded decision against a GNUBG oracle at ``--oracle-ply`` (default 0): the
loss is the oracle-equity difference between the oracle's best move and the
move actually played, from the mover's perspective.

The point is to turn the headline equity gap into a *profile*: how much of the
gap comes from doubles (where the supervised policy head has no training
signal), from contact vs race positions, and so on. GNUBG's own decisions act
as a calibration baseline — it played 2-ply in the logged games, so its
0-ply-measured loss should be near zero.

Usage:
    python3 scripts/error_profile.py \
        experiments/pretrain-gnubg-v4-10x256/logs/gnubg_eval_20260604_075512.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from raccoon.data.bgmatch_replay import _chance_tables
from raccoon.env.game_wrapper import GameState, GameWrapper
from raccoon.eval.gnubg_adapter import board_from_view, evaluate_equity
from raccoon.search.mcts import _advance_through_chance


def _our_equity_after(state: GameState, action: int, ply: int) -> float:
    """Oracle equity (mover's POV) of ``action``, mirroring the adapter.

    Doubles half-1 candidates are valued by their best half-2 continuation,
    exactly as ``gnubg_adapter._best_action_and_opp_equity`` does; terminal
    moves use the actual game return rather than a sentinel.
    """
    me = state.current_player()
    child = state.clone()
    child.apply_action(action)
    child = _advance_through_chance(child)
    if child.is_terminal():
        return child.returns()[me]
    if child.current_player() == me:
        return max(_our_equity_after(child, a, ply) for a in child.legal_actions())
    opp_view = child.board_from_perspective()
    return -evaluate_equity(board_from_view(opp_view), ply)


def _is_race(view) -> bool:
    """True if the two sides have fully passed each other (no contact)."""
    if view.my_bar or view.opp_bar:
        return False
    my_max = max((i + 1 for i in range(24) if view.my_points[i] > 0), default=0)
    opp_min = min((25 - (i + 1) for i in range(24) if view.opp_points[i] > 0), default=25)
    return my_max < opp_min


def profile_games(games: list[dict], oracle_ply: int, max_games: int | None = None):
    wrapper = GameWrapper()
    tables = _chance_tables()
    opening, midgame = tables["opening"], tables["midgame"]

    stats = defaultdict(lambda: {"loss": 0.0, "n": 0})
    worst: list[tuple[float, str]] = []
    forced = defaultdict(int)
    desynced = 0
    n_games = 0

    for gi, game in enumerate(games[:max_games] if max_games else games):
        state = wrapper.new_game()
        raccoon_is_p0 = game["raccoon_is_player0"]
        first_chance = True
        ok = True

        for mi, m in enumerate(game["moves"]):
            dice = tuple(sorted(m["dice"])) if m["dice"] else None
            while state.is_chance_node():
                key = (m["player"], dice) if first_chance else dice
                table = opening if first_chance else midgame
                if key not in table:
                    ok = False
                    break
                state.apply_action(table[key])
                first_chance = False
            if (not ok or state.is_terminal()
                    or state.current_player() != m["player"]
                    or m["action"] not in state.legal_actions()):
                ok = False
                break

            side = "raccoon" if (m["player"] == 0) == raccoon_is_p0 else "gnubg"
            legal = state.legal_actions()
            if len(legal) == 1:
                forced[side] += 1
            else:
                eqs = {a: _our_equity_after(state, a, oracle_ply) for a in legal}
                loss = max(eqs.values()) - eqs[m["action"]]
                view = state.board_from_perspective()
                kind = "doubles" if m["dice"][0] == m["dice"][1] else "non-doubles"
                phase = "race" if _is_race(view) else "contact"
                for bucket in ("total", kind, phase):
                    s = stats[(side, bucket)]
                    s["loss"] += loss
                    s["n"] += 1
                if side == "raccoon":
                    worst.append((loss, f"game {gi} move {mi}: {m['action_str']} "
                                        f"(dice {m['dice']}, {kind}, {phase}, -{loss:.3f})"))

            state.apply_action(m["action"])

        if ok:
            n_games += 1
        else:
            desynced += 1
        if (gi + 1) % 10 == 0:
            print(f"  ... {gi + 1} games replayed")

    return stats, worst, forced, desynced, n_games


def main():
    parser = argparse.ArgumentParser(description="Per-decision equity-loss profile")
    parser.add_argument("game_log", help="Path to a gnubg_eval_*.json game log")
    parser.add_argument("--oracle-ply", type=int, default=0,
                        help="GNUBG eval ply for the loss oracle (default 0)")
    parser.add_argument("--max-games", type=int, default=None)
    args = parser.parse_args()

    payload = json.loads(Path(args.game_log).read_text())
    games = payload["games"]
    print(f"Loaded {len(games)} games from {args.game_log}")
    print(f"Oracle: GNUBG {args.oracle_ply}-ply  (losses are lower bounds on true error)")

    stats, worst, forced, desynced, n_games = profile_games(
        games, args.oracle_ply, args.max_games)

    print(f"\nReplayed {n_games} games cleanly ({desynced} desynced and skipped)\n")
    print(f"{'side':<8} {'bucket':<12} {'decisions':>9} {'mean loss':>10} "
          f"{'loss/game':>10} {'share':>7}")
    for side in ("raccoon", "gnubg"):
        total_loss = stats[(side, "total")]["loss"] or 1e-9
        for bucket in ("total", "non-doubles", "doubles", "contact", "race"):
            s = stats[(side, bucket)]
            if s["n"] == 0:
                continue
            print(f"{side:<8} {bucket:<12} {s['n']:>9} {s['loss'] / s['n']:>10.4f} "
                  f"{s['loss'] / n_games:>10.3f} {s['loss'] / total_loss:>6.0%}")
        print(f"{side:<8} {'(forced)':<12} {forced[side]:>9}")

    print("\nWorst Raccoon decisions:")
    for loss, desc in sorted(worst, reverse=True)[:10]:
        print(f"  {desc}")


if __name__ == "__main__":
    main()
