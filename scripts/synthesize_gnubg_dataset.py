"""Build the v3 policy/value dataset by distilling GNUBG 4-ply analysis.

For each game in the bglab ``analyzed/`` exports we replay the moves through
OpenSpiel (reusing the proven ``bgmatch_replay`` board-signature matcher) and,
at every **non-doubles** decision, read GNUBG's labels straight off the file:

  - policy target (SOFT): for each candidate at the block's deepest ply,
    money equity = ``equity_from_wildbg(tuple)``; the soft target is
    ``softmax(money_equities / T)`` over those candidates' action indices,
    stored sorted by equity desc (col 0 = money-best).
  - value target: the money-best candidate's equity (position value under
    optimal money play, in [-1, 1]).

Doubles turns are *advanced* but not labeled: OpenSpiel splits a doubles play
into two decision nodes, while GNUBG analyses the whole 4-checker move with one
equity, so the first-half action is not uniquely determined. Non-doubles plays
are a single OpenSpiel action, so each candidate maps to exactly one index.

Verification: at every decision the replayed board is compared to the file's
GNU Position ID (decoded); any desync aborts that game (counted, never
silently mislabeled).

Output ``data/bglab/gnubg4ply_cache.npz``:
  observations  (N,17,2,12) f32
  value_targets (N,)        f32
  policy_actions(N,K)       i32  (padded -1)
  policy_probs  (N,K)       f32  (padded 0, valid entries sum to 1)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pyspiel

from raccoon.data.bganalyzed import (
    AnalyzedGame, best_candidates, parse_analyzed_file,
)
from raccoon.data.bgmatch_replay import (
    ReplayError,
    _apply_move_to_board,
    _board_signature,
    _chance_tables,
    _find_action_sequence_to_signature,
    _normalize_moves,
    _parse_moves_raw,
    _strip_action_index_prefix,
)
from raccoon.data.wildbg import decode_position_id, equity_from_wildbg
from raccoon.env.encoder import encode_state
from raccoon.env.game_wrapper import GameState

MAX_K = 6  # max candidates kept per soft target


def _advance_through_noops(state) -> None:
    """Apply empty (forfeit / unplayable-doubles-half) actions to reach chance."""
    while not state.is_chance_node() and not state.is_terminal():
        empties = [
            a for a in state.legal_actions()
            if _normalize_moves(_strip_action_index_prefix(state.action_to_string(a))) == ()
        ]
        if not empties:
            break
        state.apply_action(empties[0])


def _boards_match(pid: str, state) -> bool:
    """True if the decoded Position ID is the same board as the replayed state.

    GNUBG's exported Position ID for "X to play" is encoded from the
    perspective of the player who *just moved* (the opponent of the on-roll
    player), which relates to OpenSpiel's on-roll ``board_from_perspective``
    by a point-reversal + my/opp swap. The game-opening position is symmetric
    and matches directly. We accept either orientation — a genuinely
    misaligned board would match neither.
    """
    try:
        bf = decode_position_id(pid)
    except Exception:
        return False
    bs = GameState(state).board_from_perspective()
    direct = (
        np.array_equal(bf.my_points, bs.my_points)
        and np.array_equal(bf.opp_points, bs.opp_points)
        and bf.my_bar == bs.my_bar and bf.opp_bar == bs.opp_bar
        and bf.my_off == bs.my_off and bf.opp_off == bs.opp_off
    )
    if direct:
        return True
    return (
        np.array_equal(bf.my_points, bs.opp_points[::-1])
        and np.array_equal(bf.opp_points, bs.my_points[::-1])
        and bf.my_bar == bs.opp_bar and bf.opp_bar == bs.my_bar
        and bf.my_off == bs.opp_off and bf.opp_off == bs.my_off
    )


def _candidate_money_equity(probs) -> float:
    """Money equity in [-1,1] from a GNUBG (win,wG,wBG,lose,lG,lBG) tuple."""
    win, win_g, win_bg, _lose, lose_g, lose_bg = probs
    return equity_from_wildbg(win, win_g, win_bg, lose_g, lose_bg)


class _Stats:
    def __init__(self):
        self.games_ok = 0
        self.games_failed = 0
        self.decisions_emitted = 0
        self.doubles_skipped = 0
        self.doubles_value_only = 0
        self.nocand_skipped = 0
        self.posid_mismatch_games = 0
        self.cand_match_fail = 0


def _synthesize_game(
    game: AnalyzedGame, game_obj, tables, temperature: float, stats: _Stats,
    out_obs, out_actions, out_probs, out_values,
) -> None:
    """Replay one game, appending emitted examples to the output lists."""
    if not game.decisions:
        return
    first = game.decisions[0]
    first_player = 0 if first.player_name == game.player_x else 1
    key = (first_player, tuple(sorted(first.dice)))
    opening = tables["opening"].get(key)
    if opening is None:
        raise ReplayError(f"no opening outcome for {key}")

    state = game_obj.new_initial_state()
    state.apply_action(opening)

    for i, dec in enumerate(game.decisions):
        if i > 0:
            if not state.is_chance_node():
                raise ReplayError(f"decision {i}: expected chance node")
            mid = tables["midgame"].get(tuple(sorted(dec.dice)))
            if mid is None:
                raise ReplayError(f"decision {i}: no chance outcome for {dec.dice}")
            state.apply_action(mid)

        expected_player = 0 if dec.player_name == game.player_x else 1
        if state.current_player() != expected_player:
            raise ReplayError(
                f"decision {i}: player mismatch "
                f"(spiel {state.current_player()} vs file {expected_player})"
            )
        if dec.position_id is not None and not _boards_match(dec.position_id, state):
            stats.posid_mismatch_games += 1
            raise ReplayError(f"decision {i}: Position ID board mismatch")

        is_doubles = dec.dice[0] == dec.dice[1]
        cur_sig = _board_signature(state)
        me = state.current_player()

        # --- Emit a labeled example (non-doubles, has candidates) ---
        if not is_doubles:
            deep = best_candidates(dec)
            if not deep:
                stats.nocand_skipped += 1
            else:
                matched: list[tuple[int, float]] = []
                for c in deep:
                    moves = _parse_moves_raw(c.move_str)
                    tsig = _apply_move_to_board(cur_sig, me, moves)
                    if tsig is None:
                        continue
                    seq = _find_action_sequence_to_signature(state, tsig)
                    if seq is None or len(seq) != 1:
                        continue
                    matched.append((seq[0], _candidate_money_equity(c.probs)))
                if not matched:
                    stats.cand_match_fail += 1
                else:
                    matched.sort(key=lambda t: t[1], reverse=True)
                    matched = matched[:MAX_K]
                    actions = np.full(MAX_K, -1, dtype=np.int32)
                    probs = np.zeros(MAX_K, dtype=np.float32)
                    eqs = np.array([e for _, e in matched], dtype=np.float64)
                    w = np.exp((eqs - eqs.max()) / temperature)
                    w = w / w.sum()
                    for j, (a, _) in enumerate(matched):
                        actions[j] = a
                        probs[j] = w[j]
                    out_obs.append(
                        encode_state(GameState(state).board_from_perspective())
                    )
                    out_actions.append(actions)
                    out_probs.append(probs)
                    out_values.append(np.float32(matched[0][1]))
                    stats.decisions_emitted += 1
        else:
            # Doubles: OpenSpiel splits the play into two decision nodes while
            # GNUBG scores the whole 4-checker move with one equity, so the
            # first-half *policy* action is ambiguous. The *value* is not — the
            # best candidate's money equity is the position value — so emit a
            # value-only example (empty policy, zero probs => masked in the soft
            # CE) to give the value head doubles coverage (~18% of decisions).
            deep = best_candidates(dec)
            if not deep:
                stats.doubles_skipped += 1
            else:
                best_eq = max(_candidate_money_equity(c.probs) for c in deep)
                out_obs.append(
                    encode_state(GameState(state).board_from_perspective())
                )
                out_actions.append(np.full(MAX_K, -1, dtype=np.int32))
                out_probs.append(np.zeros(MAX_K, dtype=np.float32))
                out_values.append(np.float32(best_eq))
                stats.doubles_value_only += 1

        # --- Advance replay with the played move ---
        if dec.played_move_str is None:
            _advance_through_noops(state)
        else:
            pmoves = _parse_moves_raw(dec.played_move_str)
            ptsig = _apply_move_to_board(cur_sig, me, pmoves)
            pseq = _find_action_sequence_to_signature(state, ptsig) if ptsig else None
            if pseq is None:
                raise ReplayError(
                    f"decision {i}: cannot match played move "
                    f"{dec.played_move_str!r}"
                )
            for a in pseq:
                state.apply_action(a)
                _advance_through_noops(state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="data/bglab/gnubg4ply_cache.npz")
    parser.add_argument(
        "--analyzed-root", type=str, default="data/bglab/data-raw",
        help="Root holding lasse/analyzed and Llabba/analyzed.",
    )
    parser.add_argument(
        "--policy-temperature", type=float, default=0.02,
        help="Softmax temperature over candidate money equities (in [-1,1]). "
             "0.02 puts ~0.63 mean mass on the best move (sharp on clear-best "
             "positions, soft on near-ties).",
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=1000)
    args = parser.parse_args()

    root = Path(args.analyzed_root)
    # File list: all lasse/Llabba 4-ply + lasse 2-ply games not present at 4-ply.
    lasse4 = sorted((root / "lasse/analyzed/4-ply").glob("*.txt"))
    llabba4 = sorted((root / "Llabba/analyzed/4-ply").glob("*.txt"))
    have = {p.name for p in lasse4}
    lasse2_extra = [
        p for p in sorted((root / "lasse/analyzed/2-ply").glob("*.txt"))
        if p.name not in have
    ]
    files = lasse4 + llabba4 + lasse2_extra
    if args.max_files is not None:
        files = files[:args.max_files]
    print(f"Files: {len(lasse4)} lasse-4ply + {len(llabba4)} Llabba-4ply + "
          f"{len(lasse2_extra)} lasse-2ply-extra = {len(files)}", flush=True)

    game_obj = pyspiel.load_game("backgammon(scoring_type=full_scoring)")
    tables = _chance_tables()
    stats = _Stats()
    out_obs: list = []
    out_actions: list = []
    out_probs: list = []
    out_values: list = []
    t0 = time.time()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _save(final: bool) -> None:
        if not out_obs:
            return
        np.savez_compressed(
            out_path,
            observations=np.stack(out_obs).astype(np.float32),
            policy_actions=np.stack(out_actions).astype(np.int32),
            policy_probs=np.stack(out_probs).astype(np.float32),
            value_targets=np.array(out_values, dtype=np.float32),
            meta=np.array(json.dumps({
                "source": "gnubg-4ply-analysis",
                "analyzed_root": str(root),
                "policy_temperature": args.policy_temperature,
                "n_files": len(files),
                "games_ok": stats.games_ok,
                "games_failed": stats.games_failed,
                "decisions_emitted": stats.decisions_emitted,
                "doubles_skipped": stats.doubles_skipped,
                "doubles_value_only": stats.doubles_value_only,
                "posid_mismatch_games": stats.posid_mismatch_games,
                "cand_match_fail": stats.cand_match_fail,
                "wall_time_sec": round(time.time() - t0, 1),
                "hostname": platform.node(),
                "final": final,
            })),
        )

    for idx, f in enumerate(files):
        try:
            game = parse_analyzed_file(f)
        except Exception as e:
            stats.games_failed += 1
            if stats.games_failed <= 5:
                print(f"  parse error {f.name}: {e}", flush=True)
            continue
        try:
            _synthesize_game(
                game, game_obj, tables, args.policy_temperature, stats,
                out_obs, out_actions, out_probs, out_values,
            )
            stats.games_ok += 1
        except ReplayError:
            stats.games_failed += 1
        except Exception as e:
            stats.games_failed += 1
            if stats.games_failed <= 5:
                print(f"  unexpected error {f.name}: {type(e).__name__}: {e}", flush=True)

        if (idx + 1) % args.log_every == 0 or idx + 1 == len(files):
            el = time.time() - t0
            eta = (len(files) - idx - 1) / max(idx + 1, 1) * el / 60
            print(
                f"  [{idx+1:>5d}/{len(files)}] ok={stats.games_ok} "
                f"fail={stats.games_failed} emitted={stats.decisions_emitted:,} "
                f"dbl_skip={stats.doubles_skipped:,} posid_mm={stats.posid_mismatch_games} "
                f"el={el/60:.1f}min eta={eta:.0f}min", flush=True,
            )
        if (idx + 1) % args.save_every == 0:
            _save(final=False)

    _save(final=True)
    vals = np.array(out_values, dtype=np.float32) if out_values else np.array([0.0])
    best_mass = (
        float(np.mean([p[0] for p in out_probs])) if out_probs else 0.0
    )
    print(
        f"\nWrote {out_path}\n"
        f"  examples: {len(out_obs):,}\n"
        f"  games ok={stats.games_ok} fail={stats.games_failed} "
        f"(posid_mismatch={stats.posid_mismatch_games})\n"
        f"  doubles value-only={stats.doubles_value_only:,} "
        f"doubles skipped={stats.doubles_skipped:,} "
        f"nocand={stats.nocand_skipped:,} cand_match_fail={stats.cand_match_fail:,}\n"
        f"  value mean={vals.mean():.3f} std={vals.std():.3f} "
        f"min={vals.min():.3f} max={vals.max():.3f}\n"
        f"  mean prob mass on best move: {best_mass:.3f} "
        f"(temperature={args.policy_temperature})\n"
        f"  total {(time.time()-t0)/60:.1f} min", flush=True,
    )


if __name__ == "__main__":
    main()
