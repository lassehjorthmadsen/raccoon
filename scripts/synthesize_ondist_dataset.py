"""Build an ON-DISTRIBUTION GNUBG policy/value cache (exp008 expert iteration).

Where ``scripts/synthesize_gnubg_dataset.py`` distils *pre-computed* GNUBG 4-ply
analysis of the human-match archive (off-distribution), this script generates
positions by **playing the current network** (DAgger / expert iteration) and
labels each decision with the live ``gnubg-nn`` oracle at ``--ply``. This fills
the on-distribution coverage gap behind the −1.59 ppg plateau while keeping the
targets purely supervised (continuous GNUBG money equity + GNUBG soft policy),
so the exp007 self-play "regression trap" cannot recur.

At every decision the move that *advances* the game is sampled from the net's
own policy (with temperature for variety) — so the state distribution is the
learner's — but the *labels* come from GNUBG:

  - non-doubles: SOFT policy = ``softmax(candidate_money_equities / T)`` over the
    top-K candidate actions (money equity in [-1, 1]); value = the money-best
    candidate's equity (the position's value under best play, in [-1, 1]).
  - doubles: value-only by default (OpenSpiel splits the play into two decision
    nodes; matches the v5 ``_dbl`` cache and trains the mid-doubles channel).
    ``--doubles-policy`` additionally emits a per-half-node policy label — which
    the offline 4-ply archive could not (it scored the whole 4-checker move with
    one equity), so this is the only path to doubles *policy* recovery.

Game outcomes are never used (all targets come from the oracle), so games may be
truncated at any time — the run stops at ``--max-decisions`` / ``--max-minutes``.

Output npz (identical schema to the soft cache, consumed unchanged by
``scripts/pretrain_policy.py``):
  observations  (N,26,2,12) f32
  value_targets (N,)        f32  in [-1, 1]
  policy_actions(N,K)       i32  (padded -1; an all-(-1) row is value-only)
  policy_probs  (N,K)       f32  (padded 0, valid rows sum to 1)
"""

from __future__ import annotations

import argparse
import json
import os

# Idle OpenMP threads sleep instead of busy-spinning (see docs/.. OMP note); set
# before torch is imported (transitively, via the raccoon imports below).
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import platform
import time
from pathlib import Path

import numpy as np

from raccoon.env.encoder import NUM_CHANNELS, channels_for_network, encode_state
from raccoon.env.game_wrapper import GameWrapper
from raccoon.eval.gnubg_adapter import candidate_equities
from raccoon.model.network import load_model
from raccoon.search.mcts import _advance_through_chance, select_action

MAX_K = 6  # max candidates kept per soft target (matches synthesize_gnubg_dataset)


class _Stats:
    def __init__(self):
        self.games = 0
        self.decisions_emitted = 0
        self.doubles_value_only = 0
        self.doubles_policy_emitted = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net", type=str, required=True,
                        help="Checkpoint whose policy rolls out the games (round r-1 net).")
    parser.add_argument("--out", type=str, required=True, help="Output npz path.")
    parser.add_argument("--ply", type=int, default=2,
                        help="GNUBG search depth for the labels (2 = 'world').")
    parser.add_argument("--max-decisions", type=int, default=None,
                        help="Stop after emitting this many labeled examples.")
    parser.add_argument("--max-minutes", type=float, default=None,
                        help="Stop after this many wall-clock minutes.")
    parser.add_argument("--max-games", type=int, default=None,
                        help="Stop after this many games (rarely the binding cap).")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Rollout sampling temperature for the net's move.")
    parser.add_argument("--temp-threshold", type=int, default=30,
                        help="Use greedy (temp=0) rollout moves after this ply.")
    parser.add_argument("--policy-temperature", type=float, default=0.02,
                        help="Softmax temperature over candidate money equities "
                             "(in [-1,1]); 0.02 puts ~0.63 mass on the best move.")
    parser.add_argument("--doubles-policy", action="store_true",
                        help="Also emit policy labels for doubles half-nodes "
                             "(default: value-only, matching the v5 cache).")
    parser.add_argument("--save-every", type=int, default=5000,
                        help="Flush the npz every N emitted examples.")
    parser.add_argument("--log-every", type=int, default=25, help="Log every N games.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)

    network = load_model(args.net)
    network.eval()
    # Encode observations to match the rollout net's channel layout so the cache
    # is consistent with the warm-start checkpoint (v5 is 17-ch base, not 26-ch).
    channels = channels_for_network(network.config)
    n_ch = len(channels) if channels is not None else NUM_CHANNELS
    print(f"Rollout net: {args.net} "
          f"({network.config['num_blocks']}x{network.config['channels']})  "
          f"oracle ply={args.ply}  encode_channels={n_ch}", flush=True)

    wrapper = GameWrapper()
    stats = _Stats()
    out_obs: list = []
    out_actions: list = []
    out_probs: list = []
    out_values: list = []
    t0 = time.time()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write incrementally to a .partial.npz and atomically promote to the real
    # path only on the final save, so a killed run never leaves a partial cache
    # at out_path (the pipeline treats an existing out_path as a complete cache).
    tmp_path = out_path.with_name(out_path.stem + ".partial.npz")

    def _save(final: bool) -> None:
        if not out_obs:
            return
        np.savez_compressed(
            tmp_path,
            observations=np.stack(out_obs).astype(np.float32),
            policy_actions=np.stack(out_actions).astype(np.int32),
            policy_probs=np.stack(out_probs).astype(np.float32),
            value_targets=np.array(out_values, dtype=np.float32),
            meta=np.array(json.dumps({
                "source": "gnubg-ondist",
                "rollout_net": args.net,
                "ply": args.ply,
                "policy_temperature": args.policy_temperature,
                "rollout_temperature": args.temperature,
                "temp_threshold": args.temp_threshold,
                "doubles_policy": args.doubles_policy,
                "seed": args.seed,
                "games": stats.games,
                "decisions_emitted": stats.decisions_emitted,
                "doubles_value_only": stats.doubles_value_only,
                "doubles_policy_emitted": stats.doubles_policy_emitted,
                "wall_time_sec": round(time.time() - t0, 1),
                "hostname": platform.node(),
                "final": final,
            })),
        )
        if final:
            os.replace(tmp_path, out_path)

    def _budget_done() -> bool:
        if args.max_decisions is not None and len(out_obs) >= args.max_decisions:
            return True
        if args.max_minutes is not None and (time.time() - t0) / 60.0 >= args.max_minutes:
            return True
        if args.max_games is not None and stats.games >= args.max_games:
            return True
        return False

    def _emit_value_only(obs, value_norm: float) -> None:
        out_obs.append(obs)
        out_actions.append(np.full(MAX_K, -1, dtype=np.int32))
        out_probs.append(np.zeros(MAX_K, dtype=np.float32))
        out_values.append(np.float32(value_norm))

    def _emit_soft(obs, cands: list[tuple[int, float]]) -> float:
        # cands: [(action, my_equity_points)]. Normalise to [-1,1], keep top-K,
        # soft target = softmax(eq / policy_temperature); value = best eq.
        ranked = sorted(cands, key=lambda t: t[1], reverse=True)[:MAX_K]
        eqs = np.array([e / 3.0 for _, e in ranked], dtype=np.float64)
        w = np.exp((eqs - eqs.max()) / args.policy_temperature)
        w = w / w.sum()
        actions = np.full(MAX_K, -1, dtype=np.int32)
        probs = np.zeros(MAX_K, dtype=np.float32)
        for j, (a, _) in enumerate(ranked):
            actions[j] = a
            probs[j] = w[j]
        out_obs.append(obs)
        out_actions.append(actions)
        out_probs.append(probs)
        out_values.append(np.float32(eqs[0]))
        return float(eqs[0])

    while not _budget_done():
        state = wrapper.new_game()
        state = _advance_through_chance(state)
        move_count = 0

        while not state.is_terminal():
            if _budget_done():
                break

            board_view = state.board_from_perspective()
            obs = encode_state(board_view, channels=channels)
            is_doubles = (
                board_view.dice is not None
                and board_view.dice[0] == board_view.dice[1]
            )

            # GNUBG labels for this position (expensive step).
            cands = candidate_equities(state, args.ply)

            if is_doubles and not args.doubles_policy:
                value_norm = max(e for _, e in cands) / 3.0
                _emit_value_only(obs, value_norm)
                stats.doubles_value_only += 1
            else:
                _emit_soft(obs, cands)
                if is_doubles:
                    stats.doubles_policy_emitted += 1
                else:
                    stats.decisions_emitted += 1

            if len(out_obs) % args.save_every == 0:
                _save(final=False)

            # Advance the game by sampling the net's own policy (on-distribution).
            policy, _ = network.predict(obs, state.legal_actions())
            temp = args.temperature if move_count < args.temp_threshold else 0.0
            action = select_action(policy, temperature=temp)
            state.apply_action(action)
            state = _advance_through_chance(state)
            move_count += 1

        stats.games += 1
        if stats.games % args.log_every == 0:
            el = (time.time() - t0) / 60.0
            n = len(out_obs)
            rate = n / max(el, 1e-9)
            print(
                f"  games={stats.games} emitted={n:,} "
                f"(nondbl={stats.decisions_emitted:,} dbl_vo={stats.doubles_value_only:,} "
                f"dbl_pol={stats.doubles_policy_emitted:,}) "
                f"el={el:.1f}min rate={rate:.0f}/min", flush=True,
            )

    _save(final=True)
    vals = np.array(out_values, dtype=np.float32) if out_values else np.array([0.0])
    best_mass = float(np.mean([p[0] for p in out_probs])) if out_probs else 0.0
    print(
        f"\nWrote {out_path}\n"
        f"  examples: {len(out_obs):,}  (games={stats.games})\n"
        f"  non-doubles policy={stats.decisions_emitted:,}  "
        f"doubles value-only={stats.doubles_value_only:,}  "
        f"doubles policy={stats.doubles_policy_emitted:,}\n"
        f"  value mean={vals.mean():.3f} std={vals.std():.3f} "
        f"min={vals.min():.3f} max={vals.max():.3f}\n"
        f"  mean prob mass on best move: {best_mass:.3f} "
        f"(policy_temperature={args.policy_temperature})\n"
        f"  total {(time.time() - t0) / 60.0:.1f} min", flush=True,
    )


if __name__ == "__main__":
    main()
