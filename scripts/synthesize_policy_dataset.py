"""Build the policy-pretraining dataset by 1-ply V-lookahead on bglab matches.

For each decision state reached while replaying a match, we:

  1. Encode the state ``s`` (with its dice) → ``observation``.
  2. Enumerate legal actions; for each, clone the state, apply, and encode
     the resulting position. Resulting positions live at chance nodes
     (opponent to roll), mid-doubles decisions, or terminal states; we
     normalise each to a pre-roll BoardView from the relevant player's
     perspective so the V head — which was trained on pre-roll wildbg
     positions — gets in-distribution inputs.
  3. Batch-evaluate V on the child observations. Pick ``argmax_a V(child)``
     (negated where the child is opponent-to-move) as the **policy target**.
  4. Evaluate V on the state itself; record as the **value target**
     (knowledge distillation — keeps the value head anchored to its
     wildbg-calibrated predictions while the trunk learns the dice
     channels for the first time).

Output: ``data/bglab/policy_cache.npz`` with three arrays:

  - ``observations`` : (N, 17, 2, 12) float32
  - ``policy_targets``: (N,) int32  — chosen action index in 0..1351
  - ``value_targets`` : (N,) float32 in [-1, 1]

Typical run on the full corpus (~5000 games, ~240k decisions):

    python scripts/synthesize_policy_dataset.py \\
        --pretrained experiments/pretrain-wildbg-v1/checkpoints/pretrained.pt \\
        --out data/bglab/policy_cache.npz
"""

from __future__ import annotations

import argparse
import glob
import json
import platform
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import pyspiel

from raccoon.data.bgmatch import parse_match_file
from raccoon.data.bgmatch_replay import ReplayError, replay_game
from raccoon.env.encoder import encode_state
from raccoon.env.game_wrapper import GameState
from raccoon.model.network import load_model


def _state_after_apply(
    state: pyspiel.BackgammonState, action: int,
) -> tuple[pyspiel.BackgammonState, int, bool]:
    """Apply ``action`` to a clone, auto-advancing through trailing no-ops.

    Returns ``(child_state, decision_player_after, is_terminal)``. For
    chance nodes (opponent about to roll) we pick decision_player_after as
    1 - current; the child board is the same regardless of which chance
    outcome we pick, so the perspective is what matters.
    """
    sc = state.clone()
    me = sc.current_player()
    sc.apply_action(action)

    while not sc.is_chance_node() and not sc.is_terminal():
        # Skip trailing no-op (forfeit) actions so we land on the resting
        # state — same way replay_game does.
        empties = []
        from raccoon.data.bgmatch_replay import (
            _normalize_moves, _strip_action_index_prefix,
        )
        for ea in sc.legal_actions():
            if _normalize_moves(
                _strip_action_index_prefix(sc.action_to_string(ea))
            ) == ():
                empties.append(ea)
        if not empties:
            break
        sc.apply_action(empties[0])

    if sc.is_terminal():
        return sc, me, True
    if sc.is_chance_node():
        # After my move, opponent is up next.
        return sc, 1 - me, False
    # Mid-doubles: same player still on move.
    return sc, sc.current_player(), False


def _encode_pre_roll(
    state: pyspiel.BackgammonState, perspective_player: int,
) -> np.ndarray:
    """Encode ``state`` from ``perspective_player``'s POV with dice cleared.

    The V head was trained on wildbg pre-roll positions, so we feed it
    pre-roll inputs (no dice, no mid-doubles flag) regardless of the
    actual chance/decision status of the source state. The board itself
    is unchanged.

    For chance nodes we can't call ``board_from_perspective`` directly
    (no current player), so we advance one chance step on a clone — any
    chance outcome leaves the board untouched, only the dice differ, and
    we wipe those right after.
    """
    sc = state.clone()
    if sc.is_chance_node():
        sc.apply_action(0)
    gs = GameState(sc)
    bv = gs.board_from_perspective()
    # If the resulting decision is not the perspective player, the
    # BoardView is already from the wrong side — flip it.
    if sc.current_player() != perspective_player:
        bv = replace(
            bv,
            my_points=bv.opp_points,
            opp_points=bv.my_points,
            my_bar=bv.opp_bar,
            opp_bar=bv.my_bar,
            my_off=bv.opp_off,
            opp_off=bv.my_off,
        )
    bv = replace(bv, dice=None, mid_doubles=False)
    return encode_state(bv)


def _terminal_value(
    state: pyspiel.BackgammonState, perspective_player: int,
) -> float:
    """Return the terminal equity from ``perspective_player``'s POV in [-1, 1].

    ``state.returns()`` gives ±1/±2/±3 under full_scoring; divide by 3 to
    match the [-1, 1] convention used by the value head.
    """
    return state.returns()[perspective_player] / 3.0


@torch.no_grad()
def _eval_values_batch(
    network, observations: np.ndarray, device: torch.device,
) -> np.ndarray:
    """Batched value forward pass — returns shape (N,) in [-1, 1]."""
    if len(observations) == 0:
        return np.array([], dtype=np.float32)
    x = torch.from_numpy(observations).float().to(device, non_blocking=True)
    _, value = network(x)
    return value.squeeze(-1).cpu().numpy()


def _process_decision(
    state: pyspiel.BackgammonState, network, device, max_actions_per_batch: int,
) -> tuple[np.ndarray, int, float]:
    """Run 1-ply lookahead at one decision; returns (obs, argmax_action, V(state)).

    The observation is the encoding of ``state`` AS-IS (including dice and
    mid-doubles flag) — that's what the policy head trains on. The argmax
    action is selected by evaluating V on every legal child position, all
    from the perspective of whoever's the *next* decision-maker.
    """
    me = state.current_player()
    gs = GameState(state)
    obs_state = encode_state(gs.board_from_perspective())  # with dice intact

    legal = state.legal_actions()
    child_obs: list[np.ndarray] = []
    child_meta: list[tuple[int, int, bool, float]] = []
    # (action, decision_player_after, is_terminal, terminal_value)
    for a in legal:
        child_state, dec_player, is_term = _state_after_apply(state, a)
        if is_term:
            tv = _terminal_value(child_state, me)
            child_meta.append((a, dec_player, True, tv))
            child_obs.append(np.zeros((17, 2, 12), dtype=np.float32))
        else:
            obs_c = _encode_pre_roll(child_state, dec_player)
            child_meta.append((a, dec_player, False, 0.0))
            child_obs.append(obs_c)

    # Include obs_state as the LAST entry so we can read V(state) in the
    # same batch — saves a separate forward pass.
    obs_state_pre_roll = _encode_pre_roll(state, me)
    all_obs = np.stack(child_obs + [obs_state_pre_roll])
    values = _eval_values_batch(network, all_obs, device)

    # Convert each child value to "my POV": negate when the next decision
    # was opponent's; for terminal, use the precomputed terminal value.
    child_values = np.empty(len(legal), dtype=np.float32)
    for i, (_, dec_player, is_term, tv) in enumerate(child_meta):
        if is_term:
            child_values[i] = tv
        elif dec_player == me:
            child_values[i] = values[i]
        else:
            child_values[i] = -values[i]

    best_idx = int(np.argmax(child_values))
    best_action = child_meta[best_idx][0]
    value_target = float(values[-1])  # V(state) — note: includes dice in input

    return obs_state, best_action, value_target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretrained", type=str, required=True,
        help="Checkpoint whose V head drives the 1-ply lookahead.",
    )
    parser.add_argument(
        "--out", type=str, default="data/bglab/policy_cache.npz",
        help="Output npz path. Directory will be created if needed.",
    )
    parser.add_argument(
        "--patterns", type=str, nargs="+",
        default=[
            "data/bglab/data-raw/lasse/raw/*.txt",
            "data/bglab/data-raw/Llabba/raw/*.txt",
        ],
        help="Glob patterns to find match files.",
    )
    parser.add_argument("--max-files", type=int, default=None,
                        help="Cap files processed (for smoke runs).")
    parser.add_argument("--log-every", type=int, default=50,
                        help="Print progress every N files.")
    parser.add_argument("--save-every", type=int, default=200,
                        help="Save an intermediate npz every N files so a "
                             "crash or interruption doesn't lose everything.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    print(f"Loading network from {args.pretrained}", flush=True)
    network = load_model(args.pretrained).to(device)
    network.eval()

    game_obj = pyspiel.load_game("backgammon(scoring_type=full_scoring)")
    files: list[str] = []
    for p in args.patterns:
        files.extend(sorted(glob.glob(p)))
    if args.max_files is not None:
        files = files[:args.max_files]
    print(f"Found {len(files)} match files", flush=True)

    all_obs: list[np.ndarray] = []
    all_pol: list[int] = []
    all_val: list[float] = []
    n_games_ok = n_games_fail = n_step_errors = 0
    t0 = time.time()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _save(final: bool) -> None:
        if not all_obs:
            return
        obs_arr = np.stack(all_obs).astype(np.float32)
        pol_arr = np.array(all_pol, dtype=np.int32)
        val_arr = np.array(all_val, dtype=np.float32)
        np.savez_compressed(
            out_path,
            observations=obs_arr,
            policy_targets=pol_arr,
            value_targets=val_arr,
            meta=np.array(json.dumps({
                "pretrained": args.pretrained,
                "patterns": args.patterns,
                "n_files": len(files),
                "n_games_ok": n_games_ok,
                "n_games_fail": n_games_fail,
                "n_step_errors": n_step_errors,
                "n_decisions": int(obs_arr.shape[0]),
                "wall_time_sec": round(time.time() - t0, 1),
                "hostname": platform.node(),
                "device": str(device),
                "final": final,
            })),
        )

    for i, f in enumerate(files):
        try:
            match = parse_match_file(f)
        except Exception as e:
            print(f"  parse error on {Path(f).name}: {e}", flush=True)
            continue

        for game in match.games:
            try:
                steps = list(replay_game(game, game_obj))
            except ReplayError:
                n_games_fail += 1
                continue
            n_games_ok += 1
            for step in steps:
                try:
                    obs, pol, val = _process_decision(
                        step.state, network, device,
                        max_actions_per_batch=64,
                    )
                except Exception as e:
                    n_step_errors += 1
                    if n_step_errors <= 5:
                        print(
                            f"  step error in {Path(f).name} game "
                            f"{game.game_number}: {type(e).__name__}: {e}",
                            flush=True,
                        )
                    continue
                all_obs.append(obs)
                all_pol.append(pol)
                all_val.append(val)

        if (i + 1) % args.log_every == 0 or i + 1 == len(files):
            elapsed = time.time() - t0
            rate = len(all_obs) / max(elapsed, 1e-6)
            eta_min = (len(files) - i - 1) / max(i + 1, 1) * elapsed / 60
            print(
                f"  [{i + 1:>5d}/{len(files)}] games_ok={n_games_ok} "
                f"games_fail={n_games_fail} step_err={n_step_errors} "
                f"decisions={len(all_obs):,} elapsed={elapsed/60:.1f}min "
                f"rate={rate:.0f}/s eta={eta_min:.0f}min",
                flush=True,
            )

        if (i + 1) % args.save_every == 0:
            _save(final=False)
            print(f"    -> intermediate save: {out_path}", flush=True)

    _save(final=True)
    obs_arr = np.stack(all_obs).astype(np.float32) if all_obs else np.empty((0, 17, 2, 12), np.float32)
    pol_arr = np.array(all_pol, dtype=np.int32)
    val_arr = np.array(all_val, dtype=np.float32)
    elapsed = time.time() - t0
    print(
        f"\nWrote {out_path}\n"
        f"  observations: {obs_arr.shape}\n"
        f"  policy_targets: {pol_arr.shape} "
        f"(range {pol_arr.min() if len(pol_arr) else 0}-{pol_arr.max() if len(pol_arr) else 0})\n"
        f"  value_targets: {val_arr.shape} "
        f"(mean={val_arr.mean() if len(val_arr) else 0:.3f}, "
        f"std={val_arr.std() if len(val_arr) else 0:.3f})\n"
        f"  games ok={n_games_ok}  fail={n_games_fail}  step_errors={n_step_errors}\n"
        f"  total {elapsed/60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
