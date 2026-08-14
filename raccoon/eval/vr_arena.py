"""Net-vs-GNUBG matches with a variance-reduced ppg estimator.

Same games as :func:`raccoon.train.td_selfplay.gnubg_arena` — net plays 0-ply greedy
value lookahead, GNUBG plays at ``gnubg_ply``, seats alternate — but every pre-roll
state is intercepted before the dice are sampled so the realised **luck** of the roll
can be recorded (see :mod:`raccoon.eval.luck` for the estimator and why it is exactly
unbiased). Each game then yields two numbers:

    raw = the game result in points, from the net's POV
    vr  = raw - (accumulated luck)

``mean(vr)`` estimates the same quantity as ``mean(raw)`` with far less variance, which
is what makes sub-0.05-ppg differences measurable at a practical number of games.

Turning variance reduction off (``vr=False``) plays *identical* games from the same
seed: the control variate reads gnubg through a pure C path and touches no RNG. That is
what lets a plain run and a variance-reduced run be compared as two independent samples
of one process rather than two different processes.
"""

from __future__ import annotations

import numpy as np

from raccoon.env.game_wrapper import GameWrapper
from raccoon.eval.luck import pre_roll_values
from raccoon.train.lookahead import select_move


def gnubg_arena_vr(
    net, device, games: int, gnubg_ply: int = 0, cv_ply: int = 0,
    seed: int = 0, max_moves: int = 2000, vr: bool = True,
    joint_doubles: bool = True,
) -> dict:
    """Play ``games`` net-vs-GNUBG games, returning per-game raw and VR results.

    Returns ``{"games", "net_wins", "game_pts", "game_luck", "game_vr", "game_rolls"}``.
    ``game_pts`` is the raw result in points from the net's POV (±1/±2/±3 under
    ``full_scoring``); ``game_luck`` is the accumulated dice luck, also from the net's
    POV; ``game_vr = game_pts - game_luck``. With ``vr=False`` the luck array is all
    zeros and ``game_vr`` equals ``game_pts``.

    ``cv_ply`` is the ply of the control-variate evaluator and must be 0 (gnubg-nn's
    ``best_move`` segfaults deeper); ``gnubg_ply`` is the *opponent's* strength and is
    unconstrained. The two are deliberately independent — the control variate stays a
    fixed function of (position, roll) whatever the opponent does, which is what keeps
    the estimator unbiased.

    ``joint_doubles`` selects how the net executes a doubles turn — jointly over all
    four half-moves (the default, and what GNUBG's side always did) or by the older
    greedy two-step path. It is the exp020 A/B; see
    :func:`raccoon.train.lookahead.child_values`. The control variate is unaffected
    either way: it is a fixed function of (position, roll) and never of the move
    played, so both arms are measured on the same unbiased ruler.

    Seeds the global numpy RNG (dice), like the other arenas in this project.
    """
    from raccoon.eval.gnubg_adapter import pick_move

    np.random.seed(seed)
    wrapper = GameWrapper()
    game_pts: list[float] = []
    game_luck: list[float] = []
    game_rolls: list[int] = []
    wins = 0

    for g in range(games):
        net_is_p0 = (g % 2 == 0)
        net_player = 0 if net_is_p0 else 1
        state = wrapper.new_game()
        luck_total = 0.0
        rolls = 0
        moves = 0

        while not state.is_terminal() and moves < max_moves:
            if state.is_chance_node():
                # Pre-roll: value every outcome, then sample one. The baseline is the
                # mean over the node's *own* dice distribution, so the recorded luck is
                # zero-mean by construction. Both branches draw the roll with the same
                # call on the same distribution, so the dice stream is identical whether
                # or not variance reduction is on.
                if vr:
                    actions, probs, values = pre_roll_values(state, net_player, cv_ply)
                else:
                    outcomes = state.chance_outcomes()
                    actions = [a for a, _ in outcomes]
                    probs = np.array([p for _, p in outcomes], dtype=np.float64)
                idx = int(np.random.choice(len(actions), p=probs))
                if vr:
                    luck_total += float(values[idx]) - float(probs @ values)
                state.apply_action(actions[idx])
                rolls += 1
                continue

            if state.current_player() == net_player:
                action, _ = select_move(
                    state._state, net, device, temperature=0.0,
                    joint_doubles=joint_doubles,
                )
            else:
                action = pick_move(state, gnubg_ply)
            state.apply_action(action)
            moves += 1

        if not state.is_terminal():
            continue

        net_pts = state.returns()[net_player]
        game_pts.append(net_pts)
        game_luck.append(luck_total)
        game_rolls.append(rolls)
        wins += int(net_pts > 0)

    pts = np.array(game_pts, dtype=np.float64)
    luck = np.array(game_luck, dtype=np.float64)
    return {
        "games": len(pts),
        "net_wins": wins,
        "game_pts": pts,
        "game_luck": luck,
        "game_vr": pts - luck,
        "game_rolls": np.array(game_rolls, dtype=np.int32),
    }
