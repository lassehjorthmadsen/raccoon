"""What Raccoon's two-step doubles execution costs in play (exp020).

OpenSpiel splits a doubles turn into two consecutive decisions by the same player,
two of the four half-moves each. GNUBG's side of the arena
(:func:`raccoon.eval.gnubg_adapter.candidate_equities`) recurses through both halves
and optimises the turn jointly. Raccoon's side historically did not: it ranked half 1
by the static value of the *intermediate* position — which :func:`encode_pre_roll`
hands to the value head with the dice and the mid-doubles flag cleared, so the net is
told "you are about to roll" about a position where the mover still owes two
half-moves — and then picked half 2 greedily. On 1/6 of turns the two engines were
not playing by the same rule, and only one of them was handicapped.

This module measures the gap directly, on the distribution where it matters:

    concession = eq_gnubg(final_joint) - eq_gnubg(final_greedy)

Both finals come from the **same** network, so the teacher's own move choice never
enters and this is purely a question of how the net's value function is *searched*.
The main line plays the greedy arm by default (``play_joint=False``), so the sample
is the exp019 playing distribution — the one whose ppg shortfall prompted the
experiment — and the joint arm is scored as a counterfactual on each turn.

Two properties keep the measurement honest:

- **Non-doubles turns are a control, not filler.** A turn that needs one OpenSpiel
  action has one leaf per legal action, so joint and greedy enumerate the identical
  set and must agree exactly. Any non-zero concession there is a harness bug — a bad
  resting signature, a sign error, a mis-judged terminal — not a strength difference.
  ``tests/test_doubles.py`` pins the same property at the ``child_values`` level.
- **Measuring does not perturb the games.** Both arms are explored on clones, and
  neither ``select_move`` at temperature 0 nor ``evaluate_equity`` touches the RNG, so
  a run with the same seed visits the same positions as a plain ``gnubg_arena``.

Judged at ``judge_ply`` (default 2 — the strongest affordable referee, and the tier
the distillation teacher used). GNUBG is only consulted when the two arms actually
disagree; when they land on the same position the concession is exactly zero and no
evaluation is needed, which is what keeps the run affordable.
"""
from __future__ import annotations

import numpy as np

from raccoon.env.game_wrapper import GameState, GameWrapper
from raccoon.search.mcts import _advance_through_chance
from raccoon.train.lookahead import select_move, state_after_apply


def _resting_signature(state, is_terminal: bool) -> tuple:
    """Canonical identity of the position a completed turn comes to rest on.

    Compared instead of the action sequence because OpenSpiel enumerates *ordered*
    half-move pairs: two different action sequences routinely play the same four
    half-moves in a different order and reach the same board, which is agreement,
    not disagreement.
    """
    from raccoon.eval.gnubg_adapter import board_from_view

    if is_terminal:
        return ("terminal", tuple(state.returns()))
    resolved = state.clone()
    if resolved.is_chance_node():
        resolved.apply_action(0)
    board = board_from_view(GameState(resolved).board_from_perspective())
    return tuple(tuple(side) for side in board)


def _judged_equity(state, is_terminal: bool, mover: int, ply: int) -> float:
    """GNUBG cubeless equity of a resting position, from the mover's POV, in points."""
    from raccoon.eval.gnubg_adapter import board_from_view, evaluate_equity

    if is_terminal:
        return float(state.returns()[mover])
    resolved = state.clone()
    if resolved.is_chance_node():
        resolved.apply_action(0)
    equity = evaluate_equity(
        board_from_view(GameState(resolved).board_from_perspective()), ply
    )
    return equity if resolved.current_player() == mover else -equity


def _play_turn(state, net, device, joint_doubles: bool):
    """Play one whole turn on a clone. Returns ``(resting, is_terminal, n_actions)``."""
    mover = state.current_player()
    cur = state.clone()
    n_actions = 0
    while True:
        action, _ = select_move(cur, net, device, joint_doubles=joint_doubles)
        nxt, dec_player, is_term = state_after_apply(cur, action)
        n_actions += 1
        if is_term or dec_player != mover:
            return nxt, is_term, n_actions
        cur = nxt


def doubles_concession_arena(
    net, device, games: int, gnubg_ply: int = 0, judge_ply: int = 2,
    seed: int = 0, max_moves: int = 2000, play_joint: bool = False,
) -> dict:
    """Play ``net`` vs GNUBG and score both doubles arms at every net turn.

    Returns per-turn arrays (``two_step``, ``disagreed``, ``concession``,
    ``concession_0ply``, ``n_paths``) plus per-game points, so a caller can compute an
    empirical CI on the concession and cross-check the raw ppg against exp019. Seats
    alternated; seeds the global numpy RNG (dice), matching
    :func:`raccoon.train.td_selfplay.gnubg_arena`.
    """
    from raccoon.eval.gnubg_adapter import pick_move

    np.random.seed(seed)
    wrapper = GameWrapper()
    two_step: list[bool] = []
    disagreed: list[bool] = []
    concession: list[float] = []
    concession_0ply: list[float] = []
    n_paths: list[int] = []
    game_pts: list[float] = []

    for g in range(games):
        net_is_p0 = (g % 2 == 0)
        state = _advance_through_chance(wrapper.new_game())
        prev_decision_player: int | None = None
        moves = 0
        while not state.is_terminal() and moves < max_moves:
            me = state.current_player()
            if (me == 0) != net_is_p0:
                action = pick_move(state, gnubg_ply)
            else:
                if prev_decision_player != me:  # start of a turn, not mid-doubles
                    _score_turn(
                        state._state, net, device, me, judge_ply,
                        two_step, disagreed, concession, concession_0ply, n_paths,
                    )
                action, _ = select_move(
                    state._state, net, device, joint_doubles=play_joint,
                )
            prev_decision_player = me
            state.apply_action(action)
            state = _advance_through_chance(state)
            moves += 1

        if not state.is_terminal():
            continue
        pts_p0 = state.returns()[0]
        game_pts.append(pts_p0 if net_is_p0 else -pts_p0)

    return {
        "games": len(game_pts),
        "game_pts": np.array(game_pts, dtype=np.float64),
        "two_step": np.array(two_step, dtype=bool),
        "disagreed": np.array(disagreed, dtype=bool),
        "concession": np.array(concession, dtype=np.float64),
        "concession_0ply": np.array(concession_0ply, dtype=np.float64),
        "n_paths": np.array(n_paths, dtype=np.int64),
    }


def _score_turn(
    raw_state, net, device, mover: int, judge_ply: int,
    two_step: list, disagreed: list, concession: list, concession_0ply: list,
    n_paths: list,
) -> None:
    """Compare the two doubles arms at one start-of-turn state and append a record.

    Priced twice — at ``judge_ply`` (the headline) and at 0-ply — because a
    disagreement costs two extra GNUBG evaluations against a whole turn's worth of
    network enumeration, so the second referee is essentially free and makes the
    result checkable against a weaker, faster judge.
    """
    greedy, greedy_term, greedy_actions = _play_turn(raw_state, net, device, False)
    joint, joint_term, joint_actions = _play_turn(raw_state, net, device, True)

    is_two_step = max(greedy_actions, joint_actions) > 1
    same = (
        _resting_signature(greedy, greedy_term)
        == _resting_signature(joint, joint_term)
    )
    loss = loss_0ply = 0.0
    if not same:
        loss = (
            _judged_equity(joint, joint_term, mover, judge_ply)
            - _judged_equity(greedy, greedy_term, mover, judge_ply)
        )
        loss_0ply = (
            _judged_equity(joint, joint_term, mover, 0)
            - _judged_equity(greedy, greedy_term, mover, 0)
        )

    paths = 0
    if is_two_step:
        for a1 in raw_state.legal_actions():
            child, dec_player, is_term = state_after_apply(raw_state, a1)
            paths += (
                len(child.legal_actions())
                if (not is_term) and dec_player == mover else 1
            )

    two_step.append(is_two_step)
    disagreed.append(not same)
    concession.append(loss)
    concession_0ply.append(loss_0ply)
    n_paths.append(paths)
