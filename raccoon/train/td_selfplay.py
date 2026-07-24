"""TD(λ) self-play for backgammon value learning (exp010).

The seed net's scalar value head is money-equity/3 in [-1, 1] from the to-move
player's POV (see ``raccoon/train/lookahead.py``). This module plays self-play
games choosing moves by 0-ply value lookahead (dice supply exploration,
TD-Gammon style) and computes forward-view TD(λ) value targets so the value head
can be regressed toward them.

Three pieces, all pure/testable; the training loop lives in ``scripts/train_td.py``:

- ``play_td_game``   — one self-play game → per-decision (pre-roll obs, player,
                       V(state)) plus the terminal ``returns()``.
- ``lambda_returns`` — forward-view λ-returns with per-step sign handling so the
                       alternation between opponents (and the same-player
                       consecutive decisions of a doubles turn) are both correct.
- ``net_arena``   — net-vs-net match, both playing 0-ply greedy, for eval that
                       is apples-to-apples with how TD actually plays.
"""
from __future__ import annotations

import numpy as np

from raccoon.env.game_wrapper import GameWrapper
from raccoon.search.mcts import _advance_through_chance
from raccoon.train.lookahead import encode_pre_roll, select_move


def play_td_game(
    network, device, temperature: float = 0.0,
    rng: np.random.Generator | None = None, max_moves: int = 2000,
) -> tuple[list[np.ndarray], list[int], list[float], list[float]] | None:
    """Play one TD self-play game.

    Returns ``(obs, players, values, terminal_returns)`` where, for each decision
    t: ``obs[t]`` is the pre-roll encoding from the to-move player's POV,
    ``players[t]`` is that player, and ``values[t]`` = V(obs[t]) from the same
    net used to select the move. ``terminal_returns`` is ``state.returns()``
    (±1/±2/±3). Returns ``None`` if the game somehow doesn't terminate (safety
    valve; should not happen in practice).
    """
    wrapper = GameWrapper()
    state = wrapper.new_game()
    state = _advance_through_chance(state)

    obs_list: list[np.ndarray] = []
    players: list[int] = []
    values: list[float] = []
    moves = 0
    while not state.is_terminal() and moves < max_moves:
        me = state.current_player()
        obs_list.append(encode_pre_roll(state._state, me))
        action, v_state = select_move(
            state._state, network, device, temperature=temperature, rng=rng,
        )
        players.append(me)
        values.append(v_state)
        state.apply_action(action)
        state = _advance_through_chance(state)
        moves += 1

    if not state.is_terminal():
        return None
    return obs_list, players, values, list(state.returns())


def lambda_returns(
    players: list[int], values: list[float],
    terminal_returns: list[float], lam: float,
) -> list[float]:
    """Forward-view TD(λ) targets, one per decision, in the decision player's POV.

    Undiscounted (γ=1), zero intermediate reward. The last decision bootstraps
    off the terminal outcome ``returns()[player]/3``; earlier decisions blend the
    one-step value of the next state with the recursive λ-return, converted into
    the current player's frame via ``sign`` (+1 when the next decision is the same
    player — a doubles half-move — else −1 for the opponent). λ=0 is one-step TD,
    λ=1 is Monte-Carlo (regress to the final outcome).
    """
    n = len(players)
    if n == 0:
        return []
    g = [0.0] * n
    g[-1] = terminal_returns[players[-1]] / 3.0
    for t in range(n - 2, -1, -1):
        sign = 1.0 if players[t] == players[t + 1] else -1.0
        g[t] = sign * ((1.0 - lam) * values[t + 1] + lam * g[t + 1])
    return g


def gnubg_arena(
    net, device, games: int, gnubg_ply: int = 0, seed: int = 0,
    max_moves: int = 2000,
) -> dict:
    """Play ``net`` (0-ply value) vs GNUBG (``pick_move`` at ``gnubg_ply``).

    This is the loop's real eval: GNUBG at a fixed ply is an external reference
    the net never trains against, so — unlike a vs-seed arena — it can't be gamed
    by self-play overfitting (the exp007 trap). 0-ply GNUBG is ~0.007 ppg weaker
    than 2-ply but ~100x faster, so it's the default. Returns
    ``{"games", "net_wins", "equity_per_game"}``, equity in points/game from the
    net's POV. Seats alternated; seeds the global numpy RNG (dice).

    ``gnubg_nn`` is imported lazily so the pure TD helpers above stay importable
    without it (e.g. in CI).
    """
    from raccoon.eval.gnubg_adapter import pick_move

    np.random.seed(seed)
    wrapper = GameWrapper()
    total = 0.0
    wins = 0
    completed = 0
    for g in range(games):
        net_is_p0 = (g % 2 == 0)
        state = wrapper.new_game()
        state = _advance_through_chance(state)
        moves = 0
        while not state.is_terminal() and moves < max_moves:
            me = state.current_player()
            if (me == 0) == net_is_p0:
                action, _ = select_move(state._state, net, device, temperature=0.0)
            else:
                action = pick_move(state, gnubg_ply)
            state.apply_action(action)
            state = _advance_through_chance(state)
            moves += 1
        if not state.is_terminal():
            continue
        pts_p0 = state.returns()[0]
        net_pts = pts_p0 if net_is_p0 else -pts_p0
        total += net_pts
        wins += int(net_pts > 0)
        completed += 1
    return {
        "games": completed,
        "net_wins": wins,
        "equity_per_game": total / completed if completed else 0.0,
    }


def net_arena(
    net_a, net_b, device, games: int, seed: int = 0, max_moves: int = 2000,
) -> dict:
    """Play ``games`` net-vs-net games, both sides 0-ply greedy, seats alternated.

    Returns ``{"games", "net_a_wins", "equity_per_game"}`` with equity in points
    per game from net_a's POV (+ = net_a ahead). Seat alternation removes
    first-player bias. Seeds the global numpy RNG (used for dice) for
    reproducibility across the match.
    """
    np.random.seed(seed)
    wrapper = GameWrapper()
    total = 0.0
    a_wins = 0
    completed = 0
    for g in range(games):
        a_is_p0 = (g % 2 == 0)
        nets = {0: net_a, 1: net_b} if a_is_p0 else {0: net_b, 1: net_a}
        state = wrapper.new_game()
        state = _advance_through_chance(state)
        moves = 0
        while not state.is_terminal() and moves < max_moves:
            me = state.current_player()
            action, _ = select_move(state._state, nets[me], device, temperature=0.0)
            state.apply_action(action)
            state = _advance_through_chance(state)
            moves += 1
        if not state.is_terminal():
            continue
        pts_p0 = state.returns()[0]
        a_pts = pts_p0 if a_is_p0 else -pts_p0
        total += a_pts
        a_wins += int(a_pts > 0)
        completed += 1
    return {
        "games": completed,
        "net_a_wins": a_wins,
        "equity_per_game": total / completed if completed else 0.0,
    }


def gnubg_arena_scored(
    net, device, games: int, ref_ply: int = 0, seed: int = 0,
    max_moves: int = 2000, top_k: int = 0,
) -> dict:
    """Net (0-ply value) vs GNUBG, with per-decision **error-rate** scoring.

    Like :func:`gnubg_arena` (raw ppg from net-vs-GNUBG games) but ALSO scores every
    *net* decision against GNUBG at ``ref_ply``: ``error = V_best - V_played`` where
    both come from ``candidate_equities`` (GNUBG's cubeless equity of each legal move,
    from the net's POV). This is the standard backgammon "error rate" — an **exact,
    low-variance** measure of how much equity the net's move choices concede to GNUBG,
    with none of the control-variate bias a naive luck-adjustment carries (a 0-ply
    control variate isn't self-consistent with its own 1-ply lookahead).

    GNUBG is called on the net's moves too (via ``candidate_equities``), so this is
    ~2x the GNUBG cost of :func:`gnubg_arena` — use it offline, not in the inner loop.
    The opponent plays at ``ref_ply`` as well, so raw ppg and the error rate are both
    measured against the same GNUBG reference (opponent error is then structurally 0).

    Returns per-game net points AND per-game net error (for tight CIs on both), the
    total net decisions, and — if ``top_k`` — the highest-error ``(obs, error)`` pairs
    (the DAgger set). ``obs`` is the net-POV pre-roll encoding at that decision.
    """
    from raccoon.eval.gnubg_adapter import pick_move, candidate_equities

    np.random.seed(seed)
    wrapper = GameWrapper()
    game_pts: list[float] = []
    game_err: list[float] = []
    wins = 0
    decisions = 0
    top: list[tuple[np.ndarray, float]] = []
    for g in range(games):
        net_is_p0 = (g % 2 == 0)
        state = wrapper.new_game()
        state = _advance_through_chance(state)
        moves = 0
        err_this_game = 0.0
        while not state.is_terminal() and moves < max_moves:
            me = state.current_player()
            if (me == 0) == net_is_p0:
                # net's decision — score it vs GNUBG (same actions the net ranks)
                eq_by_action = {a: e for a, e in candidate_equities(state, ref_ply)}
                v_best = max(eq_by_action.values())
                action, _ = select_move(state._state, net, device, temperature=0.0)
                err = v_best - eq_by_action.get(action, v_best)
                if err < 0.0:
                    err = 0.0  # net's move can't beat GNUBG's max; guard rounding
                err_this_game += err
                decisions += 1
                if top_k:
                    top.append((encode_pre_roll(state._state, me), float(err)))
            else:
                action = pick_move(state, ref_ply)
            state.apply_action(action)
            state = _advance_through_chance(state)
            moves += 1
        if not state.is_terminal():
            continue
        pts_p0 = state.returns()[0]
        net_pts = pts_p0 if net_is_p0 else -pts_p0
        game_pts.append(net_pts)
        game_err.append(err_this_game)
        wins += int(net_pts > 0)

    if top_k and len(top) > top_k:
        top.sort(key=lambda t: t[1], reverse=True)
        top = top[:top_k]
    completed = len(game_pts)
    return {
        "games": completed,
        "net_wins": wins,
        "equity_per_game": (sum(game_pts) / completed) if completed else 0.0,
        "game_pts": np.array(game_pts, dtype=np.float64),
        "game_err": np.array(game_err, dtype=np.float64),
        "decisions": decisions,
        "err_total": float(sum(game_err)),
        "top_error": top,
    }
