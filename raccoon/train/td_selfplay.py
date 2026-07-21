"""TD(λ) self-play for backgammon value learning (exp010).

The seed net's scalar value head is money-equity/3 in [-1, 1] from the to-move
player's POV (see ``raccoon/train/lookahead.py``). This module plays self-play
games choosing moves by 1-ply value lookahead (dice supply exploration,
TD-Gammon style) and computes forward-view TD(λ) value targets so the value head
can be regressed toward them.

Three pieces, all pure/testable; the training loop lives in ``scripts/train_td.py``:

- ``play_td_game``   — one self-play game → per-decision (pre-roll obs, player,
                       V(state)) plus the terminal ``returns()``.
- ``lambda_returns`` — forward-view λ-returns with per-step sign handling so the
                       alternation between opponents (and the same-player
                       consecutive decisions of a doubles turn) are both correct.
- ``oneply_arena``   — net-vs-net match, both playing 1-ply greedy, for eval that
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
    """Play ``net`` (1-ply value) vs GNUBG (``pick_move`` at ``gnubg_ply``).

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


def oneply_arena(
    net_a, net_b, device, games: int, seed: int = 0, max_moves: int = 2000,
) -> dict:
    """Play ``games`` net-vs-net games, both sides 1-ply greedy, seats alternated.

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
