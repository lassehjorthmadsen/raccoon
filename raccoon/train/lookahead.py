"""0-ply value lookahead over backgammon positions.

The value head is trained on **pre-roll** board positions (dice cleared) and
represents equity from the to-move player's POV in [-1, 1] (money-equity / 3:
win = ±1/3, gammon = ±2/3, backgammon = ±1). Given a decision state, a 0-ply
lookahead enumerates the legal moves, evaluates V on each resulting pre-roll
child (negating when the child is the opponent's to move), and ranks them.

"0-ply" here is GNUBG's convention: static value evaluation of the candidate
moves, with no further search (0 additional plies of lookahead). It matches
GNUBG's own 0-ply move selection, so a net playing this way is directly
comparable to `gnubg` at ply 0. (TD-Gammon's papers call the same operation
"1-ply"; we use GNUBG's numbering throughout since GNUBG is the benchmark.)

These helpers were originally private to ``scripts/synthesize_policy_dataset.py``
(DAgger policy distillation). They are shared here so TD(λ) self-play
(``raccoon/train/td_selfplay.py``) reuses exactly the same move-selection and
value convention — the perspective/negation logic is subtle and belongs in one
place. All functions take a raw ``pyspiel`` state (they build ``GameState``
internally); callers holding a ``GameState`` pass ``gs._state``.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
import pyspiel

from raccoon.env.encoder import encode_state
from raccoon.env.game_wrapper import GameState


def state_after_apply(
    state: pyspiel.BackgammonState, action: int,
) -> tuple[pyspiel.BackgammonState, int, bool]:
    """Apply ``action`` to a clone, auto-advancing through trailing no-ops.

    Returns ``(child_state, decision_player_after, is_terminal)``. For chance
    nodes (opponent about to roll) we pick decision_player_after as 1 - current;
    the child board is the same regardless of which chance outcome we pick, so
    the perspective is what matters.
    """
    sc = state.clone()
    me = sc.current_player()
    sc.apply_action(action)

    while not sc.is_chance_node() and not sc.is_terminal():
        # Skip trailing no-op (forfeit / can't-play) actions so we land on the
        # resting state — same way replay_game does.
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


def encode_pre_roll(
    state: pyspiel.BackgammonState, perspective_player: int,
) -> np.ndarray:
    """Encode ``state`` from ``perspective_player``'s POV with dice cleared.

    The V head was trained on pre-roll positions, so we feed it pre-roll inputs
    (no dice, no mid-doubles flag) regardless of the actual chance/decision
    status of the source state. The board itself is unchanged.

    For chance nodes we can't call ``board_from_perspective`` directly (no
    current player), so we advance one chance step on a clone — any chance
    outcome leaves the board untouched, only the dice differ, and we wipe those
    right after.
    """
    sc = state.clone()
    if sc.is_chance_node():
        sc.apply_action(0)
    gs = GameState(sc)
    bv = gs.board_from_perspective()
    # If the resulting decision is not the perspective player, the BoardView is
    # from the wrong side — flip it.
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


def terminal_value(
    state: pyspiel.BackgammonState, perspective_player: int,
) -> float:
    """Terminal equity from ``perspective_player``'s POV in [-1, 1].

    ``state.returns()`` gives ±1/±2/±3 under full_scoring; divide by 3 to match
    the [-1, 1] convention used by the value head.
    """
    return state.returns()[perspective_player] / 3.0


@torch.no_grad()
def eval_values_batch(
    network, observations: np.ndarray, device: torch.device,
) -> np.ndarray:
    """Batched value forward pass — returns equity/3 in [-1, 1], shape (N,).

    Uses ``network.value_equity`` so a scalar-head net and a six-outcome net are
    interchangeable here (the latter derives equity from its softmax).
    """
    if len(observations) == 0:
        return np.array([], dtype=np.float32)
    x = torch.from_numpy(observations).float().to(device, non_blocking=True)
    return network.value_equity(x).cpu().numpy()


def child_values(
    state: pyspiel.BackgammonState, network, device: torch.device,
) -> tuple[list[int], np.ndarray, float]:
    """0-ply lookahead at one decision.

    Returns ``(legal_actions, child_values, v_state)`` where ``child_values[i]``
    is the equity of ``legal_actions[i]`` from the to-move player's POV (V on the
    resulting pre-roll child, negated when the opponent is next; terminal children
    use the exact terminal value), and ``v_state`` is V on the current pre-roll
    state, also from the to-move player's POV. Children and the state itself are
    evaluated in a single batched forward pass.
    """
    me = state.current_player()
    obs_state_pre_roll = encode_pre_roll(state, me)

    legal = state.legal_actions()
    child_obs: list[np.ndarray] = []
    child_meta: list[tuple[int, bool, float]] = []  # (dec_player, is_term, tv)
    for a in legal:
        child_state, dec_player, is_term = state_after_apply(state, a)
        if is_term:
            child_meta.append((dec_player, True, terminal_value(child_state, me)))
            child_obs.append(np.zeros_like(obs_state_pre_roll))
        else:
            child_obs.append(encode_pre_roll(child_state, dec_player))
            child_meta.append((dec_player, False, 0.0))

    all_obs = np.stack(child_obs + [obs_state_pre_roll])
    values = eval_values_batch(network, all_obs, device)

    cv = np.empty(len(legal), dtype=np.float32)
    for i, (dec_player, is_term, tv) in enumerate(child_meta):
        if is_term:
            cv[i] = tv
        elif dec_player == me:
            cv[i] = values[i]
        else:
            cv[i] = -values[i]
    return legal, cv, float(values[-1])


def process_decision(
    state: pyspiel.BackgammonState, network, device,
    max_actions_per_batch: int = 64,
) -> tuple[np.ndarray, int, float]:
    """Policy-distillation view of a decision (used by synthesize_policy_dataset).

    Returns ``(obs_state, argmax_action, V(state))`` where ``obs_state`` is the
    encoding of the state AS-IS (dice + mid-doubles flag intact — the policy head
    trains on that), the action is the 0-ply argmax, and the value target is V on
    the pre-roll state. ``max_actions_per_batch`` is accepted for backwards
    compatibility and unused (all children batch in one pass).
    """
    obs_state = encode_state(GameState(state).board_from_perspective())
    legal, cv, v_state = child_values(state, network, device)
    best_action = legal[int(np.argmax(cv))]
    return obs_state, best_action, v_state


def select_move(
    state: pyspiel.BackgammonState, network, device,
    temperature: float = 0.0, rng: np.random.Generator | None = None,
) -> tuple[int, float]:
    """Choose a move by 0-ply value lookahead. Returns ``(action, V(state))``.

    ``temperature == 0`` picks the argmax child (greedy, TD-Gammon style — the
    dice supply exploration). ``temperature > 0`` samples from a softmax over the
    child equities, which requires ``rng``.
    """
    legal, cv, v_state = child_values(state, network, device)
    if temperature and temperature > 0.0:
        if rng is None:
            raise ValueError("select_move: temperature > 0 requires an rng")
        logits = cv / temperature
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        idx = int(rng.choice(len(legal), p=probs))
    else:
        idx = int(np.argmax(cv))
    return legal[idx], v_state
