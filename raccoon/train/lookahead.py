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

from raccoon.data.bgmatch_replay import _normalize_moves, _strip_action_index_prefix
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

    Changing sides costs an index reversal as well as a label swap: point
    indices in a ``BoardView`` run from the *to-move* player's bearoff outwards,
    so the two sides number the same board in opposite directions. Swapping the
    labels alone leaves a mirrored position that looks legal and evaluates as
    noise — the net scores such inputs at R^2 = -0.06 against reference equity
    where a correctly-sided board scores 0.9964. Production callers never take
    this path (``child_values`` asks for each child's own to-move player and
    negates instead), but it is the natural thing for deeper search to want.
    """
    sc = state.clone()
    if sc.is_chance_node():
        sc.apply_action(0)
    gs = GameState(sc)
    bv = gs.board_from_perspective()
    # If the resulting decision is not the perspective player, the BoardView is
    # from the wrong side — swap the labels *and* reverse the indexing.
    if sc.current_player() != perspective_player:
        bv = replace(
            bv,
            my_points=bv.opp_points[::-1].copy(),
            opp_points=bv.my_points[::-1].copy(),
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
    joint_doubles: bool = True,
) -> tuple[list[int], np.ndarray, float]:
    """0-ply lookahead at one decision.

    Returns ``(legal_actions, child_values, v_state)`` where ``child_values[i]``
    is the equity of ``legal_actions[i]`` from the to-move player's POV (V on the
    resulting pre-roll child, negated when the opponent is next; terminal children
    use the exact terminal value), and ``v_state`` is V on the current pre-roll
    state, also from the to-move player's POV. Every position needed is evaluated
    in a single batched forward pass.

    **Doubles (``joint_doubles``, exp020).** OpenSpiel splits a double into two
    consecutive decisions by the same player, two of the four half-moves each. With
    ``joint_doubles=True`` (the default) a half-1 action is scored by the *best
    half-2 continuation*, so the whole turn is optimised jointly — exactly what
    :func:`raccoon.eval.gnubg_adapter.candidate_equities` already does for GNUBG.

    ``joint_doubles=False`` restores the older behaviour: rank half-1 by the static
    value of the *intermediate* position and then pick half-2 greedily. That is
    unsound, because :func:`encode_pre_roll` clears the dice and the mid-doubles
    flag, so the value head is told "you are about to roll" about a position where
    the mover still owes two half-moves of a known die — and it was trained on
    pre-roll positions only, having never seen such a state (``gen_gnubg_selfplay.py``
    emits one record per turn and skips mid-doubles states). exp020 measured what
    that costs in play; the flag is kept so the two arms can be run from one code
    path and so the regression stays testable.

    Turn leaves are deduplicated by encoded position before evaluation: OpenSpiel
    enumerates *ordered* half-move pairs, so a doubles turn yields ~550 paths onto
    only ~58 distinct positions. Non-doubles decisions have one leaf per action and
    are unaffected by ``joint_doubles`` either way.

    Callers inherit the joint behaviour, :func:`process_decision` (policy
    distillation) included — an improvement to its doubles half-1 action, and no
    reason to regenerate any existing cache.
    """
    me = state.current_player()
    obs_state_pre_roll = encode_pre_roll(state, me)

    # Distinct non-terminal leaf positions, evaluated once each. A leaf token is
    # (is_terminal, terminal_value_or_slot, sign_for_the_mover).
    unique_obs: list[np.ndarray] = []
    slot_of: dict[bytes, int] = {}

    def leaf_token(child, dec_player: int) -> tuple[bool, float | int, float]:
        if child.is_terminal():
            return True, terminal_value(child, me), 1.0
        obs = encode_pre_roll(child, dec_player)
        key = obs.tobytes()
        slot = slot_of.get(key)
        if slot is None:
            slot = len(unique_obs)
            slot_of[key] = slot
            unique_obs.append(obs)
        return False, slot, (1.0 if dec_player == me else -1.0)

    legal = state.legal_actions()
    groups: list[list[tuple[bool, float | int, float]]] = []
    for a in legal:
        child_state, dec_player, is_term = state_after_apply(state, a)
        if joint_doubles and not is_term and dec_player == me:
            groups.append([
                leaf_token(*state_after_apply(child_state, a2)[:2])
                for a2 in child_state.legal_actions()
            ])
        else:
            groups.append([leaf_token(child_state, dec_player)])

    all_obs = np.stack(unique_obs + [obs_state_pre_roll])
    values = eval_values_batch(network, all_obs, device)

    cv = np.empty(len(legal), dtype=np.float32)
    for i, tokens in enumerate(groups):
        cv[i] = max(
            payload if is_term else sign * values[payload]
            for is_term, payload, sign in tokens
        )
    return legal, cv, float(values[-1])


def process_decision(
    state: pyspiel.BackgammonState, network, device,
    max_actions_per_batch: int = 64, joint_doubles: bool = True,
) -> tuple[np.ndarray, int, float]:
    """Policy-distillation view of a decision (used by synthesize_policy_dataset).

    Returns ``(obs_state, argmax_action, V(state))`` where ``obs_state`` is the
    encoding of the state AS-IS (dice + mid-doubles flag intact — the policy head
    trains on that), the action is the 0-ply argmax, and the value target is V on
    the pre-roll state. ``max_actions_per_batch`` is accepted for backwards
    compatibility and unused (all children batch in one pass).
    """
    obs_state = encode_state(GameState(state).board_from_perspective())
    legal, cv, v_state = child_values(state, network, device, joint_doubles)
    best_action = legal[int(np.argmax(cv))]
    return obs_state, best_action, v_state


def select_move(
    state: pyspiel.BackgammonState, network, device,
    temperature: float = 0.0, rng: np.random.Generator | None = None,
    joint_doubles: bool = True,
) -> tuple[int, float]:
    """Choose a move by 0-ply value lookahead. Returns ``(action, V(state))``.

    ``temperature == 0`` picks the argmax child (greedy, TD-Gammon style — the
    dice supply exploration). ``temperature > 0`` samples from a softmax over the
    child equities, which requires ``rng``. See :func:`child_values` for what
    ``joint_doubles`` controls.
    """
    legal, cv, v_state = child_values(state, network, device, joint_doubles)
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
