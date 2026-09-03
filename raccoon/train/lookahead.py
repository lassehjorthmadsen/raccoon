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

from raccoon.cube.janowski import (
    cl2cf_money, cube_action, jacoby_active, probs6_to_cumulative5,
)
from raccoon.cube.state import CubeState, flip_label, x_for
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
    legal, groups, all_obs = enumerate_leaves(state, joint_doubles)
    values = eval_values_batch(network, all_obs, device)

    cv = np.empty(len(legal), dtype=np.float32)
    for i, tokens in enumerate(groups):
        cv[i] = max(
            payload / 3.0 if is_term else (-values[payload] if opp else values[payload])
            for is_term, payload, opp in tokens
        )
    return legal, cv, float(values[-1])


# A leaf token: ``(is_terminal, terminal_points_or_slot, opponent_is_to_move)``.
# Terminal payloads are in **points** (±1/±2/±3) so that both the cubeless caller
# (which divides by 3) and the cubeful one (which does not) can read the same
# enumeration; see the scale note on :func:`child_cubeful_values`.
LeafToken = tuple[bool, "float | int", bool]


def enumerate_leaves(
    state: pyspiel.BackgammonState, joint_doubles: bool = True,
) -> tuple[list[int], list[list[LeafToken]], np.ndarray]:
    """The shared skeleton of every 0-ply lookahead: what to evaluate, and where.

    Returns ``(legal_actions, groups, observations)``. ``groups[i]`` holds the
    leaf tokens for ``legal_actions[i]`` — one token per candidate turn-ending
    position, several when ``joint_doubles`` expands the second half of a doubles
    turn. ``observations`` is the batch to run the network on: the distinct
    non-terminal leaves, **with the pre-roll encoding of ``state`` itself
    appended last**, so ``values[-1]`` is always V(state).

    Split out of :func:`child_values` so the cubeful ranking in
    :func:`child_cubeful_values` reuses the identical enumeration, dedup and
    doubles handling rather than growing a second copy of the subtle part.
    """
    me = state.current_player()
    obs_state_pre_roll = encode_pre_roll(state, me)

    unique_obs: list[np.ndarray] = []
    slot_of: dict[bytes, int] = {}

    def leaf_token(child, dec_player: int) -> LeafToken:
        if child.is_terminal():
            return True, child.returns()[me], False
        obs = encode_pre_roll(child, dec_player)
        key = obs.tobytes()
        slot = slot_of.get(key)
        if slot is None:
            slot = len(unique_obs)
            slot_of[key] = slot
            unique_obs.append(obs)
        return False, slot, dec_player != me

    legal = state.legal_actions()
    groups: list[list[LeafToken]] = []
    for a in legal:
        child_state, dec_player, is_term = state_after_apply(state, a)
        if joint_doubles and not is_term and dec_player == me:
            groups.append([
                leaf_token(*state_after_apply(child_state, a2)[:2])
                for a2 in child_state.legal_actions()
            ])
        else:
            groups.append([leaf_token(child_state, dec_player)])

    return legal, groups, np.stack(unique_obs + [obs_state_pre_roll])


@torch.no_grad()
def eval_probs6_batch(
    network, observations: np.ndarray, device: torch.device,
) -> np.ndarray:
    """Batched six-outcome forward pass, shape ``(N, 6)``.

    The cubeful sibling of :func:`eval_values_batch`. ``network.value_probs6``
    raises on a ``scalar``-head net, which is correct: Janowski needs the whole
    distribution to compute W and L, and a single equity number cannot supply it.
    """
    if len(observations) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    x = torch.from_numpy(observations).float().to(device, non_blocking=True)
    return network.value_probs6(x).cpu().numpy()


def child_cubeful_values(
    state: pyspiel.BackgammonState, network, device: torch.device,
    cube_label: str, x: float, jacoby: bool = False,
    joint_doubles: bool = True,
) -> tuple[list[int], np.ndarray, float]:
    """0-ply lookahead ranking candidates by **cubeful** equity.

    The cube-aware twin of :func:`child_values`, sharing its enumeration through
    :func:`enumerate_leaves`. Returns ``(legal_actions, child_values, v_state)``
    with the same shapes and the same "from the to-move player's POV" convention.

    ``cube_label`` is where the cube sits *as the player to move sees it* —
    ``CENTERED``/``PLAYER``/``OPPONENT`` from :mod:`raccoon.cube.janowski`; a
    caller holding a :class:`~raccoon.cube.state.CubeState` gets it from
    ``cube.label_for(mover)``. ``x`` is the cube life index, normally
    :func:`raccoon.cube.state.x_for` of the root position. ``jacoby`` is the raw
    rule setting: **False for Raccoon's target** (money without Jacoby, per
    ``goal.md``), True only when scoring against the BGSage benchmark, whose
    references were generated with the rule on.

    Two things differ from the cubeless path and both are easy to get wrong:

    **Scale.** These values are money points at cube value 1, roughly [-3, 3] —
    *not* the equity/3 in [-1, 1] that :func:`child_values` returns. Terminal
    children are therefore ``state.returns()[me]`` undivided. Never compare a
    number from this function with one from that one.

    **The negation flips the cube owner too.** A leaf where the opponent is on
    roll is scored ``-cl2cf_money(probs, flip_label(cube_label), ...)``: the
    equity changes sign *and* a cube we own becomes a cube they own. Negating
    alone would credit the opponent with our cube ownership.

    The cube *value* never enters: it multiplies every candidate by the same
    constant, so it cannot change a ranking. Callers scale at the end.
    """
    legal, groups, all_obs = enumerate_leaves(state, joint_doubles)
    probs6 = eval_probs6_batch(network, all_obs, device)

    # Each distinct leaf is priced twice — once as ours to move, once as theirs —
    # because the dedup key is the encoded position and two tokens sharing a slot
    # can still sit on opposite sides of a negation.
    flipped = flip_label(cube_label)
    # `jacoby` is the raw rule setting; cl2cf_money wants it already resolved
    # against the cube position, and the two sides of a flip do not resolve the
    # same way once the cube has been turned.
    jac_mine = jacoby_active(cube_label, jacoby)
    jac_theirs = jacoby_active(flipped, jacoby)
    mine = np.empty(len(probs6), dtype=np.float64)
    theirs = np.empty(len(probs6), dtype=np.float64)
    for j, p6 in enumerate(probs6):
        probs5 = probs6_to_cumulative5(p6)
        mine[j] = cl2cf_money(probs5, cube_label, x, jac_mine)
        theirs[j] = cl2cf_money(probs5, flipped, x, jac_theirs)

    cv = np.empty(len(legal), dtype=np.float32)
    for i, tokens in enumerate(groups):
        cv[i] = max(
            payload if is_term else (-theirs[payload] if opp else mine[payload])
            for is_term, payload, opp in tokens
        )
    return legal, cv, float(mine[-1])


@torch.no_grad()
def net_cube_action(
    state: pyspiel.BackgammonState, network, device, cube_label: str,
    x: float, jacoby: bool = False,
) -> tuple[bool, bool]:
    """``(should_double, should_take)`` from the net, for the player on roll.

    The net-side counterpart of
    :func:`raccoon.eval.gnubg_adapter.gnubg_cube_action`, and it works the same
    way: one pre-roll evaluation of the board **with the doubler on roll**, put
    through Janowski. ``cube_label`` is from the doubler's point of view, so it
    is ``CENTERED`` or ``PLAYER``.

    A receiver answers by calling this on the same position with its own net and
    reading ``should_take`` — see the note on ``gnubg_cube_action``. ``state``
    may still carry dice; :func:`encode_pre_roll` clears them, which is what the
    value head was trained on.
    """
    obs = encode_pre_roll(state, state.current_player())
    p6 = eval_probs6_batch(network, obs[None], device)[0]
    return cube_action(probs6_to_cumulative5(p6), cube_label, x, jacoby=jacoby)


def select_move_cubeful(
    state: pyspiel.BackgammonState, network, device,
    cube: CubeState, jacoby: bool = False, joint_doubles: bool = True,
) -> tuple[int, float]:
    """Choose a move by cubeful 0-ply lookahead. Returns ``(action, E(state))``.

    Greedy only — the cube path is a playing and scoring path, not a training
    one, so it has no use for the exploration temperature :func:`select_move`
    carries. ``x`` is read once from the root position via
    :func:`raccoon.cube.state.x_for`: race-ness does not flip inside a single
    turn, and a per-root constant keeps this the same cost as the cubeless path.
    """
    view = GameState(state).board_from_perspective()
    legal, cv, e_state = child_cubeful_values(
        state, network, device, cube.label_for(state.current_player()),
        x_for(view), jacoby, joint_doubles,
    )
    return legal[int(np.argmax(cv))], e_state


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
