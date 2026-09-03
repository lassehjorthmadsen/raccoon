"""Net-vs-GNUBG **cubeful** money games, with a variance-reduced ppg estimator.

.. warning::

   **The opponent here is a stand-in, and a ppg out of this module is not a
   strength result against GNU Backgammon.** gnubg-nn cannot make a money cube
   decision at all, and it ranks checker plays cubelessly, so both halves of its
   cubeful game are reconstructed by bolting *our* Janowski layer onto its
   probabilities. Both sides then share one cube model — while real GNU
   Backgammon uses the closed form only as a leaf evaluator inside a cubeful
   search that makes cube decisions at every node. That makes this opponent
   weaker than real GNUBG precisely on the cube.

   What this module is genuinely for is the **estimator**: it is where the
   cube-scaled control variate was built and shown to stay unbiased with a live
   cube (exp025's supporting measurement, ``experiments/exp025-cube-ranking``).
   exp026 keeps the estimator and replaces the opponent with the real
   ``/usr/games/gnubg``, which does answer money cube decisions. Until then, do
   not quote a ppg from here.

The cube-aware sibling of :func:`raccoon.eval.vr_arena.gnubg_arena_vr`, and it
keeps that function's properties on purpose: seats alternate, dice come from the
global numpy RNG, and turning variance reduction off plays the *identical* games
from the same seed, because the control variate reads gnubg through a pure C path
and touches no RNG.

Three things are new.

**The cube is a layer above OpenSpiel.** OpenSpiel's backgammon has no cube, so
the state machine here carries a :class:`~raccoon.cube.state.CubeState` beside the
game state. A money cube decision happens at the top of a turn, *before* the roll
— which is an OpenSpiel chance node — so that is where the loop asks. The mover
and their board are recovered by cloning, applying one outcome and reading the
result: applying a chance action only sets the dice, and dice are not part of a
gnubg board. The **opening** chance node is skipped, because it picks the starter
as well as the roll and nobody is on roll before it.

**Both sides play cubefully, and both use the same cube model.** gnubg-nn cannot
answer a money cube decision at all (``evaluate_cube_decision`` raises ``Not
implemented for money``), so GNUBG's cube and cubeful checker play are
reconstructed from its own probabilities through the Janowski model exp024
measured — see :func:`raccoon.eval.gnubg_adapter.gnubg_cube_action`. The match
therefore compares evaluation and checker play, **not** cube models. Any number
out of this module has to be reported with that sentence attached.

**This is money without Jacoby**, per ``goal.md`` — ``jacoby=False`` throughout.
Beavers are not implemented.
"""

from __future__ import annotations

import numpy as np

from raccoon.cube.janowski import X_CONTACT
from raccoon.cube.state import CubeState, x_for
from raccoon.env.game_wrapper import GameWrapper
from raccoon.eval.luck import (
    OPENING_OUTCOMES, pre_roll_cubeful_values, pre_roll_values,
)

# Which control variate to subtract. Both are exactly unbiased; see the section
# comment in raccoon/eval/luck.py for why, and the exp025 pilot for which wins.
CV_CUBELESS = "cubeless"
CV_CUBEFUL = "cubeful"


def cubeful_match(
    player, opponent, games: int, cv_ply: int = 0,
    seed: int = 0, max_moves: int = 2000, vr: bool = True,
    cv: str = CV_CUBEFUL, jacoby: bool = False,
) -> dict:
    """Play ``games`` cubeful money games between two opponents.

    ``player`` is the side every number is reported from (normally the net);
    ``opponent`` is the other seat. Both satisfy
    :class:`raccoon.eval.opponents.Opponent`, so who is playing is not this
    function's business — which is what lets the same loop, and the same
    estimator, measure against the stand-in or against the real GNUBG binary.

    **The control variate stays gnubg-nn whoever the opponent is.** ``h`` only
    has to be a *fixed* function of (pre-roll state, roll) to keep the estimator
    unbiased; it does not have to be, and should not be, the opponent's own
    evaluator. Keeping it fixed across opponents also makes two matches
    measurable on one ruler.

    Returns ``{"games", "net_wins", "game_pts", "game_luck", "game_vr",
    "game_rolls", "game_cube", "game_ended_by_drop", "net_doubles",
    "net_takes", "net_doubles_offered", "net_takes_offered"}``.

    ``game_pts`` is the result in points from the net's POV, already multiplied
    by the cube; ``game_luck`` is the accumulated dice luck on the same scale;
    ``game_vr = game_pts - game_luck``. With ``vr=False`` the luck array is zero
    and ``game_vr`` equals ``game_pts``.

    ``cv_ply`` is the control variate's evaluator and must be 0 (gnubg-nn's
    ``best_move`` segfaults deeper). It is deliberately independent of how strong
    either seat is, which is what keeps the estimator unbiased whatever they do.

    Seeds the global numpy RNG (dice), like every other arena in this project.
    """
    if cv not in (CV_CUBELESS, CV_CUBEFUL):
        raise ValueError(f"cv must be {CV_CUBELESS!r} or {CV_CUBEFUL!r}, got {cv!r}")

    np.random.seed(seed)
    wrapper = GameWrapper()
    game_pts: list[float] = []
    game_luck: list[float] = []
    game_rolls: list[int] = []
    game_cube: list[int] = []
    ended_by_drop: list[bool] = []
    wins = 0
    # Cube diagnostics, pooled over the run: how often each side was offered the
    # decision and how often it acted. A run where the net never doubles is a
    # broken run that still produces a plausible-looking ppg, so these are not
    # optional colour.
    counts = {"net_doubles": 0, "net_doubles_offered": 0,
              "net_takes": 0, "net_takes_offered": 0}

    for g in range(games):
        net_player = 0 if g % 2 == 0 else 1   # `player`'s seat this game
        state = wrapper.new_game()
        cube = CubeState()
        luck_total = 0.0
        rolls = 0
        moves = 0
        dropped_by: int | None = None

        while not state.is_terminal() and moves < max_moves:
            if state.is_chance_node():
                is_opening = len(state.chance_outcomes()) == OPENING_OUTCOMES
                if not is_opening:
                    probe = state.clone()
                    probe.apply_action(state.chance_outcomes()[0][0])
                    mover = probe.current_player()
                    if cube.may_double(mover):
                        seats = {net_player: player, 1 - net_player: opponent}
                        if mover == net_player:
                            counts["net_doubles_offered"] += 1
                        doubles = seats[mover].wants_to_double(probe, cube)
                        if doubles:
                            if mover == net_player:
                                counts["net_doubles"] += 1
                            receiver = 1 - mover
                            if receiver == net_player:
                                counts["net_takes_offered"] += 1
                            takes = seats[receiver].accepts_double(probe, cube)
                            if takes:
                                if receiver == net_player:
                                    counts["net_takes"] += 1
                                cube = cube.after_double(mover)
                            else:
                                dropped_by = receiver
                                break

                if vr:
                    if cv == CV_CUBEFUL:
                        # The opening position is contact and the cube is centred
                        # there in every game, so the index is fixed; elsewhere it
                        # comes from the board we just probed.
                        x_cv = (X_CONTACT if is_opening
                                else x_for(probe.board_from_perspective()))
                        actions, probs, values = pre_roll_cubeful_values(
                            state, net_player, cube, x_cv, jacoby, cv_ply,
                        )
                    else:
                        actions, probs, values = pre_roll_values(
                            state, net_player, cv_ply,
                        )
                else:
                    outcomes = state.chance_outcomes()
                    actions = [a for a, _ in outcomes]
                    probs = np.array([p for _, p in outcomes], dtype=np.float64)

                idx = int(np.random.choice(len(actions), p=probs))
                if vr:
                    luck_total += cube.value * (
                        float(values[idx]) - float(probs @ values)
                    )
                state.apply_action(actions[idx])
                rolls += 1
                continue

            mover = state.current_player()
            seat = player if mover == net_player else opponent
            for action in seat.turn_actions(state, cube):
                state.apply_action(action)
                moves += 1

        if dropped_by is not None:
            # A pass ends the game at the current stake, before it was doubled.
            net_pts = float(cube.value) * (1.0 if dropped_by != net_player else -1.0)
        elif state.is_terminal():
            net_pts = state.returns()[net_player] * cube.value
        else:
            continue   # hit max_moves: no result to record

        game_pts.append(net_pts)
        game_luck.append(luck_total)
        game_rolls.append(rolls)
        game_cube.append(cube.value)
        ended_by_drop.append(dropped_by is not None)
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
        "game_cube": np.array(game_cube, dtype=np.int32),
        "game_ended_by_drop": np.array(ended_by_drop, dtype=bool),
        **counts,
    }


def gnubg_cube_arena(
    net, device, games: int, gnubg_ply: int = 2, cv_ply: int = 0,
    seed: int = 0, max_moves: int = 2000, vr: bool = True,
    cv: str = CV_CUBEFUL, joint_doubles: bool = True, jacoby: bool = False,
) -> dict:
    """The net against the gnubg-nn STAND-IN — exp025's estimator harness.

    Kept so exp025's ``results/pilot_cv_*.json`` stay reproducible from one
    command. See this module's warning: the opponent cannot make a money cube
    decision of its own, so a ppg from here is not a strength result. For a real
    opponent build a :class:`~raccoon.eval.opponents.GnubgCliOpponent` and call
    :func:`cubeful_match` directly.
    """
    from raccoon.eval.opponents import GnubgNnOpponent, NetOpponent

    return cubeful_match(
        NetOpponent(net, device, jacoby=jacoby, joint_doubles=joint_doubles),
        GnubgNnOpponent(ply=gnubg_ply, jacoby=jacoby),
        games, cv_ply=cv_ply, seed=seed, max_moves=max_moves, vr=vr, cv=cv,
        jacoby=jacoby,
    )
