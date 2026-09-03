"""Players a cubeful money match can seat, behind one interface.

A cubeful game asks each side two questions — *what do you play* and *what do
you do with the cube* — and the arena in :mod:`raccoon.eval.cube_arena` should
not care who answers them. This module holds the three answerers Raccoon has:

* :class:`NetOpponent` — the shipped net, ranking afterstates by Janowski
  cubeful equity (:func:`raccoon.train.lookahead.child_cubeful_values`) and
  deciding the cube with the same closed form.
* :class:`GnubgCliOpponent` — **the real GNU Backgammon binary**, whose cube
  decisions come out of a cubeful search that decides the cube at every node.
  This is the honest opponent for ``goal.md``.
* :class:`GnubgNnOpponent` — a **stand-in**, kept only so the exp025 estimator
  work stays reproducible. The ``gnubg-nn`` package cannot make a money cube
  decision (``evaluate_cube_decision`` raises *Not implemented for money*) and
  ranks checker plays cubelessly, so its cubeful game is our Janowski layer
  bolted onto its probabilities. Both sides then share one cube model, which is
  weaker than real GNUBG on exactly the dimension a cube experiment tests. Do
  not report a strength result against it.

**Turns, not actions.** OpenSpiel splits a doubles roll into two consecutive
decisions by the same player; GNUBG answers for the whole turn at once. So the
interface is a *sequence* of actions for one turn, which both kinds can express
and which leaves the arena loop with nothing to reason about.

**Cube questions are asked on a pre-roll decision state** — the probe the arena
builds by applying one chance outcome — with the cube state as the doubler sees
it. Both halves of a cube decision are read off the same position: Janowski
derives the take from the doubler's three equities, so a receiver answers by
evaluating that same on-roll position with its own engine. That is what makes a
cube decision a real disagreement between two evaluations rather than one engine
deciding for both.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from raccoon.cube.janowski import CENTERED, OPPONENT, PLAYER
from raccoon.cube.state import CubeState, x_for
from raccoon.env.game_wrapper import GameState


class Opponent(Protocol):
    """One seat in a cubeful money game."""

    name: str

    def turn_actions(self, state: GameState, cube: CubeState) -> list[int]:
        """OpenSpiel actions for the whole of this turn, in order."""

    def wants_to_double(self, probe: GameState, cube: CubeState) -> bool:
        """Whether to turn the cube before rolling, at this pre-roll position."""

    def accepts_double(self, probe: GameState, cube: CubeState) -> bool:
        """Whether to take, evaluating the same on-roll position as the doubler."""


class NetOpponent:
    """Raccoon: cubeful 0-ply lookahead, Janowski cube."""

    def __init__(self, net, device, jacoby: bool = False,
                 joint_doubles: bool = True, name: str = "raccoon") -> None:
        self.net, self.device = net, device
        self.jacoby, self.joint_doubles = jacoby, joint_doubles
        self.name = name

    def turn_actions(self, state: GameState, cube: CubeState) -> list[int]:
        from raccoon.train.lookahead import child_cubeful_values

        mover = state.current_player()
        legal, values, _ = child_cubeful_values(
            state._state, self.net, self.device, cube.label_for(mover),
            x_for(state.board_from_perspective()), self.jacoby, self.joint_doubles,
        )
        # One action at a time: with joint_doubles the half-1 choice is already
        # scored by its best half-2 continuation, so the arena calling us again
        # for half 2 lands on that continuation.
        return [legal[int(np.argmax(values))]]

    def _cube_action(self, probe: GameState, cube: CubeState) -> tuple[bool, bool]:
        from raccoon.train.lookahead import net_cube_action

        return net_cube_action(
            probe._state, self.net, self.device,
            cube.label_for(probe.current_player()),
            x_for(probe.board_from_perspective()), self.jacoby,
        )

    def wants_to_double(self, probe: GameState, cube: CubeState) -> bool:
        return self._cube_action(probe, cube)[0]

    def accepts_double(self, probe: GameState, cube: CubeState) -> bool:
        return self._cube_action(probe, cube)[1]


class GnubgCliOpponent:
    """The real ``gnubg`` binary, with its own cubeful search.

    Holds one :class:`~raccoon.eval.gnubg_cli.GnubgCli` subprocess. Every move it
    returns is put through :func:`~raccoon.eval.gnubg_cli.actions_reaching`, which
    refuses a board no legal turn reaches — so a notation-parsing slip stops the
    run instead of quietly playing a different game.
    """

    def __init__(self, ply: int = 2, jacoby: bool = False,
                 name: str = "gnubg-cli") -> None:
        from raccoon.eval.gnubg_cli import GnubgCli

        self.cli = GnubgCli(ply=ply, jacoby=jacoby)
        self.ply = ply
        self.name = f"{name}-{ply}ply"

    @staticmethod
    def _owner(cube: CubeState, mover: int) -> str:
        """Cube owner in the CLI's seating, where the mover is always player 1."""
        from raccoon.eval.gnubg_cli import CENTRE

        label = cube.label_for(mover)
        return {CENTERED: CENTRE, PLAYER: "1", OPPONENT: "0"}[label]

    def turn_actions(self, state: GameState, cube: CubeState) -> list[int]:
        from raccoon.eval.gnubg_adapter import board_from_view
        from raccoon.eval.gnubg_cli import actions_reaching, target_board

        mover = state.current_player()
        view = state.board_from_perspective()
        board = board_from_view(view)
        notation = self.cli.best_move(
            board, view.dice, cube.value, self._owner(cube, mover),
        )
        if not notation:
            # A dance: OpenSpiel still requires the forfeit action to be played.
            return [state.legal_actions()[0]]
        return actions_reaching(state, target_board(board, notation))

    def _analysis(self, probe: GameState, cube: CubeState):
        from raccoon.eval.gnubg_adapter import board_from_view

        mover = probe.current_player()
        return self.cli.cube_analysis(
            board_from_view(probe.board_from_perspective()),
            cube.value, self._owner(cube, mover),
        )

    def wants_to_double(self, probe: GameState, cube: CubeState) -> bool:
        return self._analysis(probe, cube).should_double

    def accepts_double(self, probe: GameState, cube: CubeState) -> bool:
        return self._analysis(probe, cube).should_take

    def close(self) -> None:
        self.cli.close()


class GnubgNnOpponent:
    """The ``gnubg-nn`` package with our Janowski layer bolted on — a STAND-IN.

    See the module docstring: this cannot make a money cube decision of its own,
    so a match against it compares evaluation and checker play with one shared
    cube model. Kept for reproducing exp025's estimator measurements only.
    """

    def __init__(self, ply: int = 2, jacoby: bool = False,
                 name: str = "gnubg-nn-standin") -> None:
        self.ply, self.jacoby = ply, jacoby
        self.name = f"{name}-{ply}ply"

    def turn_actions(self, state: GameState, cube: CubeState) -> list[int]:
        from raccoon.eval.gnubg_adapter import pick_move_cubeful

        mover = state.current_player()
        return [pick_move_cubeful(
            state, self.ply, cube.label_for(mover),
            x_for(state.board_from_perspective()), self.jacoby,
        )]

    def _cube_action(self, probe: GameState, cube: CubeState) -> tuple[bool, bool]:
        from raccoon.eval.gnubg_adapter import board_from_view, gnubg_cube_action

        mover = probe.current_player()
        return gnubg_cube_action(
            board_from_view(probe.board_from_perspective()), self.ply,
            cube.label_for(mover), x_for(probe.board_from_perspective()),
            self.jacoby,
        )

    def wants_to_double(self, probe: GameState, cube: CubeState) -> bool:
        return self._cube_action(probe, cube)[0]

    def accepts_double(self, probe: GameState, cube: CubeState) -> bool:
        return self._cube_action(probe, cube)[1]
