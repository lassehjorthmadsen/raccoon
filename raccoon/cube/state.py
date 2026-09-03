"""Cube state for a money game, and the seat/label bridge the model needs.

:mod:`raccoon.cube.janowski` speaks in labels *relative to the player on roll*
-- ``CENTERED``, ``PLAYER`` (we own it), ``OPPONENT`` (they own it). A game loop
speaks in OpenSpiel seat indices, 0 and 1. Translating between the two is a
one-line operation that is silent when it is backwards, which is exactly the
kind of thing this project keeps in one place with a test on it (see the
perspective-flip note in :func:`raccoon.train.lookahead.encode_pre_roll` for the
same lesson learned the expensive way).

Money play only. There is no cube cap in the rules, so :data:`MAX_CUBE` is a
guard against a runaway loop rather than a rule -- reaching it means something is
wrong, not that the game is over.
"""

from __future__ import annotations

from dataclasses import dataclass

from raccoon.cube.janowski import CENTERED, OPPONENT, PLAYER, X_CONTACT, X_RACE

# A 1024-cube is a 10-double game. Real money games do not get there; a loop that
# does has a bug, and stopping at a finite number makes that visible.
MAX_CUBE = 1024


@dataclass(frozen=True)
class CubeState:
    """Where the cube is and what it is worth. Immutable; transitions return new ones."""

    value: int = 1
    owner: int | None = None   # seat index of the owner, None while centred

    def label_for(self, mover: int) -> str:
        """This cube from ``mover``'s point of view, as janowski labels it."""
        if self.owner is None:
            return CENTERED
        return PLAYER if self.owner == mover else OPPONENT

    def may_double(self, mover: int) -> bool:
        """Whether ``mover`` is allowed to turn the cube."""
        return (self.owner is None or self.owner == mover) and self.value < MAX_CUBE

    def after_double(self, doubler: int) -> CubeState:
        """The cube after ``doubler`` doubles and the opponent takes."""
        if not self.may_double(doubler):
            raise ValueError(f"seat {doubler} cannot double cube {self}")
        return CubeState(value=self.value * 2, owner=1 - doubler)


def flip_label(label: str) -> str:
    """The same cube seen from the other side of the board.

    Used wherever a value is negated to change perspective: negating the equity
    alone is not enough, because the *owner* is stated relative to whoever is on
    roll. Getting this wrong flips the cube's value to the wrong side and is
    invisible in the aggregate.
    """
    if label == PLAYER:
        return OPPONENT
    if label == OPPONENT:
        return PLAYER
    if label == CENTERED:
        return CENTERED
    raise ValueError(f"unknown cube owner label {label!r}")


def is_race(view) -> bool:
    """True if the two sides have fully passed each other, so no contact remains.

    ``view`` is a :class:`raccoon.env.game_wrapper.BoardView`, indexed from the
    current player's perspective. Moved here from ``scripts/error_profile.py``,
    which now imports it, so the cube's race/contact split and the error profile's
    phase split cannot drift apart.
    """
    if view.my_bar or view.opp_bar:
        return False
    my_max = max((i + 1 for i in range(24) if view.my_points[i] > 0), default=0)
    opp_min = min((25 - (i + 1) for i in range(24) if view.opp_points[i] > 0), default=25)
    return my_max < opp_min


def x_for(view) -> float:
    """The cube life index to use for this position.

    GNU Backgammon documents 0.68 for contact and 0.60 for short races, and
    exp024 recovered both constants from the BGSage rollouts without being told
    either (``docs/cube.qmd``). Nothing here is fitted.
    """
    return X_RACE if is_race(view) else X_CONTACT
