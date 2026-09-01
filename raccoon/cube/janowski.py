"""Janowski's cube model: cubeless probabilities -> cubeful equity.

A network evaluates a position cubelessly -- six outcome probabilities for a
game played to conclusion with no doubling cube. Under real rules that game
never happens, so the number has to be converted into a *cubeful* equity before
it can drive a double or a take. Every strong engine does that conversion with
the model in Rick Janowski, "Take-Points in Money Games" (reproduced at
https://bkgm.com/articles/Janowski/cubeformulae.pdf).

The model interpolates between two solvable extremes:

* the **dead cube**, where the cube can never be turned again, so the game is
  simply played out at the current stake, and
* the **live cube**, where every future double lands exactly on the opponent's
  take point.

Real positions sit between the two, at a fraction ``x`` -- the *cube life
index*, 0.0 for a dead cube and 1.0 for a perfectly live one. GNU Backgammon
documents 0.68 for contact positions and 0.60 for short races.

Two variants of the model exist and they are not interchangeable. The paper's
Appendix 1 gives a closed form for the cube-centred equity; GNU Backgammon, XG
and BGSage instead use a **piecewise-linear** live-cube equity (GNUBG's
``MoneyLive``), interpolating through the known points at the take point and
cash point. This module implements the piecewise-linear variant, because that is
what the engines we benchmark against actually ship -- see
``bgsage/cpp/src/cube.cpp: money_live`` and ``cl2cf_money``. The two agree on an
owned cube and diverge on a centred cube under the Jacoby rule.

Conventions used throughout, and worth stating because two of them are easy to
get backwards:

* **Probabilities are the cumulative 5-vector** GNUBG and BGSage use --
  ``(P(win), P(win gammon incl. bg), P(win bg), P(lose gammon incl. bg),
  P(lose bg))``. ``P(lose)`` is the implicit ``1 - probs[0]``. This is *not* the
  network's mutually-exclusive six-outcome order; ``probs6_to_cumulative5``
  bridges the two.
* **Equities are normalised to a cube value of 1**, from the on-roll player's
  point of view. The conversion is cube-value independent for money play, so a
  caller holding a 4-cube multiplies by 4 at the end rather than passing the
  cube value in.
* **The double/take equity is doubled.** ``equity_dt`` is ``2 x E_opponent`` so
  that it compares directly against ``equity_dp = 1.0``, matching how XG and
  GNUBG report a double/take line.
"""

from dataclasses import dataclass
from typing import Sequence

# Cube owner labels, matching the BGSage benchmark's ``cube_owner`` field.
CENTERED = "centered"
PLAYER = "player"
OPPONENT = "opponent"

# GNU Backgammon's documented cube life indices.
X_CONTACT = 0.68
X_RACE = 0.60


@dataclass(frozen=True)
class CubeEquities:
    """The three cubeful equities a cube decision compares, cube value 1.

    All from the player-on-roll's point of view, so the doubler maximises:
    double when ``min(dt, dp) > nd``, and the receiver takes when ``dt < dp``.
    """

    nd: float   # no double -- play on with the cube where it is
    dt: float   # double, opponent takes (already doubled: 2 x E_opponent)
    dp: float   # double, opponent passes -- always exactly 1.0


def compute_wl(probs: Sequence[float]) -> tuple[float, float]:
    """Average cubeless value of a win (W) and of a loss (L).

    Janowski's model compresses the whole gammon/backgammon distribution into
    these two numbers: given that you win, how many points on average; given
    that you lose, how many. A gammonless position has W = L = 1.

    Both are floored at 1, and not defensively -- rolled-out labels violate it in
    practice. Variance reduction produces corrected means rather than counts, so a
    rare outcome can come back very slightly negative; divided by a minute win
    probability that turns into a large negative W. Over the BGSage money
    benchmark this affects 550 rows, 10 of them with W < 0 and the worst at
    -5.29, which would put the take point and both envelope endpoints through the
    floor. A win cannot be worth less than a point, so clamping here repairs
    every consumer at once.
    """
    p_win = probs[0]
    W = 1.0 + (probs[1] + probs[2]) / p_win if p_win > 1e-7 else 1.0
    p_lose = 1.0 - p_win
    L = 1.0 + (probs[3] + probs[4]) / p_lose if p_lose > 1e-7 else 1.0
    return max(1.0, W), max(1.0, L)


def cubeless_equity(probs: Sequence[float]) -> float:
    """Expected points per game with no cube, from the on-roll player's POV."""
    p_win, p_wg, p_wbg, p_lg, p_lbg = probs
    return (
        (p_win - p_wg) + 2.0 * (p_wg - p_wbg) + 3.0 * p_wbg
        - ((1.0 - p_win) - p_lg) - 2.0 * (p_lg - p_lbg) - 3.0 * p_lbg
    )


def take_point(W: float, L: float, x: float) -> float:
    """Cubeless winning probability at which a take becomes correct.

    Janowski equation (3): the dead-cube take point (L - 0.5) / (W + L) with the
    cube-ownership bonus 0.5x added to the denominator. At x = 0 this is the
    familiar risk/(risk + gain); at x = 1 and no gammons it is the continuous
    model's 20%.
    """
    return _over_denominator(W, L, x, L - 0.5)


def cash_point(W: float, L: float, x: float) -> float:
    """Winning probability above which doubling the opponent out beats playing on."""
    return _over_denominator(W, L, x, L + 0.5 + 0.5 * x)


def too_good_point(W: float, L: float, x: float) -> float:
    """Winning probability above which playing on for a gammon beats cashing."""
    return _over_denominator(W, L, x, L + 1.0)


def redouble_point(W: float, L: float, x: float) -> float:
    """Winning probability at which redoubling an owned cube becomes correct."""
    return _over_denominator(W, L, x, L + x)


def beaver_point(W: float, L: float, x: float) -> float:
    """Winning probability below which a beaver is correct."""
    return _over_denominator(W, L, x, L)


def _over_denominator(W: float, L: float, x: float, numerator: float) -> float:
    """Every cube action point in the paper shares the denominator W + L + 0.5x."""
    denom = W + L + 0.5 * x
    if abs(denom) < 1e-12:
        return float("nan")
    return numerator / denom


def money_live(W: float, L: float, p_win: float, owner: str,
               jacoby: bool) -> float:
    """Live-cube (x = 1) money equity, piecewise-linear in the winning chance.

    The live-cube equity is pinned at four known points -- lose the cube-drop
    (-1) at the take point, win it (+1) at the cash point, and the raw cubeless
    values -L and +W at the extremes -- and interpolated linearly between them.
    Which segments exist depends on where the cube sits: an owned cube has no
    take point (the opponent cannot double), an opponent-owned cube has no cash
    point.

    Under the Jacoby rule with the cube still centred, gammons are worthless
    until someone turns it, so the two outer segments flatten to exactly +/-1:
    above the cash point you double and get passed, below the take point you get
    doubled and pass. There is no "too good" premium to be had. The middle
    segment -- the doubling window itself -- is *unchanged*, because doubling is
    precisely what re-activates gammons. That identity matters: it means the
    Jacoby rule cannot explain any modelling error that lives inside the window.
    """
    tp = take_point(W, L, 1.0)
    cp = cash_point(W, L, 1.0)
    p = p_win

    if owner == CENTERED:
        if p < tp:
            return -1.0 if jacoby else -L + (-1.0 + L) * p / tp
        if p < cp:
            return -1.0 + 2.0 * (p - tp) / (cp - tp)
        return 1.0 if jacoby else 1.0 + (W - 1.0) * (p - cp) / (1.0 - cp)

    if owner == PLAYER:
        # We own the cube: the opponent can never double us out, so there is no
        # take-point segment -- we run from -L all the way up to our cash point.
        if p < cp:
            return -L + (1.0 + L) * p / cp
        return 1.0 + (W - 1.0) * (p - cp) / (1.0 - cp)

    if owner == OPPONENT:
        # Opponent owns it: we can never cash, so there is no cash-point segment.
        if p < tp:
            return -L + (-1.0 + L) * p / tp
        return -1.0 + (W + 1.0) * (p - tp) / (1.0 - tp)

    raise ValueError(f"unknown cube owner {owner!r}")


def jacoby_active(cube_owner: str, jacoby: bool) -> bool:
    """Whether the Jacoby rule is currently suppressing gammons.

    Jacoby only bites while the cube has never been turned, which for money play
    is exactly while it is still centred. Once someone doubles, gammons count
    again for the rest of the game.
    """
    return jacoby and cube_owner == CENTERED


def cl2cf_money(probs: Sequence[float], owner: str, x: float,
                jacoby: bool = False) -> float:
    """Cubeless probabilities -> cubeful money equity, cube value 1.

    The cube life index ``x`` blends the two solvable extremes: a dead cube,
    where the game is played out at the current stake, and a live cube, where
    every double is perfectly timed.

    ``jacoby`` here is the *already-resolved* flag -- pass ``jacoby_active(...)``
    rather than the raw rule setting, since the rule stops applying the moment
    the cube is turned.
    """
    W, L = compute_wl(probs)
    # Dead cube under Jacoby: it stays centred forever, so gammons never count
    # and the game is worth 2p - 1. Otherwise the full cubeless equity applies.
    e_dead = (2.0 * probs[0] - 1.0) if jacoby else cubeless_equity(probs)
    e_live = money_live(W, L, probs[0], owner, jacoby)
    return e_dead * (1.0 - x) + e_live * x


def cube_equities(probs: Sequence[float], cube_owner: str, x: float,
                  jacoby: bool = True, nd_offset: float = 0.0,
                  x_dt: float | None = None) -> CubeEquities:
    """The three equities a money cube decision compares, cube value 1.

    Args:
        probs: cumulative 5-vector, on-roll player's POV.
        cube_owner: "centered" or "player" -- you cannot double a cube the
            opponent owns, so "opponent" is not a legal cube-decision state.
        x: cube life index.
        jacoby: whether the Jacoby rule is in force (resolved internally
            against the cube position).
        nd_offset: additive correction to the no-double equity. Exists so an
            experiment can test whether the model's error is a flat shift the
            cube life index could absorb; leave at 0.0 for plain Janowski.
        x_dt: cube life index for the double/take line, when it should differ
            from the one used for no-double. Janowski's Appendix 2 already
            allows the two sides of a decision their own index; defaults to
            ``x``, which is the textbook single-index model.

    Returns:
        CubeEquities(nd, dt, dp). ``dt`` is already doubled and ``dp`` is 1.0,
        so the three are directly comparable.
    """
    if cube_owner == OPPONENT:
        raise ValueError("cannot make a cube decision when the opponent owns the cube")

    nd = cl2cf_money(probs, cube_owner, x, jacoby_active(cube_owner, jacoby))
    # After a double the opponent owns the cube at twice the stake, and the cube
    # has been turned, so Jacoby no longer applies whatever it was before.
    dt = 2.0 * cl2cf_money(probs, OPPONENT, x if x_dt is None else x_dt,
                           jacoby=False)
    return CubeEquities(nd=nd + nd_offset, dt=dt, dp=1.0)


def cube_action(probs: Sequence[float], cube_owner: str, x: float,
                jacoby: bool = True, nd_offset: float = 0.0,
                x_dt: float | None = None) -> tuple[bool, bool]:
    """(should_double, should_take) under the model.

    The doubler prefers whichever of no-double and double is worth more, where
    doubling is worth ``min(dt, dp)`` because the opponent picks their own best
    reply. The receiver takes when doing so costs the doubler less than a pass.
    """
    eq = cube_equities(probs, cube_owner, x, jacoby, nd_offset, x_dt)
    should_double = min(eq.dt, eq.dp) > eq.nd
    should_take = eq.dt < eq.dp
    return should_double, should_take


def probs6_to_cumulative5(probs6: Sequence[float]) -> tuple[float, ...]:
    """Network six-outcome order -> the cumulative 5-vector used cube-side.

    The value head emits six *mutually exclusive* outcomes in the order
    ``[win, win_g, win_bg, lose, lose_g, lose_bg]``; every cube formula here and
    in GNUBG expects the *nested* 5-vector instead. The inverse of this mapping
    already exists twice in the scripts (``relabel_2ply.py`` and
    ``gen_gnubg_selfplay.py``) -- the duplication is known, not overlooked.
    """
    win_s, win_g, win_bg, lose_s, lose_g, lose_bg = probs6
    return (
        win_s + win_g + win_bg,   # P(win)
        win_g + win_bg,           # P(win gammon, incl. backgammon)
        win_bg,                   # P(win backgammon)
        lose_g + lose_bg,         # P(lose gammon, incl. backgammon)
        lose_bg,                  # P(lose backgammon)
    )
