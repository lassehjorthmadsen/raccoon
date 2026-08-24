"""Tests for the Janowski cube model.

The strongest available oracle is Janowski's own paper: it works one position
end to end and tabulates every cube action point across a grid of W and L, so
the numbers below are the published ones rather than values this implementation
produced. Where the paper is silent, the cross-checks come from what GNU
Backgammon reports for the same position.
"""

import pytest

from raccoon.cube import janowski as J

# The Reno Masters position worked through on pp. 6-7 of the paper, from the
# side considering the take: Black wins 65% (17% gammon, 1% backgammon), White
# wins 35% (4% gammon). W and L below are White's.
RENO_W = 1.1142857142857143
RENO_L = 1.2923076923076924
RENO_PROBS_BLACK = (0.65, 0.18, 0.01, 0.04, 0.0)


def test_reno_wl_matches_paper():
    W, L = J.compute_wl(RENO_PROBS_BLACK)
    assert W == pytest.approx(1.292, abs=5e-4)   # paper: W = 1.292 for Black
    assert L == pytest.approx(1.114, abs=5e-4)   # paper: L = 1.114 for Black


@pytest.mark.parametrize("x,tp_paper,eq_paper", [
    (0.0, 0.3292, -0.500),      # dead cube
    (1.0, 0.2725, -0.636),      # live cube
    (2.0 / 3.0, 0.2892, -0.596),  # "normal" cube
])
def test_reno_take_points_match_paper(x, tp_paper, eq_paper):
    """Paper equations (1)-(4) on the worked example."""
    tp = J.take_point(RENO_W, RENO_L, x)
    assert tp == pytest.approx(tp_paper, abs=5e-4)
    # Equation (4): the cubeless equity of a take.
    assert tp * (RENO_W + RENO_L) - RENO_L == pytest.approx(eq_paper, abs=1e-3)


# --- Cubeless take-point tables 1a / 1b / 1c ------------------------------

@pytest.mark.parametrize("x,W,L,expected", [
    # Table 1a, dead cube (x = 0.0)
    (0.0, 1.00, 1.00, 0.250), (0.0, 2.00, 1.00, 0.167),
    (0.0, 1.00, 2.00, 0.500), (0.0, 2.00, 2.00, 0.375),
    # Table 1b, live cube (x = 1.0)
    (1.0, 1.00, 1.00, 0.200), (1.0, 2.00, 1.00, 0.143),
    (1.0, 1.00, 2.00, 0.429), (1.0, 2.00, 2.00, 0.333),
    # Table 1c, normal cube (x = 2/3)
    (2 / 3, 1.00, 1.00, 0.214), (2 / 3, 2.00, 1.00, 0.150),
    (2 / 3, 1.00, 2.00, 0.450), (2 / 3, 2.00, 2.00, 0.346),
])
def test_take_point_tables(x, W, L, expected):
    assert J.take_point(W, L, x) == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize("x,expected", [
    (0.0, -0.500),   # Table 2a: the dead-cube take equity is -0.5 everywhere
    (1.0, -0.600),   # Table 2b at W = L = 1.00
    (2 / 3, -0.571),  # Table 2c at W = L = 1.00
])
def test_take_equity_tables(x, expected):
    tp = J.take_point(1.0, 1.0, x)
    assert tp * 2.0 - 1.0 == pytest.approx(expected, abs=1e-3)


# --- Appendix 4 equity tables, all at W = L = 1.00 and x = 2/3 ------------

@pytest.mark.parametrize("fn,expected", [
    (J.beaver_point, -0.143),    # Table A2
    (J.redouble_point, 0.429),   # Table A7
    (J.cash_point, 0.571),       # Table A8
    (J.too_good_point, 0.714),   # Table A9
])
def test_appendix4_equity_tables(fn, expected):
    """Appendix 4 tabulates the *cubeless equity* at each action point."""
    p = fn(1.0, 1.0, 2 / 3)
    assert p * 2.0 - 1.0 == pytest.approx(expected, abs=1e-3)


def test_gammonless_live_take_point_is_the_continuous_model():
    """With no gammons a live cube gives the classic 20% take point."""
    assert J.take_point(1.0, 1.0, 1.0) == pytest.approx(0.20)


# --- The Jacoby branch ----------------------------------------------------
#
# These pin the structure the exp024 diagnostic rests on: the Jacoby rule
# changes only the two outer segments of the centred-cube curve, never the
# doubling window between the take point and the cash point.

def test_jacoby_leaves_the_doubling_window_untouched():
    W, L = 1.4, 1.2
    tp = J.take_point(W, L, 1.0)
    cp = J.cash_point(W, L, 1.0)
    for frac in (0.01, 0.25, 0.5, 0.75, 0.99):
        p = tp + frac * (cp - tp)
        on = J.money_live(W, L, p, J.CENTERED, jacoby=True)
        off = J.money_live(W, L, p, J.CENTERED, jacoby=False)
        assert on == pytest.approx(off), f"window differs at p={p}"


def test_jacoby_clamps_the_outer_segments():
    W, L = 1.4, 1.2
    tp = J.take_point(W, L, 1.0)
    cp = J.cash_point(W, L, 1.0)
    assert J.money_live(W, L, tp / 2, J.CENTERED, jacoby=True) == -1.0
    assert J.money_live(W, L, (cp + 1.0) / 2, J.CENTERED, jacoby=True) == 1.0
    # Without Jacoby the same points carry a too-good / too-weak premium.
    assert J.money_live(W, L, tp / 2, J.CENTERED, jacoby=False) < -1.0
    assert J.money_live(W, L, (cp + 1.0) / 2, J.CENTERED, jacoby=False) > 1.0


def test_dead_cube_term_switches_on_jacoby():
    """At x = 0 the model is purely the dead-cube equity."""
    p = (0.55, 0.20, 0.02, 0.10, 0.01)
    assert J.cl2cf_money(p, J.CENTERED, 0.0, jacoby=True) == pytest.approx(
        2.0 * p[0] - 1.0)
    assert J.cl2cf_money(p, J.CENTERED, 0.0, jacoby=False) == pytest.approx(
        J.cubeless_equity(p))


def test_live_cube_is_exactly_money_live():
    p = (0.55, 0.20, 0.02, 0.10, 0.01)
    W, L = J.compute_wl(p)
    for owner in (J.CENTERED, J.PLAYER, J.OPPONENT):
        assert J.cl2cf_money(p, owner, 1.0, jacoby=False) == pytest.approx(
            J.money_live(W, L, p[0], owner, jacoby=False))


def test_jacoby_only_active_while_the_cube_is_centred():
    assert J.jacoby_active(J.CENTERED, jacoby=True)
    assert not J.jacoby_active(J.PLAYER, jacoby=True)
    assert not J.jacoby_active(J.CENTERED, jacoby=False)


# --- Cube decisions -------------------------------------------------------

def test_cube_equities_shape_and_conventions():
    p = (0.55, 0.20, 0.02, 0.10, 0.01)
    eq = J.cube_equities(p, J.CENTERED, 0.68)
    assert eq.dp == 1.0
    # dt is already doubled, so it compares directly against dp.
    assert eq.dt == pytest.approx(
        2.0 * J.cl2cf_money(p, J.OPPONENT, 0.68, jacoby=False))


def test_cannot_decide_when_opponent_owns_the_cube():
    with pytest.raises(ValueError):
        J.cube_equities((0.5, 0.1, 0.0, 0.1, 0.0), J.OPPONENT, 0.68)


def test_nd_offset_shifts_only_the_no_double_line():
    p = (0.55, 0.20, 0.02, 0.10, 0.01)
    base = J.cube_equities(p, J.CENTERED, 0.68)
    bumped = J.cube_equities(p, J.CENTERED, 0.68, nd_offset=0.05)
    assert bumped.nd == pytest.approx(base.nd + 0.05)
    assert bumped.dt == base.dt and bumped.dp == base.dp


def test_huge_advantage_is_a_double_and_a_pass():
    p = (0.95, 0.40, 0.02, 0.0, 0.0)
    should_double, should_take = J.cube_action(p, J.CENTERED, 0.68)
    assert should_double and not should_take


def test_dead_even_position_is_no_double_and_a_take():
    p = (0.50, 0.12, 0.005, 0.12, 0.005)
    should_double, should_take = J.cube_action(p, J.CENTERED, 0.68)
    assert not should_double and should_take


# --- Probability-vector plumbing -----------------------------------------

def test_probs6_round_trips_against_the_existing_inverse():
    """The scripts already convert cumulative-5 -> 6; this is its inverse."""
    from scripts.relabel_2ply import _outcomes6_and_equity

    original = (0.6123, 0.2011, 0.0154, 0.0872, 0.0033)
    six, _ = _outcomes6_and_equity(original)
    assert J.probs6_to_cumulative5(six) == pytest.approx(original, abs=1e-6)


def test_cubeless_equity_matches_the_cumulative_shortcut():
    """win + wg + wbg - lose - lg - lbg is the same sum, as used in relabel_2ply."""
    p = (0.6123, 0.2011, 0.0154, 0.0872, 0.0033)
    win, wg, wbg, lg, lbg = p
    lose = 1.0 - win
    assert J.cubeless_equity(p) == pytest.approx(
        win + wg + wbg - lose - lg - lbg)


# --- Cross-check against a shipping engine --------------------------------

def test_matches_gnubg_reported_equities_for_the_opening():
    """GNUBG 2-ply on the position after an opening 21 slot.

    Tolerance is loose because the reference numbers are read off GNUBG's UI at
    a rounded ply evaluation, not computed from these exact probabilities; the
    point is that all three cube placements land in the right place.
    """
    p = (0.4990, 0.1443, 0.0097, 0.1428, 0.0075)
    assert J.cl2cf_money(p, J.PLAYER, 0.68) == pytest.approx(0.1722, abs=5e-3)
    assert J.cl2cf_money(p, J.CENTERED, 0.68) == pytest.approx(0.0036, abs=5e-3)
    assert 2.0 * J.cl2cf_money(p, J.OPPONENT, 0.68) == pytest.approx(
        -0.3373, abs=5e-3)
