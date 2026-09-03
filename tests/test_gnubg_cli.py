"""Driving the real GNU Backgammon binary as a cubeful decision oracle.

Skipped when the binary is not installed, so CI without it stays green.

The tests split into two kinds, and the distinction matters. The **notation**
tests are pure and exact: they pin how GNUBG's answers turn back into moves, and
they are what stands between a parsing slip and an engine that quietly plays a
different game for thousands of positions. The **oracle** tests talk to the real
binary and are deliberately loose about its numbers, because the binary and the
``gnubg-nn`` package ship different weights and are not the same evaluator.
"""

import os
import shutil

import numpy as np
import pytest

from raccoon.env.game_wrapper import GameWrapper
from raccoon.search.mcts import _advance_through_chance

gnubg_nn = pytest.importorskip("gnubg_nn")

from raccoon.cube.janowski import CENTERED
from raccoon.eval.gnubg_adapter import board_from_view, pick_move
from raccoon.eval.gnubg_cli import (
    BAR, CENTRE, DEFAULT_EXE, GnubgCli, GnubgProtocolError, OFF,
    actions_reaching, apply_move, parse_move, resting_boards, target_board,
)

HAS_GNUBG = os.path.exists(DEFAULT_EXE) or shutil.which("gnubg") is not None
needs_gnubg = pytest.mark.skipif(not HAS_GNUBG, reason="gnubg binary not installed")

START_ID = "4HPwATDgc/ABMA"


def _positions(n=8, seed=11):
    """Real, asymmetric positions from a GNUBG-vs-GNUBG game."""
    np.random.seed(seed)
    wrapper = GameWrapper()
    state = _advance_through_chance(wrapper.new_game())
    out = []
    while len(out) < n and not state.is_terminal():
        out.append(state.clone())
        state.apply_action(pick_move(state, 0))
        _advance_through_chance(state)
    return out


# --- notation: pure, exact ----------------------------------------------------

@pytest.mark.parametrize("notation,hops", [
    ("8/5 6/5", [(8, 5), (6, 5)]),
    ("24/13", [(24, 13)]),
    ("13/3(2)", [(13, 3), (13, 3)]),
    ("bar/20*", [(BAR, 20)]),
    ("13/off", [(13, OFF)]),
    ("24/18*/13", [(24, 18), (18, 13)]),
    ("13/11 13/11 6/4 6/4", [(13, 11), (13, 11), (6, 4), (6, 4)]),
    ("6/off 5/off", [(6, OFF), (5, OFF)]),
])
def test_parse_move_covers_every_shape_gnubg_emits(notation, hops):
    assert parse_move(notation) == hops


def test_a_chained_play_expands_to_one_hop_per_die():
    """`24/18*/13` must not collapse to a single 24->13 hop: the intermediate
    point is where the hit happens, and dropping it loses the hit."""
    assert parse_move("24/18*/13") == [(24, 18), (18, 13)]


def test_unparseable_notation_is_refused_rather_than_guessed():
    for bad in ("sideways", "8-5", "8/", ""):
        with pytest.raises(GnubgProtocolError):
            parse_move(bad) if bad else parse_move("8/")


def test_apply_move_reads_a_hit_off_the_board_not_off_the_asterisk():
    """The `*` marker is informational. A hop onto a lone opposing checker sends
    it to the bar whether or not GNUBG wrote the star, so the two can't disagree."""
    board = gnubg_nn.board_from_position_id(START_ID)
    board = [list(board[0]), list(board[1])]
    board[0][25 - 5 - 1] = 1          # a lone opponent blot on our 5-point
    before_bar = board[0][24]
    after = apply_move(board, parse_move("8/5"))   # no star in the notation
    assert after[0][25 - 5 - 1] == 0
    assert after[0][24] == before_bar + 1


def test_apply_move_refuses_to_land_on_a_made_point():
    board = gnubg_nn.board_from_position_id(START_ID)
    board = [list(board[0]), list(board[1])]
    board[0][25 - 5 - 1] = 2          # opponent owns our 5-point
    with pytest.raises(GnubgProtocolError):
        apply_move(board, parse_move("8/5"))


def test_apply_move_refuses_to_move_from_an_empty_point():
    board = gnubg_nn.board_from_position_id(START_ID)
    with pytest.raises(GnubgProtocolError):
        apply_move([list(board[0]), list(board[1])], parse_move("11/5"))


def test_resting_boards_covers_every_legal_turn_including_doubles():
    for state in _positions(6):
        view = state.board_from_perspective()
        boards = resting_boards(state)
        assert boards, "a decision with no reachable resting board"
        # A doubles turn needs two OpenSpiel actions; a normal one needs one.
        lengths = {len(seq) for seq in boards.values()}
        expected = {1, 2} if view.dice and view.dice[0] == view.dice[1] else {1}
        assert lengths <= expected


def test_actions_reaching_refuses_an_unreachable_board():
    """The safety net: a notation slip must fail loudly, not play something else."""
    state = _positions(1)[0]
    board = board_from_view(state.board_from_perspective())
    impossible = [list(board[0]), [0] * 25]
    with pytest.raises(GnubgProtocolError):
        actions_reaching(state, impossible)


# --- the oracle ---------------------------------------------------------------

@needs_gnubg
def test_position_id_round_trips_exactly():
    """The exact harness-correctness control.

    It compares an encoding against itself, so unlike a probability comparison it
    stays valid despite the binary and the package shipping different weights —
    and a perspective or seating bug shows up as a mismatched ID rather than as a
    plausible-looking equity.
    """
    with GnubgCli(ply=0) as cli:
        for state in _positions(8):
            board = board_from_view(state.board_from_perspective())
            assert cli.echo_position_id(board) == gnubg_nn.position_id(board)


@needs_gnubg
@pytest.mark.parametrize("dice,book", [
    ((3, 1), "8/5 6/5"),
    ((4, 2), "8/4 6/4"),
    ((6, 1), "13/7 8/7"),
])
def test_opening_rolls_get_their_book_plays(dice, book):
    """A semantic check that survives the weight difference: these three openers
    are not controversial in any GNUBG build."""
    with GnubgCli(ply=2) as cli:
        assert cli.best_move(gnubg_nn.board_from_position_id(START_ID), dice) == book


@needs_gnubg
def test_every_move_the_oracle_returns_is_playable():
    """The end-to-end contract: parse, apply, and find the matching legal turn.

    Runs whole games rather than sampled positions, so bar entries, hits,
    bear-offs and doubles all appear.
    """
    np.random.seed(3)
    wrapper = GameWrapper()
    applied = 0
    with GnubgCli(ply=0) as cli:
        for _ in range(2):
            state = _advance_through_chance(wrapper.new_game())
            guard = 0
            while not state.is_terminal() and guard < 400:
                guard += 1
                view = state.board_from_perspective()
                if state.current_player() == 0:
                    notation = cli.best_move(board_from_view(view), view.dice)
                    if not notation:
                        state.apply_action(state.legal_actions()[0])
                    else:
                        for action in actions_reaching(
                            state, target_board(board_from_view(view), notation)
                        ):
                            state.apply_action(action)
                        applied += 1
                else:
                    state.apply_action(pick_move(state, 0))
                _advance_through_chance(state)
        assert cli.restarts == 0
    assert applied > 30, "too few oracle turns to be a meaningful check"


@needs_gnubg
def test_cube_analysis_parses_and_orders_correctly():
    """Owning the cube cannot be worth less than it being centred, at one cube
    value. Nothing in the parse enforces that, so it is a real check."""
    with GnubgCli(ply=0) as cli:
        for state in _positions(4):
            board = board_from_view(state.board_from_perspective())
            centred = cli.cube_analysis(board, cube_value=1, cube_owner=CENTRE)
            owned = cli.cube_analysis(board, cube_value=1, cube_owner="1")
            assert owned.nd >= centred.nd - 1e-9
            assert centred.dp == pytest.approx(1.0)
            assert centred.action


@needs_gnubg
def test_the_binary_and_the_package_track_each_other_but_are_not_identical():
    """They ship different weights, so this is a sanity band, not an equality.

    A perspective or encoding bug would blow far past this tolerance; the genuine
    net difference sits around a couple of units in the third decimal.
    """
    with GnubgCli(ply=2) as cli:
        gaps = []
        for state in _positions(6):
            board = board_from_view(state.board_from_perspective())
            cli_p = cli.cubeless_probs(board)
            nn_p = gnubg_nn.probabilities(board, 2)
            gaps.append(max(abs(a - b) for a, b in zip(cli_p, nn_p)))
        assert max(gaps) < 0.02


@needs_gnubg
def test_a_dead_process_is_restarted_rather_than_ending_the_run():
    """GNUBG segfaults on malformed input; a 7,000-game run must survive it."""
    with GnubgCli(ply=0) as cli:
        board = gnubg_nn.board_from_position_id(START_ID)
        assert cli.best_move(board, (3, 1)) == "8/5 6/5"
        cli._proc.kill()
        cli._proc.wait(timeout=5)
        assert cli.best_move(board, (3, 1)) == "8/5 6/5"
        assert cli.restarts == 1


# --- a whole cubeful match ------------------------------------------------------

@needs_gnubg
def test_a_cubeful_match_against_the_real_binary_plays_through():
    """End-to-end: the oracle seated in the arena, cube live, Jacoby off.

    Deliberately says nothing about who wins — ten games measures nothing, and
    the point of the check is that every turn, cube decision and settlement
    survives the round trip through GNUBG's notation.
    """
    import torch

    from raccoon.eval.cube_arena import cubeful_match
    from raccoon.eval.opponents import GnubgCliOpponent, NetOpponent
    from raccoon.model.network import RaccoonNet

    torch.manual_seed(5)
    net = RaccoonNet(channels=16, num_blocks=1, value_head="outcomes6")
    net.eval()
    opp = GnubgCliOpponent(ply=0)
    try:
        r = cubeful_match(NetOpponent(net, torch.device("cpu")), opp,
                          games=4, seed=2)
    finally:
        opp.close()

    assert r["games"] == 4
    assert opp.cli.restarts == 0, "GNUBG needed restarting during a clean match"
    # Settlement arithmetic, same contract as tests/test_cube_arena.py.
    dropped = r["game_ended_by_drop"]
    if dropped.any():
        np.testing.assert_array_equal(
            np.abs(r["game_pts"][dropped]), r["game_cube"][dropped])
    played = ~dropped
    if played.any():
        ratio = np.abs(r["game_pts"][played]) / r["game_cube"][played]
        assert set(np.unique(ratio)) <= {1.0, 2.0, 3.0}
    cubes = r["game_cube"]
    assert np.all((cubes & (cubes - 1)) == 0), "cube held a non-power of two"


@needs_gnubg
def test_the_binary_prices_the_cube_differently_from_our_closed_form():
    """The whole reason exp026 exists, as an assertion.

    Our cube equity is Janowski applied once at the root; GNUBG's comes out of a
    cubeful search that decides the cube at every node. This guards that the
    harness actually reaches that search rather than bouncing back something
    equivalent to our own formula. Comparing the *equities* rather than the
    double/take verdicts is what makes it bite: on quiet early positions both
    correctly say "no double", so agreement on the verdict proves nothing.

    Note what the numbers say, because it bears on how big the exp026 effect can
    be: on these eight ordinary early-game positions the two agree to within
    about 0.006 of a point, and usually far closer. The closed form is a good
    approximation of the search where the cube is dead; the place it is expected
    to part company is the volatile run-up to a doubling window, which is a
    measurement for the experiment rather than for a unit test.
    """
    from raccoon.cube.janowski import X_CONTACT, cube_equities
    from raccoon.eval.gnubg_adapter import outcome_probs

    with GnubgCli(ply=0) as cli:
        gaps = []
        for state in _positions(8):
            board = board_from_view(state.board_from_perspective())
            theirs = cli.cube_analysis(board, cube_value=1, cube_owner=CENTRE)
            ours = cube_equities(
                outcome_probs(board, 0), CENTERED, X_CONTACT, jacoby=False,
            )
            gaps.append(abs(theirs.nd - ours.nd))

    assert max(gaps) > 1e-3, (
        "GNUBG's cubeful search and our closed form priced every position "
        "identically -- either the harness is not reaching the search, or there "
        "is nothing for exp026 to measure"
    )
