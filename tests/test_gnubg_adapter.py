"""Tests for the gnubg-nn adapter. Skipped if gnubg-nn isn't installed."""

import pytest

gnubg_nn = pytest.importorskip("gnubg_nn")

from raccoon.env.game_wrapper import GameWrapper
from raccoon.eval import gnubg_adapter
from raccoon.search.mcts import _advance_through_chance


def test_board_from_view_matches_reference_starting_position():
    """BoardView at game start should serialise to the standard gnubg board."""
    wrapper = GameWrapper()
    state = wrapper.new_game()
    state = _advance_through_chance(state)
    view = state.board_from_perspective()

    our_board = gnubg_adapter.board_from_view(view)
    ref_board = gnubg_nn.board_from_position_id("4HPwATDgc/ABMA")

    assert our_board == ref_board


def test_evaluate_equity_starting_position_is_small_positive():
    """Side to move at game start has a small positive equity (~0.068 at ply 0)."""
    board = gnubg_nn.board_from_position_id("4HPwATDgc/ABMA")
    eq = gnubg_adapter.evaluate_equity(board, ply=0)
    assert 0.0 < eq < 0.2


def test_pick_move_returns_legal_action_from_start():
    wrapper = GameWrapper()
    state = wrapper.new_game()
    state = _advance_through_chance(state)

    action = gnubg_adapter.pick_move(state, ply=0)
    assert action in state.legal_actions()


def test_pick_move_plays_canonical_openings():
    """Regression for gnubg-nn slot-1 = on-roll perspective bug.

    Symmetric tests can't catch a flipped perspective. This forces an
    asymmetric position by injecting specific opening rolls and checks that
    pick_move agrees with the textbook best play. If this regresses, the
    adapter is evaluating equities from the wrong side again.
    """
    import pyspiel
    from raccoon.env.game_wrapper import GameState

    game = pyspiel.load_game("backgammon")
    expected = {
        "51": "24/23 13/8",
        "31": "8/5 6/5",
        "65": "24/18/13",
        "42": "8/4 6/4",
        "61": "13/7 8/7",
    }
    for roll, want in expected.items():
        s = game.new_initial_state()
        for o, _ in s.chance_outcomes():
            label = s.action_to_string(s.current_player(), o)
            if "O starts" in label and f"roll: {roll}" in label:
                s.apply_action(o)
                break
        action = gnubg_adapter.pick_move(GameState(s), ply=0)
        body = s.action_to_string(s.current_player(), action).split(" - ", 1)[1]
        assert body == want, f"opening {roll}: expected {want}, got {body}"


def test_pick_move_handles_doubles_like_gnubg_native():
    """Regression for the OpenSpiel doubles-split bug.

    OpenSpiel splits a doubles roll into two consecutive half-turns by the
    same player with no intervening chance node. A naive ``pick_move`` that
    always reads equity from the board after a single half-turn ends up
    minimising its own equity on half 1 (because the "child" still has the
    same player to move) and picks the *worst* half-1 option. The fix does
    a 2-half lookahead. This test asserts agreement with gnubg-nn's own
    ``best_move`` on a concrete doubles position.
    """
    import pyspiel
    from raccoon.env.game_wrapper import GameState

    game = pyspiel.load_game("backgammon")
    s = game.new_initial_state()

    # Seed: O wins opening roll 5-1, plays the book move 24/23 13/8.
    for o, _ in s.chance_outcomes():
        if "O starts" in s.action_to_string(s.current_player(), o) and "roll: 51" in s.action_to_string(s.current_player(), o):
            s.apply_action(o)
            break
    for a in s.legal_actions():
        if s.action_to_string(s.current_player(), a).split(" - ", 1)[1] == "24/23 13/8":
            s.apply_action(a)
            break

    # Find a doubles chance outcome (any will do); 3-3 is a common one.
    assert s.is_chance_node()
    picked = None
    for o, _ in s.chance_outcomes():
        if "roll: 33" in s.action_to_string(s.current_player(), o):
            picked = o
            break
    assert picked is not None
    s.apply_action(picked)

    # Compute the equity our pick_move associates with its chosen full turn.
    gs = GameState(s)
    _, ours_opp_equity = gnubg_adapter._best_action_and_opp_equity(gs, ply=0)
    # ``_best_action_and_opp_equity`` returns opponent-POV equity; flip it.
    ours_me_equity = -ours_opp_equity

    # Ask gnubg-nn for the best full 3-3 turn from the same starting board.
    start_board = gnubg_adapter.board_from_view(gs.board_from_perspective())
    _, candidates = gnubg_nn.best_move(
        pos=start_board, dice1=3, dice2=3, n=0, s=b"O",
        b=False, r=False, list=True, reduced=False,
    )
    native_top_eq = candidates[0][3]

    # A buggy adapter (minimising own equity on half 1) would land on the
    # *worst* legal full turn — ~30+ centi-equity below native best. The fix
    # should agree with gnubg native within rounding.
    assert abs(ours_me_equity - native_top_eq) < 5e-3, (
        f"pick_move doubles turn diverges from gnubg native best: "
        f"ours={ours_me_equity:+.4f} vs native={native_top_eq:+.4f}"
    )


def test_level_to_ply_mapping():
    assert gnubg_adapter.level_to_ply("beginner") == 0
    assert gnubg_adapter.level_to_ply("world") == 2
    assert gnubg_adapter.level_to_ply("World Class") == 2
    with pytest.raises(ValueError):
        gnubg_adapter.level_to_ply("super-duper")


# --- the cubeful side (exp025) -------------------------------------------------

def _mid_game_states(n=6):
    """A few real decision states, reached by letting GNUBG play itself at 0-ply."""
    import numpy as np

    np.random.seed(17)
    wrapper = GameWrapper()
    state = _advance_through_chance(wrapper.new_game())
    out = []
    while len(out) < n and not state.is_terminal():
        out.append(state.clone())
        state.apply_action(gnubg_adapter.pick_move(state, 0))
        state = _advance_through_chance(state)
    return out


def test_best_move_probs_reproduces_best_move_equity_exactly():
    """The two read the same ``probabilities`` call, one collapsed and one not,
    so the cubeless equity of the distribution must be the equity to the bit.
    That is what makes the cubeful control variate free rather than a second
    evaluator that could drift."""
    from raccoon.cube.janowski import cubeless_equity
    from raccoon.eval.luck import ROLL_KEYS

    for state in _mid_game_states(4):
        board = gnubg_adapter.board_from_view(state.board_from_perspective())
        for die1, die2 in ROLL_KEYS:
            eq = gnubg_adapter.best_move_equity(board, die1, die2, 0)
            probs = gnubg_adapter.best_move_probs(board, die1, die2, 0)
            assert cubeless_equity(probs) == pytest.approx(eq, abs=1e-9)


def test_best_move_probs_refuses_deeper_ply_like_its_sibling():
    board = gnubg_nn.board_from_position_id("4HPwATDgc/ABMA")
    with pytest.raises(ValueError):
        gnubg_adapter.best_move_probs(board, 3, 1, 1)


def test_invert_probs_is_an_involution_and_swaps_the_sides():
    from raccoon.cube.janowski import cubeless_equity

    board = gnubg_nn.board_from_position_id("4HPwATDgc/ABMA")
    probs = gnubg_adapter.outcome_probs(board, 0)
    flipped = gnubg_adapter._invert_probs(probs)
    assert cubeless_equity(flipped) == pytest.approx(-cubeless_equity(probs), abs=1e-12)
    assert gnubg_adapter._invert_probs(flipped) == pytest.approx(tuple(probs), abs=1e-12)


def test_dead_cube_ranking_matches_the_cubeless_ranking():
    """x = 0 makes Janowski the identity on cubeless equity, so GNUBG's cubeful
    candidate list must reproduce its cubeless one. Terminal children are the
    one exception: ``candidate_equities`` hard-codes +3.0 for them where the
    cubeful path takes the exact 1/2/3, so they are compared separately."""
    from raccoon.cube.janowski import CENTERED

    for state in _mid_game_states(6):
        cubeless = dict(gnubg_adapter.candidate_equities(state, 0))
        cubeful = dict(gnubg_adapter.candidate_cubeful_equities(
            state, 0, CENTERED, 0.0, False,
        ))
        assert cubeless.keys() == cubeful.keys()
        for action, eq in cubeless.items():
            if eq == 3.0:
                continue   # terminal shortcut, deliberately different
            assert cubeful[action] == pytest.approx(eq, abs=1e-9)


def test_gnubg_cube_action_does_not_double_the_opening_position():
    """A no-brainer sanity anchor: the opening position is nobody's double."""
    from raccoon.cube.janowski import CENTERED, X_CONTACT

    board = gnubg_nn.board_from_position_id("4HPwATDgc/ABMA")
    should_double, should_take = gnubg_adapter.gnubg_cube_action(
        board, 0, CENTERED, X_CONTACT, jacoby=False,
    )
    assert not should_double
    assert should_take


def test_pick_move_cubeful_returns_a_legal_action():
    from raccoon.cube.janowski import CENTERED, PLAYER, X_CONTACT

    for state in _mid_game_states(3):
        for label in (CENTERED, PLAYER):
            action = gnubg_adapter.pick_move_cubeful(state, 0, label, X_CONTACT, False)
            assert action in state.legal_actions()
