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
