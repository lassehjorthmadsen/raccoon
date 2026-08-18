"""Filtered expectimax search (exp021).

The search runs on gnubg-nn's 2x25 boards rather than OpenSpiel states, because
the BGSage benchmark stores raw boards and OpenSpiel cannot build a state from
one. That buys speed and costs us OpenSpiel's guarantees, so the gates here are
about equivalence rather than plausibility:

* :func:`test_move_generation_matches_openspiel` — gnubg's whole-turn enumeration
  produces exactly OpenSpiel's set of resulting positions, doubles and bar and
  bear-off included. This is the test that would catch a must-use-both-dice or
  play-the-larger-die discrepancy.
* :func:`test_depth0_matches_lookahead` — at depth 0 the new path reproduces the
  shipped 0-ply ranking, so the whole board/encoding chain is pinned to code that
  already scored PR 0.9498.
* :func:`test_search_matches_naive_reference` — the batched, deduplicating,
  filter-chained implementation equals an independent textbook expectimax.

Like ``test_doubles.py`` these use a hash-based value function: an untrained
ResNet is nearly constant, which would make ordering assertions vacuous.
"""
import numpy as np
import pyspiel
import pytest
import torch

pytest.importorskip("gnubg_nn")

from raccoon.env.game_wrapper import GameState
from raccoon.eval.gnubg_adapter import board_from_view, board_to_view
from raccoon.search.expectimax import (
    ROLLS, SearchConfig, _children, _Evaluator, _terminal_value,
    board26_to_slots, gate_skips, pass_turn, search_values,
)
from raccoon.train.lookahead import child_values

CPU = torch.device("cpu")


class _HashNet:
    """Deterministic pseudo-random value function, row-wise so batching is inert.

    An untrained ResNet is nearly constant, which would make ordering assertions
    vacuous. The projection weights are random rather than ``arange``-like: evenly
    spaced integer weights alias badly through a sine (moving a checker shifts the
    sum by a fixed step, and some steps land near a multiple of 2*pi), which made
    distinct positions collide on the same value.
    """

    def __init__(self, seed: int = 0, size: int = 26 * 2 * 12):
        generator = torch.Generator().manual_seed(seed)
        self.weights = torch.randn(size, generator=generator, dtype=torch.float64)

    def value_equity(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.shape[0], -1).double()
        return torch.sin(flat @ self.weights[:flat.shape[1]]).float()


def _random_states(n, seed=0, skip=0):
    """Whole-turn decision states from uniform-random playouts.

    Mid-doubles states (half 1 already played) are excluded: there OpenSpiel owes
    two more half-moves of a known die while gnubg would enumerate a fresh turn,
    so the two are answering different questions. Whole-turn doubles decisions are
    kept, and are where the enumeration is hardest.
    """
    game = pyspiel.load_game("backgammon(scoring_type=full_scoring)")
    rng = np.random.default_rng(seed)
    out, seen = [], 0
    while len(out) < n:
        # Play through GameState, not the raw state: mid_doubles is derived from
        # the previous decision player, which only the wrapper tracks.
        gs = GameState(game.new_initial_state())
        while not gs.is_terminal() and len(out) < n:
            if gs.is_chance_node():
                actions, probs = zip(*gs.chance_outcomes())
                gs.apply_action(int(rng.choice(actions, p=probs)))
                continue
            if not gs.board_from_perspective().mid_doubles:
                seen += 1
                if seen > skip:
                    out.append(gs.clone())
            gs.apply_action(int(rng.choice(gs.legal_actions())))
    return out


def _openspiel_child_boards(state):
    """Distinct whole-turn results of ``state``, as gnubg boards, opponent on roll.

    OpenSpiel splits a doubles turn into two decisions by the same player, so a
    whole turn means recursing until the move passes to the opponent. Returns
    ``None`` if any continuation ends the game: a terminal OpenSpiel state has no
    current player and therefore no board to compare, and exact terminal scoring
    is covered by :func:`test_terminal_values_are_exact`.
    """
    me = state.current_player()
    state = state._state
    boards = set()

    def collect(node):
        if node.is_terminal():
            raise _GameEnded
        if node.is_chance_node():
            # No current player at a chance node; step through it (any outcome
            # leaves the board untouched) so the opponent is on roll.
            node = node.clone()
            node.apply_action(0)
        if node.current_player() != me:
            slots = board_from_view(GameState(node).board_from_perspective())
            boards.add((tuple(slots[0]), tuple(slots[1])))
            return
        for action in node.legal_actions():
            child = node.clone()
            child.apply_action(action)
            collect(child)

    try:
        for action in state.legal_actions():
            child = state.clone()
            child.apply_action(action)
            collect(child)
    except _GameEnded:
        return None
    return boards


class _GameEnded(Exception):
    """A continuation finished the game, so board comparison does not apply."""


def _state_to_board(state):
    """The gnubg board for a decision state, mover on roll."""
    return tuple(tuple(s) for s in board_from_view(state.board_from_perspective()))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_move_generation_matches_openspiel(seed):
    """gnubg's whole-turn enumeration equals OpenSpiel's, position for position."""
    checked = 0
    for state in _random_states(60, seed=seed):
        if state.is_terminal():
            continue
        dice = state.board_from_perspective().dice
        if dice is None:
            continue
        theirs = _openspiel_child_boards(state)
        if theirs is None:
            continue
        board = _state_to_board(state)
        ours = set(_children(board, *sorted(dice, reverse=True)))
        assert ours == theirs, (
            f"seed={seed} dice={dice}: {len(ours)} gnubg vs {len(theirs)} OpenSpiel"
        )
        checked += 1
    assert checked > 20, f"only {checked} positions actually compared"


def test_terminal_values_are_exact():
    """A finished game is scored exactly, never handed to the network."""
    game = pyspiel.load_game("backgammon(scoring_type=full_scoring)")
    rng = np.random.default_rng(3)
    found = 0
    for _ in range(200):
        gs = GameState(game.new_initial_state())
        while not gs.is_terminal():
            if gs.is_chance_node():
                actions, probs = zip(*gs.chance_outcomes())
                gs.apply_action(int(rng.choice(actions, p=probs)))
                continue
            me = gs.current_player()
            before = gs.clone()
            gs.apply_action(int(rng.choice(gs.legal_actions())))
            if gs.is_terminal():
                points = abs(gs.returns()[me])
                board = _state_to_board(before)
                terminals = [
                    v for d1, d2, _ in ROLLS for v in
                    [_terminal_value(c) for c in _children(board, d1, d2)] if v is not None
                ]
                if terminals:
                    # A finished game is a loss for whoever would be on roll:
                    # -1/3 plain, -2/3 gammon, -1 backgammon.
                    assert min(terminals) >= -1.0 and max(terminals) <= -1.0 / 3.0
                    assert points in (1.0, 2.0, 3.0)
                    found += 1
                break
        if found > 5:
            break
    assert found > 0, "no terminal position exercised"


def test_depth0_matches_lookahead():
    """Depth 0 reproduces the shipped 0-ply ranking on real decisions."""
    net = _HashNet()
    for state in _random_states(25, seed=7):
        legal, reference, _ = child_values(state._state, net, CPU)
        board = _state_to_board(state)
        dice = state.board_from_perspective().dice
        if dice is None:
            continue
        candidates = _children(board, *sorted(dice, reverse=True))
        values, _ = search_values(candidates, net, CPU, SearchConfig(depth=0))
        # Both rank the same whole-turn positions; compare the best value reached.
        assert values.max() == pytest.approx(reference.max(), abs=2e-6)


def _static(board, net, cache):
    """Unbatched single-position evaluation, the reference for the batched path."""
    if board not in cache:
        from raccoon.env.encoder import encode_state
        obs = encode_state(board_to_view([list(board[0]), list(board[1])]))
        with torch.no_grad():
            cache[board] = float(net.value_equity(torch.from_numpy(obs[None]).float())[0])
    return cache[board]


def _naive(board, depth, net, cache):
    """Textbook expectimax: no batching, no dedup, no filters, no leaf reuse.

    Models the same **greedy opponent** the real search uses: the reply is chosen
    on static value, and only that reply is searched deeper. Choosing on the deep
    value instead would be the full-width opponent, a different (more accurate,
    far costlier) algorithm — see docs/search.qmd.
    """
    if depth == 0:
        return _static(board, net, cache)
    total = 0.0
    for d1, d2, weight in ROLLS:
        children = _children(board, d1, d2)
        statics = [
            _terminal_value(c) if _terminal_value(c) is not None else _static(c, net, cache)
            for c in children
        ]
        pick = min(range(len(children)), key=lambda j: statics[j])
        terminal = _terminal_value(children[pick])
        if terminal is not None:
            value = terminal
        elif depth == 1:
            value = statics[pick]
        else:
            value = _naive(children[pick], depth - 1, net, cache)
        total += weight * (-value)
    return total


@pytest.mark.parametrize("depth", [1, 2])
def test_search_matches_naive_reference(depth):
    """The optimised search equals an independent reference implementation.

    Filters are opened wide (large k, no gate) so the two search the same tree —
    filtering is a deliberate approximation, tested separately.
    """
    net = _HashNet()
    cfg = SearchConfig(depth=depth, k=99, k2=99, threshold=9.9, gate=0.0)
    states = _random_states(6 if depth == 1 else 2, seed=11, skip=8)
    for state in states:
        dice = state.board_from_perspective().dice
        if dice is None:
            continue
        candidates = _children(_state_to_board(state), *sorted(dice, reverse=True))[:4]
        values, _ = search_values(candidates, net, CPU, cfg)
        expected = [
            -(_terminal_value(c) if _terminal_value(c) is not None
              else _naive(c, depth, net, {}))
            for c in candidates
        ]
        np.testing.assert_allclose(values, expected, atol=1e-9)


def test_batch_size_does_not_change_values():
    """Batch size is a cost knob and must not move a single number."""
    net = _HashNet()
    state = _random_states(1, seed=5, skip=12)[0]
    dice = state.board_from_perspective().dice
    candidates = _children(_state_to_board(state), *sorted(dice, reverse=True))
    runs = [
        search_values(candidates, net, CPU, SearchConfig(depth=1, batch_size=bs))[0]
        for bs in (1, 7, 512)
    ]
    np.testing.assert_array_equal(runs[0], runs[1])
    np.testing.assert_array_equal(runs[0], runs[2])


def test_dedup_actually_fires():
    """Transpositions collapse: fewer evaluations than raw leaves."""
    net = _HashNet()
    state = _random_states(1, seed=9, skip=14)[0]
    dice = state.board_from_perspective().dice
    candidates = _children(_state_to_board(state), *sorted(dice, reverse=True))
    _, evaluated = search_values(candidates, net, CPU, SearchConfig(depth=1, k=99, gate=0.0))
    raw = sum(
        len(_children(c, d1, d2)) for c in candidates for d1, d2, _ in ROLLS
    )
    assert evaluated < raw, f"no dedup: {evaluated} evaluated vs {raw} leaves"


def test_gate_skips_lopsided_decisions():
    """The gate fires only when the static pick is already clear."""
    assert gate_skips(np.array([0.5, 0.1, 0.0]), gate=0.08)
    assert not gate_skips(np.array([0.5, 0.49, 0.0]), gate=0.08)
    assert not gate_skips(np.array([0.5, 0.1]), gate=0.0)
    assert not gate_skips(np.array([0.5]), gate=0.08)


def test_gate_short_circuits_to_static_values():
    """A gated decision returns exactly the static values, unsearched."""
    net = _HashNet()
    state = _random_states(1, seed=21, skip=10)[0]
    dice = state.board_from_perspective().dice
    candidates = _children(_state_to_board(state), *sorted(dice, reverse=True))
    static, static_evals = search_values(candidates, net, CPU, SearchConfig(depth=0))
    # Any positive gap clears a hair-thin gate, so every decision short-circuits.
    gated, gated_evals = search_values(
        candidates, net, CPU, SearchConfig(depth=2, gate=1e-12)
    )
    np.testing.assert_array_equal(static, gated)
    assert gated_evals == static_evals


def test_board26_conversion_matches_benchmark_encoder():
    """Our 26-array conversion agrees with the scorer's existing route."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from eval_benchmark_pr import flip_board_to_opp, flipped_to_board_view
    from raccoon.env.encoder import encode_state

    rng = np.random.default_rng(0)
    for state in _random_states(10, seed=4):
        view = state.board_from_perspective()
        board26 = [int(view.opp_bar)]
        board26 += [int(view.my_points[i]) - int(view.opp_points[i]) for i in range(24)]
        board26 += [int(view.my_bar)]
        theirs = encode_state(flipped_to_board_view(flip_board_to_opp(board26)))
        ours = encode_state(board_to_view(
            [list(s) for s in pass_turn(board26_to_slots(board26))]
        ))
        np.testing.assert_array_equal(ours, theirs)
