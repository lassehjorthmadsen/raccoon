"""Tests for raccoon.cli.display."""

import pytest

from raccoon.cli.display import (
    compute_pips,
    format_analysis,
    format_legal_moves,
    format_result,
    render_board,
)
from raccoon.env.game_wrapper import GameWrapper
from raccoon.search.mcts import Analysis, Candidate, _advance_through_chance


@pytest.fixture
def initial_state():
    wrapper = GameWrapper()
    state = wrapper.new_game()
    state = _advance_through_chance(state)
    return state


def test_compute_pips_initial(initial_state):
    """Standard opening position has 167 pips for each player."""
    pips_x, pips_o = compute_pips(initial_state)
    assert pips_x == 167
    assert pips_o == 167


def test_render_board_initial_layout(initial_state):
    board = render_board(initial_state)
    lines = board.splitlines()

    # Borders with point numbers present
    assert any("13-14-15-16-17-18" in line for line in lines)
    assert any("12-11-10--9--8--7" in line for line in lines)
    assert any("-6--5--4--3--2--1" in line for line in lines)
    assert any("19-20-21-22-23-24" in line for line in lines)

    # BAR gutter on the middle row
    assert any("|BAR|" in line for line in lines)

    # Right-side labels
    assert any("O: Raccoon" in line for line in lines)
    assert any("X: You" in line for line in lines)

    # Pip count footer
    assert "Pip counts: O 167, X 167" in board

    # 15 checkers per side on the board area (excludes labels / borders)
    board_lines = [
        line for line in lines
        if line.lstrip().startswith("|") or line.lstrip().startswith("^")
        or line.lstrip().startswith("v")
    ]
    # Filter to just the cell area (strip off trailing info columns)
    cell_area = "\n".join(
        line.split("|")[1] + line.split("|")[3]
        for line in board_lines
        if line.count("|") >= 4 and "BAR" not in line
    )
    assert cell_area.count("X") == 15
    assert cell_area.count("O") == 15


def test_render_board_widths_consistent(initial_state):
    """All drawing rows (borders, stacks, middle) share a common inner width."""
    lines = render_board(initial_state).splitlines()
    drawing_rows = [
        line for line in lines
        if ("|" in line and "Pip counts" not in line)
    ]
    # Strip trailing info after the last '|'
    trimmed = [line[: line.rindex("|") + 1] for line in drawing_rows]
    widths = {len(line) for line in trimmed}
    assert len(widths) == 1, f"Inconsistent row widths: {widths}"


def test_format_legal_moves(initial_state):
    out = format_legal_moves(initial_state)
    assert "Legal moves:" in out
    # At least one numbered entry
    assert "[ 0]" in out


def test_format_analysis_sorted_and_marked(initial_state):
    """Top row should be prefixed with '*' and rows sorted by visits desc."""
    candidates = [
        Candidate(action=0, visits=5, visit_prob=0.25, prior=0.1, q_value=-0.02),
        Candidate(action=1, visits=10, visit_prob=0.5, prior=0.3, q_value=0.15),
        Candidate(action=2, visits=5, visit_prob=0.25, prior=0.2, q_value=0.08),
    ]
    # Caller is responsible for sorting, but Analysis from analyze() is sorted;
    # here we pre-sort to mirror that contract.
    candidates.sort(key=lambda c: c.visits, reverse=True)
    analysis = Analysis(candidates=candidates, root_value=0.12, num_simulations=20)

    out = format_analysis(initial_state, analysis, top_n=3)
    lines = out.splitlines()
    # Header + 3 rows
    assert len(lines) == 4
    # Best row marked
    assert lines[1].lstrip().startswith("*1.")
    assert lines[2].lstrip().startswith("2.")
    # Equity shown as signed float
    assert "+0.150" in lines[1]
    # Diff column present on non-best row
    assert "(-0.170)" in lines[2]


def test_format_analysis_empty(initial_state):
    analysis = Analysis(candidates=[], root_value=0.0, num_simulations=0)
    assert format_analysis(initial_state, analysis) == "(no candidates)"


def test_render_board_turn_arrow(initial_state):
    """Middle (BAR) row starts with `^` or `v` indicating side to move."""
    board = render_board(initial_state)
    lines = board.splitlines()
    middle = [line for line in lines if "|BAR|" in line][0]
    current = initial_state.current_player()
    expected = "^" if current == 0 else "v"
    assert middle.startswith(expected)


def test_format_result_terminal_only():
    """format_result asserts on terminal state."""
    wrapper = GameWrapper()
    state = wrapper.new_game()
    state = _advance_through_chance(state)
    with pytest.raises(AssertionError):
        format_result(state)
