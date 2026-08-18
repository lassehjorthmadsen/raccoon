"""Filtered expectimax search over backgammon positions (exp021).

The design study is ``docs/search.qmd``. In one paragraph: rank candidate moves
by pushing the evaluation *n* rolls deeper into the game tree instead of asking
the value head about the afterstate directly. Every chance node is expanded
**full width** over the 21 distinct rolls (doubles weight 1/36, non-doubles
2/36), the opponent replies greedily, and each tree level is evaluated in one
batched forward pass.

Ply numbering is GNUBG's, as everywhere in this project (see
``raccoon/train/lookahead.py``): ``depth=0`` is static evaluation of the
candidate afterstates — exactly what ``lookahead.child_values`` already does —
and each further ply adds one roll of lookahead.

**Boards are gnubg-nn's 2x25 layout, not OpenSpiel states.** The BGSage
benchmark stores raw boards and OpenSpiel cannot build a state from a board, so
the search generates moves with ``gnubg_nn.moves()``: whole-turn (all four
half-moves of a double), already deduplicated to distinct resulting positions,
~1.6 us per call. A board is ``[slot0, slot1]`` with **slot 1 on roll**; each
25-list holds that player's own points 1..24 then their bar, borne-off implicit.

Two conventions are easy to get backwards, and both are pinned by tests:

* ``gnubg_nn.moves()`` keys decode with the mover in **slot 1** (slot 0 is the
  untouched opponent). Advancing the turn therefore means *swapping* the slots,
  which :func:`_children` does exactly once.
* :func:`gnubg_adapter.terminal_equity_after_move` wants the just-moved player in
  **slot 0** — which is the orientation a child already has after that swap.

The value convention matches the rest of the codebase: equity/3 in [-1, 1] from
the point of view of the player **on roll**, so ``S(board)`` is what the position
is worth to whoever is about to roll, and a candidate afterstate is worth
``-S(child)`` to the player who created it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from raccoon.env.encoder import encode_state
from raccoon.eval.gnubg_adapter import board_to_view, terminal_equity_after_move

import gnubg_nn


# The 21 distinct rolls with their weights out of 36.
ROLLS: tuple[tuple[int, int, float], ...] = tuple(
    (d1, d2, 1.0 / 36.0 if d1 == d2 else 2.0 / 36.0)
    for d1 in range(1, 7)
    for d2 in range(d1, 7)
)

Board = tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class SearchConfig:
    """Search parameters. Frozen so it can be pickled to worker processes.

    ``depth`` and the two filter widths set strength; everything else is either
    a cost knob that cannot change the answer (``batch_size``) or the ``gate``,
    whose worst-case cost is bounded in advance (see :func:`gate_skips`).

    ``k`` candidates survive the static filter and are searched at ``depth - 1``;
    the best ``k2`` of those are then searched at the full ``depth`` (BGSage's
    iterative-deepening filter chain, MULTI-PLY.md section 8). ``k2`` is unused
    at ``depth <= 1``, where one pass answers the question.
    """

    depth: int = 1
    k: int = 8
    k2: int = 2
    threshold: float = 0.16
    gate: float = 0.08
    batch_size: int = 512

    def __post_init__(self) -> None:
        if self.depth < 0:
            raise ValueError(f"depth must be >= 0, got {self.depth}")
        if self.k < 1 or self.k2 < 1:
            raise ValueError(f"k and k2 must be >= 1, got k={self.k}, k2={self.k2}")

    def tag(self) -> str:
        """Short identifier for filenames and result records."""
        return f"d{self.depth}_k{self.k}_k2{self.k2}_t{self.threshold}_g{self.gate}"


def board26_to_slots(board26: list[int]) -> Board:
    """BGSage's 26-array (mover's POV) to a gnubg board with the mover on roll.

    The 26-array is ``[opp_bar, points 1..24 (mover positive), mover_bar]`` with
    point ``k`` numbered from the mover's side. The mover's own 25-list is those
    counts directly; the opponent's is the negative entries read from their end
    of the board, which is the same index reversal every other converter here
    applies.
    """
    me = tuple(max(0, board26[k]) for k in range(1, 25)) + (board26[25],)
    opp = tuple(max(0, -board26[25 - k]) for k in range(1, 25)) + (board26[0],)
    return (opp, me)


def pass_turn(board: Board) -> Board:
    """Hand the roll to the other player without moving a checker."""
    return (board[1], board[0])


def _children(board: Board, die1: int, die2: int) -> list[Board]:
    """Legal whole-turn continuations, each with the opponent on roll.

    A dance (no legal move) returns the single position that simply passes the
    turn, which keeps the caller's expectation over the 21 rolls complete.
    """
    keys = gnubg_nn.moves([list(board[0]), list(board[1])], die1, die2, 0)
    if not keys:
        return [pass_turn(board)]
    out = []
    for key in keys:
        after = gnubg_nn.board_from_position_key(key)
        # moves() leaves the mover in slot 1; swap so the opponent is on roll.
        out.append((tuple(after[1]), tuple(after[0])))
    return out


def _terminal_value(board: Board) -> float | None:
    """Value of a finished game to the player **on roll**, in [-1, 1].

    ``board`` has the player who just moved in slot 0, so a finished game is a
    loss for the side on roll: -1/3 for a plain loss, -2/3 gammon, -1 backgammon.
    """
    points = terminal_equity_after_move([list(board[0]), list(board[1])])
    return None if points is None else -points / 3.0


class _Evaluator:
    """Batched, deduplicating static evaluation of boards.

    The cache lives for one :func:`search_values` call and is what collapses the
    transpositions that different (candidate, roll) paths share — the same
    ``slot_of`` idiom ``lookahead.child_values`` uses at depth 0, widened to the
    whole tree.
    """

    def __init__(self, network, device, channels, batch_size: int):
        self.network = network
        self.device = device
        self.channels = channels
        self.batch_size = batch_size
        self.cache: dict[Board, float] = {}
        self.evaluated = 0

    @torch.no_grad()
    def values(self, boards: list[Board]) -> np.ndarray:
        """Static value of each board to the player on roll, in [-1, 1]."""
        todo = [b for b in dict.fromkeys(boards) if b not in self.cache]
        for start in range(0, len(todo), self.batch_size):
            chunk = todo[start:start + self.batch_size]
            obs = np.stack([
                encode_state(board_to_view([list(b[0]), list(b[1])]), channels=self.channels)
                for b in chunk
            ])
            x = torch.from_numpy(obs).float().to(self.device, non_blocking=True)
            out = self.network.value_equity(x).cpu().numpy()
            for board, value in zip(chunk, out):
                self.cache[board] = float(value)
        self.evaluated += len(todo)
        return np.array([self.cache[b] for b in boards], dtype=np.float64)


def _expand(boards: list[Board]) -> tuple[list[list[list[Board]]], list[list[list[float]]]]:
    """Per board, per roll: the child positions and any exact terminal values.

    Returns ``(children, terminals)`` where ``terminals[i][r][j]`` is the exact
    value of ``children[i][r][j]`` to the player on roll there, or ``None`` when
    the position needs evaluating.
    """
    children, terminals = [], []
    for board in boards:
        per_roll_children, per_roll_terminal = [], []
        for die1, die2, _ in ROLLS:
            kids = _children(board, die1, die2)
            per_roll_children.append(kids)
            per_roll_terminal.append([_terminal_value(k) for k in kids])
        children.append(per_roll_children)
        terminals.append(per_roll_terminal)
    return children, terminals


def _search(boards: list[Board], depth: int, ev: _Evaluator) -> np.ndarray:
    """Value of each board to the player on roll, searching ``depth`` plies.

    ``depth == 0`` is the static head. Deeper, the player on roll rolls each of
    the 21 dice in turn, picks the reply that is best for them (equivalently: the
    child worth least to their opponent), and the results are weighted-averaged.
    The reply is *chosen* on static value — the greedy opponent model GNUBG and
    BGSage both use — and only the chosen one is searched deeper. At
    ``depth == 1`` that static value **is** the answer for the sub-position, so
    the pick batch doubles as the leaf batch and no second pass runs.
    """
    if depth == 0:
        return ev.values(boards)

    children, terminals = _expand(boards)

    # One batch for every non-terminal child of every board and roll.
    to_eval: list[Board] = []
    for i, per_roll in enumerate(children):
        for r, kids in enumerate(per_roll):
            for j, kid in enumerate(kids):
                if terminals[i][r][j] is None:
                    to_eval.append(kid)
    ev.values(to_eval)

    def static_of(i: int, r: int, j: int) -> float:
        term = terminals[i][r][j]
        return term if term is not None else ev.cache[children[i][r][j]]

    # Best reply per (board, roll): the child worth least to the opponent.
    best: list[list[int]] = []
    for i, per_roll in enumerate(children):
        best.append([
            int(np.argmin([static_of(i, r, j) for j in range(len(kids))]))
            for r, kids in enumerate(per_roll)
        ])

    if depth == 1:
        deeper = None
    else:
        # Only the chosen replies are searched deeper, and terminals stop here.
        pending = [
            children[i][r][best[i][r]]
            for i in range(len(boards))
            for r in range(len(ROLLS))
            if terminals[i][r][best[i][r]] is None
        ]
        values = _search(pending, depth - 1, ev) if pending else np.array([])
        deeper = dict(zip(pending, values))

    out = np.zeros(len(boards), dtype=np.float64)
    for i in range(len(boards)):
        total = 0.0
        for r, (_, _, weight) in enumerate(ROLLS):
            j = best[i][r]
            term = terminals[i][r][j]
            if term is not None:
                child_value = term
            elif deeper is None:
                child_value = ev.cache[children[i][r][j]]
            else:
                child_value = deeper[children[i][r][j]]
            # The child's value is to *its* player on roll; negate for the mover.
            total += weight * (-child_value)
        out[i] = total
    return out


def gate_skips(static_values: np.ndarray, gate: float) -> bool:
    """Whether a decision is lopsided enough that searching cannot change it.

    ``static_values`` are the candidates' values to the *mover* (higher is
    better). When the best candidate leads the runner-up by more than ``gate``
    the search is skipped and the static pick stands. The cost of this is
    bounded in advance: on the exp018/ep22 benchmark dump, a gate of 0.08 skips
    26% of decisions while capping the PR sacrificed at 0.002.
    """
    if gate <= 0.0 or len(static_values) < 2:
        return False
    top2 = np.partition(static_values, -2)[-2:]
    return bool(abs(top2[1] - top2[0]) > gate)


def search_values(
    candidates: list[Board], network, device, cfg: SearchConfig, channels=None,
) -> tuple[np.ndarray, int]:
    """Rank one decision's candidate afterstates. Returns ``(values, n_evals)``.

    ``candidates`` are the positions *after* each legal move, with the opponent
    on roll. The returned values are from the **mover's** point of view in
    [-1, 1] (higher is better), directly comparable with the static values
    ``lookahead.child_values`` produces.

    Candidates that survive neither filter keep their static value, which is
    sound because the filters only ever drop moves that are already behind: a
    dropped move cannot win the argmax either way.
    """
    ev = _Evaluator(network, device, channels, cfg.batch_size)

    terminal = [_terminal_value(c) for c in candidates]
    live = [c for c, t in zip(candidates, terminal) if t is None]
    ev.values(live)

    static = np.array([
        -(t if t is not None else ev.cache[c])
        for c, t in zip(candidates, terminal)
    ], dtype=np.float64)

    if cfg.depth == 0 or gate_skips(static, cfg.gate):
        return static, ev.evaluated

    values = static.copy()
    order = np.argsort(-static)
    best = static[order[0]]
    keep = [i for i in order[:cfg.k] if best - static[i] <= cfg.threshold]
    # Terminal candidates are already exact; searching them would be wasted work.
    keep = [i for i in keep if terminal[i] is None]

    if not keep:
        return static, ev.evaluated

    first_depth = 1 if cfg.depth > 1 else cfg.depth
    searched = _search([candidates[i] for i in keep], first_depth, ev)
    for i, value in zip(keep, searched):
        values[i] = -value

    if cfg.depth > 1:
        # Filter chain: only the best few get the expensive full-depth pass.
        deep = sorted(keep, key=lambda i: -values[i])[:cfg.k2]
        searched = _search([candidates[i] for i in deep], cfg.depth, ev)
        for i, value in zip(deep, searched):
            values[i] = -value

    return values, ev.evaluated
