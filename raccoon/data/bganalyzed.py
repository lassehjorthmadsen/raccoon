"""Parse GNU Backgammon text-export analysis files (bglab ``analyzed/`` dirs).

Each file is one game annotated by GNUBG at a fixed lookahead (2-ply / 4-ply).
For every checker decision the export gives the exact position (GNU Position
ID), the dice, the move actually played, and a ranked list of candidate moves
each with a cubeful equity and the full outcome-probability tuple:

    Move number 7:  Magic to play 44
     GNU Backgammon  Position ID: 4HPwATDgc/ABMA
     ...ascii board...                      O: Magic
                                            X: Lasse
    * Magic moves bar/21 24/20 13/9(2)
    Rolled 44:
    *    1. Cubeful 4-ply  bar/21 24/20 13/9(2)  Eq.: +0.223
           0.553 0.150 0.007 - 0.447 0.120 0.005
         2. Cubeful 4-ply  ...                   Eq.: +0.135 (-0.088)
           ...

This module is pure text parsing — it does not touch OpenSpiel. The move
strings are matched to action indices later by ``bgmatch_replay`` during
replay. The probability tuple is nested like wildbg
(``win >= win_g >= win_bg``) from the mover's perspective, so
``raccoon.data.wildbg.equity_from_wildbg`` converts it to money equity.

Player mapping: GNUBG labels the two players ``X`` and ``O`` in the board
header (``X: <name>`` / ``O: <name>``). OpenSpiel's player 0 is X and player 1
is O (matching ``game_wrapper``), so ``player_x``/``player_o`` give the
column→player mapping for replay.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


_MOVE_HEADER_RE = re.compile(r"^Move number (\d+):\s+(.+?)\s+to play\s+(\d+)\s*$")
_POSITION_ID_RE = re.compile(r"Position ID:\s+(\S+)")
_XO_RE = re.compile(r"\b([XO]):\s+(\S+)\s*$")
_PLAYED_MOVES_RE = re.compile(r"^\*\s+(.+?)\s+moves\s+(.+?)\s*$")
_PLAYED_CANNOT_RE = re.compile(r"^\*\s+(.+?)\s+cannot move\s*$")
_ROLLED_RE = re.compile(r"^Rolled (\d+):\s*$")
# Candidate: optional leading '*', rank '.', "Cubeful N-ply", move, "Eq.: value".
_CAND_RE = re.compile(
    r"^(\*?)\s*(\d+)\.\s+Cubeful\s+(\d+)-ply\s+(.+?)\s+Eq\.:\s+([+-][\d.]+)"
)
_TUPLE_RE = re.compile(
    r"^\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+-\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$"
)
_MATCH_LEN_RE = re.compile(r"(\d+) point match|match to (\d+) points")
# Lines that mark the end of game play (statistics sections, export footer).
_END_MARKERS = (
    "Game statistics", "Cube statistics", "Overall statistics",
    "Chequerplay rating", "Output generated",
)


@dataclass
class Candidate:
    """One ranked candidate move at a decision."""
    rank: int
    ply: int
    move_str: str
    probs: tuple[float, float, float, float, float, float]  # win wG wBG lose lG lBG
    is_played: bool = False


@dataclass
class AnalyzedDecision:
    """One checker decision: position, dice, played move, ranked candidates."""
    move_number: int
    player_name: str
    dice: tuple[int, int]
    position_id: str | None = None
    played_move_str: str | None = None   # None == cannot move (forfeit)
    candidates: list[Candidate] = field(default_factory=list)


@dataclass
class AnalyzedGame:
    """A parsed game: player↔X/O mapping and the ordered decisions."""
    player_x: str
    player_o: str
    match_length: int | None
    decisions: list[AnalyzedDecision]


def parse_analyzed(text: str) -> AnalyzedGame:
    """Parse one GNUBG analysis export into an ``AnalyzedGame``."""
    lines = text.splitlines()
    player_x = ""
    player_o = ""
    match_length: int | None = None
    decisions: list[AnalyzedDecision] = []
    cur: AnalyzedDecision | None = None
    in_rolled = False
    pending_cand: Candidate | None = None  # candidate awaiting its tuple line

    def _flush_pending() -> None:
        nonlocal pending_cand
        if pending_cand is not None and cur is not None:
            cur.candidates.append(pending_cand)
        pending_cand = None

    for raw in lines:
        line = raw.rstrip("\n")

        if any(m in line for m in _END_MARKERS):
            _flush_pending()
            in_rolled = False
            # End of play; ignore the rest (stats/footer).
            break

        if match_length is None:
            m = _MATCH_LEN_RE.search(line)
            if m:
                match_length = int(m.group(1) or m.group(2))

        # Capture X/O → name mapping (first occurrence of each).
        mxo = _XO_RE.search(line)
        if mxo:
            who, name = mxo.group(1), mxo.group(2)
            if who == "X" and not player_x:
                player_x = name
            elif who == "O" and not player_o:
                player_o = name

        m = _MOVE_HEADER_RE.match(line)
        if m:
            _flush_pending()
            in_rolled = False
            num = int(m.group(1))
            name = m.group(2).strip()
            dd = m.group(3)
            cur = AnalyzedDecision(
                move_number=num, player_name=name,
                dice=(int(dd[0]), int(dd[1])),
            )
            decisions.append(cur)
            continue

        if cur is None:
            continue

        m = _POSITION_ID_RE.search(line)
        if m and cur.position_id is None:
            cur.position_id = m.group(1)
            continue

        m = _PLAYED_CANNOT_RE.match(line)
        if m:
            cur.played_move_str = None  # forfeit
            continue
        m = _PLAYED_MOVES_RE.match(line)
        if m:
            cur.played_move_str = m.group(2).strip()
            continue

        if _ROLLED_RE.match(line):
            _flush_pending()
            in_rolled = True
            continue

        if in_rolled:
            m = _CAND_RE.match(line)
            if m:
                _flush_pending()
                pending_cand = Candidate(
                    rank=int(m.group(2)),
                    ply=int(m.group(3)),
                    move_str=m.group(4).strip(),
                    probs=(0, 0, 0, 0, 0, 0),
                    is_played=(m.group(1) == "*"),
                )
                continue
            m = _TUPLE_RE.match(line)
            if m and pending_cand is not None:
                pending_cand.probs = tuple(float(m.group(i)) for i in range(1, 7))
                _flush_pending()
                continue

    _flush_pending()
    return AnalyzedGame(
        player_x=player_x, player_o=player_o,
        match_length=match_length, decisions=decisions,
    )


def parse_analyzed_file(path: str | Path) -> AnalyzedGame:
    """Read a file and parse it into an ``AnalyzedGame``."""
    return parse_analyzed(Path(path).read_text(encoding="utf-8", errors="replace"))


def best_candidates(decision: AnalyzedDecision) -> list[Candidate]:
    """Return the candidates at the deepest ply present in this decision.

    GNUBG lists the strongest moves at full depth and may tail off to
    shallower ply for lower-ranked moves. We keep only the deepest ply so
    every kept candidate is equally well-evaluated.
    """
    if not decision.candidates:
        return []
    max_ply = max(c.ply for c in decision.candidates)
    return [c for c in decision.candidates if c.ply == max_ply]


def _main() -> None:
    """``python -m raccoon.data.bganalyzed <file>`` debug helper."""
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m raccoon.data.bganalyzed <analyzed_file>", file=sys.stderr)
        sys.exit(2)
    game = parse_analyzed_file(sys.argv[1])
    print(f"X={game.player_x!r} (player 0)  O={game.player_o!r} (player 1)  "
          f"match_length={game.match_length}")
    print(f"decisions: {len(game.decisions)}")
    n_forfeit = sum(1 for d in game.decisions if d.played_move_str is None)
    n_nocand = sum(1 for d in game.decisions if not d.candidates)
    print(f"  forfeits (cannot move): {n_forfeit}   no-candidate blocks: {n_nocand}")
    for d in game.decisions[:4]:
        deep = best_candidates(d)
        print(f"  #{d.move_number} {d.player_name} dice={d.dice} "
              f"pid={d.position_id} played={d.played_move_str!r} "
              f"{len(d.candidates)} cands (deepest ply {deep[0].ply if deep else '-'}):")
        for c in deep[:3]:
            star = "*" if c.is_played else " "
            print(f"     {star} r{c.rank} {c.move_str!r} probs={c.probs}")


if __name__ == "__main__":
    _main()
