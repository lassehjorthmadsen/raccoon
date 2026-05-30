"""Parse Backgammon Studio / Backgammon Galaxy match files into structured data.

Match file format (one match per file):

    ; [Site "BackgammonGalaxy"]
    ; [Match ID "..."]
    ...

    5 point match

     Game 1
     lasse : 0                       mglough : 0
      1)                             51: 24/23 13/8
      2) 21: 13/11 24/23             22: 6/4 6/4 13/11 13/11
      ...
      8)  Doubles => 2                Drops
          Wins 1 point

     Game 2
     ...

Two columns, one per player. Either column may be empty on a given turn
(opening roll where only one player moves, or a forfeit "DD:" with no
checkers movable). Cube actions ("Doubles => N", "Takes", "Drops") and
game-end annotations ("Wins N point") are skipped — we only emit checker
decisions.

Move notation per turn: ``DD: from/to from/to ...`` where DD is the two
dice digits, ``from`` is a point (1-24) or 25 for bar, and ``to`` is a
point or 0 for bear-off. A trailing ``*`` marks a hit (we ignore the
asterisk because OpenSpiel reconstructs hit-ness from the board state).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


# Column convention: column 0 is the LEFT player, column 1 is the RIGHT player.
# We map column index -> OpenSpiel player id below in the replay layer.

# A move line begins with " N)" where N is 1+ digits. The two move-columns
# are whitespace-padded; we split them by finding the second occurrence of
# a "DD:" pattern (or the rightmost portion of the line beyond column-30ish).
_TURN_LINE_RE = re.compile(r"^\s*(\d+)\)(.*)$")

# A single column entry is either empty, a cube/take/drop annotation, or
# "DD: moves" where DD is two digits and moves is space-separated point pairs.
_COL_DICE_RE = re.compile(r"^\s*(\d{2}):\s*(.*?)\s*$")

# Cube/game-end annotations we ignore.
_IGNORE_PHRASES = ("Doubles =>", "Takes", "Drops", "Wins ", "Beavers", "Raccoons")

# "Cannot Move" is a forfeit annotation — keep as an empty-move decision so
# the replay layer still advances through that turn's chance + no-op.
_CANNOT_MOVE = "Cannot Move"


@dataclass
class Decision:
    """One checker-play decision in a game."""
    column: int               # 0 = left column, 1 = right column
    dice: tuple[int, int]
    move_str: str             # e.g. "24/23 13/8" — empty string means forfeit


@dataclass
class Game:
    """One game within a match — just the ordered list of checker decisions."""
    game_number: int
    decisions: list[Decision]


@dataclass
class Match:
    """A parsed match. Only carries what we need for replay."""
    metadata: dict[str, str]   # parsed ; [Key "Val"] header pairs
    match_length: int | None
    player_names: tuple[str, str]  # (left, right) — for diagnostics only
    games: list[Game]


def _split_two_columns(rest: str) -> tuple[str, str]:
    """Split the body of a turn line into (left_col, right_col).

    The two columns are visually aligned with a wide whitespace gap. The
    simplest robust split is to look for the second "DD:" pattern (which
    starts the right column). If only one column has a dice entry, the
    other side is empty.
    """
    # Find all "DD:" positions (start of a column entry).
    dice_starts = [m.start() for m in re.finditer(r"\b\d{2}:", rest)]
    if not dice_starts:
        # No dice in either column — could be "Doubles => N" / "Drops" / etc.
        # Try to detect that this row is purely cube/annotation. Split on the
        # widest whitespace gap to separate columns; both halves will likely
        # match _IGNORE_PHRASES.
        for sep in re.finditer(r"\s{3,}", rest):
            return rest[:sep.start()].strip(), rest[sep.end():].strip()
        return rest.strip(), ""

    if len(dice_starts) == 1:
        # Only one column has dice. The dice-bearing column could be either
        # left or right depending on indentation. Heuristic: if the dice
        # starts past column ~30, it's the right column; else the left.
        idx = dice_starts[0]
        if idx >= 25:
            return rest[:idx].strip(), rest[idx:].strip()
        else:
            # Left column has dice; right column may have a cube annotation.
            # Find the first wide whitespace gap after the left column.
            after_left_match = re.search(r"\s{3,}", rest[idx:])
            if after_left_match:
                split = idx + after_left_match.start()
                return rest[:split].strip(), rest[split:].strip()
            return rest.strip(), ""

    # Two or more dice positions — right column starts at the second one.
    left = rest[:dice_starts[1]].strip()
    right = rest[dice_starts[1]:].strip()
    return left, right


def _parse_column(col: str) -> Decision | None:
    """Parse one column's content into a Decision (or None if not a play).

    Empty strings, cube annotations, and game-end annotations return None.
    """
    col = col.strip()
    if not col:
        return None
    for ignore in _IGNORE_PHRASES:
        if ignore in col:
            return None
    m = _COL_DICE_RE.match(col)
    if not m:
        return None
    dd, moves = m.group(1), m.group(2).strip()
    d1, d2 = int(dd[0]), int(dd[1])
    if not (1 <= d1 <= 6 and 1 <= d2 <= 6):
        return None
    # "Cannot Move" -> forfeit (empty move). Strip hit markers ('*');
    # they're recoverable from the board state.
    if _CANNOT_MOVE in moves:
        moves_clean = ""
    else:
        moves_clean = moves.replace("*", "")
    return Decision(column=-1, dice=(d1, d2), move_str=moves_clean)


def parse_match(text: str) -> Match:
    """Parse a full match file text into a Match."""
    metadata: dict[str, str] = {}
    games: list[Game] = []
    match_length: int | None = None
    player_names = ("", "")

    current_game: Game | None = None
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Header metadata: "; [Key \"Value\"]"
        if stripped.startswith(";"):
            m = re.match(r';\s*\[(\S+(?:\s+\S+)*)\s+"([^"]*)"\]', stripped)
            if m:
                metadata[m.group(1)] = m.group(2)
            i += 1
            continue

        # Match length declaration.
        m = re.match(r"(\d+)\s+point match", stripped)
        if m:
            match_length = int(m.group(1))
            i += 1
            continue

        # Money game declaration (just in case).
        if "Money session" in stripped or "Money game" in stripped:
            match_length = 0
            i += 1
            continue

        # Game header: " Game N"
        m = re.match(r"Game\s+(\d+)\s*$", stripped)
        if m:
            if current_game is not None and current_game.decisions:
                games.append(current_game)
            current_game = Game(game_number=int(m.group(1)), decisions=[])
            i += 1
            # Next line is usually the score header. Parse player names from it.
            if i < len(lines):
                score_m = re.match(
                    r"\s*(\S.*?)\s*:\s*\d+\s+(\S.*?)\s*:\s*\d+\s*$", lines[i]
                )
                if score_m and not player_names[0]:
                    player_names = (score_m.group(1), score_m.group(2))
                if score_m:
                    i += 1
            continue

        # Turn line: " N) ... ... "
        m = _TURN_LINE_RE.match(line)
        if m and current_game is not None:
            rest = m.group(2)
            left_str, right_str = _split_two_columns(rest)
            for col_idx, col_str in enumerate((left_str, right_str)):
                dec = _parse_column(col_str)
                if dec is not None:
                    dec.column = col_idx
                    current_game.decisions.append(dec)
            i += 1
            continue

        i += 1

    if current_game is not None and current_game.decisions:
        games.append(current_game)

    return Match(
        metadata=metadata,
        match_length=match_length,
        player_names=player_names,
        games=games,
    )


def parse_match_file(path: str | Path) -> Match:
    """Convenience: read a file and return the parsed Match."""
    return parse_match(Path(path).read_text(encoding="utf-8", errors="replace"))


def _main() -> None:
    """``python -m raccoon.data.bgmatch <file>`` debug helper."""
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m raccoon.data.bgmatch <match_file>", file=sys.stderr)
        sys.exit(2)
    match = parse_match_file(sys.argv[1])
    print(f"Match length: {match.match_length}")
    print(f"Players: {match.player_names}")
    print(f"Games: {len(match.games)}")
    for g in match.games:
        print(f"  Game {g.game_number}: {len(g.decisions)} decisions")
        for d in g.decisions[:3]:
            print(f"    col{d.column} dice={d.dice} move='{d.move_str}'")
        if len(g.decisions) > 3:
            print(f"    ... and {len(g.decisions) - 3} more")


if __name__ == "__main__":
    _main()
