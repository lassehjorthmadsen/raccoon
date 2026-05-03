"""Standard backgammon match notation writer.

Takes the ``GameRecord`` instances produced by ``raccoon.eval.game_log`` and
emits a text file in the format used by eXtreme Gammon (XG), GNU Backgammon,
Snowie, and BackgammonGalaxy match transcripts. The output is designed to be
imported directly into XG for analysis.

Example output::

    ; [Site "Raccoon benchmark"]
    ; [Player 1 "Raccoon"]
    ; [Player 2 "GNUBG"]
    ; [EventDate "2026.04.07"]
    ; [Variation "Backgammon"]
    ; [Unrated "Off"]
    ; [Crawford "Off"]
    ; [CubeLimit "1"]

    0 point match

     Game 1
     Raccoon : 0                     GNUBG : 0
      1)                             51: 24/23 13/8
      2) 21: 13/11 24/23             22: 6/4 6/4 13/11 13/11
     ...
          Wins 1 point

Notation conventions match XG's importer:
- Each die-use is its own sub-move (no chains, no fold counts).
- Bar = point 25, Off = point 0.
- Hits keep the ``*`` marker on the destination point.
- A turn with no legal moves renders as just ``dd:`` (no body).
- The match-header line is ``0 point match`` for cubeless money games.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from raccoon.eval.game_log import GameRecord, MoveRecord


# Layout constants tuned to match the XG/BackgammonGalaxy reference width.
_NUM_WIDTH = 3                          # turn-number column (right-justified)
_PREFIX_WIDTH = _NUM_WIDTH + 2          # "  N) " — 5 chars
_COL_WIDTH = 28                         # left-column content width
_RIGHT_COL_START = _PREFIX_WIDTH + _COL_WIDTH  # = 33

# OpenSpiel emits multi-checker plays as ``token(N)`` (e.g. ``8/5(2)``).
_FOLD_RE = re.compile(r"^(.*)\((\d+)\)$")


def _strip_action_prefix(action_str: str) -> str:
    """Drop the OpenSpiel ``<idx> - `` prefix if present."""
    if " - " in action_str:
        return action_str.split(" - ", 1)[1]
    return action_str


def _normalize_segment(seg: str) -> tuple[str, bool]:
    """Translate one ``/``-separated segment to XG point notation.

    Returns ``(point_str, has_hit_marker)``. ``Bar`` becomes ``25``, ``Off``
    becomes ``0``, and a trailing ``*`` (hit marker) is reported separately.
    """
    has_hit = seg.endswith("*")
    base = seg[:-1] if has_hit else seg
    if base == "Bar":
        base = "25"
    elif base == "Off":
        base = "0"
    return base, has_hit


def _split_token_to_die_moves(token: str) -> list[str]:
    """Split one OpenSpiel checker token into per-die sub-moves.

    Examples::

        '13/9'         -> ['13/9']
        '13/9/3'       -> ['13/9', '9/3']
        'Bar/20*/17'   -> ['25/20*', '20/17']
        '8/3*/Off'     -> ['8/3*', '3/0']
    """
    segs = token.split("/")
    if len(segs) < 2:
        return []
    sub_moves: list[str] = []
    for i in range(len(segs) - 1):
        src, _ = _normalize_segment(segs[i])
        dst, dst_hit = _normalize_segment(segs[i + 1])
        marker = "*" if dst_hit else ""
        sub_moves.append(f"{src}/{dst}{marker}")
    return sub_moves


def format_move_body(action_str: str) -> str:
    """Convert an OpenSpiel action string to XG-compatible move text.

    Examples::

        '189 - 8/7 8/6'         -> '8/7 8/6'
        '432 - 8/5(2)'          -> '8/5 8/5'
        '11 - 24/22* 13/8'      -> '24/22* 13/8'
        '674 - Bar/20* Pass'    -> '25/20*'
        '233 - Bar/20*/17'      -> '25/20* 20/17'
        '14 - 5/Off'            -> '5/0'
        'Pass'                  -> ''
    """
    body = _strip_action_prefix(action_str).strip()
    if not body:
        return ""

    sub_moves: list[str] = []
    for token in body.split():
        if not token or token == "Pass":
            continue
        # Expand any pre-folded ``token(N)`` notation into N copies.
        m = _FOLD_RE.match(token)
        if m:
            base = m.group(1)
            count = int(m.group(2))
            expanded = [base] * count
        else:
            expanded = [token]
        for tok in expanded:
            sub_moves.extend(_split_token_to_die_moves(tok))
    return " ".join(sub_moves)


def _format_dice(dice: tuple[int, int] | None) -> str:
    if dice is None:
        return ""
    return f"{dice[0]}{dice[1]}"


def _merge_doubles_halves(moves: list[MoveRecord]) -> list[tuple[int, str]]:
    """Merge OpenSpiel's split-doubles half-turns into one logical turn.

    OpenSpiel splits a doubles roll into two consecutive same-player half-
    turns of two dice each. XG notation expects one row per *roll*, with all
    four die-uses listed individually (e.g. ``66: 24/18 18/12 8/2 13/7``).
    We concatenate the per-die sub-moves of each half (after Pass-stripping
    and ``(N)``-expansion) into a single body. No folding.
    """
    merged: list[tuple[int, str]] = []
    i = 0
    while i < len(moves):
        j = i + 1
        while (
            j < len(moves)
            and moves[j].player == moves[i].player
            and moves[j].dice == moves[i].dice
        ):
            j += 1

        all_subs: list[str] = []
        for k in range(i, j):
            body = format_move_body(moves[k].action_str)
            if body:
                all_subs.append(body)
        body_text = " ".join(all_subs)

        dice = _format_dice(moves[i].dice)
        if dice:
            cell = f"{dice}: {body_text}" if body_text else f"{dice}:"
        else:
            cell = body_text
        merged.append((moves[i].player, cell))
        i = j
    return merged


def _pad(cell: str, width: int) -> str:
    """Left-justify ``cell`` in ``width`` columns. Long cells get a trailing space."""
    if len(cell) >= width:
        return cell + " "
    return cell.ljust(width)


def _format_game(
    game: GameRecord,
    game_idx: int,
    player1_name: str,
    player2_name: str,
    score_p1_before: int,
    score_p2_before: int,
) -> str:
    """Render a single game in two-column XG-compatible format.

    ``player1_name`` is always the *left* column. The function maps
    ``MoveRecord.player`` (an OpenSpiel index) to a column using
    ``game.raccoon_is_player0`` plus the convention that Raccoon is player 1.
    Cumulative scores are passed in by the caller.
    """
    lines: list[str] = []
    lines.append(f" Game {game_idx}")

    # Score header — leading single space, then label padded so the right
    # label starts at column _RIGHT_COL_START.
    p1_label = f"{player1_name} : {score_p1_before}"
    p2_label = f"{player2_name} : {score_p2_before}"
    lines.append(" " + _pad(p1_label, _RIGHT_COL_START - 1) + p2_label)

    # Decide which OpenSpiel player index belongs to player 1 (left = Raccoon).
    raccoon_openspiel_idx = 0 if game.raccoon_is_player0 else 1
    turns = _merge_doubles_halves(game.moves)

    # Group turns into (left, right) row pairs.
    rows: list[tuple[str, str]] = []
    pending_left: str | None = None
    for player, cell in turns:
        if player == raccoon_openspiel_idx:
            if pending_left is not None:
                rows.append((pending_left, ""))
            pending_left = cell
        else:
            rows.append((pending_left or "", cell))
            pending_left = None
    if pending_left is not None:
        rows.append((pending_left, ""))

    # Render move rows with turn numbers.
    for n, (left, right) in enumerate(rows, start=1):
        prefix = f"{n:>{_NUM_WIDTH}}) "
        lines.append(prefix + _pad(left, _COL_WIDTH) + right)

    # Wins line — its own row, indented one extra space inside the cell.
    if game.result != 0:
        wins_text = f"Wins {int(abs(game.result))} point"
        if game.result > 0:
            # Left wins: 6 leading spaces (= prefix + 1 cell-indent).
            lines.append(" " * (_PREFIX_WIDTH + 1) + wins_text)
        else:
            # Right wins: pad past left col + 1 extra space.
            lines.append(" " * (_RIGHT_COL_START + 1) + wins_text)

    return "\n".join(lines) + "\n"


def format_match(
    games: list[GameRecord],
    player1_name: str = "Raccoon",
    player2_name: str = "GNUBG",
    header_fields: dict[str, str] | None = None,
) -> str:
    """Render a full match transcript as a single string.

    ``header_fields`` is merged into a default set of headers. Pass extra
    metadata (e.g. ``{"Round": "iter_0289 vs ply 2"}``) and it will be
    written verbatim. Cumulative scores in each game header are computed
    from the per-game absolute equity values, exactly like a real match.
    """
    headers: dict[str, str] = {
        "Site": "Raccoon benchmark",
        "Player 1": player1_name,
        "Player 2": player2_name,
        "EventDate": datetime.now(timezone.utc).strftime("%Y.%m.%d"),
        "Variation": "Backgammon",
        "Unrated": "Off",
        "Crawford": "Off",
        "CubeLimit": "1",
        "Jacoby": "Off",
        "Beaver": "Off",
    }
    if header_fields:
        headers.update(header_fields)

    out: list[str] = []
    for key, value in headers.items():
        out.append(f'; [{key} "{value}"]')
    out.append("")
    out.append("0 point match")
    out.append("")

    score_p1 = 0
    score_p2 = 0
    for idx, game in enumerate(games, start=1):
        out.append(_format_game(
            game, idx, player1_name, player2_name,
            score_p1_before=score_p1,
            score_p2_before=score_p2,
        ))
        out.append("")
        if game.result > 0:
            score_p1 += int(abs(game.result))
        elif game.result < 0:
            score_p2 += int(abs(game.result))

    return "\n".join(out)


def save_match_text(
    games: list[GameRecord],
    path: str,
    player1_name: str = "Raccoon",
    player2_name: str = "GNUBG",
    header_fields: dict[str, str] | None = None,
) -> None:
    """Write a list of games to ``path`` in XG-compatible match notation."""
    text = format_match(games, player1_name, player2_name, header_fields)
    with open(path, "w") as f:
        f.write(text)
