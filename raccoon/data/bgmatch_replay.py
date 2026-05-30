"""Replay parsed match decisions through OpenSpiel, yielding decision states.

The parser (``raccoon.data.bgmatch``) converts a match file into ordered
``Decision`` records — but the moves are still text. To get OpenSpiel's
internal action indices (and legal_actions, observation tensor, etc.) we
need to walk a fresh ``BackgammonState`` through the chance + decision
sequence, matching each text move against the legal action list.

A "yield" returns ``(state_clone, action_idx)``: the state immediately
before the move is applied, and the matched OpenSpiel action index that
was actually played. Caller does whatever it wants with the state (encode,
do V-lookahead, etc.) and the replay loop has already moved on to the
next position internally.

Match notation has several variants we normalise:

  - bar entry: ``25/X``, ``bar/X``, ``Bar/X`` → all map to point 25
  - bear off: ``X/0``, ``X/off``, ``X/Off`` → all map to point 0
  - hits: trailing ``*`` stripped (recoverable from board state)
  - count suffix: ``13/9(2)`` → two copies of ``13/9``
  - chains: ``24/22/21`` → ``(24,22)`` and ``(22,21)``

We compare each candidate via a sorted multiset of ``(from, to)`` pairs
which is invariant to within-turn move ordering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

import pyspiel

from raccoon.data.bgmatch import Decision, Game

# Chance outcome lookup tables — built once from a fresh game instance.
_CHANCE_TABLES_CACHE: dict[str, dict] = {}


def _build_chance_tables() -> dict:
    """Map (player, sorted_dice) -> chance action index for opening + mid-game.

    Opening has 30 outcomes (15 non-double dice pairs × 2 first-player
    choices). Mid-game has 36 (30 redundant non-double pairs + 6 doubles).
    For mid-game, the player whose turn it is to roll is determined by
    history — we don't need to encode it in the lookup. Multiple chance
    actions can map to the same (dice) since OpenSpiel emits both die
    orderings; either works, so we keep just one.
    """
    g = pyspiel.load_game("backgammon(scoring_type=full_scoring)")

    opening: dict[tuple[int, tuple[int, int]], int] = {}
    s = g.new_initial_state()
    for outcome, _ in s.chance_outcomes():
        sc = s.clone()
        sc.apply_action(outcome)
        p = sc.current_player()
        cms = sc.spiel_move_to_checker_moves(p, sc.legal_actions()[0])
        dice = tuple(sorted(cm.num for cm in cms[:2]))
        opening.setdefault((p, dice), outcome)

    midgame: dict[tuple[int, int], int] = {}
    # Walk to any mid-game chance node to enumerate.
    s = g.new_initial_state()
    s.apply_action(0)  # opening chance
    s.apply_action(s.legal_actions()[0])  # one move -> next chance
    for outcome, _ in s.chance_outcomes():
        sc = s.clone()
        sc.apply_action(outcome)
        p = sc.current_player()
        cms = sc.spiel_move_to_checker_moves(p, sc.legal_actions()[0])
        dice = tuple(sorted(cm.num for cm in cms[:2]))
        midgame.setdefault(dice, outcome)

    return {"opening": opening, "midgame": midgame}


def _chance_tables() -> dict:
    if "tables" not in _CHANCE_TABLES_CACHE:
        _CHANCE_TABLES_CACHE["tables"] = _build_chance_tables()
    return _CHANCE_TABLES_CACHE["tables"]


def _norm_point(p: str) -> int:
    """Normalise a point string to its integer point number.

    25 = bar, 0 = off. Accepts "bar"/"Bar"/"BAR"/"25" interchangeably and
    "off"/"Off"/"OFF"/"0" interchangeably.
    """
    p = p.strip().lower()
    if p in ("bar", "25"):
        return 25
    if p in ("off", "0"):
        return 0
    return int(p)


def _combine_journeys(moves: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge consecutive same-checker moves into single (start, end) journeys.

    Iteratively combines (A, B) + (B, C) → (A, C) where B is a normal point
    (not bear-off 0 — a borne-off checker can't continue moving). Lets us
    treat ``23/18 18/14`` (OpenSpiel) and ``23/14`` (collapsed match
    notation) as the same play.
    """
    work = list(moves)
    changed = True
    while changed:
        changed = False
        for i in range(len(work)):
            mid = work[i][1]
            if mid == 0:
                continue  # bear-off is terminal — no through-traffic
            for j in range(len(work)):
                if i == j:
                    continue
                if work[j][0] == mid:
                    combined = (work[i][0], work[j][1])
                    rest = [m for k, m in enumerate(work) if k != i and k != j]
                    work = rest + [combined]
                    changed = True
                    break
            if changed:
                break
    return work


def _parse_moves_raw(move_str: str) -> list[tuple[int, int]]:
    """Parse a move string into a list of (from, to) pairs as written.

    Does NOT combine chained journeys — preserves the explicit segmentation
    so the multiset comparison matches OpenSpiel's split notation exactly
    when it can. Empty/whitespace input → empty list.
    """
    if not move_str.strip():
        return []
    s = move_str.replace("*", "").strip()
    parts = s.split()
    moves: list[tuple[int, int]] = []
    for part in parts:
        n = 1
        m = re.match(r"^(.+?)\((\d+)\)$", part)
        if m:
            part = m.group(1)
            n = int(m.group(2))
        segs = part.split("/")
        if len(segs) < 2:
            continue
        points = [_norm_point(p) for p in segs]
        for i in range(len(points) - 1):
            for _ in range(n):
                moves.append((points[i], points[i + 1]))
    return moves


def _normalize_moves(move_str: str) -> tuple[tuple[int, int], ...]:
    """Sorted multiset of (from, to) pairs as literally written in ``move_str``."""
    return tuple(sorted(_parse_moves_raw(move_str)))


def _normalize_moves_journeys(move_str: str) -> tuple[tuple[int, int], ...]:
    """Sorted multiset after merging chained same-checker moves into journeys.

    Used as a fallback when literal matching fails — handles collapsed
    notation like ``23/14`` (Studio) vs ``23/18/14`` (OpenSpiel). Beware:
    over-merges when two distinct checkers happen to share an intermediate
    point (e.g., ``13/8 8/5`` could be one chain or two disjoint moves), so
    only use this as a tiebreaker, never as the primary canonical form.
    """
    return tuple(sorted(_combine_journeys(_parse_moves_raw(move_str))))


def _strip_action_index_prefix(action_str: str) -> str:
    """OpenSpiel formats actions as ``"<idx> - <moves>"``; strip the prefix."""
    if " - " in action_str:
        return action_str.split(" - ", 1)[1]
    return action_str


def _board_signature(
    state: pyspiel.BackgammonState,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return ``(player_0_board, player_1_board)`` — 24 entries per player.

    Two states with the same signature have the same checker placement. We
    use this to verify that a candidate OpenSpiel action produces the same
    final position as the parsed human move, robust to notation
    differences (chained vs. split, etc.).
    """
    return (
        tuple(state.board(0, i) for i in range(24)),
        tuple(state.board(1, i) for i in range(24)),
    )


def _legal_action_index_for_moves(
    state: pyspiel.BackgammonState, target_moves: tuple[tuple[int, int], ...],
) -> int | None:
    """Return the legal action whose moves multiset equals ``target_moves``.

    Tries the literal multiset first; if no match, retries with journeys
    (chained-move) canonicalisation. The journey form is more permissive —
    used only as a fallback for collapsed notation like ``23/14`` instead
    of ``23/18/14``.
    """
    for a in state.legal_actions():
        a_str = _strip_action_index_prefix(state.action_to_string(a))
        if _normalize_moves(a_str) == target_moves:
            return a
    # Fallback: compare via journey-combined form.
    target_journeys = tuple(sorted(_combine_journeys(list(target_moves))))
    for a in state.legal_actions():
        a_str = _strip_action_index_prefix(state.action_to_string(a))
        if _normalize_moves_journeys(a_str) == target_journeys:
            return a
    return None


def _find_action_sequence_to_signature(
    state: pyspiel.BackgammonState,
    target_board: tuple[tuple[int, ...], tuple[int, ...]],
    max_depth: int = 4,
) -> list[int] | None:
    """Return a sequence of legal actions whose application yields ``target_board``.

    Recursive search up to ``max_depth`` plies; doubles need depth 2-4 since
    OpenSpiel splits them into 2-die sub-actions. Returns the shortest
    matching sequence found, or None.
    """
    if state.is_chance_node() or state.is_terminal():
        return None
    if max_depth <= 0:
        return None
    for a in state.legal_actions():
        sc = state.clone()
        sc.apply_action(a)
        # Auto-advance through trailing no-ops so single-action plays
        # land on the resting state we compare against.
        while not sc.is_chance_node() and not sc.is_terminal():
            empties = [
                ea for ea in sc.legal_actions()
                if _normalize_moves(
                    _strip_action_index_prefix(sc.action_to_string(ea))
                ) == ()
            ]
            if not empties:
                break
            sc.apply_action(empties[0])
        if _board_signature(sc) == target_board:
            return [a]
        if not sc.is_chance_node() and not sc.is_terminal():
            rest = _find_action_sequence_to_signature(
                sc, target_board, max_depth - 1,
            )
            if rest is not None:
                return [a] + rest
    return None


def _apply_move_to_board(
    sig: tuple[tuple[int, ...], tuple[int, ...]],
    current_player: int,
    moves: list[tuple[int, int]],
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Apply parsed human moves to a board signature.

    Returns the resulting signature, or None if the moves are inconsistent
    with the starting board (e.g., moving from a point with no checker).

    Move semantics:
      - ``from = 25``: bar entry — decrement bar count (encoded externally,
        so the OpenSpiel signature alone is insufficient; we approximate by
        skipping the source decrement, which is fine for signature
        comparison since OpenSpiel will also produce the same final
        ``board()`` values regardless of bar accounting).
      - ``to = 0``: bear off — increment off count (also not part of the
        signature; we skip the target increment).
      - hit: if destination had exactly 1 opponent checker, send to bar
        (zero out that point; OpenSpiel will mirror this).

    OpenSpiel's ``state.board(player, index)`` indexes both players with
    point N at index ``24 - N`` (i.e., player's own ace point at the high
    index). We use that mapping for both players.
    """
    p_board = [list(sig[0]), list(sig[1])]
    me = current_player
    opp = 1 - current_player

    def _idx_for(point: int, player: int) -> int:
        # Player 0's point P is at index 24 - P.
        # Player 1's point P (their own perspective) is at index P - 1.
        if player == 0:
            return 24 - point
        else:
            return point - 1

    for from_pt, to_pt in moves:
        if from_pt != 25:  # 25 = bar, not in signature
            idx = _idx_for(from_pt, me)
            if not (0 <= idx < 24) or p_board[me][idx] <= 0:
                return None
            p_board[me][idx] -= 1
        if to_pt != 0:  # 0 = off, not in signature
            idx = _idx_for(to_pt, me)
            if not (0 <= idx < 24):
                return None
            # Hit: opponent's checker sits at the SAME physical index (both
            # players' board() arrays use the same physical positions).
            if p_board[opp][idx] == 1:
                p_board[opp][idx] = 0
            p_board[me][idx] += 1

    return (tuple(p_board[0]), tuple(p_board[1]))


def _multiset_subtract(
    big: list[tuple[int, int]], small: tuple[tuple[int, int], ...],
) -> list[tuple[int, int]] | None:
    """Return ``big`` with each element of ``small`` removed once.

    Returns None if ``small`` isn't a multiset-subset of ``big``.
    """
    result = list(big)
    for m in small:
        try:
            result.remove(m)
        except ValueError:
            return None
    return result


def _legal_action_subset_of(
    state: pyspiel.BackgammonState, remaining: list[tuple[int, int]],
) -> tuple[int, tuple[tuple[int, int], ...]] | None:
    """Find a legal action whose moves are a non-empty multiset-subset of ``remaining``.

    Used when one parsed decision must be split across multiple OpenSpiel
    actions (doubles, where the engine asks the player for two 2-die plays
    rather than one 4-die play). We prefer the largest matching subset so
    the split terminates in as few iterations as possible.
    """
    best: tuple[int, tuple[tuple[int, int], ...]] | None = None
    for a in state.legal_actions():
        a_str = _strip_action_index_prefix(state.action_to_string(a))
        a_moves = _normalize_moves(a_str)
        if not a_moves:
            continue
        if _multiset_subtract(remaining, a_moves) is None:
            continue
        if best is None or len(a_moves) > len(best[1]):
            best = (a, a_moves)
    return best


@dataclass
class ReplayStep:
    """One yielded decision: state before move, the OpenSpiel action applied."""
    state: pyspiel.BackgammonState  # clone — safe to keep across iterations
    action: int
    decision: Decision              # source decision (for diagnostics)


class ReplayError(Exception):
    """Raised when a parsed move can't be matched against legal_actions."""


def replay_game(
    game: Game, game_obj: pyspiel.Game | None = None,
) -> Iterator[ReplayStep]:
    """Walk a parsed ``Game`` through OpenSpiel, yielding each decision step.

    The first decision in the game determines the opening-roll outcome:
    whichever column owns that decision (col 0 or 1) is the OpenSpiel
    player that moves first. Subsequent dice are applied as mid-game
    chance outcomes matching the next decision's recorded dice.

    Raises ``ReplayError`` if a parsed move doesn't match any legal action
    (most often a parser bug or an unusual notation variant we haven't
    handled). Caller should typically skip the offending game and move on.
    """
    if not game.decisions:
        return

    if game_obj is None:
        game_obj = pyspiel.load_game("backgammon(scoring_type=full_scoring)")

    tables = _chance_tables()

    state = game_obj.new_initial_state()
    # Opening: first decision tells us (player, dice).
    first = game.decisions[0]
    first_player = first.column  # column 0 -> player 0, column 1 -> player 1
    key = (first_player, tuple(sorted(first.dice)))
    opening_outcome = tables["opening"].get(key)
    if opening_outcome is None:
        raise ReplayError(
            f"Game {game.game_number}: no opening chance outcome for "
            f"player={first_player} dice={first.dice}"
        )
    state.apply_action(opening_outcome)

    for idx, dec in enumerate(game.decisions):
        # If the state is at a chance node, apply the dice from this decision.
        # (Skipped for the very first decision where we already set the
        # opening outcome above.)
        if idx > 0:
            if not state.is_chance_node():
                raise ReplayError(
                    f"Game {game.game_number} decision {idx}: expected chance "
                    f"node but state has player={state.current_player()}"
                )
            dice_key = tuple(sorted(dec.dice))
            mid_outcome = tables["midgame"].get(dice_key)
            if mid_outcome is None:
                raise ReplayError(
                    f"Game {game.game_number} decision {idx}: no chance "
                    f"outcome for dice={dec.dice}"
                )
            state.apply_action(mid_outcome)

        # Cross-check that OpenSpiel's current player matches the column.
        if state.current_player() != dec.column:
            raise ReplayError(
                f"Game {game.game_number} decision {idx}: player mismatch "
                f"(OpenSpiel says {state.current_player()}, "
                f"match says column {dec.column})"
            )

        target = _normalize_moves(dec.move_str)
        remaining = list(target)

        # Compute target board signature: what the board looks like after
        # applying the human's moves. Used as the final fallback when
        # notation-based matching fails — robust to all chain/collapse
        # notation variants since it compares the actual resulting position.
        parsed_moves = _parse_moves_raw(dec.move_str)
        initial_sig = _board_signature(state)
        target_sig = _apply_move_to_board(initial_sig, state.current_player(), parsed_moves)

        # Try the board-signature path FIRST when the human's notation isn't
        # an obvious literal multiset match (chains, collapsed counts,
        # etc.). If we can identify a single-action sequence by signature
        # we use it directly; multi-action sequences (doubles) yield one
        # ReplayStep per OpenSpiel action and then bail out of the per-
        # decision loop. This bypasses the notation-matching code below
        # for cases where notations disagree but final boards agree.
        if target_sig is not None:
            seq = _find_action_sequence_to_signature(state, target_sig)
            if seq is not None:
                for sub_a in seq:
                    yield ReplayStep(state=state.clone(), action=sub_a, decision=dec)
                    state.apply_action(sub_a)
                    while not state.is_chance_node() and not state.is_terminal():
                        empties = [
                            ea for ea in state.legal_actions()
                            if _normalize_moves(
                                _strip_action_index_prefix(state.action_to_string(ea))
                            ) == ()
                        ]
                        if not empties:
                            break
                        state.apply_action(empties[0])
                continue

        # A single match decision may map to multiple OpenSpiel actions
        # (doubles, where OpenSpiel asks the player twice for two 2-die
        # plays). Yield one ReplayStep per OpenSpiel decision so each
        # state — including mid-doubles states — becomes a training example.
        steps_for_decision = 0
        while True:
            if state.is_chance_node():
                # OpenSpiel has consumed all dice and moved on. If the match
                # said more moves were played, that's a mismatch.
                if remaining:
                    raise ReplayError(
                        f"Game {game.game_number} decision {idx}: OpenSpiel "
                        f"advanced past decision but {len(remaining)} moves "
                        f"unconsumed ({remaining}); move='{dec.move_str}'"
                    )
                break

            target_remaining = tuple(sorted(remaining))
            action = _legal_action_index_for_moves(state, target_remaining)
            if action is not None:
                # Exact match consumes all remaining moves in one action.
                remaining = []
            elif target_remaining != ():
                # Try a partial subset (doubles split case).
                found = _legal_action_subset_of(state, remaining)
                if found is not None:
                    action, consumed = found
                    remaining = _multiset_subtract(remaining, consumed)  # type: ignore[assignment]

            if action is None:
                # Forfeit: pick the unique legal empty-move action if one exists.
                if target_remaining == ():
                    empties = [
                        a for a in state.legal_actions()
                        if _normalize_moves(
                            _strip_action_index_prefix(state.action_to_string(a))
                        ) == ()
                    ]
                    if len(empties) == 1:
                        action = empties[0]

            if action is None:
                legal_strs = [
                    _strip_action_index_prefix(state.action_to_string(a))
                    for a in state.legal_actions()[:5]
                ]
                raise ReplayError(
                    f"Game {game.game_number} decision {idx}: no legal action "
                    f"matches remaining={target_remaining} "
                    f"(full move='{dec.move_str}'); first legal: {legal_strs}"
                )

            yield ReplayStep(state=state.clone(), action=action, decision=dec)
            state.apply_action(action)
            steps_for_decision += 1
            if target_remaining == ():
                # Forfeit: one OpenSpiel action consumed; advance state and stop.
                break
            if not remaining:
                break
            if steps_for_decision > 4:
                raise ReplayError(
                    f"Game {game.game_number} decision {idx}: needed >4 "
                    f"OpenSpiel actions for one decision; bug"
                )

        # After consuming all match-listed moves: doubles where only 2 of 4
        # dice were playable will leave the state at a mid-doubles decision
        # node with only empty (no-op) legal actions. Auto-apply them so we
        # land on the next chance node before the loop iterates.
        while not state.is_chance_node() and not state.is_terminal():
            empties = [
                a for a in state.legal_actions()
                if _normalize_moves(
                    _strip_action_index_prefix(state.action_to_string(a))
                ) == ()
            ]
            if not empties:
                # Mismatch: OpenSpiel still wants real moves but the match
                # gave us no more. Raise so the caller can skip this game.
                non_empty = [
                    _strip_action_index_prefix(state.action_to_string(a))
                    for a in state.legal_actions()[:3]
                ]
                raise ReplayError(
                    f"Game {game.game_number} decision {idx}: match exhausted "
                    f"but OpenSpiel still expects a play; first legal: {non_empty}"
                )
            state.apply_action(empties[0])


def _main() -> None:
    """``python -m raccoon.data.bgmatch_replay <file>`` debug helper."""
    import sys
    from raccoon.data.bgmatch import parse_match_file

    if len(sys.argv) != 2:
        print(
            "Usage: python -m raccoon.data.bgmatch_replay <match_file>",
            file=sys.stderr,
        )
        sys.exit(2)

    match = parse_match_file(sys.argv[1])
    game_obj = pyspiel.load_game("backgammon(scoring_type=full_scoring)")
    total_steps = 0
    failures = 0
    for g in match.games:
        try:
            steps = list(replay_game(g, game_obj))
            total_steps += len(steps)
            print(f"Game {g.game_number}: {len(steps)} steps replayed OK")
        except ReplayError as e:
            failures += 1
            print(f"Game {g.game_number}: FAILED — {e}")
    print(f"\nTotal: {total_steps} steps across {len(match.games)} games "
          f"({failures} games failed)")


if __name__ == "__main__":
    _main()
