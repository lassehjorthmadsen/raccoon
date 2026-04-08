"""Game logging for benchmark matches.

Records per-move data in a format compatible with backgammon notation, plus
a per-game summary (result, gammon/bg, who went first, opponent info).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


@dataclass
class MoveRecord:
    player: int                          # OpenSpiel player index that moved
    dice: tuple[int, int] | None         # dice used for this move
    action: int                          # OpenSpiel action index
    action_str: str                      # human-readable move (e.g. "8/5 6/5")


@dataclass
class GameRecord:
    moves: list[MoveRecord] = field(default_factory=list)
    result: float = 0.0                  # equity from Raccoon's perspective
    result_type: str = "normal"          # normal | gammon | backgammon
    raccoon_is_player0: bool = True      # whether Raccoon played as OpenSpiel P0
    timestamp: str = ""
    raccoon_version: str = ""
    opponent: str = "gnubg"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_game_record(
    raccoon_is_player0: bool,
    raccoon_version: str,
    opponent: str = "gnubg",
) -> GameRecord:
    return GameRecord(
        raccoon_is_player0=raccoon_is_player0,
        timestamp=_now_iso(),
        raccoon_version=raccoon_version,
        opponent=opponent,
    )


def save_match_log(games: list[GameRecord], path: str) -> None:
    """Write a list of GameRecord to ``path`` as JSON."""
    payload = {
        "written_at": _now_iso(),
        "num_games": len(games),
        "games": [asdict(g) for g in games],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
