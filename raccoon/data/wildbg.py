"""Load wildbg-training labeled positions for supervised pretraining.

wildbg-training (https://github.com/carsten-wenderdel/wildbg-training, CC0)
ships CSVs with one row per position:

    position_id, win, win_g, win_bg, lose_g, lose_bg

where ``position_id`` is the 14-character GNU Backgammon Position ID and the
five labels are nested rollout probabilities (``win >= win_g >= win_bg``,
``1-win >= lose_g >= lose_bg``). Positions are pre-roll, so the encoder's
dice channels stay at zero.

Format references:
  https://www.gnu.org/software/gnubg/manual/html_node/A-technical-description-of-the-Position-ID.html
"""

import base64
import csv
import sys
from pathlib import Path

import numpy as np

from raccoon.env.game_wrapper import BoardView


def decode_position_id(pid: str) -> BoardView:
    """Decode a GNUBG Position ID into a perspective-relative ``BoardView``.

    The ID encodes 80 bits packed LSB-first into 10 bytes. For each of the
    two players (on-roll first, then opponent), 25 locations are encoded
    (points 1..24 then bar); each location contributes N consecutive ``1``
    bits (= checker count) terminated by a single ``0``.

    Each player is encoded from their own perspective (point 1 = ace point
    closest to that player's bearoff). To express the opponent's checkers
    in the on-roll player's coordinates, we reverse the opponent's 24-point
    array: opponent's point ``i`` lives at on-roll's point ``25 - i``.
    """
    raw = base64.b64decode(pid + "==")
    if len(raw) != 10:
        raise ValueError(f"Position ID {pid!r} decodes to {len(raw)} bytes, expected 10")

    bits = [(b >> i) & 1 for b in raw for i in range(8)]

    counts: list[list[int]] = []
    pos = 0
    for _ in range(2):
        locations = []
        for _ in range(25):
            n = 0
            while pos < len(bits) and bits[pos] == 1:
                n += 1
                pos += 1
            if pos >= len(bits):
                raise ValueError(f"Position ID {pid!r} truncated while decoding")
            pos += 1  # consume terminating 0
            locations.append(n)
        counts.append(locations)

    me, opp = counts
    my_points = np.array(me[:24], dtype=np.float32)
    opp_points = np.array(opp[:24][::-1], dtype=np.float32)
    my_bar = int(me[24])
    opp_bar = int(opp[24])
    my_off = 15 - int(my_points.sum()) - my_bar
    opp_off = 15 - int(opp_points.sum()) - opp_bar

    if my_off < 0 or opp_off < 0:
        raise ValueError(
            f"Position ID {pid!r} has more than 15 checkers per side "
            f"(my_off={my_off}, opp_off={opp_off})"
        )

    return BoardView(
        my_points=my_points,
        opp_points=opp_points,
        my_bar=my_bar,
        opp_bar=opp_bar,
        my_off=my_off,
        opp_off=opp_off,
        dice=None,
        mid_doubles=False,
    )


def equity_from_wildbg(
    win: float, win_g: float, win_bg: float, lose_g: float, lose_bg: float,
) -> float:
    """Convert wildbg's 5 nested outcome probabilities to a scalar equity in [-1, 1].

    Standard cubeless equity decomposition (in points-per-game, range [-3, 3]):

        equity = 1*P(win normal) + 2*P(win gammon) + 3*P(win backgammon)
               - 1*P(lose normal) - 2*P(lose gammon) - 3*P(lose backgammon)

    With wildbg's nested encoding (``win`` is total win prob, ``win_g`` is
    gammon-or-better win prob, ``win_bg`` is backgammon prob), the disjoint
    probabilities are ``win - win_g``, ``win_g - win_bg``, ``win_bg``. The
    full expression simplifies to:

        equity = (win - lose) + (win_g + win_bg) - (lose_g + lose_bg)

    where ``lose = 1 - win``. We then divide by 3 to match the [-1, 1]
    convention used by the existing self-play value targets (raw outcomes
    of +-1/+-2/+-3 normalised by /3.0).
    """
    lose = 1.0 - win
    equity = (win - lose) + (win_g + win_bg) - (lose_g + lose_bg)
    return equity / 3.0


def load_wildbg_csv(path: str | Path) -> list[tuple[BoardView, float]]:
    """Load a wildbg CSV into a list of ``(BoardView, value_target)`` tuples.

    Skips the header row. Raises if any row has malformed labels or
    Position IDs (caller probably wants this to be loud).
    """
    out: list[tuple[BoardView, float]] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            bv = decode_position_id(row["position_id"])
            value = equity_from_wildbg(
                float(row["win"]),
                float(row["win_g"]),
                float(row["win_bg"]),
                float(row["lose_g"]),
                float(row["lose_bg"]),
            )
            out.append((bv, value))
    return out


def load_wildbg_dir(data_dir: str | Path) -> list[tuple[BoardView, float]]:
    """Load every ``*.csv`` under ``data_dir`` (recursive) into one list."""
    data_dir = Path(data_dir)
    csv_paths = sorted(data_dir.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSVs found under {data_dir}")
    out: list[tuple[BoardView, float]] = []
    for p in csv_paths:
        out.extend(load_wildbg_csv(p))
    return out


def _main() -> None:
    """``python -m raccoon.data.wildbg <position_id>`` debug helper.

    Decodes the ID and pretty-prints the resulting ``BoardView`` plus the
    encoder tensor dump.
    """
    from raccoon.env.encoder import dump_tensor

    if len(sys.argv) != 2:
        print("Usage: python -m raccoon.data.wildbg <position_id>", file=sys.stderr)
        sys.exit(2)
    pid = sys.argv[1]
    bv = decode_position_id(pid)
    print(f"Position ID: {pid}")
    print(f"my_points  : {bv.my_points.astype(int).tolist()}")
    print(f"opp_points : {bv.opp_points.astype(int).tolist()}")
    print(f"my_bar/off : {bv.my_bar} / {bv.my_off}")
    print(f"opp_bar/off: {bv.opp_bar} / {bv.opp_off}")
    print()
    print(dump_tensor(bv))


if __name__ == "__main__":
    _main()
