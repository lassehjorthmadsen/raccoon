#!/usr/bin/env python3
"""Export OpenSpiel/encoder fixtures for the browser engine's differential tests.

The web engine at raccoonbg.com reimplements two things in JavaScript that this
repo gets from C++ and numpy: legal move generation (OpenSpiel) and the
26-channel encoder (raccoon/env/encoder.py). Both are ported, neither is
trusted — these fixtures are the contract the JS is tested against, so a
divergence fails CI instead of quietly making the demo play a different game
from the one the benchmark scored.

Three files, gzipped JSON, written into the website repo's test/fixtures/:

  movegen.json.gz  board + dice + mid_doubles -> the SET of resulting boards,
                   in the mover's perspective, after every legal action. Sets,
                   not lists: two actions reaching the same position are the
                   same move as far as a 0-ply engine is concerned, and
                   OpenSpiel's action indices are deliberately not ported.
  encoder.json.gz  board (with and without dice) -> the full (26, 2, 12) tensor.
  engine.json.gz   board + dice -> the move the Python engine picks and the
                   equity it gives every candidate, under the shipped net.

Cases are stratified so the awkward paths — checkers on the bar, bear-off,
doubles, and the second half of a doubles turn — are all covered rather than
left to chance.

``--out-dir`` points into the separate raccoon-website checkout. It defaults to
the sibling layout (``../raccoon-website/test/fixtures``) and honours
``$RACCOON_WEBSITE``; where neither fits, pass it explicitly. A wrong guess is
refused rather than created — see ``raccoon/web_export.py``.

Usage:
    python scripts/export_web_fixtures.py \\
        --checkpoint experiments/exp018-distill/checkpoints/ep22.pt
"""
from __future__ import annotations

import argparse
import gzip
import json
import os

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")  # avoid CPU spin-collapse

import time
from pathlib import Path

import numpy as np
import pyspiel

from raccoon.env.encoder import encode_state
from raccoon.env.game_wrapper import BoardView, GameState, GameWrapper
from raccoon.web_export import ensure_out_dir, website_path

# Per-bucket quotas for the movegen fixture. "plain" is ordinary contact play;
# the rest are the cases a hand-written move generator gets wrong.
QUOTAS = {
    "bar": 400,        # checkers on the bar — entry is forced
    "bearoff": 400,    # borne-off checkers present — exact-roll and overshoot rules
    "doubles": 400,    # four half-moves, staged as two OpenSpiel actions
    "middoubles": 400, # the second stage, same player still on roll
    "plain": 400,
}


def canonical_board(state: pyspiel.BackgammonState, player: int) -> dict:
    """Board from ``player``'s perspective, independent of who is on roll.

    Uses the wrapper's own accessors (``GameState.board`` indexes as
    ``24 - point``; ``parse_bar_and_off`` reads the state string) so the fixture
    can't drift from what the encoder sees. Works on terminal states too, which
    is why it doesn't go through ``board_from_perspective``.
    """
    gs = GameState(state)
    points = [
        gs.board(player, (23 - i) if player == 0 else i) for i in range(24)
    ]
    opp = 1 - player
    opp_points = [
        gs.board(opp, (23 - i) if player == 0 else i) for i in range(24)
    ]
    my_bar, opp_bar, my_off, opp_off = gs.parse_bar_and_off(player)
    return {
        "my": points,
        "opp": opp_points,
        "my_bar": my_bar,
        "opp_bar": opp_bar,
        "my_off": my_off,
        "opp_off": opp_off,
    }


def board_key(board: dict) -> str:
    """Canonical string for set comparison. The JS port must format it identically."""
    return "{}|{}|{},{}|{},{}".format(
        ",".join(str(int(v)) for v in board["my"]),
        ",".join(str(int(v)) for v in board["opp"]),
        int(board["my_bar"]), int(board["opp_bar"]),
        int(board["my_off"]), int(board["opp_off"]),
    )


def board_view(board: dict, dice, mid_doubles: bool) -> BoardView:
    return BoardView(
        my_points=np.array(board["my"], dtype=np.float32),
        opp_points=np.array(board["opp"], dtype=np.float32),
        my_bar=int(board["my_bar"]),
        opp_bar=int(board["opp_bar"]),
        my_off=int(board["my_off"]),
        opp_off=int(board["opp_off"]),
        dice=tuple(dice) if dice else None,
        mid_doubles=mid_doubles,
    )


def bucket_of(view, dice, mid_doubles: bool) -> str:
    if mid_doubles:
        return "middoubles"
    if view.my_bar > 0:
        return "bar"
    if view.my_off > 0 or view.opp_off > 0:
        return "bearoff"
    if dice and dice[0] == dice[1]:
        return "doubles"
    return "plain"


def collect_cases(seed: int, quotas: dict[str, int]) -> list[dict]:
    """Play random games, recording decisions until every bucket is full."""
    from raccoon.train.lookahead import state_after_apply

    rng = np.random.default_rng(seed)
    wrapper = GameWrapper()
    counts = {k: 0 for k in quotas}
    cases: list[dict] = []
    games = 0

    while any(counts[k] < quotas[k] for k in quotas):
        state = wrapper.new_game()
        games += 1
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                idx = rng.choice(len(outcomes), p=[p for _, p in outcomes])
                state.apply_action(outcomes[idx][0])
                continue

            me = state.current_player()
            view = state.board_from_perspective()
            bucket = bucket_of(view, view.dice, view.mid_doubles)

            if counts[bucket] < quotas[bucket]:
                raw = state._state
                children = sorted({
                    board_key(canonical_board(state_after_apply(raw, a)[0], me))
                    for a in state.legal_actions()
                })
                cases.append({
                    "board": canonical_board(raw, me),
                    "dice": list(view.dice) if view.dice else None,
                    "mid_doubles": bool(view.mid_doubles),
                    "bucket": bucket,
                    "children": children,
                })
                counts[bucket] += 1

            legal = state.legal_actions()
            state.apply_action(int(legal[rng.integers(len(legal))]))

    print(f"  {len(cases)} cases from {games} random games: "
          + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return cases


def encoder_cases(cases: list[dict], n: int) -> list[dict]:
    """Encoder fixtures: the tensor for each case as-is, and pre-roll (dice cleared).

    Both variants matter — the dice and mid-doubles planes are only exercised by
    the first, and the value head only ever sees the second.
    """
    out = []
    for case in cases[:n]:
        for dice, mid in ((case["dice"], case["mid_doubles"]), (None, False)):
            view = board_view(case["board"], dice, mid)
            tensor = encode_state(view)
            out.append({
                "board": case["board"],
                "dice": list(dice) if dice else None,
                "mid_doubles": bool(mid),
                "tensor": [round(float(v), 7) for v in tensor.reshape(-1)],
            })
    return out


def collect_engine_cases(seed: int, n: int, checkpoint: str) -> list[dict]:
    """Play games with the net choosing, recording its actual decisions.

    Positions come from the net's own play rather than from the movegen cases:
    OpenSpiel has no board setter, so a fixture board cannot be replayed into a
    state. Running the same ``child_values`` lookahead the benchmark scores
    means a JS engine that reproduces these has reproduced the benchmarked
    move-selection rule, not merely a plausible one.
    """
    import torch
    from raccoon.model.network import load_model
    from raccoon.train.lookahead import child_values, state_after_apply

    torch.set_flush_denormal(True)
    network = load_model(checkpoint)
    network.eval()
    device = torch.device("cpu")

    rng = np.random.default_rng(seed)
    wrapper = GameWrapper()
    cases: list[dict] = []
    t0 = time.perf_counter()

    while len(cases) < n:
        state = wrapper.new_game()
        while not state.is_terminal() and len(cases) < n:
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                idx = rng.choice(len(outcomes), p=[p for _, p in outcomes])
                state.apply_action(outcomes[idx][0])
                continue

            me = state.current_player()
            view = state.board_from_perspective()
            raw = state._state
            legal, values, v_state = child_values(raw, network, device)

            # Keep one entry per distinct resulting board: that is the unit the
            # JS engine ranks, and duplicates would make agreement ill-defined.
            by_board: dict[str, float] = {}
            for action, value in zip(legal, values):
                key = board_key(canonical_board(state_after_apply(raw, action)[0], me))
                by_board[key] = max(by_board.get(key, -9.0), float(value))

            best_key = max(by_board, key=lambda k: by_board[k])
            cases.append({
                "board": canonical_board(raw, me),
                "dice": list(view.dice) if view.dice else None,
                "mid_doubles": bool(view.mid_doubles),
                "v_state": round(float(v_state), 6),
                "best_child": best_key,
                "children": {k: round(v, 6) for k, v in sorted(by_board.items())},
            })

            best_action = int(legal[int(np.argmax(values))])
            state.apply_action(best_action)

    print(f"  {len(cases)} engine cases in {time.perf_counter() - t0:.0f}s")
    return cases


def write_gz(path: Path, payload: dict) -> None:
    # No mkdir here: main() has already validated out_dir. Recreating it would
    # reintroduce exactly the silent-success path ensure_out_dir exists to close.
    with gzip.open(path, "wt", compresslevel=9) as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="experiments/exp018-distill/checkpoints/ep22.pt")
    # Defaults to the sibling layout; $RACCOON_WEBSITE overrides. See
    # raccoon/web_export.py for why the guess is validated rather than mkdir -p'd.
    ap.add_argument("--out-dir", default=website_path("test/fixtures"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--encoder-n", type=int, default=300)
    ap.add_argument("--engine-n", type=int, default=1000)
    args = ap.parse_args()

    out_dir = ensure_out_dir(args.out_dir)
    provenance = {
        "generator": "scripts/export_web_fixtures.py",
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "openspiel_game": "backgammon(scoring_type=full_scoring)",
        "board_key_format": "my24|opp24|my_bar,opp_bar|my_off,opp_off",
    }

    print("Collecting movegen cases...")
    cases = collect_cases(args.seed, QUOTAS)
    write_gz(out_dir / "movegen.json.gz", {"meta": provenance, "cases": cases})

    print("Encoding tensor fixtures...")
    write_gz(
        out_dir / "encoder.json.gz",
        {"meta": {**provenance, "channels": 26, "shape": [26, 2, 12]},
         "cases": encoder_cases(cases, args.encoder_n)},
    )

    print("Collecting engine cases (net in the loop)...")
    write_gz(
        out_dir / "engine.json.gz",
        {"meta": {**provenance,
                  "note": "equities are equity/3 in [-1,1] from the mover's POV"},
         "cases": collect_engine_cases(args.seed + 1, args.engine_n, args.checkpoint)},
    )


if __name__ == "__main__":
    main()
