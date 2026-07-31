"""Time gnubg-nn labeling cost per decision at several plies.

Prices the DAgger teacher: how much slower are 3-ply/4-ply labels than the
2-ply ones exp008 used? Rolls out games with a checkpoint's raw policy
(same as ``synthesize_ondist_dataset.py``), collects decision positions, then
times ``candidate_equities`` per ply over the same positions.

    python scripts/bench_gnubg_ply.py --net <ckpt.pt> --positions 30 --plies 0 2 3 4
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import numpy as np

from raccoon.env.encoder import NUM_CHANNELS, channels_for_network, encode_state
from raccoon.env.game_wrapper import GameWrapper
from raccoon.eval.gnubg_adapter import candidate_equities
from raccoon.model.network import load_model
from raccoon.search.mcts import _advance_through_chance, select_action


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net", type=str, required=True,
                        help="Checkpoint whose policy rolls out the games.")
    parser.add_argument("--positions", type=int, default=30,
                        help="Decision positions to label per ply.")
    parser.add_argument("--plies", type=int, nargs="+", default=[0, 2, 3, 4])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    network = load_model(args.net)
    network.eval()
    channels = channels_for_network(network.config)

    # Collect a fixed set of on-distribution decision states (with clones so
    # each ply times the identical positions).
    wrapper = GameWrapper()
    states = []
    while len(states) < args.positions:
        state = _advance_through_chance(wrapper.new_game())
        while not state.is_terminal() and len(states) < args.positions:
            states.append(state.clone())
            obs = encode_state(state.board_from_perspective(), channels=channels)
            policy, _ = network.predict(obs, state.legal_actions())
            state.apply_action(select_action(policy, temperature=args.temperature))
            state = _advance_through_chance(state)
    n_moves = [len(s.legal_actions()) for s in states]
    print(f"{len(states)} positions (mean {np.mean(n_moves):.1f} legal moves), "
          f"net {args.net}")

    base = None
    for ply in args.plies:
        t0 = time.time()
        for s in states:
            candidate_equities(s, ply)
        per = (time.time() - t0) / len(states)
        rel = f"  ({per / base:.1f}x 2-ply)" if base is not None and ply != 2 else ""
        if ply == 2:
            base = per
        print(f"ply={ply}: {per:8.3f} s/decision"
              f"  ->  {3600 / per:,.0f} labels/h/core{rel}", flush=True)


if __name__ == "__main__":
    main()
