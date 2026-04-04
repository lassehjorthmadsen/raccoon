"""Raccoon Game Protocol (RGP) — text-based engine communication.

Inspired by UCI (chess). Communication via stdin/stdout.

Controller -> Engine commands:
    rgp                     identify protocol
    isready                 ready check
    newgame                 start new game
    position <state_str>    set board position (OpenSpiel state string)
    dice <d1> <d2>          set dice for next move
    go simulations <N>      search with N simulations
    quit                    exit

Engine -> Controller responses:
    id name Raccoon v0.1
    rgpok                   protocol acknowledged
    readyok                 ready
    bestmove <action>       chosen move (OpenSpiel action index)
    info score <value> nodes <N>    search info
"""

import sys

import pyspiel

from raccoon.env.encoder import encode_state
from raccoon.env.game_wrapper import GameWrapper, GameState
from raccoon.model.network import RaccoonNet
from raccoon.search.mcts import MCTS, select_action


class RGPEngine:
    """Raccoon Game Protocol engine."""

    def __init__(self, network: RaccoonNet, default_simulations: int = 100):
        self.network = network
        self.default_simulations = default_simulations
        self.wrapper = GameWrapper()
        self.state: GameState | None = None
        self.running = True

    def run(self, input_stream=None, output_stream=None):
        """Main loop: read commands from stdin, write responses to stdout."""
        inp = input_stream or sys.stdin
        out = output_stream or sys.stdout

        for line in inp:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            cmd = parts[0]

            if cmd == "rgp":
                out.write("id name Raccoon v0.1\n")
                out.write("rgpok\n")
                out.flush()

            elif cmd == "isready":
                out.write("readyok\n")
                out.flush()

            elif cmd == "newgame":
                self.state = self.wrapper.new_game()
                # Advance through the initial dice roll
                from raccoon.search.mcts import _advance_through_chance
                self.state = _advance_through_chance(self.state)
                out.write("readyok\n")
                out.flush()

            elif cmd == "dice" and len(parts) >= 3:
                # Dice are set by advancing through chance with specific values
                # For now, just acknowledge — dice are part of the OpenSpiel state
                out.write("readyok\n")
                out.flush()

            elif cmd == "go":
                if self.state is None:
                    out.write("info error no game in progress\n")
                    out.flush()
                    continue

                sims = self.default_simulations
                if "simulations" in parts:
                    idx = parts.index("simulations")
                    if idx + 1 < len(parts):
                        sims = int(parts[idx + 1])

                mcts = MCTS(self.network, num_simulations=sims)
                action_probs = mcts.search(self.state)

                if action_probs:
                    action = select_action(action_probs, temperature=0)
                    move_str = self.state.action_to_string(action)
                    out.write(f"info score {action_probs[action]:.4f} nodes {sims}\n")
                    out.write(f"bestmove {action} {move_str}\n")
                else:
                    out.write("info error no legal moves\n")
                out.flush()

            elif cmd == "quit":
                self.running = False
                break

            out.flush()

        if not self.running:
            return
