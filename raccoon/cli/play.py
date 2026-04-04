"""Terminal interface for playing against Raccoon."""

import random
import sys

from raccoon.env.game_wrapper import GameWrapper, GameState
from raccoon.model.network import RaccoonNet
from raccoon.search.mcts import MCTS, select_action, _advance_through_chance


def display_board(state: GameState) -> str:
    """Return the board display string from OpenSpiel."""
    return str(state._state)


def play_interactive(network: RaccoonNet, num_simulations: int = 100):
    """Play a game against Raccoon in the terminal."""
    wrapper = GameWrapper()
    mcts = MCTS(network, num_simulations=num_simulations)

    print("=== Raccoon Backgammon ===")
    print("You are player X. Raccoon is player O.")
    print("Enter move as an action number, or 'hint' for analysis, 'quit' to exit.\n")

    state = wrapper.new_game()
    state = _advance_through_chance(state)

    human_player = 0  # X

    while not state.is_terminal():
        print(display_board(state))
        print()

        current = state.current_player()
        legal = state.legal_actions()

        if current == human_player:
            # Show legal moves
            print("Legal moves:")
            for i, a in enumerate(legal):
                print(f"  [{i}] {a}: {state.action_to_string(a)}")
            print()

            while True:
                try:
                    inp = input("Your move (number or index): ").strip()
                except EOFError:
                    return

                if inp == "quit":
                    return
                if inp == "hint":
                    action_probs = mcts.search(state)
                    if action_probs:
                        best = max(action_probs, key=action_probs.get)
                        print(
                            f"Hint: {state.action_to_string(best)} "
                            f"(visit prob {action_probs[best]:.3f})"
                        )
                    continue

                try:
                    val = int(inp)
                    # Try as index first, then as action number
                    if 0 <= val < len(legal):
                        action = legal[val]
                    elif val in legal:
                        action = val
                    else:
                        print("Invalid move. Try again.")
                        continue
                    break
                except ValueError:
                    print("Enter a number. Try again.")
        else:
            # Raccoon's turn
            print("Raccoon is thinking...")
            action_probs = mcts.search(state)
            if not action_probs:
                print("Raccoon has no legal moves!")
                break
            action = select_action(action_probs, temperature=0)
            print(f"Raccoon plays: {state.action_to_string(action)}")

        state.apply_action(action)
        state = _advance_through_chance(state)
        print()

    if state.is_terminal():
        print(display_board(state))
        equity, result_type = state.terminal_result()
        if equity > 0:
            print(f"Player X wins ({result_type})!")
        else:
            print(f"Player O wins ({result_type})!")
