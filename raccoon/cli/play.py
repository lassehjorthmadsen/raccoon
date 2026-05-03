"""Terminal interface for playing against Raccoon."""

from raccoon.cli import display
from raccoon.env.game_wrapper import GameWrapper
from raccoon.model.network import RaccoonNet
from raccoon.search.mcts import MCTS, _advance_through_chance


def play_interactive(network: RaccoonNet, num_simulations: int = 100):
    """Play a game against Raccoon in the terminal."""
    wrapper = GameWrapper()
    mcts = MCTS(network, num_simulations=num_simulations)

    print("=== Raccoon Backgammon ===")
    print("You are X, Raccoon is O. Commands: <number>, hint, quit.\n")

    state = wrapper.new_game()
    state = _advance_through_chance(state)

    human_player = 0

    while not state.is_terminal():
        print(display.render_board(state, human_player=human_player))
        print()

        current = state.current_player()
        legal = state.legal_actions()

        if current == human_player:
            print(display.format_legal_moves(state))
            print()

            action = None
            while action is None:
                try:
                    inp = input("Your move (index or action number): ").strip()
                except EOFError:
                    return
                if inp == "quit":
                    return
                if inp == "hint":
                    analysis = mcts.analyze(state)
                    print()
                    print(display.format_analysis(state, analysis))
                    print()
                    continue
                try:
                    val = int(inp)
                except ValueError:
                    print("Enter a number, 'hint', or 'quit'.")
                    continue
                if 0 <= val < len(legal):
                    action = legal[val]
                elif val in legal:
                    action = val
                else:
                    print("Invalid move. Try again.")
        else:
            print("Raccoon is thinking...")
            analysis = mcts.analyze(state)
            if not analysis.candidates:
                print("Raccoon has no legal moves!")
                break
            action = analysis.candidates[0].action
            print(f"Raccoon plays: {display.format_move(state, action)}")
            print()
            print(display.format_analysis(state, analysis))

        state.apply_action(action)
        state = _advance_through_chance(state)
        print()

    print(display.render_board(state, human_player=human_player))
    print()
    print(display.format_result(state, human_player=human_player))
