"""Self-play game generation for training data."""

from dataclasses import dataclass

import numpy as np

from raccoon.env.encoder import encode_state
from raccoon.env.game_wrapper import GameWrapper, GameState
from raccoon.search.mcts import MCTS, select_action, _advance_through_chance


@dataclass
class TrainingExample:
    observation: np.ndarray     # (16, 2, 12)
    policy_target: np.ndarray   # (1352,)
    # Game outcome from this player's perspective, normalised to [-1, 1] by
    # dividing OpenSpiel's raw returns (±1/±2/±3 for normal/gammon/backgammon)
    # by 3. This lets the tanh-bounded value head represent the full range.
    value_target: float


@dataclass
class GameResult:
    examples: list[TrainingExample]
    num_moves: int
    outcome: float       # +1/-1 from player 0's perspective
    result_type: str     # 'normal', 'gammon', or 'backgammon'


def play_one_game(
    network,
    num_simulations: int = 100,
    temperature: float = 1.0,
    temp_threshold: int = 30,
    virtual_loss_count: int = 1,
) -> GameResult:
    """Play a complete self-play game and return training examples + stats."""
    wrapper = GameWrapper()
    mcts = MCTS(
        network, num_simulations=num_simulations,
        virtual_loss_count=virtual_loss_count,
    )

    state = wrapper.new_game()
    state = _advance_through_chance(state)

    # Collect (observation, policy, player) during the game
    history: list[tuple[np.ndarray, np.ndarray, int]] = []
    move_count = 0

    while not state.is_terminal():
        player = state.current_player()
        board_view = state.board_from_perspective()
        obs = encode_state(board_view)

        temp = temperature if move_count < temp_threshold else 0.0
        action_probs = mcts.search(state)

        if not action_probs:
            break

        # Convert action_probs dict to a full policy vector
        policy = np.zeros(1352, dtype=np.float32)
        for a, p in action_probs.items():
            policy[a] = p

        history.append((obs, policy, player))

        action = select_action(action_probs, temperature=temp)
        state.apply_action(action)
        state = _advance_through_chance(state)
        move_count += 1

    if not state.is_terminal():
        return GameResult(examples=[], num_moves=0, outcome=0.0, result_type="unknown")

    # Fill in value targets from the terminal result. Divide by 3 (the
    # backgammon return maximum) so targets land in [-1, 1] and can be
    # matched by the tanh-bounded value head.
    returns = state.returns()
    equity, result_type = state.terminal_result()
    examples = []
    for obs, policy, player in history:
        examples.append(TrainingExample(
            observation=obs,
            policy_target=policy,
            value_target=returns[player] / 3.0,
        ))

    return GameResult(
        examples=examples,
        num_moves=move_count,
        outcome=equity,
        result_type=result_type,
    )
