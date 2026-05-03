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
    # Value target in [-1, 1] from this player's perspective. Blends the
    # terminal outcome (returns/3) with the MCTS root Q-value according to
    # value_bootstrap_alpha: 1.0 = pure outcome, 0.0 = pure Q.
    value_target: float


@dataclass
class GameResult:
    examples: list[TrainingExample]
    num_moves: int
    outcome: float       # +1/-1 from player 0's perspective
    result_type: str     # 'normal', 'gammon', or 'backgammon'
    avg_visit_entropy: float = 0.0


def play_one_game(
    network,
    num_simulations: int = 100,
    temperature: float = 1.0,
    temp_threshold: int = 30,
    virtual_loss_count: int = 1,
    dirichlet_alpha: float = 0.0,
    noise_eps: float = 0.25,
    value_bootstrap_alpha: float = 1.0,
) -> GameResult:
    """Play a complete self-play game and return training examples + stats."""
    wrapper = GameWrapper()
    mcts = MCTS(
        network, num_simulations=num_simulations,
        virtual_loss_count=virtual_loss_count,
        dirichlet_alpha=dirichlet_alpha,
        noise_eps=noise_eps,
    )

    state = wrapper.new_game()
    state = _advance_through_chance(state)

    # Collect (observation, policy, player, root_q) during the game
    history: list[tuple[np.ndarray, np.ndarray, int, float]] = []
    entropies: list[float] = []
    move_count = 0

    while not state.is_terminal():
        player = state.current_player()
        board_view = state.board_from_perspective()
        obs = encode_state(board_view)

        temp = temperature if move_count < temp_threshold else 0.0
        action_probs, move_entropy, root_q = mcts.search_with_value(state)
        entropies.append(move_entropy)

        if not action_probs:
            break

        # Convert action_probs dict to a full policy vector
        policy = np.zeros(1352, dtype=np.float32)
        for a, p in action_probs.items():
            policy[a] = p

        history.append((obs, policy, player, root_q))

        action = select_action(action_probs, temperature=temp)
        state.apply_action(action)
        state = _advance_through_chance(state)
        move_count += 1

    if not state.is_terminal():
        return GameResult(examples=[], num_moves=0, outcome=0.0, result_type="unknown")

    # Blend terminal outcome with MCTS root Q-value.
    # alpha=1.0: pure terminal outcome (original behaviour).
    # alpha=0.0: pure MCTS Q (zero-variance bootstrap, but biased early in training).
    # Intermediate alpha reduces noise for early-game positions while retaining
    # some terminal signal. Both terms are already in [-1, 1].
    returns = state.returns()
    equity, result_type = state.terminal_result()
    examples = []
    for obs, policy, player, root_q in history:
        outcome = returns[player] / 3.0
        value_target = float(np.clip(
            value_bootstrap_alpha * outcome + (1.0 - value_bootstrap_alpha) * root_q,
            -1.0, 1.0,
        ))
        examples.append(TrainingExample(
            observation=obs,
            policy_target=policy,
            value_target=value_target,
        ))

    return GameResult(
        examples=examples,
        num_moves=move_count,
        outcome=equity,
        result_type=result_type,
        avg_visit_entropy=float(np.mean(entropies)) if entropies else 0.0,
    )
