"""AlphaZero-style MCTS with chance-node handling for backgammon."""

from __future__ import annotations

import math

import numpy as np

from raccoon.env.encoder import encode_state
from raccoon.env.game_wrapper import GameState
from raccoon.model.network import RaccoonNet


class MCTSNode:
    """A node in the MCTS tree. Only decision and terminal nodes are stored."""

    __slots__ = (
        "state", "parent", "parent_action", "children",
        "visit_count", "value_sum", "prior",
    )

    def __init__(
        self,
        state: GameState,
        parent: MCTSNode | None = None,
        parent_action: int | None = None,
        prior: float = 0.0,
    ):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children: dict[int, MCTSNode] = {}
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.prior = prior

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        return len(self.children) > 0


def _advance_through_chance(state: GameState) -> GameState:
    """Sample dice and advance until we reach a decision or terminal node."""
    while state.is_chance_node():
        outcomes = state.chance_outcomes()
        actions, probs = zip(*outcomes)
        chosen = np.random.choice(actions, p=probs)
        state.apply_action(chosen)
    return state


class MCTS:
    """Monte Carlo Tree Search with PUCT selection."""

    def __init__(
        self,
        network: RaccoonNet,
        num_simulations: int = 100,
        c_puct: float = 1.5,
    ):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def search(self, state: GameState) -> dict[int, float]:
        """Run MCTS and return action -> visit count proportion."""
        root_state = state.clone()
        root_state = _advance_through_chance(root_state)

        if root_state.is_terminal():
            return {}

        root = MCTSNode(root_state)
        self._expand(root)

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # Select
            while node.is_expanded() and not node.state.is_terminal():
                action, node = self._select_child(node)
                search_path.append(node)

            # Evaluate
            if node.state.is_terminal():
                # Value from the perspective of the node's player
                returns = node.state.returns()
                # The parent made the move, so we need the value from the
                # perspective of whoever is "to move" at this terminal.
                # Since the game is over, use the return for the player who
                # moved last (the parent's player).
                if node.parent is not None:
                    parent_player = node.parent.state.current_player()
                    value = returns[parent_player]
                else:
                    value = returns[0]
            else:
                value = self._expand(node)

            # Backup
            self._backup(search_path, value)

        # Build action probabilities from visit counts
        total_visits = sum(
            child.visit_count for child in root.children.values()
        )
        if total_visits == 0:
            return {}
        return {
            action: child.visit_count / total_visits
            for action, child in root.children.items()
        }

    def _expand(self, node: MCTSNode) -> float:
        """Expand a leaf node using the network. Returns the value estimate."""
        state = node.state
        board_view = state.board_from_perspective()
        obs = encode_state(board_view)
        legal_actions = state.legal_actions()

        policy, value = self.network.predict(obs, legal_actions)

        for action, prob in policy.items():
            child_state = state.clone()
            child_state.apply_action(action)
            child_state = _advance_through_chance(child_state)
            child = MCTSNode(
                state=child_state,
                parent=node,
                parent_action=action,
                prior=prob,
            )
            node.children[action] = child

        return value

    def _select_child(self, node: MCTSNode) -> tuple[int, MCTSNode]:
        """Select the child with the highest PUCT score."""
        best_score = float("-inf")
        best_action = -1
        best_child = None

        sqrt_parent = math.sqrt(node.visit_count)

        for action, child in node.children.items():
            # PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N(s,a))
            q = child.q_value
            exploration = (
                self.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            )
            # Negate Q because child's value is from the opponent's perspective
            score = -q + exploration
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _backup(self, search_path: list[MCTSNode], value: float) -> None:
        """Propagate the value back up the search path, negating at each level."""
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value


def select_action(
    action_probs: dict[int, float], temperature: float = 1.0
) -> int:
    """Sample an action from the visit distribution."""
    actions = list(action_probs.keys())
    probs = np.array([action_probs[a] for a in actions])

    if temperature == 0:
        return actions[np.argmax(probs)]

    # Apply temperature
    probs = probs ** (1.0 / temperature)
    probs = probs / probs.sum()
    return int(np.random.choice(actions, p=probs))
