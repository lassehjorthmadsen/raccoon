"""AlphaZero-style MCTS with chance-node handling for backgammon."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from raccoon.env.encoder import encode_state
from raccoon.env.game_wrapper import GameState


@dataclass
class Candidate:
    """A single candidate move from MCTS analysis."""

    action: int
    visits: int
    visit_prob: float
    prior: float
    q_value: float  # from the side-to-move's perspective


@dataclass
class Analysis:
    """Full MCTS analysis of a position, for display/debugging."""

    candidates: list[Candidate]  # sorted by visits desc
    root_value: float            # network value estimate at the root
    num_simulations: int


class MCTSNode:
    """A node in the MCTS tree. Only decision and terminal nodes are stored.

    Uses lazy child creation: when a node is expanded (evaluated by the
    network), only the prior probabilities are stored. Child nodes are
    created on-demand when first selected by PUCT, avoiding expensive
    state cloning for actions that are never visited.
    """

    __slots__ = (
        "state", "parent", "parent_action", "children",
        "visit_count", "value_sum", "prior",
        "_unvisited",  # action → prior for actions not yet visited
        "_expanded",   # True after network evaluation
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
        self._unvisited: dict[int, float] = {}
        self._expanded: bool = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        return self._expanded


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
        network,
        num_simulations: int = 100,
        c_puct: float = 1.5,
        virtual_loss_count: int = 1,
    ):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.virtual_loss_count = virtual_loss_count

    def search(self, state: GameState) -> dict[int, float]:
        """Run MCTS and return action -> visit count proportion."""
        root, _ = self._run(state)
        if root is None:
            return {}

        total_visits = sum(
            child.visit_count for child in root.children.values()
        )
        if total_visits == 0:
            return {}
        return {
            action: child.visit_count / total_visits
            for action, child in root.children.items()
        }

    def analyze(self, state: GameState) -> Analysis:
        """Run MCTS and return a full Analysis with per-candidate stats."""
        root, root_value = self._run(state)
        if root is None:
            return Analysis(candidates=[], root_value=0.0, num_simulations=0)

        total_visits = sum(
            child.visit_count for child in root.children.values()
        )
        candidates: list[Candidate] = []
        for action, child in root.children.items():
            visit_prob = (
                child.visit_count / total_visits if total_visits else 0.0
            )
            # child.q_value is from the child's side-to-move perspective,
            # which is the opponent of the root player; negate to express
            # the candidate's value from the root player's perspective.
            q_from_root = -child.q_value
            candidates.append(
                Candidate(
                    action=action,
                    visits=child.visit_count,
                    visit_prob=visit_prob,
                    prior=child.prior,
                    q_value=q_from_root,
                )
            )
        candidates.sort(key=lambda c: c.visits, reverse=True)
        return Analysis(
            candidates=candidates,
            root_value=root_value,
            num_simulations=self.num_simulations,
        )

    def _run(self, state: GameState) -> tuple[MCTSNode | None, float]:
        """Run the full MCTS loop and return (root, root_value).

        Returns (None, 0.0) if the root state is terminal (no moves).
        """
        root_state = state.clone()
        root_state = _advance_through_chance(root_state)

        if root_state.is_terminal():
            return None, 0.0

        root = MCTSNode(root_state)
        root_value = self._expand(root)

        if self.virtual_loss_count <= 1:
            self._run_sequential(root)
        else:
            self._run_batched(root)

        return root, root_value

    def _run_sequential(self, root: MCTSNode) -> None:
        """Original unbatched MCTS loop."""
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            while node.is_expanded() and not node.state.is_terminal():
                action, node = self._select_child(node)
                search_path.append(node)

            if node.state.is_terminal():
                value = self._terminal_value(node)
            else:
                value = self._expand(node)

            self._backup(search_path, value)

    def _run_batched(self, root: MCTSNode) -> None:
        """Batched MCTS loop using virtual loss for leaf diversification.

        Selects V leaves simultaneously (applying virtual loss to encourage
        diverse paths), evaluates them in one batched network call, then
        reverts virtual loss and backs up real values.
        """
        sim = 0
        while sim < self.num_simulations:
            V = min(self.virtual_loss_count, self.num_simulations - sim)

            pending: list[tuple[list[MCTSNode], float | None]] = []
            needs_eval: list[int] = []
            seen_leaves: set[int] = set()

            for _ in range(V):
                node = root
                search_path = [node]
                while node.is_expanded() and not node.state.is_terminal():
                    action, node = self._select_child(node)
                    search_path.append(node)

                if node.state.is_terminal():
                    pending.append((search_path, self._terminal_value(node)))
                elif id(node) not in seen_leaves:
                    seen_leaves.add(id(node))
                    pending.append((search_path, None))
                    needs_eval.append(len(pending) - 1)
                else:
                    continue

                # Apply virtual loss: inflate visit count, add pessimistic value
                for n in search_path:
                    n.visit_count += 1
                    n.value_sum -= 1.0

            if not pending:
                break

            if needs_eval:
                obs_list = []
                legal_list = []
                for idx in needs_eval:
                    leaf = pending[idx][0][-1]
                    obs_list.append(
                        encode_state(leaf.state.board_from_perspective())
                    )
                    legal_list.append(leaf.state.legal_actions())
                eval_results = self.network.predict_batch(obs_list, legal_list)

            eval_i = 0
            for search_path, terminal_value in pending:
                # Revert virtual loss
                for n in search_path:
                    n.visit_count -= 1
                    n.value_sum += 1.0

                if terminal_value is not None:
                    value = terminal_value
                else:
                    policy, value = eval_results[eval_i]
                    eval_i += 1
                    self._expand_with_policy(search_path[-1], policy)

                self._backup(search_path, value)

            sim += len(pending)

    @staticmethod
    def _terminal_value(node: MCTSNode) -> float:
        """Get the backup value for a terminal node."""
        returns = node.state.returns()
        parent_player = node.parent.state.current_player()
        return -returns[parent_player] / 3.0

    def _expand(self, node: MCTSNode) -> float:
        """Expand a leaf node using the network. Returns the value estimate."""
        state = node.state
        obs = encode_state(state.board_from_perspective())
        legal_actions = state.legal_actions()
        policy, value = self.network.predict(obs, legal_actions)
        self._expand_with_policy(node, policy)
        return value

    def _expand_with_policy(
        self, node: MCTSNode, policy: dict[int, float],
    ) -> None:
        """Mark a leaf as expanded by storing its policy priors.

        Child nodes are NOT created here — they are created lazily in
        _select_child when first visited, avoiding state cloning for
        actions that never get explored.
        """
        node._unvisited = dict(policy)
        node._expanded = True

    def _select_child(self, node: MCTSNode) -> tuple[int, MCTSNode]:
        """Select the child with the highest PUCT score.

        Handles both visited children (in node.children) and unvisited
        actions (in node._unvisited). Creates the child node lazily on
        first visit.
        """
        best_score = float("-inf")
        best_action = -1
        best_child = None

        sqrt_parent = math.sqrt(node.visit_count)

        # Score visited children
        for action, child in node.children.items():
            q = child.q_value
            exploration = (
                self.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            )
            score = -q + exploration
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        # Score unvisited actions (Q=0, N=0)
        for action, prior in node._unvisited.items():
            exploration = self.c_puct * prior * sqrt_parent
            if exploration > best_score:
                best_score = exploration
                best_action = action
                best_child = None  # signal: needs creation

        if best_child is None:
            # First visit — create the child node
            prior = node._unvisited.pop(best_action)
            child_state = node.state.clone()
            child_state.apply_action(best_action)
            child_state = _advance_through_chance(child_state)
            best_child = MCTSNode(
                state=child_state, parent=node,
                parent_action=best_action, prior=prior,
            )
            node.children[best_action] = best_child

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
