"""Training orchestrator: self-play -> replay buffer -> SGD -> checkpoint."""

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from raccoon.model.network import RaccoonNet, save_checkpoint
from raccoon.train.replay_buffer import ReplayBuffer
from raccoon.train.self_play import play_one_game


class Coach:
    """Runs the AlphaZero training loop."""

    def __init__(
        self,
        network: RaccoonNet,
        optimizer: torch.optim.Optimizer,
        replay_buffer: ReplayBuffer,
        num_simulations: int = 100,
        batch_size: int = 256,
        games_per_iteration: int = 50,
        training_steps_per_iteration: int = 100,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
    ):
        self.network = network
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.games_per_iteration = games_per_iteration
        self.training_steps_per_iteration = training_steps_per_iteration
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = uuid.uuid4().hex[:8]

    def run_iteration(self, iteration: int) -> dict:
        """One full iteration: self-play -> train -> checkpoint -> log."""
        t0 = time.time()

        # Self-play phase
        sp_start = time.time()
        game_results = self.self_play_phase()
        sp_time = time.time() - sp_start
        num_positions = sum(len(g.examples) for g in game_results)

        # Training phase
        tr_start = time.time()
        if len(self.replay_buffer) >= self.batch_size:
            metrics = self.training_phase()
        else:
            metrics = {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}
        tr_time = time.time() - tr_start

        # Checkpoint
        self.save_checkpoint(iteration)

        # Game stats
        game_lengths = [g.num_moves for g in game_results]
        outcomes = [g.outcome for g in game_results]
        result_types = [g.result_type for g in game_results]

        # Logging
        total_time = time.time() - t0
        metrics.update({
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "iteration": iteration,
            "num_games": self.games_per_iteration,
            "num_positions": num_positions,
            "replay_buffer_size": len(self.replay_buffer),
            "avg_game_length": round(sum(game_lengths) / len(game_lengths), 1),
            "avg_outcome": round(sum(outcomes) / len(outcomes), 3),
            "gammons": sum(1 for r in result_types if r == "gammon"),
            "backgammons": sum(1 for r in result_types if r == "backgammon"),
            "self_play_time": round(sp_time, 1),
            "training_time": round(tr_time, 1),
            "total_time": round(total_time, 1),
        })
        self.log_metrics(iteration, metrics)
        return metrics

    def self_play_phase(self):
        """Generate training data through self-play."""
        game_results = []
        for g in range(self.games_per_iteration):
            print(
                f"\r  Self-play: game {g + 1}/{self.games_per_iteration}",
                end="", flush=True,
            )
            result = play_one_game(
                self.network,
                num_simulations=self.num_simulations,
            )
            self.replay_buffer.add_game(result.examples)
            game_results.append(result)
        print()
        return game_results

    def training_phase(self) -> dict[str, float]:
        """Train the network on sampled positions from the replay buffer.

        ``target_value`` is already normalised to [-1, 1] by ``self_play``
        (raw backgammon returns ±1/±2/±3 divided by 3), so it can be matched
        directly against the tanh-bounded value head output.
        """
        self.network.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        steps = 0

        for _ in range(self.training_steps_per_iteration):
            obs, target_policy, target_value = self.replay_buffer.sample_batch(
                self.batch_size
            )

            policy_logits, value = self.network(obs)
            value = value.squeeze(-1)

            policy_loss = -(
                target_policy * F.log_softmax(policy_logits, dim=1)
            ).sum(dim=1).mean()
            value_loss = F.mse_loss(value, target_value)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            steps += 1

        return {
            "policy_loss": round(total_policy_loss / max(steps, 1), 4),
            "value_loss": round(total_value_loss / max(steps, 1), 4),
            "total_loss": round(
                (total_policy_loss + total_value_loss) / max(steps, 1), 4
            ),
        }

    def save_checkpoint(self, iteration: int) -> None:
        path = self.checkpoint_dir / f"iter_{iteration:04d}.pt"
        training = {
            "games_per_iteration": self.games_per_iteration,
            "training_steps_per_iteration": self.training_steps_per_iteration,
            "num_simulations": self.num_simulations,
            "batch_size": self.batch_size,
            "total_games": (iteration + 1) * self.games_per_iteration,
        }
        save_checkpoint(
            self.network, self.optimizer, step=iteration, path=str(path),
            training=training,
        )

    def log_metrics(self, iteration: int, metrics: dict) -> None:
        log_path = self.log_dir / "training_log.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")
