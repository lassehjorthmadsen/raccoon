"""CLI entry point for training."""

import argparse

import torch

from raccoon.model.network import RaccoonNet
from raccoon.train.coach import Coach
from raccoon.train.replay_buffer import ReplayBuffer


def main():
    parser = argparse.ArgumentParser(description="Train Raccoon via self-play")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--games-per-iter", type=int, default=50)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--training-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--replay-size", type=int, default=100_000)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    args = parser.parse_args()

    network = RaccoonNet()
    optimizer = torch.optim.Adam(
        network.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    start_iter = 0

    if args.resume:
        from raccoon.model.network import load_checkpoint
        checkpoint = load_checkpoint(args.resume, network, optimizer)
        start_iter = checkpoint.get("step", 0) + 1
        print(f"Resumed from iteration {start_iter - 1}")

    replay_buffer = ReplayBuffer(max_size=args.replay_size)

    coach = Coach(
        network=network,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        num_simulations=args.simulations,
        batch_size=args.batch_size,
        games_per_iteration=args.games_per_iter,
        training_steps_per_iteration=args.training_steps,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )

    for i in range(start_iter, start_iter + args.iterations):
        metrics = coach.run_iteration(i)
        print(
            f"Iter {i}: "
            f"games={metrics['num_games']}, "
            f"positions={metrics['num_positions']}, "
            f"buffer={metrics['replay_buffer_size']}, "
            f"p_loss={metrics['policy_loss']:.4f}, "
            f"v_loss={metrics['value_loss']:.4f}, "
            f"time={metrics['total_time']:.1f}s"
        )


if __name__ == "__main__":
    main()
