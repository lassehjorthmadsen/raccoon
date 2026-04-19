"""CLI entry point for training."""

import argparse

import torch

from raccoon.model.network import RaccoonNet, load_checkpoint, load_model
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
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="Save checkpoint every N iterations")
    parser.add_argument("--experiment-name", type=str, required=True,
                        help="Required. Outputs go to experiments/<name>/{checkpoints,logs}/")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Concurrent self-play games (default: 8)")
    parser.add_argument("--inference-batch-size", type=int, default=32,
                        help="Batch size for batched NN inference (default: 32)")
    args = parser.parse_args()

    exp_root = f"experiments/{args.experiment_name}"
    checkpoint_dir = f"{exp_root}/checkpoints"
    log_dir = f"{exp_root}/logs"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_iter = 0

    if args.resume:
        network = load_model(args.resume)
        network.to(device)
        optimizer = torch.optim.Adam(
            network.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        checkpoint = load_checkpoint(args.resume, network, optimizer)
        # Move optimizer state tensors to the correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        start_iter = checkpoint.get("step", 0) + 1
        print(
            f"Resumed from iteration {start_iter - 1} "
            f"({network.config['num_blocks']}x{network.config['channels']})"
        )
    else:
        network = RaccoonNet(channels=args.channels, num_blocks=args.num_blocks)
        network.to(device)
        optimizer = torch.optim.Adam(
            network.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    print(f"Device: {device}")
    replay_buffer = ReplayBuffer(max_size=args.replay_size)

    coach = Coach(
        network=network,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        num_simulations=args.simulations,
        batch_size=args.batch_size,
        games_per_iteration=args.games_per_iter,
        training_steps_per_iteration=args.training_steps,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        experiment_name=args.experiment_name,
        checkpoint_every=args.checkpoint_every,
        num_workers=args.num_workers,
        inference_batch_size=args.inference_batch_size,
    )

    last_iter = start_iter + args.iterations - 1
    for i in range(start_iter, start_iter + args.iterations):
        metrics = coach.run_iteration(i, last_iteration=last_iter)
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
