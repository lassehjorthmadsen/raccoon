"""CLI entry point for interactive play."""

import argparse

from raccoon.cli.play import play_interactive
from raccoon.model.network import RaccoonNet, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Play against Raccoon")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    parser.add_argument("--simulations", type=int, default=100)
    args = parser.parse_args()

    network = RaccoonNet()
    if args.checkpoint:
        load_checkpoint(args.checkpoint, network)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint provided — using random network")

    play_interactive(network, num_simulations=args.simulations)


if __name__ == "__main__":
    main()
