"""CLI entry point for checkpoint evaluation."""

import argparse

import torch

from raccoon.eval.arena import Arena
from raccoon.model.network import RaccoonNet, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints")
    parser.add_argument("--checkpoint1", required=True, help="Path to first checkpoint")
    parser.add_argument("--checkpoint2", default=None, help="Path to second checkpoint")
    parser.add_argument("--random", action="store_true", help="Compare against random network")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--simulations", type=int, default=50)
    args = parser.parse_args()

    model1 = RaccoonNet()
    load_checkpoint(args.checkpoint1, model1)
    print(f"Loaded checkpoint 1: {args.checkpoint1}")

    if args.random:
        model2 = RaccoonNet()
        print("Player 2: random (untrained) network")
    elif args.checkpoint2:
        model2 = RaccoonNet()
        load_checkpoint(args.checkpoint2, model2)
        print(f"Loaded checkpoint 2: {args.checkpoint2}")
    else:
        parser.error("Provide --checkpoint2 or --random")

    arena = Arena(
        player1=model1,
        player2=model2,
        num_games=args.games,
        num_simulations=args.simulations,
    )

    print(f"Playing {args.games} games with {args.simulations} simulations each...")
    result = arena.play_match()
    print(result.summary())


if __name__ == "__main__":
    main()
