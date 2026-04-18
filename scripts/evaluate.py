"""CLI entry point for checkpoint evaluation."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from raccoon.eval.arena import Arena
from raccoon.model.network import RaccoonNet, load_model


def latest_checkpoint(checkpoint_dir="checkpoints"):
    """Return the path to the highest-numbered checkpoint, or None."""
    files = sorted(Path(checkpoint_dir).glob("iter_*.pt"))
    return str(files[-1]) if files else None


def training_metadata(checkpoint_path):
    """Extract training metadata from a checkpoint file."""
    cp = torch.load(checkpoint_path, weights_only=False)
    meta = {"step": cp.get("step")}
    meta.update(cp.get("training", {}))
    return meta


def log_dir_from_checkpoint(checkpoint_path):
    """Derive the experiment log directory from a checkpoint path.

    If checkpoint is .../experiments/NAME/checkpoints/iter_XXXX.pt,
    returns .../experiments/NAME/logs. Otherwise returns 'logs' (cwd).
    """
    cp = Path(checkpoint_path).resolve()
    if cp.parent.name == "checkpoints" and cp.parent.parent.name != ".":
        return cp.parent.parent / "logs"
    return Path("logs")


def log_result(result, model1, args, log_dir=None):
    """Append evaluation result to eval_log.jsonl in the experiment's log dir."""
    if log_dir is None:
        log_dir = log_dir_from_checkpoint(args.checkpoint1)
    log_path = Path(log_dir) / "eval_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint1,
        "opponent": "random" if args.random else args.checkpoint2,
        "network": model1.config,
        "training": training_metadata(args.checkpoint1),
        "num_games": result.num_games,
        "simulations": args.simulations,
        "p1_wins": result.wins_p1,
        "p2_wins": result.wins_p2,
        "p1_points": result.p1_points,
        "p2_points": result.p2_points,
        "equity": round(result.equity, 4),
        "p1_gammons_won": result.p1_gammons_won,
        "p1_backgammons_won": result.p1_backgammons_won,
        "p2_gammons_won": result.p2_gammons_won,
        "p2_backgammons_won": result.p2_backgammons_won,
        "avg_game_length": round(result.avg_game_length, 1),
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"Result logged to {log_path}")


def main():
    default_cp = latest_checkpoint()

    parser = argparse.ArgumentParser(description="Evaluate checkpoints")
    parser.add_argument(
        "--checkpoint1", default=default_cp,
        help="Path to first checkpoint (default: latest)",
    )
    parser.add_argument("--checkpoint2", default=None, help="Path to second checkpoint")
    parser.add_argument(
        "--random", action="store_true", default=True,
        help="Compare against random network (default)",
    )
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--simulations", type=int, default=50)
    args = parser.parse_args()

    # --checkpoint2 implies not random
    if args.checkpoint2:
        args.random = False

    if not args.checkpoint1:
        parser.error("No checkpoints found and --checkpoint1 not specified")

    model1 = load_model(args.checkpoint1)
    print(f"Player 1: {args.checkpoint1} ({model1.config['num_blocks']}x{model1.config['channels']})")

    if args.random:
        model2 = RaccoonNet(**model1.config)
        print("Player 2: random (untrained) network")
    else:
        model2 = load_model(args.checkpoint2)
        print(f"Player 2: {args.checkpoint2} ({model2.config['num_blocks']}x{model2.config['channels']})")

    arena = Arena(
        player1=model1,
        player2=model2,
        num_games=args.games,
        num_simulations=args.simulations,
    )

    print(f"Playing {args.games} games with {args.simulations} simulations each...")
    result = arena.play_session()
    print(result.summary())
    log_result(result, model1, args)


if __name__ == "__main__":
    main()
