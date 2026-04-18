"""CLI entry point for GNUBG benchmark evaluation (M6).

Plays a cubeless money-game session between a Raccoon checkpoint and GNUBG
(via the `gnubg-nn` package) and logs the results.

Usage:
    python scripts/eval_gnubg.py --checkpoint checkpoints/best.pt \
        --games 1000 --gnubg-level world --simulations 200
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from raccoon.eval.game_log import save_match_log
from raccoon.eval.gnubg_harness import GnubgHarness
from raccoon.eval.match_log import save_match_text
from raccoon.model.network import load_model


def latest_checkpoint(checkpoint_dir: str = "checkpoints") -> str | None:
    files = sorted(Path(checkpoint_dir).glob("iter_*.pt"))
    return str(files[-1]) if files else None


def log_dir_from_checkpoint(checkpoint_path):
    """Derive the experiment log directory from a checkpoint path.

    If checkpoint is .../experiments/NAME/checkpoints/iter_XXXX.pt,
    returns .../experiments/NAME/logs. Otherwise returns 'logs' (cwd).
    """
    cp = Path(checkpoint_path).resolve()
    if cp.parent.name == "checkpoints" and cp.parent.parent.name != ".":
        return cp.parent.parent / "logs"
    return Path("logs")


def log_summary(result, args, checkpoint_path: str, log_dir: str | None = None) -> Path:
    """Append a one-line JSON summary to gnubg_eval_log.jsonl in the experiment's log dir."""
    if log_dir is None:
        log_dir = log_dir_from_checkpoint(checkpoint_path)
    log_path = Path(log_dir) / "gnubg_eval_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint_path,
        "opponent": f"gnubg(level={args.gnubg_level})",
        "num_games": result.num_games,
        "simulations": args.simulations,
        "raccoon_wins": result.raccoon_wins,
        "gnubg_wins": result.gnubg_wins,
        "raccoon_equity": round(result.raccoon_equity, 4),
        "equity_per_game": round(result.equity_per_game, 4),
        "win_rate": round(result.raccoon_win_rate, 4),
        "ci_95": round(result.confidence_interval_95, 4),
        "raccoon_gammons_won": result.raccoon_gammons_won,
        "raccoon_backgammons_won": result.raccoon_backgammons_won,
        "gnubg_gammons_won": result.gnubg_gammons_won,
        "gnubg_backgammons_won": result.gnubg_backgammons_won,
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return log_path


def main():
    default_cp = latest_checkpoint()

    parser = argparse.ArgumentParser(description="Evaluate Raccoon against GNUBG")
    parser.add_argument(
        "--checkpoint", default=default_cp,
        help="Path to model checkpoint (default: latest iter_*.pt)",
    )
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument(
        "--gnubg-level", default="world",
        help="Friendly level name (beginner/intermediate/advanced/world/supremo)",
    )
    parser.add_argument(
        "--ply", type=int, default=None,
        help="Override ply directly (0/1/2). Takes precedence over --gnubg-level.",
    )
    parser.add_argument(
        "--no-log-games", action="store_true",
        help="Skip writing per-game move logs (summary is still logged).",
    )
    args = parser.parse_args()

    if not args.checkpoint:
        parser.error("No checkpoint found and --checkpoint not specified")

    model = load_model(args.checkpoint)
    print(
        f"Raccoon: {args.checkpoint} "
        f"({model.config['num_blocks']}x{model.config['channels']})"
    )
    print(
        f"GNUBG: level={args.gnubg_level}"
        + (f" ply={args.ply}" if args.ply is not None else "")
    )
    print(
        f"Playing {args.games} games, "
        f"{args.simulations} MCTS simulations per Raccoon move..."
    )

    harness = GnubgHarness(
        raccoon_network=model,
        gnubg_level=args.gnubg_level,
        num_simulations=args.simulations,
        ply=args.ply,
        log_games=not args.no_log_games,
        raccoon_version=args.checkpoint,
    )

    result = harness.play_match(num_games=args.games)
    print(result.summary())

    summary_path = log_summary(result, args, args.checkpoint)
    print(f"Summary appended to {summary_path}")

    if not args.no_log_games:
        log_dir = log_dir_from_checkpoint(args.checkpoint)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        games_path = log_dir / f"gnubg_eval_{timestamp}.json"
        save_match_log(result.games, str(games_path))
        print(f"Per-game JSON log saved to {games_path}")

        text_path = log_dir / f"gnubg_eval_{timestamp}.txt"
        save_match_text(
            result.games,
            str(text_path),
            player1_name="Raccoon",
            player2_name="GNUBG",
            header_fields={
                "Round": Path(args.checkpoint).stem,
                "Event": f"Raccoon vs GNUBG ({args.gnubg_level})",
            },
        )
        print(f"Standard-format match log saved to {text_path}")


if __name__ == "__main__":
    main()
