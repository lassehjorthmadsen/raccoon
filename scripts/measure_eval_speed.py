"""How many positions per second can each engine evaluate, and at what cost?

Every strength number in this project compares engines at matched *search
depth*. None compares them at matched *time*. This script measures the missing
quantity: the throughput of the network evaluation each engine's search is built
on, and the arithmetic that explains the difference.

It matters because search cost is evaluation cost times tree size. Two engines at
the same depth do a similar number of evaluations, so a per-evaluation gap of
three or four orders of magnitude is a per-decision gap of the same size — which
decides what depth each engine can afford in a real game.

Three things are recorded:

* **Measured throughput.** Raccoon's network at several thread counts and batch
  sizes, and ``gnubg_nn.probabilities``, on the same machine and in the same run,
  so the ratio does not depend on comparing published figures from different
  hardware.
* **Arithmetic cost.** Multiply-accumulates per position for each architecture,
  from its shape alone. If the measured ratio matches the arithmetic ratio, the
  difference is the network rather than the implementation, and no amount of
  optimisation will close it.
* **Hypothetical architectures.** What smaller networks would cost, scaled from
  the measured throughput of the one we have. These are arithmetic projections,
  not measurements: a real network has to be trained before its accuracy is
  known, and a smaller one is only useful if it stays accurate enough.

The numbers are hardware-specific. What travels between machines is the ratio
and the MAC counts, so both are recorded alongside the machine's identity.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import numpy as np
import torch


def resnet_macs(in_channels: int, channels: int, blocks: int,
                board: tuple[int, int] = (2, 12)) -> int:
    """Multiply-accumulates for one forward pass of the trunk.

    A 3x3 convolution with ``c_in`` inputs and ``c_out`` outputs costs
    ``c_in * c_out * 9`` per board square, and every residual block holds two of
    them at full width. The heads are small enough to ignore beside the trunk.
    """
    squares = board[0] * board[1]
    stem = in_channels * channels * 9 * squares
    body = blocks * 2 * (channels * channels * 9 * squares)
    return stem + body


def mlp_macs(inputs: int, hidden: int, outputs: int = 5) -> int:
    """Multiply-accumulates for a one-hidden-layer network, the shape every
    strong backgammon engine uses (GNU Backgammon 250-128-5, BGSage 244-400-5)."""
    return inputs * hidden + hidden * outputs


def time_network(net, in_channels: int, threads: int, batch: int,
                 seconds: float = 2.0) -> float:
    torch.set_num_threads(threads)
    x = torch.randn(batch, in_channels, 2, 12)
    with torch.no_grad():
        for _ in range(2):
            net.value_probs6(x)
        start = time.time()
        done = 0
        while time.time() - start < seconds:
            net.value_probs6(x)
            done += batch
        return done / (time.time() - start)


def time_gnubg(seconds: float = 2.0) -> float:
    import gnubg_nn

    board = gnubg_nn.board_from_position_id("4HPwATDgc/ABMA")
    start = time.time()
    done = 0
    while time.time() - start < seconds:
        gnubg_nn.probabilities(board, 0)
        done += 1
    return done / (time.time() - start)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="experiments/exp018-distill/checkpoints/ep22.pt")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--output", help="directory to write eval_speed.json into")
    a = ap.parse_args()

    from raccoon.model.network import load_model

    torch.set_flush_denormal(True)
    net = load_model(a.checkpoint)
    net.eval()
    cfg = net.config
    channels, blocks = cfg["channels"], cfg["num_blocks"]
    in_channels = cfg.get("in_channels", 26)
    params = sum(p.numel() for p in net.parameters())

    measured = {}
    for threads in (1, a.threads):
        for batch in (64, 512):
            measured[f"threads{threads}_batch{batch}"] = time_network(
                net, in_channels, threads, batch)
    best = max(measured.values())
    gnubg = time_gnubg()

    macs = resnet_macs(in_channels, channels, blocks)
    out = {
        "machine": {
            "node": platform.node(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "raccoon": {
            "checkpoint": a.checkpoint,
            "architecture": f"{blocks}x{channels} ResNet",
            "parameters": int(params),
            "in_channels": in_channels,
            "macs_per_position": macs,
            "boards_per_second": measured,
            "boards_per_second_best": best,
            "gflops_achieved": best * macs * 2 / 1e9,
        },
        "gnubg": {
            "architecture": "250-128-5 MLP (one net per position class)",
            "macs_per_position": mlp_macs(250, 128),
            "boards_per_second": gnubg,
        },
        "ratios": {
            "throughput_gnubg_over_raccoon": gnubg / best,
            "macs_raccoon_over_gnubg": macs / mlp_macs(250, 128),
        },
        "projected_architectures": [
            {"name": f"ResNet {b}x{c}", "macs_per_position": resnet_macs(in_channels, c, b),
             "projected_boards_per_second": best * macs / resnet_macs(in_channels, c, b)}
            for b, c in ((blocks, channels), (6, 128), (4, 64), (2, 64))
        ] + [
            {"name": f"MLP 250x{h}", "macs_per_position": mlp_macs(250, h),
             "projected_boards_per_second": best * macs / mlp_macs(250, h)}
            for h in (128, 400)
        ],
        "note": (
            "Projected rows scale the measured throughput by the arithmetic cost "
            "ratio. They say what an architecture would cost, not what it would "
            "score; accuracy needs training and measuring."
        ),
    }

    r, g = out["raccoon"], out["gnubg"]
    print(f"{r['architecture']}, {params/1e6:.1f}M params, {macs/1e6:.0f}M MACs/position")
    for key, value in r["boards_per_second"].items():
        print(f"  {key:<20} {value:>12,.0f} boards/s")
    print(f"  achieved {r['gflops_achieved']:.0f} GFLOPS")
    print(f"\nGNUBG {g['architecture']}: {g['boards_per_second']:>12,.0f} boards/s, "
          f"{g['macs_per_position']/1e6:.3f}M MACs/position")
    print(f"\nGNUBG is {out['ratios']['throughput_gnubg_over_raccoon']:,.0f}x faster; "
          f"Raccoon does {out['ratios']['macs_raccoon_over_gnubg']:,.0f}x the arithmetic")

    if a.output:
        Path(a.output).mkdir(parents=True, exist_ok=True)
        path = Path(a.output) / "eval_speed.json"
        path.write_text(json.dumps(out, indent=1) + "\n")
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
