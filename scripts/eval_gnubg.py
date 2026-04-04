"""CLI entry point for GNUBG benchmark evaluation.

This script will be completed when the GNUBG harness is implemented (M6).
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Evaluate against GNUBG")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--gnubg-path", default="gnubg")
    parser.add_argument("--gnubg-level", default="world")
    args = parser.parse_args()

    print("GNUBG harness not yet implemented. See raccoon/eval/gnubg_harness.py")


if __name__ == "__main__":
    main()
