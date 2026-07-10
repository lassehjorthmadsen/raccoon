"""Re-encode stored policy/value caches to a different encoder channel set.

The pretraining caches store *encoded* observations, not raw boards. The 17
base channels are lossless (``decode_base_planes``), so a cache produced for a
17-channel net (v5 / exp008 lineage) can be upgraded to the full 26-channel
Fix-N encoder — handcrafted features recomputed from the decoded board —
without touching GNUBG. All other arrays (policy targets, values, meta) pass
through unchanged.

Typical use (upgrade the archive + all exp008 on-dist caches):

    python scripts/reencode_cache.py --out-dir data/caches_26ch --verify \
        data/bglab/gnubg4ply_cache_dbl.npz \
        experiments/exp008-ondist-10x256-2ply/caches/ondist_round_*.npz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from raccoon.env.encoder import (
    FEATURE_GROUPS,
    NUM_CHANNELS,
    decode_base_planes,
    encode_state,
)


def reencode(observations: np.ndarray, verify: bool) -> np.ndarray:
    """Return ``observations`` re-encoded to the full 26-channel Fix-N layout."""
    base = FEATURE_GROUPS["base"]
    out = np.empty((observations.shape[0], NUM_CHANNELS, 2, 12), dtype=np.float32)
    for i in range(observations.shape[0]):
        view = decode_base_planes(observations[i])
        out[i] = encode_state(view)
        if verify and not np.array_equal(out[i][base], observations[i][base]):
            raise AssertionError(f"round-trip mismatch at row {i}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Cache .npz files to re-encode.")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Directory for the re-encoded caches "
                             "(<stem>_26ch.npz; refuses to overwrite an input).")
    parser.add_argument("--verify", action="store_true",
                        help="Assert the re-encoded base planes exactly match "
                             "the stored ones for every row.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for in_path in map(Path, args.inputs):
        out_path = out_dir / f"{in_path.stem}_26ch.npz"
        if out_path.resolve() == in_path.resolve():
            raise SystemExit(f"refusing to overwrite input {in_path}")
        t0 = time.time()
        with np.load(in_path) as z:
            arrays = {k: z[k] for k in z.files}
        obs = arrays["observations"]
        if obs.shape[1] == NUM_CHANNELS:
            print(f"{in_path.name}: already {NUM_CHANNELS}-channel, skipping")
            continue
        arrays["observations"] = reencode(obs, verify=args.verify)
        meta = json.loads(str(arrays["meta"])) if "meta" in arrays else {}
        meta["reencoded"] = {"from_channels": int(obs.shape[1]),
                             "to_channels": NUM_CHANNELS,
                             "source": str(in_path)}
        arrays["meta"] = np.array(json.dumps(meta))
        np.savez_compressed(out_path, **arrays)
        print(f"{in_path.name} -> {out_path}  "
              f"({obs.shape[0]:,} rows, {time.time() - t0:.0f}s"
              f"{', verified' if args.verify else ''})")


if __name__ == "__main__":
    main()
