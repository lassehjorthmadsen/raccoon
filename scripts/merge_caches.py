"""Concatenate several SOFT policy/value caches into one combined npz.

All inputs must be soft caches (``policy_actions`` (N,K) + ``policy_probs`` (N,K),
same K) as produced by ``synthesize_gnubg_dataset.py`` /
``synthesize_ondist_dataset.py``. The exp008 pipeline uses this to train each
DAgger round on the 4-ply archive + all accumulated on-distribution caches.

    python scripts/merge_caches.py --out combined.npz archive.npz ondist_1.npz ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def merge_caches(inputs: list[str], out: str) -> int:
    """Concatenate soft caches ``inputs`` into ``out``; return the example count."""
    if not inputs:
        raise ValueError("merge_caches: no input caches given")

    obs_l, act_l, prob_l, val_l = [], [], [], []
    k = None
    for p in inputs:
        d = np.load(p, allow_pickle=True)
        if "policy_actions" not in d.files:
            raise ValueError(f"{p}: not a soft cache (no policy_actions)")
        acts = d["policy_actions"]
        if k is None:
            k = acts.shape[1]
        elif acts.shape[1] != k:
            raise ValueError(f"{p}: K={acts.shape[1]} != {k} (caches disagree on MAX_K)")
        obs_l.append(d["observations"].astype(np.float32, copy=False))
        act_l.append(acts.astype(np.int32, copy=False))
        prob_l.append(d["policy_probs"].astype(np.float32, copy=False))
        val_l.append(d["value_targets"].astype(np.float32, copy=False))
        print(f"  {p}: {len(d['value_targets']):,} examples", flush=True)

    observations = np.concatenate(obs_l)
    policy_actions = np.concatenate(act_l)
    policy_probs = np.concatenate(prob_l)
    value_targets = np.concatenate(val_l)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        observations=observations,
        policy_actions=policy_actions,
        policy_probs=policy_probs,
        value_targets=value_targets,
        meta=np.array(json.dumps({
            "source": "merge_caches",
            "inputs": list(inputs),
            "n": int(len(value_targets)),
        })),
    )
    print(f"Wrote {out_path}: {len(value_targets):,} examples (K={k})", flush=True)
    return int(len(value_targets))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="Output combined npz path.")
    ap.add_argument("inputs", nargs="+", help="Soft cache npz files to concatenate.")
    args = ap.parse_args()
    merge_caches(args.inputs, args.out)


if __name__ == "__main__":
    main()
