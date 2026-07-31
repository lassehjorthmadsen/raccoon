"""Deterministic train/holdout split for GNUBG-label distillation caches.

exp014 compares two nets trained on the *same* 6M positions — one on GNUBG-0-ply
labels (``data/distill/0ply/run1/``), one on GNUBG-2-ply labels
(``data/distill/2ply/run1/``) — holding out the same 2M positions from both, so
R² is measured on an identical, unseen set for both arms.

The two caches are shard-for-shard byte-identical in ``observations`` (only the
labels differ), so a split that depends only on ``(split_seed, shard_name, row
index)`` — never on cache content — automatically lines up row-for-row across
``0ply/run1/`` and ``2ply/run1/`` without needing to store or share a mask file.
``scripts/train_distill.py`` (excludes holdout rows from training) and
``scripts/eval_r2.py`` (scores only holdout rows) both call :func:`held_out_mask`
so they can never drift apart.
"""
from __future__ import annotations

import zlib

import numpy as np


def held_out_mask(shard_name: str, n: int, holdout_frac: float,
                   split_seed: int) -> np.ndarray:
    """Boolean mask, length ``n``, True where the row is held out.

    Deterministic in ``(split_seed, shard_name)`` only — same shard name always
    yields the same mask, independent of which cache directory it's read from,
    the machine, or call order. ``holdout_frac <= 0`` disables holdout (all
    False), preserving old callers that train on 100% of a cache.
    """
    if holdout_frac <= 0:
        return np.zeros(n, dtype=bool)
    # crc32 (not the builtin hash()) so the seed is stable across processes/
    # machines — Python's str hash is randomized per-process by default.
    seed = zlib.crc32(f"{split_seed}:{shard_name}".encode()) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    return rng.random(n) < holdout_frac
