"""Tests for the exp014 deterministic train/holdout split (raccoon.data.cache_split)."""
import numpy as np

from raccoon.data.cache_split import held_out_mask


def test_disabled_by_default_frac():
    mask = held_out_mask("shard_w0_0000.npz", 1000, 0.0, split_seed=14)
    assert mask.dtype == bool
    assert not mask.any()


def test_deterministic_same_inputs():
    a = held_out_mask("shard_w1_0003.npz", 5000, 0.25, split_seed=14)
    b = held_out_mask("shard_w1_0003.npz", 5000, 0.25, split_seed=14)
    assert np.array_equal(a, b)


def test_independent_of_cache_content():
    # The whole point: the mask depends only on (shard name, n, frac, seed) —
    # never on the array contents — so cache/ and cache_2ply/ (byte-identical
    # shard-for-shard) get the identical split without sharing any data.
    n = 12345
    m1 = held_out_mask("shard_w2_0004.npz", n, 0.25, split_seed=14)
    m2 = held_out_mask("shard_w2_0004.npz", n, 0.25, split_seed=14)
    assert np.array_equal(m1, m2)


def test_fraction_is_roughly_right():
    mask = held_out_mask("shard_w0_0000.npz", 500_000, 0.25, split_seed=14)
    frac = mask.mean()
    assert 0.24 < frac < 0.26


def test_different_shards_get_different_masks():
    m1 = held_out_mask("shard_w0_0000.npz", 1000, 0.25, split_seed=14)
    m2 = held_out_mask("shard_w0_0001.npz", 1000, 0.25, split_seed=14)
    assert not np.array_equal(m1, m2)


def test_different_seed_gives_different_mask():
    m1 = held_out_mask("shard_w0_0000.npz", 1000, 0.25, split_seed=14)
    m2 = held_out_mask("shard_w0_0000.npz", 1000, 0.25, split_seed=99)
    assert not np.array_equal(m1, m2)
