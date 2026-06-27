"""Tests for exp008 on-distribution GNUBG synthesis + cache merge.

The gnubg-dependent tests are skipped when ``gnubg_nn`` isn't installed; the
merge test is pure numpy and always runs.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def test_candidate_equities_matches_pick_move_and_covers_actions():
    pytest.importorskip("gnubg_nn")
    from raccoon.env.game_wrapper import GameWrapper
    from raccoon.eval import gnubg_adapter as G
    from raccoon.search.mcts import _advance_through_chance

    np.random.seed(0)
    wrapper = GameWrapper()
    state = wrapper.new_game()
    state = _advance_through_chance(state)
    for _ in range(4):
        legal = state.legal_actions()
        cands = G.candidate_equities(state, ply=0)
        # Every legal action, in order, with a finite equity.
        assert [a for a, _ in cands] == legal
        assert all(np.isfinite(e) for _, e in cands)
        # pick_move == argmax candidate; value (best/3) is a sane money equity.
        best = max(cands, key=lambda t: t[1])[0]
        assert best == G.pick_move(state, ply=0)
        assert -1.0001 <= max(e for _, e in cands) / 3.0 <= 1.0001
        state.apply_action(best)
        state = _advance_through_chance(state)


def _write_fake_soft_cache(path, n, k=6, seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((n, 26, 2, 12)).astype(np.float32)
    actions = np.full((n, k), -1, dtype=np.int32)
    probs = np.zeros((n, k), dtype=np.float32)
    pol_rows = n // 2  # half value-only (doubles), half one-hot policy
    actions[:pol_rows, 0] = rng.integers(0, 1352, pol_rows)
    probs[:pol_rows, 0] = 1.0
    values = rng.uniform(-1, 1, n).astype(np.float32)
    np.savez_compressed(
        path, observations=obs, policy_actions=actions, policy_probs=probs,
        value_targets=values, meta=np.array(json.dumps({"source": "test"})),
    )


def test_merge_caches_roundtrip(tmp_path):
    a, b, out = tmp_path / "a.npz", tmp_path / "b.npz", tmp_path / "c.npz"
    _write_fake_soft_cache(a, 10, seed=1)
    _write_fake_soft_cache(b, 7, seed=2)
    subprocess.run(
        [sys.executable, "scripts/merge_caches.py", "--out", str(out), str(a), str(b)],
        check=True, cwd=REPO,
    )
    d = np.load(out, allow_pickle=True)
    assert d["observations"].shape == (17, 26, 2, 12)
    assert d["policy_actions"].shape == (17, 6)
    assert d["value_targets"].shape == (17,)
    valid = d["policy_actions"][:, 0] >= 0
    assert int(valid.sum()) == 5 + 3  # half of 10 + half of 7 carry a policy
    assert float(d["value_targets"].min()) >= -1.0
    assert float(d["value_targets"].max()) <= 1.0


def test_synthesize_ondist_end_to_end(tmp_path):
    pytest.importorskip("gnubg_nn")
    import torch
    from raccoon.model.network import RaccoonNet

    net = RaccoonNet(channels=8, num_blocks=1)  # tiny + fast; schema is what matters
    ckpt = tmp_path / "tiny.pt"
    torch.save(
        {"model_state_dict": net.state_dict(), "config": net.config, "step": -1}, ckpt,
    )

    out = tmp_path / "ondist.npz"
    subprocess.run(
        [sys.executable, "scripts/synthesize_ondist_dataset.py",
         "--net", str(ckpt), "--out", str(out), "--ply", "0",
         "--max-decisions", "12", "--seed", "0", "--save-every", "1000"],
        check=True, cwd=REPO,
    )
    d = np.load(out, allow_pickle=True)
    n = int(d["value_targets"].shape[0])
    assert n >= 12
    assert d["observations"].shape == (n, 26, 2, 12)
    assert d["policy_actions"].shape == (n, 6)
    assert d["policy_probs"].shape == (n, 6)
    assert float(d["value_targets"].min()) >= -1.0001
    assert float(d["value_targets"].max()) <= 1.0001
    acts, probs = d["policy_actions"], d["policy_probs"]
    for i in range(n):
        valid = acts[i] >= 0
        if valid.any():
            assert abs(float(probs[i][valid].sum()) - 1.0) < 1e-4  # soft policy row
        else:
            assert float(probs[i].sum()) == 0.0  # value-only (doubles) row
