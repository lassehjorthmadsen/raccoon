"""Tests for the batched inference server."""

import threading

import numpy as np
import pytest

from raccoon.model.network import RaccoonNet
from raccoon.train.inference_server import InferenceServer


@pytest.fixture
def network():
    return RaccoonNet()


@pytest.fixture
def server(network):
    s = InferenceServer(network, batch_size=8)
    s.start()
    yield s
    s.stop()


def test_single_request(server):
    obs = np.random.randn(17, 2, 12).astype(np.float32)
    legal_actions = [0, 1, 2, 3]
    policy, value = server.predict(obs, legal_actions)
    assert set(policy.keys()) == set(legal_actions)
    assert abs(sum(policy.values()) - 1.0) < 1e-5
    assert -1.0 <= value <= 1.0


def test_concurrent_requests(server):
    """Multiple threads get correct, independent results."""
    num_threads = 16
    results = [None] * num_threads
    legal = [0, 10, 100, 500]

    def worker(idx):
        obs = np.random.randn(17, 2, 12).astype(np.float32)
        results[idx] = server.predict(obs, legal)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for r in results:
        assert r is not None
        policy, value = r
        assert set(policy.keys()) == set(legal)
        assert abs(sum(policy.values()) - 1.0) < 1e-5
        assert -1.0 <= value <= 1.0


def test_matches_direct_predict(network):
    """InferenceServer produces the same output as RaccoonNet.predict."""
    obs = np.random.randn(17, 2, 12).astype(np.float32)
    legal = [0, 5, 10, 50, 100]

    direct_policy, direct_value = network.predict(obs, legal)

    server = InferenceServer(network, batch_size=4)
    server.start()
    try:
        server_policy, server_value = server.predict(obs, legal)
    finally:
        server.stop()

    assert abs(direct_value - server_value) < 1e-5
    for a in legal:
        assert abs(direct_policy[a] - server_policy[a]) < 1e-5
