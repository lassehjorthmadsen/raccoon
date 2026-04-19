"""Tests for the batched inference server."""

import multiprocessing as mp
import queue

import numpy as np
import pytest

from raccoon.model.network import RaccoonNet
from raccoon.train.inference_server import InferenceClient, InferenceServer


@pytest.fixture
def network():
    return RaccoonNet()


def _client_worker(request_queue, response_queue, worker_idx, result_queue):
    """Worker that sends a single inference request."""
    client = InferenceClient(request_queue, response_queue, worker_idx)
    obs = np.random.randn(17, 2, 12).astype(np.float32)
    legal_actions = [0, 1, 2, 3]
    policy, value = client.predict(obs, legal_actions)
    result_queue.put((worker_idx, dict(policy), float(value)))


def test_single_request(network):
    ctx = mp.get_context("spawn")
    request_queue = ctx.Queue()
    response_queues = [ctx.Queue()]
    result_queue = ctx.Queue()

    server = InferenceServer(network, request_queue, response_queues)

    p = ctx.Process(target=_client_worker, args=(
        request_queue, response_queues[0], 0, result_queue,
    ))
    p.start()

    # Serve until worker finishes
    while p.is_alive():
        batch = server._collect_batch()
        if batch:
            server._process_batch(batch)

    p.join()
    _, policy, value = result_queue.get()
    assert set(policy.keys()) == {0, 1, 2, 3}
    assert abs(sum(policy.values()) - 1.0) < 1e-5
    assert -1.0 <= value <= 1.0


def test_concurrent_requests(network):
    num_workers = 4
    ctx = mp.get_context("spawn")
    request_queue = ctx.Queue()
    response_queues = [ctx.Queue() for _ in range(num_workers)]
    result_queue = ctx.Queue()

    server = InferenceServer(network, request_queue, response_queues, batch_size=4)

    workers = []
    for i in range(num_workers):
        p = ctx.Process(target=_client_worker, args=(
            request_queue, response_queues[i], i, result_queue,
        ))
        p.start()
        workers.append(p)

    alive = set(range(num_workers))
    while alive:
        batch = server._collect_batch()
        if batch:
            server._process_batch(batch)
        else:
            alive = {i for i in alive if workers[i].is_alive()}

    for p in workers:
        p.join()

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    assert len(results) == num_workers
    for _, policy, value in results:
        assert abs(sum(policy.values()) - 1.0) < 1e-5
        assert -1.0 <= value <= 1.0


def test_matches_direct_predict(network):
    """InferenceServer produces the same output as RaccoonNet.predict."""
    obs = np.random.randn(17, 2, 12).astype(np.float32)
    legal = [0, 5, 10, 50, 100]

    direct_policy, direct_value = network.predict(obs, legal)

    ctx = mp.get_context("spawn")
    request_queue = ctx.Queue()
    response_queues = [ctx.Queue()]
    server = InferenceServer(network, request_queue, response_queues)

    request_queue.put((0, obs, legal))
    batch = server._collect_batch()
    server._process_batch(batch)

    server_policy, server_value = response_queues[0].get()

    assert abs(direct_value - server_value) < 1e-5
    for a in legal:
        assert abs(direct_policy[a] - server_policy[a]) < 1e-5
