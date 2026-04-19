"""Parallel self-play using worker processes and batched inference."""

import multiprocessing as mp
import queue

from raccoon.model.network import RaccoonNet
from raccoon.train.inference_server import InferenceClient, InferenceServer
from raccoon.train.self_play import GameResult, play_one_game


def _worker_loop(
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    result_queue: mp.Queue,
    worker_idx: int,
    game_indices: list[int],
    num_simulations: int,
    temperature: float,
    temp_threshold: int,
) -> None:
    """Worker process: plays assigned games, sends results back."""
    client = InferenceClient(request_queue, response_queue, worker_idx)
    for game_idx in game_indices:
        result = play_one_game(
            client,
            num_simulations=num_simulations,
            temperature=temperature,
            temp_threshold=temp_threshold,
        )
        result_queue.put((game_idx, result))


def parallel_self_play(
    network: RaccoonNet,
    num_games: int,
    num_simulations: int = 100,
    temperature: float = 1.0,
    temp_threshold: int = 30,
    num_workers: int = 8,
    batch_size: int = 32,
) -> list[GameResult]:
    """Play multiple self-play games in parallel with batched inference.

    Worker processes run MCTS and game logic on CPU. The main process
    runs the inference server on GPU, batching NN requests for throughput.
    """
    num_workers = min(num_workers, num_games)
    ctx = mp.get_context("spawn")
    request_queue = ctx.Queue()
    response_queues = [ctx.Queue() for _ in range(num_workers)]
    result_queue = ctx.Queue()

    # Distribute games across workers
    assignments: list[list[int]] = [[] for _ in range(num_workers)]
    for i in range(num_games):
        assignments[i % num_workers].append(i)

    # Start worker processes
    workers = []
    for w in range(num_workers):
        p = ctx.Process(
            target=_worker_loop,
            args=(
                request_queue, response_queues[w], result_queue, w,
                assignments[w], num_simulations, temperature, temp_threshold,
            ),
        )
        p.start()
        workers.append(p)

    server = InferenceServer(
        network, request_queue, response_queues, batch_size=batch_size,
    )

    # Serve inference requests and collect results until all workers done.
    # Must drain result_queue regularly to prevent workers from blocking
    # on put() when the pipe buffer fills with large GameResult objects.
    results: list[tuple[int, GameResult]] = []
    alive = set(range(len(workers)))
    while alive:
        batch = server._collect_batch()
        if batch:
            server._process_batch(batch)
        else:
            alive = {i for i in alive if workers[i].is_alive()}
        while True:
            try:
                results.append(result_queue.get_nowait())
            except queue.Empty:
                break

    # Drain remaining inference requests and results
    while True:
        batch = server._collect_batch()
        if not batch:
            break
        server._process_batch(batch)
    while not result_queue.empty():
        results.append(result_queue.get_nowait())

    for p in workers:
        p.join()

    results.sort(key=lambda x: x[0])
    return [r for _, r in results]
