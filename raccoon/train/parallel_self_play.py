"""Parallel self-play with per-worker local inference.

Each worker process gets its own copy of the model on GPU/CPU and
does inference locally. This eliminates IPC for inference (the main
bottleneck with the previous inference-server approach) at the cost
of duplicated model memory (~9MB per worker, negligible on a T4).
"""

import multiprocessing as mp
import queue

from raccoon.train.self_play import GameResult, play_one_game


def _worker_loop(
    network_state: dict,
    network_config: dict,
    result_queue: mp.Queue,
    game_indices: list[int],
    num_simulations: int,
    temperature: float,
    temp_threshold: int,
    virtual_loss_count: int,
) -> None:
    """Worker process: creates local network, plays games, sends results."""
    import torch
    from raccoon.model.network import RaccoonNet

    # Prevent thread oversubscription on CPU: each worker gets 1 torch thread.
    # On GPU this is a no-op since inference runs on the device.
    if not torch.cuda.is_available():
        torch.set_num_threads(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = RaccoonNet(**network_config)
    net.load_state_dict(network_state)
    net.to(device)
    net.eval()

    for game_idx in game_indices:
        result = play_one_game(
            net,
            num_simulations=num_simulations,
            temperature=temperature,
            temp_threshold=temp_threshold,
            virtual_loss_count=virtual_loss_count,
        )
        result_queue.put((game_idx, result))


def parallel_self_play(
    network,
    num_games: int,
    num_simulations: int = 100,
    temperature: float = 1.0,
    temp_threshold: int = 30,
    num_workers: int = 8,
    batch_size: int = 32,
    virtual_loss_count: int = 1,
) -> list[GameResult]:
    """Play multiple self-play games in parallel with local inference.

    Each worker process gets its own model copy on GPU/CPU. No inference
    IPC — workers only send completed GameResult objects back.
    """
    num_workers = min(num_workers, num_games)
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    # Serialize network weights to CPU for transfer to workers
    net_state = {k: v.cpu() for k, v in network.state_dict().items()}
    net_config = network.config

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
                net_state, net_config,
                result_queue, assignments[w],
                num_simulations, temperature, temp_threshold,
                virtual_loss_count,
            ),
        )
        p.start()
        workers.append(p)

    # Drain result_queue while workers run to prevent pipe buffer deadlock
    results: list[tuple[int, GameResult]] = []
    alive = set(range(len(workers)))
    while alive:
        try:
            results.append(result_queue.get(timeout=0.1))
        except queue.Empty:
            alive = {i for i in alive if workers[i].is_alive()}

    # Drain any remaining results
    while not result_queue.empty():
        results.append(result_queue.get_nowait())

    for p in workers:
        p.join()

    results.sort(key=lambda x: x[0])
    return [r for _, r in results]
