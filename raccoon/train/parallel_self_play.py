"""Parallel self-play using threaded games and batched inference."""

from concurrent.futures import ThreadPoolExecutor

from raccoon.model.network import RaccoonNet
from raccoon.train.inference_server import InferenceServer
from raccoon.train.self_play import GameResult, play_one_game


def parallel_self_play(
    network: RaccoonNet,
    num_games: int,
    num_simulations: int = 100,
    temperature: float = 1.0,
    temp_threshold: int = 30,
    num_workers: int = 8,
    batch_size: int = 32,
) -> list[GameResult]:
    """Play multiple self-play games in parallel with batched inference."""
    server = InferenceServer(network, batch_size=batch_size)
    server.start()
    try:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = [
                pool.submit(
                    play_one_game,
                    server,
                    num_simulations=num_simulations,
                    temperature=temperature,
                    temp_threshold=temp_threshold,
                )
                for _ in range(num_games)
            ]
            return [f.result() for f in futures]
    finally:
        server.stop()
