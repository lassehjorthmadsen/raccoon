"""Batched inference server for parallel self-play.

Game workers run in separate processes and communicate with the inference
server (which owns the GPU) via multiprocessing queues.
"""

import multiprocessing as mp
import queue

import numpy as np
import torch
import torch.nn.functional as F

from raccoon.model.network import RaccoonNet


class InferenceClient:
    """Proxy used by worker processes to request NN inference."""

    def __init__(
        self,
        request_queue: mp.Queue,
        response_queue: mp.Queue,
        worker_idx: int,
    ):
        self._request_queue = request_queue
        self._response_queue = response_queue
        self._worker_idx = worker_idx

    def predict(
        self, obs: np.ndarray, legal_actions: list[int],
    ) -> tuple[dict[int, float], float]:
        self._request_queue.put((self._worker_idx, obs, legal_actions))
        return self._response_queue.get()


class InferenceServer:
    """Collects requests from worker processes and runs batched GPU inference.

    Runs in the main process. Workers send (worker_idx, obs, legal_actions)
    on the shared request queue. The server batches these, runs a forward
    pass, and puts (policy, value) on each worker's response queue.
    """

    def __init__(
        self,
        network: RaccoonNet,
        request_queue: mp.Queue,
        response_queues: list[mp.Queue],
        batch_size: int = 32,
        max_wait_sec: float = 0.001,
    ):
        self.network = network
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.batch_size = batch_size
        self.max_wait_sec = max_wait_sec
        self.network.eval()
        self._device = network.device
        self._num_actions = network.num_actions

    def _collect_batch(self) -> list[tuple[int, np.ndarray, list[int]]]:
        batch: list[tuple[int, np.ndarray, list[int]]] = []

        try:
            first = self.request_queue.get(timeout=0.01)
            batch.append(first)
        except queue.Empty:
            return batch

        while len(batch) < self.batch_size:
            try:
                batch.append(self.request_queue.get_nowait())
            except queue.Empty:
                if len(batch) < self.batch_size:
                    try:
                        batch.append(
                            self.request_queue.get(timeout=self.max_wait_sec)
                        )
                    except queue.Empty:
                        break

        return batch

    def _process_batch(
        self,
        batch: list[tuple[int, np.ndarray, list[int]]],
    ) -> None:
        obs_np = np.stack([obs for _, obs, _ in batch])
        x = torch.from_numpy(obs_np).float().to(self._device)

        with torch.no_grad():
            logits_batch, values_batch = self.network(x)

        logits_batch = logits_batch.cpu()
        values_batch = values_batch.cpu()

        for i, (worker_idx, _, legal_actions) in enumerate(batch):
            logits = logits_batch[i]
            mask = torch.full((self._num_actions,), float("-inf"))
            mask[legal_actions] = 0.0
            probs = F.softmax(logits + mask, dim=0).numpy()
            policy = {a: float(probs[a]) for a in legal_actions}
            value = float(values_batch[i].item())
            self.response_queues[worker_idx].put((policy, value))
