"""Batched inference server for parallel self-play."""

import queue
import threading
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from raccoon.model.network import RaccoonNet


@dataclass
class InferenceRequest:
    obs: np.ndarray
    legal_actions: list[int]
    result: tuple[dict[int, float], float] | None = field(default=None, repr=False)
    event: threading.Event = field(default_factory=threading.Event)


class InferenceServer:
    """Collects single-position requests from game threads and batches them."""

    def __init__(
        self,
        network: RaccoonNet,
        batch_size: int = 32,
        max_wait_sec: float = 0.001,
    ):
        self.network = network
        self.batch_size = batch_size
        self.max_wait_sec = max_wait_sec
        self._queue: queue.Queue[InferenceRequest] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.network.eval()
        self._stop.clear()
        self._thread = threading.Thread(target=self._server_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def predict(
        self, obs: np.ndarray, legal_actions: list[int],
    ) -> tuple[dict[int, float], float]:
        req = InferenceRequest(obs=obs, legal_actions=legal_actions)
        self._queue.put(req)
        req.event.wait()
        return req.result  # type: ignore[return-value]

    def _server_loop(self) -> None:
        device = self.network.device
        num_actions = self.network.num_actions

        while not self._stop.is_set():
            batch: list[InferenceRequest] = []

            try:
                first = self._queue.get(timeout=0.05)
                batch.append(first)
            except queue.Empty:
                continue

            while len(batch) < self.batch_size:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    if len(batch) < self.batch_size:
                        # Brief wait for more requests to arrive
                        try:
                            batch.append(self._queue.get(timeout=self.max_wait_sec))
                        except queue.Empty:
                            break

            self._process_batch(batch, device, num_actions)

        # Drain remaining requests on shutdown
        while not self._queue.empty():
            batch = []
            while not self._queue.empty() and len(batch) < self.batch_size:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            if batch:
                self._process_batch(batch, device, num_actions)

    def _process_batch(
        self,
        batch: list[InferenceRequest],
        device: torch.device,
        num_actions: int,
    ) -> None:
        obs_np = np.stack([req.obs for req in batch])
        x = torch.from_numpy(obs_np).float().to(device)

        with torch.no_grad():
            logits_batch, values_batch = self.network(x)

        logits_batch = logits_batch.cpu()
        values_batch = values_batch.cpu()

        for i, req in enumerate(batch):
            logits = logits_batch[i]
            mask = torch.full((num_actions,), float("-inf"))
            mask[req.legal_actions] = 0.0
            probs = F.softmax(logits + mask, dim=0).numpy()
            policy = {a: float(probs[a]) for a in req.legal_actions}
            value = float(values_batch[i].item())
            req.result = (policy, value)
            req.event.set()
