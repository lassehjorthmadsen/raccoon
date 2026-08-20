#!/usr/bin/env python3
"""Export a Raccoon checkpoint to ONNX for the in-browser engine.

The demo at raccoonbg.com runs the same network the BGSage benchmark scores,
through onnxruntime-web. Move selection there is 0-ply value lookahead (see
``child_values`` in raccoon/train/lookahead.py), so the exported graph needs the
value head only: ``equity`` ranks the candidate moves, and ``probs`` (the
outcomes6 softmax) feeds the win/gammon/backgammon panel. The policy head is
deliberately left out — nothing in the web engine reads it, and dropping it
keeps the browser honest about which rule actually picks the move.

Alongside ``<name>.onnx`` the script writes ``<name>.meta.json``, which carries
the encoder contract (channel subset, board shape), the value-head type, and the
provenance of the weights. Both the JS engine and
``scripts/eval_benchmark_pr.py --onnx`` read it, so the browser can never drift
from the encoder the weights were trained with.

``--out-dir`` points into the separate raccoon-website checkout. It defaults to
the sibling layout (``../raccoon-website/app/models``) and honours
``$RACCOON_WEBSITE``; where neither fits, pass it explicitly. A wrong guess is
refused rather than created — see ``raccoon/web_export.py``.

Usage:
    # fp32 (~48 MB for a 10x256 net), verified against torch:
    python scripts/export_web_model.py \\
        --checkpoint experiments/exp018-distill/checkpoints/ep22.pt

    # also emit a statically-quantised int8 copy (~4x smaller):
    python scripts/export_web_model.py ... --quantize

int8 was measured and rejected for the shipped model (2026-08-13): 12.0 MB instead
of 47.5 MB, but PR 8.53 vs 1.20 on the first 400 benchmark decisions and rollout R^2
0.78 vs 0.997 — the 10-block trunk accumulates far too much quantisation error, and
per-channel weights with percentile calibration did not rescue it (mean equity drift
0.157 points, corr 0.88). The flag stays for future nets; the site ships fp32. The
real fix for download size is a small distilled student net, not quantisation.

Scoring the exported file on the benchmark (the number the site quotes):
    python scripts/eval_benchmark_pr.py --onnx ../raccoon-website/app/models/<name>.onnx
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")  # avoid CPU spin-collapse

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from raccoon.env.encoder import channels_for_network, encode_state  # noqa: E402
from raccoon.model.network import RaccoonNet, load_model  # noqa: E402
from raccoon.web_export import ensure_out_dir, website_path  # noqa: E402

OPSET = 17


class WebValueModel(nn.Module):
    """Value-only view of a ``RaccoonNet``: (N, C, 2, 12) -> equity[, probs].

    ``equity`` is equity/3 in [-1, 1] from the position's to-move perspective —
    the same quantity ``RaccoonNet.value_equity`` returns, so a browser ranking
    moves by it reproduces the benchmarked move-selection rule exactly. For an
    ``outcomes6`` net the six-outcome softmax is exported as a second output
    (win/gammon/backgammon each way); a scalar net has no such distribution and
    exports ``equity`` alone rather than a zero-filled stand-in.
    """

    def __init__(self, net: RaccoonNet):
        super().__init__()
        self.net = net
        self.outcomes6 = net.value_head == "outcomes6"

    def forward(self, x: torch.Tensor):
        _, value_out = self.net(x)
        if not self.outcomes6:
            return value_out.squeeze(-1)
        probs = F.softmax(value_out, dim=-1)
        points = torch.tensor(
            RaccoonNet._OUTCOME_POINTS, dtype=probs.dtype, device=probs.device
        )
        equity = (probs * points).sum(dim=-1) / 3.0
        return equity, probs


def sample_positions(n: int, channels: list[int] | None) -> np.ndarray:
    """Encode ``n`` benchmark candidate positions, for verification/calibration.

    Reuses the benchmark's own board conversion so the tensors are exactly what
    ``score_raccoon`` feeds the network — no second implementation to drift.
    """
    from eval_benchmark_pr import (
        DEFAULT_BENCHMARK,
        flip_board_to_opp,
        flipped_to_board_view,
        load_benchmark,
    )

    decisions, _ = load_benchmark(DEFAULT_BENCHMARK)
    obs: list[np.ndarray] = []
    for dec in decisions:
        for move in dec["moves"]:
            bv = flipped_to_board_view(flip_board_to_opp(move["board"]))
            obs.append(encode_state(bv, channels=channels))
            if len(obs) >= n:
                return np.stack(obs)
    return np.stack(obs)


def export(model: WebValueModel, obs: np.ndarray, path: Path) -> None:
    """Write the ONNX graph with a dynamic batch axis."""
    output_names = ["equity", "probs"] if model.outcomes6 else ["equity"]
    dynamic_axes = {name: {0: "batch"} for name in ["obs"] + output_names}
    example = torch.from_numpy(obs[:4]).float()

    torch.onnx.export(
        model,
        (example,),
        str(path),
        input_names=["obs"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )


def verify(path: Path, model: WebValueModel, obs: np.ndarray) -> float:
    """Return the max abs difference in equity between ONNX and torch."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    onnx_eq = sess.run(["equity"], {"obs": obs.astype(np.float32)})[0].reshape(-1)
    with torch.no_grad():
        out = model(torch.from_numpy(obs).float())
    torch_eq = (out[0] if model.outcomes6 else out).numpy().reshape(-1)
    return float(np.abs(onnx_eq - torch_eq).max())


def quantize(src: Path, dst: Path, calib: np.ndarray) -> None:
    """Statically quantise to int8 (QDQ) using real positions for calibration.

    Dynamic quantisation is the usual first reach, but this network is almost
    entirely Conv2d and dynamic mode leaves convolutions in fp32 — no size win.
    Static QDQ quantises them properly; the accuracy cost is not assumed here,
    it is measured on the benchmark afterwards.
    """
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self, data: np.ndarray, batch: int = 32):
            self.batches = [
                {"obs": data[i:i + batch].astype(np.float32)}
                for i in range(0, len(data), batch)
            ]
            self.i = 0

        def get_next(self):
            if self.i >= len(self.batches):
                return None
            self.i += 1
            return self.batches[self.i - 1]

    prepared = src.with_suffix(".prep.onnx")
    quant_pre_process(str(src), str(prepared), skip_symbolic_shape=True)
    quantize_static(
        str(prepared),
        str(dst),
        Reader(calib),
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.Percentile,
        extra_options={"CalibMovingAverage": True},
    )
    prepared.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    # Defaults to the sibling layout; $RACCOON_WEBSITE overrides. ensure_out_dir
    # below refuses to invent the path if that guess is wrong on this machine.
    ap.add_argument("--out-dir", default=website_path("app/models"))
    ap.add_argument("--name", default=None, help="basename; default from checkpoint")
    ap.add_argument("--quantize", action="store_true", help="also emit an int8 copy")
    ap.add_argument("--verify-n", type=int, default=2000)
    ap.add_argument("--calib-n", type=int, default=512)
    args = ap.parse_args()

    torch.set_flush_denormal(True)

    # Before the checkpoint load: a bad --out-dir should not cost 47 MB of
    # deserialisation to discover.
    out_dir = ensure_out_dir(args.out_dir)

    ckpt_path = Path(args.checkpoint)
    net = load_model(str(ckpt_path))
    net.eval()
    channels = channels_for_network(net.config)
    model = WebValueModel(net).eval()

    name = args.name or f"{ckpt_path.parent.parent.name.split('-')[0]}-{ckpt_path.stem}"
    fp32_path = out_dir / f"{name}-fp32.onnx"

    print(f"checkpoint : {ckpt_path}")
    print(f"config     : {net.config}")
    print(f"params     : {sum(p.numel() for p in net.parameters()):,}")

    obs = sample_positions(max(args.verify_n, args.calib_n), channels)
    print(f"positions  : {len(obs)} encoded from the BGSage benchmark")

    t0 = time.perf_counter()
    export(model, obs, fp32_path)
    print(f"exported   : {fp32_path} "
          f"({fp32_path.stat().st_size / 1e6:.1f} MB, {time.perf_counter() - t0:.1f}s)")

    max_diff = verify(fp32_path, model, obs[:args.verify_n])
    print(f"parity     : max |onnx - torch| equity = {max_diff:.2e} over {args.verify_n}")
    if max_diff > 1e-4:
        raise SystemExit(f"fp32 parity check failed: {max_diff:.2e} > 1e-4")

    written = [fp32_path]
    if args.quantize:
        int8_path = out_dir / f"{name}-int8.onnx"
        t0 = time.perf_counter()
        quantize(fp32_path, int8_path, obs[:args.calib_n])
        diff = verify(int8_path, model, obs[:args.verify_n])
        print(f"quantised  : {int8_path} "
              f"({int8_path.stat().st_size / 1e6:.1f} MB, {time.perf_counter() - t0:.1f}s)")
        print(f"  int8 equity drift vs torch: max {diff:.2e} "
              f"(informational — the decision is PR on the benchmark, not this)")
        written.append(int8_path)

    meta = {
        "source_checkpoint": str(ckpt_path),
        "config": net.config,
        "channels": channels if channels is not None else list(range(net.config["in_channels"])),
        "board_shape": [net.config["in_channels"], net.config["board_h"], net.config["board_w"]],
        "value_head": net.value_head,
        "outputs": ["equity", "probs"] if model.outcomes6 else ["equity"],
        "outcome_points": list(RaccoonNet._OUTCOME_POINTS),
        "equity_scale": 3.0,
        "opset": OPSET,
        "files": {p.name: round(p.stat().st_size / 1e6, 1) for p in written},
        "note": (
            "equity is equity/3 in [-1,1] from the encoded position's to-move POV; "
            "multiply by equity_scale for money-game points. probs is the six-outcome "
            "softmax in order [win, win_g, win_bg, lose, lose_g, lose_bg]."
        ),
    }
    meta_path = out_dir / f"{name}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"metadata   : {meta_path}")


if __name__ == "__main__":
    main()
