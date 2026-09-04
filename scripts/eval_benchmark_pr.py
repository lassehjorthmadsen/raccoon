#!/usr/bin/env python3
"""exp015 - PR benchmark against BGSage money-game positions.

Scores checker-play skill on the BGSage money-game benchmark (14,693 checker
decisions from 500 games) using cubeless equity as the reference.

Two complementary metrics:

1. **PR** (move selection): for each decision, does the engine pick the best
   move?  PR = mean(error) * 500 where error = best_cl_eq - chosen_cl_eq.
   Uses all 14,693 decisions across all reference tiers.

2. **Eval accuracy** (R^2 / MSE on equity): how well does the engine's raw
   value estimate track the reference cubeless equity across all ~327k
   candidate post-move positions? Split by reference tier (rollout = primary,
   3T = secondary, 3P = teacher-agreement only).

Only checker decisions are scored; cube positions are skipped.

Usage:
    # Score GNUBG at 0-ply (fast, ~5s):
    python scripts/eval_benchmark_pr.py --gnubg-ply 0

    # Score GNUBG at 0-ply and 2-ply:
    python scripts/eval_benchmark_pr.py --gnubg-ply 0 --gnubg-ply 2 --workers 8

    # Score a raccoon checkpoint:
    python scripts/eval_benchmark_pr.py \\
        --checkpoint experiments/exp011b-distill/outcomes6/checkpoints/ep3.pt \\
        --engine-label "exp011b (0-ply dist)"

    # Quick smoke test (first 200 decisions):
    python scripts/eval_benchmark_pr.py --gnubg-ply 0 --max-positions 200

    # Dump per-candidate predictions for paired analysis across checkpoints
    # (see scripts/exp016_paired_mse.py), one .npz per --engine-label:
    python scripts/eval_benchmark_pr.py \\
        --checkpoint experiments/exp011b-distill/scalar/checkpoints/ep3.pt \\
        --engine-label scalar \\
        --checkpoint experiments/exp011b-distill/outcomes6/checkpoints/ep3.pt \\
        --engine-label outcomes6 \\
        --dump-dir experiments/exp016-benchmark-revisit/dumps/

    # Full exp015 comparison:
    python scripts/eval_benchmark_pr.py \\
        --gnubg-ply 0 --gnubg-ply 2 \\
        --checkpoint path/to/exp011b.pt --engine-label "exp011b (0-ply dist)" \\
        --checkpoint path/to/exp014.pt --engine-label "exp014 (2-ply dist)" \\
        --output experiments/exp015-benchmark/results/
"""
from __future__ import annotations

import argparse
import gzip
import json
import os

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")  # avoid CPU spin-collapse

import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GAME_PLANS = ["purerace", "racing", "attacking", "priming", "anchoring"]
BLUNDER_THRESHOLD = 0.08  # equity error > 0.08 is a blunder (BGSage convention)
PR_MULTIPLIER = 500
TIERS = ["rollout", "3T", "3P"]

DEFAULT_BENCHMARK = str(
    Path(__file__).resolve().parent.parent / "data" / "bgsage"
    / "money_benchmark" / "benchmark.json.gz"
)

# Which reference a move is scored against (exp025).
#
# Every checker candidate in the benchmark carries two reference equities: what
# the position is worth with no cube (`cubeless_equity`), and what it is worth
# with the cube live (`equity`, from the same rollout). Scoring against the
# second is what makes cube-aware move ranking measurable, and it is what
# `scripts/cube_blind_floor.py` scores against — so a number here is directly
# comparable to the PR 0.336 ± 0.042 floor a *perfect cubeless* player leaves on
# the table.
#
# `cubeless` stays the default so every figure published before exp025 is
# reproduced by the same command that produced it.
METRIC_CUBELESS = "cubeless"
METRIC_CUBEFUL = "cubeful"
REFERENCE_FIELD = {METRIC_CUBELESS: "cubeless_equity", METRIC_CUBEFUL: "equity"}

# The benchmark's 500 generating games were played **with the Jacoby rule on**,
# so its cubeful references price a game in which gammons are worthless while the
# cube sits in the middle. Scoring against them therefore has to model the same
# rule, which is the one place in exp025 that is not Raccoon's non-Jacoby target
# (`goal.md`); the arena in raccoon/eval/cube_arena.py uses jacoby=False. The
# `owned (Jacoby-clean)` breakdown already reported by `by_cube_owner` bounds how
# much the difference matters.
CUBE_JACOBY_DEFAULT = True


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_benchmark(path: str) -> tuple[list[dict], dict]:
    """Load benchmark.json.gz, return (checker_decisions, meta)."""
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    meta = data["meta"]
    decisions = [d for d in data["decisions"] if d["kind"] == "checker"]
    return decisions, meta


def shard(decisions: list[dict], index: int, count: int) -> list[dict]:
    """Every ``count``-th decision starting at ``index`` — a disjoint slice.

    A multi-day full-benchmark run writes nothing until it finishes, so an
    interrupted one loses everything. Splitting it into shards makes each piece
    land on disk as it completes, and the pieces recombine exactly: PR over the
    union is the summed error over the summed count, since PR is a plain mean.

    Striding rather than slicing a contiguous block matters because decisions are
    stored in game order — a contiguous block would be a handful of whole games
    with a skewed position mix, while every stride sees the whole benchmark.
    """
    if not 0 <= index < count:
        raise ValueError(f"shard index {index} out of range for count {count}")
    return decisions[index::count]


def subsample_complement(decisions: list[dict], n: int, seed: int) -> list[dict]:
    """Everything ``subsample(decisions, n, seed)`` leaves out, in benchmark order.

    A rule tuned on one sample cannot be tested on that sample, so a confirmation
    run needs the positions the tuning never saw. Deriving the holdout from the
    same (n, seed) that defined the sample keeps the two provably disjoint,
    rather than relying on a second seed to miss by luck.
    """
    keep = {id(d) for d in subsample(decisions, n, seed)}
    return [d for d in decisions if id(d) not in keep]


def subsample(decisions: list[dict], n: int, seed: int) -> list[dict]:
    """A reproducible random subset of decisions, kept in benchmark order.

    ``--max-positions`` takes a prefix, and the benchmark is stored in game-seed
    order, so a prefix is a handful of whole games with a skewed game-plan mix. A
    sweep needs a sample representative of the full set, and every config must see
    the *same* sample, hence a fixed seed rather than a fresh draw per run.
    """
    if n >= len(decisions):
        return decisions
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(decisions), size=n, replace=False))
    return [decisions[i] for i in idx]


# ---------------------------------------------------------------------------
# Board conversion utilities
# ---------------------------------------------------------------------------

def flip_board_to_opp(board26: list[int]) -> list[int]:
    """Flip a BGSage board from mover's POV to opponent's (next-to-move) POV.

    After a checker play, the opponent is next to move. This function converts
    the post-move board (still in mover's perspective) to the opponent's
    perspective so we can evaluate V(position) from their side.

    BGSage convention:
      board[0]  = opponent's bar count (positive int)
      board[k]  = checkers on point k, k=1..24 (pos=mover, neg=opponent)
      board[25] = mover's bar count (positive int)
    """
    flipped = [0] * 26
    flipped[0] = board26[25]   # mover's bar -> new opponent's bar
    flipped[25] = board26[0]   # old opponent bar -> new player's bar
    for k in range(1, 25):
        flipped[k] = -board26[25 - k]  # mirror points AND negate signs
    return flipped


def flipped_to_board_view(flipped26: list[int]):
    """Convert a flipped BGSage board (next-to-move's POV) to raccoon BoardView.

    Returns a BoardView with dice=None (pre-roll evaluation for value head).
    Computes borne-off counts from the remaining checkers.
    """
    from raccoon.env.game_wrapper import BoardView

    my_points = np.zeros(24, dtype=np.float32)
    opp_points = np.zeros(24, dtype=np.float32)

    for k in range(1, 25):
        if flipped26[k] > 0:
            my_points[k - 1] = flipped26[k]
        elif flipped26[k] < 0:
            opp_points[k - 1] = abs(flipped26[k])

    my_bar = flipped26[25]
    opp_bar = flipped26[0]
    my_off = 15 - int(my_points.sum()) - my_bar
    opp_off = 15 - int(opp_points.sum()) - opp_bar

    return BoardView(
        my_points=my_points,
        opp_points=opp_points,
        my_bar=my_bar,
        opp_bar=opp_bar,
        my_off=my_off,
        opp_off=opp_off,
        dice=None,
        mid_doubles=False,
    )


def flipped_to_gnubg(flipped26: list[int]) -> list[list[int]]:
    """Convert a flipped BGSage board to gnubg-nn [opp_25, me_25] format.

    gnubg-nn convention: board = [slot0, slot1] where slot1 is the side on roll.
    Each slot is 25 ints: indices 0..23 = points 1..24 from that player's POV,
    index 24 = bar.

    The 'flipped' board is from the next-to-move (on-roll) player's perspective:
      flipped[k] > 0 -> on-roll player's checkers on point k
      flipped[k] < 0 -> opponent's checkers on point k
      flipped[25] = on-roll player's bar
      flipped[0] = opponent's bar
    """
    # On-roll player (slot 1): points from their own POV
    me_25 = [max(0, flipped26[i + 1]) for i in range(24)]
    me_25.append(flipped26[25])  # on-roll player's bar

    # Opponent (slot 0): points from THEIR OWN POV
    # Their point j from their POV = on-roll's point (25-j) from on-roll's POV
    # = flipped26[25-j], but with negated sign (opponent's checkers are negative)
    opp_25 = [abs(min(0, flipped26[24 - i])) for i in range(24)]
    opp_25.append(flipped26[0])  # opponent's bar

    return [opp_25, me_25]


# ---------------------------------------------------------------------------
# GNUBG scoring
# ---------------------------------------------------------------------------

def _eval_gnubg_decision(args: tuple) -> dict:
    """Score one checker decision with gnubg-nn. Worker for ProcessPoolExecutor.

    Returns dict with 'error', 'game_plan', 'tier', and per-candidate
    'predicted'/'reference' equity pairs for eval-accuracy metrics.
    """
    decision, ply, metric = args
    import gnubg_nn

    field = REFERENCE_FIELD[metric]
    best_cl = max(m[field] for m in decision["moves"])

    # Evaluate each candidate from opponent's perspective
    predicted_eqs: list[float] = []
    for move in decision["moves"]:
        flipped = flip_board_to_opp(move["board"])
        gnubg_board = flipped_to_gnubg(flipped)
        win, wg, wbg, lg, lbg = gnubg_nn.probabilities(gnubg_board, ply)
        # Cubeless equity from on-roll (opponent) perspective
        opp_eq = win + wg + wbg - (1.0 - win) - lg - lbg
        # Convert to mover's equity for comparison with cubeless_equity reference
        mover_eq = -opp_eq
        predicted_eqs.append(mover_eq)

    # Engine picks the move with highest predicted mover equity
    best_idx = int(np.argmax(predicted_eqs))
    chosen_cl = decision["moves"][best_idx][field]
    error = max(0.0, best_cl - chosen_cl)

    # Collect per-candidate pairs for eval accuracy
    reference_eqs = [m[field] for m in decision["moves"]]

    return {
        "error": error,
        "game_plan": decision["game_plan"],
        "tier": decision["tier"],
        "cube_owner": decision["cube_owner"],
        "predicted": predicted_eqs,
        "reference": reference_eqs,
        "key": decision["key"],
        "seed": decision["seed"],
    }


def score_gnubg(
    decisions: list[dict],
    ply: int,
    workers: int = 1,
    max_positions: int | None = None,
    dump_predictions: str | None = None,
    metric: str = METRIC_CUBELESS,
) -> dict:
    """Score all checker decisions with gnubg-nn at given ply.

    ``dump_predictions`` writes the same per-candidate npz layout the raccoon
    path writes, so the two engines' per-decision errors can be lined up and
    compared **paired** on identical positions. Without it only summary PR
    survives, and a summary cannot be un-averaged back into per-decision errors.
    """
    if max_positions is not None:
        decisions = decisions[:max_positions]

    n_total = len(decisions)
    label = f"GNUBG {ply}-ply"
    t0 = time.perf_counter()

    all_decision_results: list[dict] = []

    if ply == 0 or workers <= 1:
        # Sequential -- fast enough for 0-ply
        for i, dec in enumerate(decisions):
            result = _eval_gnubg_decision((dec, ply, metric))
            all_decision_results.append(result)
            if (i + 1) % 500 == 0:
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / elapsed
                eta = (n_total - i - 1) / rate
                print(
                    f"  [{label}] {i+1}/{n_total} "
                    f"({elapsed:.0f}s, ~{eta:.0f}s remaining)",
                    flush=True,
                )
    else:
        # Parallel for expensive plies
        args_list = [(dec, ply, metric) for dec in decisions]
        done = 0
        results_by_idx: dict[int, dict] = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_eval_gnubg_decision, a): idx
                for idx, a in enumerate(args_list)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results_by_idx[idx] = future.result()
                done += 1
                if done % 500 == 0:
                    elapsed = time.perf_counter() - t0
                    rate = done / elapsed
                    eta = (n_total - done) / rate
                    print(
                        f"  [{label}] {done}/{n_total} "
                        f"({elapsed:.0f}s, ~{eta:.0f}s remaining)",
                        flush=True,
                    )
        # Reassemble in order
        for idx in range(n_total):
            all_decision_results.append(results_by_idx[idx])

    elapsed = time.perf_counter() - t0
    print(f"  [{label}] Done: {n_total} decisions in {elapsed:.1f}s", flush=True)

    if dump_predictions:
        os.makedirs(os.path.dirname(dump_predictions) or ".", exist_ok=True)
        np.savez_compressed(
            dump_predictions,
            pred_eq=np.array(
                [p for r in all_decision_results for p in r["predicted"]], dtype=np.float64),
            ref_eq=np.array(
                [x for r in all_decision_results for x in r["reference"]], dtype=np.float64),
            tier=np.array(
                [r["tier"] for r in all_decision_results
                 for _ in r["predicted"]], dtype=object),
            decision_key=np.array(
                [r["key"] for r in all_decision_results
                 for _ in r["predicted"]], dtype=object),
            game_seed=np.array(
                [r["seed"] for r in all_decision_results
                 for _ in r["predicted"]], dtype=np.int64),
            game_plan=np.array(
                [r["game_plan"] for r in all_decision_results
                 for _ in r["predicted"]], dtype=object),
        )
        print(f"  [{label}] Saved predictions: {dump_predictions}", flush=True)

    result = aggregate(all_decision_results, label)
    result["metric"] = metric
    result["cubeful_rank"] = False   # GNUBG always ranks with its own evaluation
    return result


# ---------------------------------------------------------------------------
# Raccoon scoring
# ---------------------------------------------------------------------------

def _onnx_evaluator(onnx_path: str):
    """Return ``(evaluate, channels)`` for a model exported by export_web_model.py.

    Scores the exact file the browser downloads, so the PR the website quotes is
    a property of the shipped artifact rather than of the checkpoint it came
    from. The encoder contract is read from the sidecar ``<name>.meta.json`` —
    the same JSON the JS engine reads — so the two can't drift apart.
    """
    import onnxruntime as ort

    stem = re.sub(r"-(fp32|int8)$", "", Path(onnx_path).stem)
    meta_path = Path(onnx_path).with_name(stem + ".meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def evaluate(obs_batch: np.ndarray) -> np.ndarray:
        out = sess.run(["equity"], {"obs": obs_batch.astype(np.float32)})[0]
        return np.asarray(out).reshape(-1)

    return evaluate, list(meta["channels"])


def score_raccoon(
    decisions: list[dict],
    checkpoint_path: str | None,
    engine_label: str,
    device_str: str = "cpu",
    max_positions: int | None = None,
    dump_predictions: str | None = None,
    onnx_path: str | None = None,
    search_cfg=None,
    metric: str = METRIC_CUBELESS,
    cubeful_rank: bool = False,
    cube_jacoby: bool = CUBE_JACOBY_DEFAULT,
    paired_dump: str | None = None,
    error_dump: str | None = None,
) -> dict:
    """Score all checker decisions with a raccoon network.

    The network is either a torch checkpoint (`checkpoint_path`) or an exported
    ONNX file (`onnx_path`, from scripts/export_web_model.py). Everything else —
    board conversion, encoding, move selection, aggregation — is shared, so the
    two are directly comparable and an export regression shows up as a PR change.

    `search_cfg` is an optional :class:`raccoon.search.expectimax.SearchConfig`.
    With none (or depth 0) each candidate is ranked by a single static evaluation
    of its afterstate — the 0-ply rule this script has always used. With a deeper
    config the same candidates are ranked by expectimax search instead; only the
    ranking rule changes, so PR stays comparable across depths. Search needs the
    torch path (it calls the network directly), not ONNX.

    `metric` selects the reference a move is scored against — see
    :data:`REFERENCE_FIELD`. `cubeful_rank` selects how the *engine* ranks
    candidates: by the value head's equity (the default, and what has always
    shipped) or by Janowski's cubeful equity built from its six-outcome
    distribution, using each decision's own recorded cube position. The two are
    independent on purpose: scoring an unchanged cubeless engine on the cubeful
    metric is the baseline the cube-aware arm is compared against, **paired on
    the same decisions with the same net**, which is what isolates the ranking
    rule from the evaluation.

    `paired_dump` (cubeful ranking only) writes a per-DECISION .npz carrying both
    rankings' errors side by side. Both are derived from the *same* six-outcome
    forward pass — for an outcomes6 net the value head's equity is exactly
    `cubeless_equity` of its own distribution — so the two arms are paired by
    construction rather than by re-running and hoping the passes matched. That
    matters: the arms agree on the great majority of decisions, so an unpaired
    comparison of two PR numbers is far too noisy to resolve the difference
    between them. `scripts/exp025_paired_pr.py` reads the file.

    `error_dump` writes one row per DECISION -- the equity thrown away, keyed by
    decision -- so two runs can be compared paired on identical positions. The
    per-candidate `dump_predictions` file cannot substitute: it stores every
    candidate's value including the pruned ones, and re-deriving a choice from it
    would let a pruned move win on its un-marked-down static value, which is the
    error exp023 measured at 0.054-0.176 PR. The choice has to be recorded where
    it was made.

    If `dump_predictions` is given, also write a per-candidate .npz with
    aligned `pred_eq`/`ref_eq`/`tier`/`decision_key`/`game_seed`/`game_plan`
    arrays (one row per candidate move, in benchmark order) for downstream
    paired analysis across checkpoints (see scripts/exp016_paired_mse.py).
    """
    import torch
    from raccoon.model.network import load_model
    from raccoon.env.encoder import encode_state, channels_for_network

    torch.set_flush_denormal(True)  # iMac CPU: avoid denormal slowdown

    if (checkpoint_path is None) == (onnx_path is None):
        raise ValueError("score_raccoon needs exactly one of checkpoint_path/onnx_path")

    searching = search_cfg is not None and search_cfg.depth > 0
    if searching and onnx_path is not None:
        raise ValueError("search requires a torch checkpoint, not an ONNX model")
    if cubeful_rank and onnx_path is not None:
        # The exported graph carries the value head, but this scorer reads it
        # through a single equity output; wiring the six-outcome tensor out of
        # ONNX is a separate job and silently ranking cubelessly would be worse.
        raise ValueError("cubeful ranking requires a torch checkpoint, not an ONNX model")
    if cubeful_rank and searching:
        raise ValueError("cubeful ranking and expectimax search are not combined yet")
    per_decision: list[tuple] = []
    if paired_dump is not None and not cubeful_rank:
        raise ValueError("--paired-dump needs --cubeful-rank: the cubeless arm is "
                         "derived from the same six-outcome pass the cubeful one uses")
    if searching:
        from raccoon.search.expectimax import (
            board26_to_slots, pass_turn, search_values,
        )

    if max_positions is not None:
        decisions = decisions[:max_positions]

    if onnx_path is not None:
        evaluate, channels = _onnx_evaluator(onnx_path)
    else:
        # Handle GCS paths
        local_path = _maybe_download_gcs(checkpoint_path)

        # Load model
        device = torch.device(device_str)
        network = load_model(local_path)
        network.to(device)
        network.eval()
        channels = channels_for_network(network.config)

        def evaluate(obs_batch: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                x = torch.from_numpy(obs_batch).float().to(device, non_blocking=True)
                return network.value_equity(x).cpu().numpy()

        def evaluate_probs6(obs_batch: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                x = torch.from_numpy(obs_batch).float().to(device, non_blocking=True)
                return network.value_probs6(x).cpu().numpy()

    if cubeful_rank:
        from raccoon.cube.janowski import (
            cl2cf_money, cubeless_equity, jacoby_active, probs6_to_cumulative5,
        )
        from raccoon.cube.state import flip_label, is_race
        from raccoon.cube.janowski import X_CONTACT, X_RACE
        from raccoon.eval.gnubg_adapter import board_to_view
        from raccoon.search.expectimax import board26_to_slots

    n_total = len(decisions)
    all_decision_results: list[dict] = []

    # Flat per-candidate accumulators for --dump-predictions (exp016).
    dump_pred: list[float] = []
    dump_ref: list[float] = []
    dump_tier: list[str] = []
    dump_key: list[str] = []
    dump_seed: list[int] = []
    dump_plan: list[str] = []

    t0 = time.perf_counter()
    total_evals = 0

    field = REFERENCE_FIELD[metric]
    paired: dict[str, list] = {
        "error_cubeful_rank": [], "error_cubeless_rank": [], "picks_differ": [],
        "decision_key": [], "game_seed": [], "tier": [], "game_plan": [],
        "cube_owner": [],
    }

    for i, dec in enumerate(decisions):
        best_cl = max(m[field] for m in dec["moves"])

        if cubeful_rank:
            # Each candidate board is post-move, so the opponent is on roll and
            # the cube reads from their side: flip the label, price it there, and
            # negate. The decision's own `cube_owner` is from the mover's POV.
            # The cube *value* is left out — it scales every candidate equally
            # and so cannot change a ranking.
            root_view = board_to_view(board26_to_slots(dec["board"]))
            x = X_RACE if is_race(root_view) else X_CONTACT
            opp_label = flip_label(dec["cube_owner"])
            # cl2cf_money wants the rule already resolved against the cube
            # position: Jacoby only bites while the cube is still centred.
            jac = jacoby_active(opp_label, cube_jacoby)
            obs_list = []
            for move in dec["moves"]:
                bv = flipped_to_board_view(flip_board_to_opp(move["board"]))
                obs_list.append(encode_state(bv, channels=channels))
            probs6 = evaluate_probs6(np.stack(obs_list))
            probs5 = [probs6_to_cumulative5(p6) for p6 in probs6]
            predicted_eqs = [
                -cl2cf_money(p5, opp_label, x, jac) for p5 in probs5
            ]
            eligible = np.ones(len(predicted_eqs), bool)
            if paired_dump is not None:
                # The cubeless ranking this net would have used, off the same
                # forward pass: for an outcomes6 head, value_equity IS
                # cubeless_equity of its own distribution.
                cubeless_pick = int(np.argmax([-cubeless_equity(p5) for p5 in probs5]))
                cubeful_pick = int(np.argmax(predicted_eqs))
                paired["error_cubeless_rank"].append(
                    max(0.0, best_cl - dec["moves"][cubeless_pick][field]))
                paired["error_cubeful_rank"].append(
                    max(0.0, best_cl - dec["moves"][cubeful_pick][field]))
                paired["picks_differ"].append(cubeless_pick != cubeful_pick)
                paired["decision_key"].append(dec["key"])
                paired["game_seed"].append(dec["seed"])
                paired["tier"].append(dec["tier"])
                paired["game_plan"].append(dec["game_plan"])
                paired["cube_owner"].append(dec["cube_owner"])
        elif searching:
            # search_values already returns the mover's value in [-1,1].
            candidates = [pass_turn(board26_to_slots(m["board"])) for m in dec["moves"]]
            result = search_values(
                candidates, network, device, search_cfg, channels=channels,
                root=board26_to_slots(dec["board"]),
            )
            total_evals += result.evaluated
            predicted_eqs = [float(v * 3.0) for v in result.values]
            # A pruned move keeps its static value, which is a fine estimate to
            # report but not comparable with a searched one — searching marks a
            # move down, so an unsearched move can overtake on that alone. Choose
            # among searched moves only, as GNUBG does.
            eligible = result.searched
        else:
            # NOTE: this path converts boards with flip_board_to_opp +
            # flipped_to_board_view, while the search path above goes through
            # gnubg_adapter.board_to_view. The two agree to within 2e-4 in
            # equity/3 (mean 4e-7) on the same positions -- consistent with
            # float32 batching rather than a semantic difference, and pinned by
            # tests/test_expectimax.py::test_board26_conversion_matches_benchmark_encoder.
            # Far below anything measured here, but it is two encoders where
            # there should be one, and a static baseline taken from one path is
            # not bit-identical to the other's.
            obs_list = []
            for move in dec["moves"]:
                flipped = flip_board_to_opp(move["board"])
                bv = flipped_to_board_view(flipped)
                obs = encode_state(bv, channels=channels)
                obs_list.append(obs)

            # Batch forward pass
            values = evaluate(np.stack(obs_list))

            # values[j] = V from opponent's perspective (in [-1,1], equity/3 scale)
            # Convert to mover's equity on the standard [-3,3] scale for comparison:
            # mover_equity = -(opponent_value * 3)
            predicted_eqs = [float(-v * 3.0) for v in values]
            eligible = np.ones(len(predicted_eqs), bool)

        # Engine picks the move with highest predicted mover equity, among those
        # whose values are on a common scale.
        masked = np.where(eligible, predicted_eqs, -np.inf)
        best_idx = int(np.argmax(masked))
        chosen_cl = dec["moves"][best_idx][field]
        error = max(0.0, best_cl - chosen_cl)

        reference_eqs = [m[field] for m in dec["moves"]]

        all_decision_results.append({
            "error": error,
            "game_plan": dec["game_plan"],
            "tier": dec["tier"],
            "cube_owner": dec["cube_owner"],
            "predicted": predicted_eqs,
            "reference": reference_eqs,
        })

        if error_dump is not None:
            per_decision.append(
                (error, dec["key"], dec["seed"], dec["tier"], dec["game_plan"])
            )

        if dump_predictions:
            n_cand = len(predicted_eqs)
            dump_pred.extend(predicted_eqs)
            dump_ref.extend(reference_eqs)
            dump_tier.extend([dec["tier"]] * n_cand)
            dump_key.extend([dec["key"]] * n_cand)
            dump_seed.extend([dec["seed"]] * n_cand)
            dump_plan.extend([dec["game_plan"]] * n_cand)

        if (i + 1) % 500 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            eta = (n_total - i - 1) / rate
            print(
                f"  [{engine_label}] {i+1}/{n_total} "
                f"({elapsed:.0f}s, ~{eta:.0f}s remaining)",
                flush=True,
            )

    elapsed = time.perf_counter() - t0
    print(
        f"  [{engine_label}] Done: {n_total} decisions in {elapsed:.1f}s",
        flush=True,
    )

    if dump_predictions:
        os.makedirs(os.path.dirname(dump_predictions) or ".", exist_ok=True)
        np.savez_compressed(
            dump_predictions,
            pred_eq=np.array(dump_pred, dtype=np.float64),
            ref_eq=np.array(dump_ref, dtype=np.float64),
            tier=np.array(dump_tier, dtype=object),
            decision_key=np.array(dump_key, dtype=object),
            game_seed=np.array(dump_seed, dtype=np.int64),
            game_plan=np.array(dump_plan, dtype=object),
        )
        print(f"  [{engine_label}] Saved predictions: {dump_predictions}", flush=True)

    if error_dump is not None:
        os.makedirs(os.path.dirname(error_dump) or ".", exist_ok=True)
        np.savez_compressed(
            error_dump,
            error=np.array([r[0] for r in per_decision], dtype=np.float64),
            decision_key=np.array([r[1] for r in per_decision], dtype=object),
            game_seed=np.array([r[2] for r in per_decision], dtype=np.int64),
            tier=np.array([r[3] for r in per_decision], dtype=object),
            game_plan=np.array([r[4] for r in per_decision], dtype=object),
        )
        print(f"  [{engine_label}] Saved per-decision errors: {error_dump}", flush=True)

    if paired_dump is not None:
        os.makedirs(os.path.dirname(paired_dump) or ".", exist_ok=True)
        np.savez_compressed(
            paired_dump,
            error_cubeful_rank=np.array(paired["error_cubeful_rank"], dtype=np.float64),
            error_cubeless_rank=np.array(paired["error_cubeless_rank"], dtype=np.float64),
            picks_differ=np.array(paired["picks_differ"], dtype=bool),
            decision_key=np.array(paired["decision_key"], dtype=object),
            game_seed=np.array(paired["game_seed"], dtype=np.int64),
            tier=np.array(paired["tier"], dtype=object),
            game_plan=np.array(paired["game_plan"], dtype=object),
            cube_owner=np.array(paired["cube_owner"], dtype=object),
            metric=np.array(metric),
            cube_jacoby=np.array(cube_jacoby),
        )
        print(f"  [{engine_label}] Saved paired errors: {paired_dump}", flush=True)

    result = aggregate(all_decision_results, engine_label)
    result["checkpoint"] = checkpoint_path or onnx_path
    result["metric"] = metric
    result["cubeful_rank"] = cubeful_rank
    if cubeful_rank:
        result["cube_jacoby"] = cube_jacoby
    if searching:
        result["search"] = {
            "depth": search_cfg.depth,
            "k": search_cfg.k,
            "k2": search_cfg.k2,
            "threshold": search_cfg.threshold,
            "window_lo": search_cfg.window_lo,
            "window_hi": search_cfg.window_hi,
            "gate": search_cfg.gate,
            "tag": search_cfg.tag(),
            "evals_per_decision": total_evals / n_total if n_total else 0.0,
            "sec_per_decision": elapsed / n_total if n_total else 0.0,
        }
    return result


# ---------------------------------------------------------------------------
# Result aggregation and display
# ---------------------------------------------------------------------------

def aggregate(decision_results: list[dict], label: str) -> dict:
    """Compute PR, per-plan breakdown, and eval-accuracy metrics from results."""
    errors = [r["error"] for r in decision_results]
    plans = [r["game_plan"] for r in decision_results]
    n = len(errors)
    total_error = sum(errors)
    pr = (total_error / n) * PR_MULTIPLIER if n > 0 else 0.0
    blunders = sum(1 for e in errors if e > BLUNDER_THRESHOLD)

    # Per game-plan PR
    by_plan: dict[str, dict] = {}
    for plan in GAME_PLANS:
        plan_errors = [e for e, p in zip(errors, plans) if p == plan]
        if plan_errors:
            plan_total = sum(plan_errors)
            by_plan[plan] = {
                "pr": (plan_total / len(plan_errors)) * PR_MULTIPLIER,
                "n": len(plan_errors),
                "total_error": plan_total,
                "blunders": sum(1 for e in plan_errors if e > BLUNDER_THRESHOLD),
            }
        else:
            by_plan[plan] = {"pr": 0.0, "n": 0, "total_error": 0.0, "blunders": 0}

    # Per cube location. The benchmark is generated with the Jacoby rule on, which
    # suppresses gammons while the cube is CENTRED and makes the reference engine
    # play boldly there -- so those labels describe a game we are not aiming at
    # (see goal.md). The owned-cube subset is Jacoby-clean by construction, and
    # reporting the split lets any comparison be redone on it without a re-score.
    by_cube_owner: dict[str, dict] = {}
    owners = [r.get("cube_owner") for r in decision_results]
    for owner in ("centered", "player", "opponent"):
        owner_errors = [e for e, o in zip(errors, owners) if o == owner]
        if not owner_errors:
            continue
        by_cube_owner[owner] = {
            "pr": (sum(owner_errors) / len(owner_errors)) * PR_MULTIPLIER,
            "n": len(owner_errors),
            "total_error": sum(owner_errors),
            "blunders": sum(1 for e in owner_errors if e > BLUNDER_THRESHOLD),
        }
    owned = [e for e, o in zip(errors, owners) if o in ("player", "opponent")]
    if owned:
        by_cube_owner["owned (Jacoby-clean)"] = {
            "pr": (sum(owned) / len(owned)) * PR_MULTIPLIER,
            "n": len(owned),
            "total_error": sum(owned),
            "blunders": sum(1 for e in owned if e > BLUNDER_THRESHOLD),
        }

    # Eval accuracy: R^2 and MSE on predicted vs reference equity, split by tier
    eval_accuracy = {}
    for tier in TIERS + ["all"]:
        if tier == "all":
            tier_results = decision_results
        else:
            tier_results = [r for r in decision_results if r["tier"] == tier]
        if not tier_results:
            eval_accuracy[tier] = {"r2": None, "mse": None, "n_positions": 0}
            continue

        pred_all = []
        ref_all = []
        for r in tier_results:
            pred_all.extend(r["predicted"])
            ref_all.extend(r["reference"])

        pred = np.array(pred_all, dtype=np.float64)
        ref = np.array(ref_all, dtype=np.float64)

        mse = float(np.mean((pred - ref) ** 2))
        ss_res = float(np.sum((pred - ref) ** 2))
        ss_tot = float(np.sum((ref - ref.mean()) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        eval_accuracy[tier] = {
            "r2": r2,
            "mse": mse,
            "n_positions": len(pred),
        }

    return {
        "engine_label": label,
        "pr": pr,
        "n": n,
        "total_error": total_error,
        "blunders": blunders,
        "by_plan": by_plan,
        "by_cube_owner": by_cube_owner,
        "eval_accuracy": eval_accuracy,
    }


def print_table(results: list[dict]) -> None:
    """Print formatted results table."""
    metric = results[0].get("metric", METRIC_CUBELESS)
    field = REFERENCE_FIELD[metric]
    ranked = ("cubeful (Janowski)" if any(r.get("cubeful_rank") for r in results)
              else "cubeless")
    print()
    print("=" * 90)
    print(f"BGSage Money Benchmark - Checker-only, {metric} reference")
    print(f"Positions: {results[0]['n']} checker decisions from 500 games")
    print(f"Reference: {field} (Sage 3P / 3T / Rollout)")
    print(f"Engines rank candidates by: {ranked} equity")
    print("=" * 90)

    # --- Move-selection PR ---
    print()
    print("MOVE SELECTION (PR = mean error x 500, lower is better)")
    print()
    plan_short = {
        "purerace": "prace", "racing": "race",
        "attacking": "atk", "priming": "prime", "anchoring": "anch",
    }
    header = f"  {'Engine':<32} {'PR':>6} {'N':>6} {'Blndr':>6}"
    for plan in GAME_PLANS:
        header += f"  {plan_short[plan]:>6}"
    print(header)
    print("  " + "-" * 86)

    for r in results:
        row = f"  {r['engine_label']:<32} {r['pr']:>6.2f} {r['n']:>6} {r['blunders']:>6}"
        for plan in GAME_PLANS:
            bp = r["by_plan"].get(plan, {})
            if bp.get("n", 0) > 0:
                row += f"  {bp['pr']:>6.2f}"
            else:
                row += f"  {'---':>6}"
        print(row)

    # Decision counts per plan
    print()
    r0 = results[0]
    counts = f"  {'(N per plan)':<32} {'':>6} {'':>6} {'':>6}"
    for plan in GAME_PLANS:
        n = r0["by_plan"].get(plan, {}).get("n", 0)
        counts += f"  {n:>6}"
    print(counts)

    # --- Eval accuracy ---
    print()
    print(f"EVAL ACCURACY (R^2 / MSE on predicted vs reference {metric} equity)")
    if metric == METRIC_CUBEFUL and not any(r.get("cubeful_rank") for r in results):
        # A cubeless prediction against a cubeful reference is two different
        # quantities, so this block is not a calibration read here. PR is.
        print("  (cubeless predictions vs cubeful references -- not comparable "
              "quantities; read PR, not R^2)")
    print()
    header2 = f"  {'Engine':<32}"
    for tier in TIERS + ["all"]:
        header2 += f"  {'R2(' + tier + ')':>12} {'MSE':>8} {'N':>8}"
    print(header2)
    print("  " + "-" * 110)

    for r in results:
        row = f"  {r['engine_label']:<32}"
        ea = r.get("eval_accuracy", {})
        for tier in TIERS + ["all"]:
            ta = ea.get(tier, {})
            r2 = ta.get("r2")
            mse = ta.get("mse")
            n_pos = ta.get("n_positions", 0)
            if r2 is not None:
                row += f"  {r2:>12.4f} {mse:>8.4f} {n_pos:>8}"
            else:
                row += f"  {'---':>12} {'---':>8} {n_pos:>8}"
        print(row)

    print()


def sanitize_label(label: str) -> str:
    """Turn an engine label into a filesystem-safe stem, e.g. for filenames."""
    fname = label.lower()
    for ch in " ()/-.,":
        fname = fname.replace(ch, "_")
    return fname.strip("_")


# What makes two results the *same measurement*, so that re-scoring is idempotent
# rather than destructive. Anything outside this list (PR, R^2, timings) is an
# outcome and may legitimately change on a re-run.
IDENTITY_FIELDS = ("n", "checkpoint", "search", "metric", "cubeful_rank")


def _describe(value: object) -> str:
    """Render an identity field compactly for an error message."""
    if isinstance(value, dict):
        skip = {"evals_per_decision", "sec_per_decision"}
        inner = ", ".join(f"{k}={v}" for k, v in value.items() if k not in skip)
        return f"{{{inner}}}"
    return repr(value)


def check_no_clobber(result: dict, path: str,
                     fields: tuple[str, ...] = IDENTITY_FIELDS) -> None:
    """Refuse to overwrite a result file that measured something else.

    Results are keyed by engine label alone, so reusing a label for a different
    measurement silently destroys the old one -- which is how a full-benchmark
    GNUBG 1-ply number was lost to a 2,000-position re-score. Re-writing the
    *same* measurement stays idempotent; a different n, checkpoint or search
    configuration is an error the caller has to resolve.

    ``fields`` names what counts as the same measurement. It defaults to this
    script's checker-play identity; a sibling scorer with different identity
    fields (see eval_cube_pr.py) passes its own.
    """
    if not os.path.exists(path):
        return
    try:
        with open(path) as f:
            existing = json.load(f)
    except (OSError, json.JSONDecodeError):
        return  # unreadable or hand-edited: nothing to protect

    clashes = [
        (field, existing.get(field), result.get(field))
        for field in fields
        if existing.get(field) != result.get(field)
    ]
    if not clashes:
        return

    lines = [
        f"refusing to overwrite {path}",
        f"  engine label {result['engine_label']!r} already names a different measurement:",
    ]
    for field, was, now in clashes:
        lines.append(f"    {field}: on disk {_describe(was)} -> incoming {_describe(now)}")
    lines.append("  Use a different --engine-label, or --overwrite to replace it.")
    raise SystemExit("\n".join(lines))


def save_results(results: list[dict], output_dir: str, overwrite: bool = False) -> None:
    """Save per-engine JSON files and summary.json."""
    os.makedirs(output_dir, exist_ok=True)

    paths = [
        (r, os.path.join(output_dir, sanitize_label(r["engine_label"]) + ".json"))
        for r in results
    ]
    if not overwrite:
        # Check every file before writing any, so a clash cannot leave the
        # directory half-updated.
        for r, path in paths:
            check_no_clobber(r, path)

    for r, path in paths:
        with open(path, "w") as f:
            json.dump(r, f, indent=2)
        print(f"  Saved: {path}")

    # Summary
    summary = {
        "benchmark": "BGSage Money Benchmark (checker-only, cubeless)",
        "n_decisions": results[0]["n"] if results else 0,
        "results": [
            {
                "engine": r["engine_label"],
                "pr": r["pr"],
                "blunders": r["blunders"],
                "eval_accuracy": r.get("eval_accuracy", {}),
            }
            for r in results
        ],
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _maybe_download_gcs(path: str) -> str:
    """If path starts with gs://, download to a temp file and return local path."""
    if not path.startswith("gs://"):
        return path
    local = os.path.join(tempfile.gettempdir(), "raccoon_checkpoint.pt")
    print(f"  Downloading {path} -> {local}...")
    result = subprocess.run(
        ["gsutil", "cp", path, local],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR: gsutil failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return local


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score engines against BGSage money-game benchmark (checker PR).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--benchmark", type=str, default=DEFAULT_BENCHMARK,
        help="Path to benchmark.json.gz",
    )
    parser.add_argument(
        "--gnubg-ply", type=int, action="append", dest="gnubg_plies",
        help="Score GNUBG at this ply (repeatable)",
    )
    parser.add_argument(
        "--checkpoint", type=str, action="append", dest="checkpoints",
        help="Path to a raccoon .pt checkpoint (repeatable)",
    )
    parser.add_argument(
        "--engine-label", type=str, action="append", dest="labels",
        help="Human label for each checkpoint (must match --checkpoint count)",
    )
    parser.add_argument(
        "--onnx", type=str, action="append", dest="onnx_models",
        help=(
            "Path to an ONNX model exported by scripts/export_web_model.py "
            "(repeatable). Scores the file the browser engine ships, using the "
            "encoder contract in its sidecar .meta.json."
        ),
    )
    parser.add_argument(
        "--onnx-label", type=str, action="append", dest="onnx_labels",
        help="Human label for each --onnx model (defaults to the file stem)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device for raccoon evaluation (default: cpu)",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Processes for GNUBG 2-ply (0 = cpu_count)",
    )
    parser.add_argument(
        "--shard", type=str, default=None, metavar="I/N",
        help="score only shard I of N (e.g. 3/8) so a long full-benchmark run "
             "lands partial results as it goes; shards recombine exactly",
    )
    parser.add_argument(
        "--subsample", type=int, default=None,
        help="score a reproducible random subset of N decisions (see --subsample-seed); "
             "unlike --max-positions this is not a game-ordered prefix",
    )
    parser.add_argument(
        "--subsample-seed", type=int, default=21,
        help="seed for --subsample; keep fixed so every config sees the same sample",
    )
    parser.add_argument(
        "--subsample-complement", action="store_true",
        help="score everything --subsample would EXCLUDE -- the holdout for a rule "
             "that was tuned on that sample. Provably disjoint from it.",
    )
    parser.add_argument(
        "--search-depth", type=int, default=0,
        help="plies of expectimax lookahead, GNUBG numbering (0 = static, the default)",
    )
    parser.add_argument("--search-k", type=int, default=8, help="root filter width")
    parser.add_argument(
        "--search-k2", type=int, default=2,
        help="candidates kept for the full-depth pass (filter chain, depth > 1 only)",
    )
    parser.add_argument(
        "--search-threshold", type=float, default=0.16,
        help="equity window for the root filter, in [-1,1] value units",
    )
    parser.add_argument(
        "--search-gate", type=float, default=0.08,
        help="skip the search when the static top-2 gap exceeds this (0 disables)",
    )
    parser.add_argument(
        "--window-lo", type=float, default=None,
        help="exp022: equity window in a pure race; with --window-hi this replaces the "
             "constant --search-threshold by one that scales with contact",
    )
    parser.add_argument(
        "--window-hi", type=float, default=None,
        help="exp022: equity window at full contact (opening position)",
    )
    parser.add_argument(
        "--max-positions", type=int, default=None,
        help="Limit number of decisions scored (for quick testing)",
    )
    parser.add_argument(
        "--metric", choices=[METRIC_CUBELESS, METRIC_CUBEFUL],
        default=METRIC_CUBELESS,
        help=(
            "Which reference to score a move against: 'cubeless' (the default, "
            "and every number published before exp025) or 'cubeful', the same "
            "rollout's equity with the cube live. The cubeful metric is what "
            "scripts/cube_blind_floor.py measures the PR 0.336 floor on."
        ),
    )
    parser.add_argument(
        "--cubeful-rank", action="store_true",
        help=(
            "Rank candidates by Janowski cubeful equity from the net's "
            "six-outcome head, using each decision's own recorded cube position, "
            "instead of by the value head's equity. Independent of --metric: the "
            "exp025 comparison is cubeless-rank vs cubeful-rank, both scored on "
            "the cubeful metric, paired on the same decisions."
        ),
    )
    parser.add_argument(
        "--no-cube-jacoby", action="store_true",
        help=(
            "Model money WITHOUT the Jacoby rule when ranking cubefully. Off by "
            "default because the benchmark's references were generated with the "
            "rule ON, so matching it is what makes the comparison fair; the flag "
            "exists to measure how much that assumption is worth."
        ),
    )
    parser.add_argument(
        "--error-dump", type=str, default=None,
        help=(
            "Write a per-DECISION .npz of the equity thrown away, keyed by "
            "decision, so two configurations can be compared paired on identical "
            "positions. Needed because a choice cannot be safely re-derived from "
            "the per-candidate dump, which includes pruned moves."
        ),
    )
    parser.add_argument(
        "--paired-dump", type=str, default=None,
        help=(
            "With --cubeful-rank, write a per-decision .npz holding BOTH "
            "rankings' errors from the same forward pass, so the two arms are "
            "paired by construction. Read it with scripts/exp025_paired_pr.py."
        ),
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Directory to save JSON results",
    )
    parser.add_argument(
        "--dump-dir", type=str, default=None,
        help=(
            "Directory to save per-candidate prediction .npz files for each "
            "raccoon checkpoint (one file per --checkpoint, named after its "
            "--engine-label), for downstream paired analysis (exp016)."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help=(
            "Allow a result file to be replaced even when its engine label "
            "already names a different measurement (different n, checkpoint or "
            "search configuration). Off by default: reusing a label is how "
            "results get lost."
        ),
    )
    args = parser.parse_args()

    if args.workers <= 0:
        import multiprocessing
        args.workers = multiprocessing.cpu_count()

    # Validate checkpoint/label pairing
    checkpoints = args.checkpoints or []
    labels = args.labels or []
    if len(labels) < len(checkpoints):
        # Auto-generate labels from checkpoint paths
        for cp in checkpoints[len(labels):]:
            labels.append(Path(cp).stem)

    # Load benchmark
    benchmark_path = os.path.realpath(args.benchmark)
    print(f"Loading benchmark: {benchmark_path}")
    decisions, meta = load_benchmark(benchmark_path)
    print(f"  {len(decisions)} checker decisions from {meta['n_games']} games")
    if args.max_positions:
        print(f"  (limiting to first {args.max_positions} decisions)")
    if args.subsample:
        if args.subsample_complement:
            decisions = subsample_complement(decisions, args.subsample, args.subsample_seed)
            print(f"  (holdout: the {len(decisions)} decisions NOT in the "
                  f"{args.subsample}-decision seed-{args.subsample_seed} sample)")
        else:
            decisions = subsample(decisions, args.subsample, args.subsample_seed)
            print(f"  (random subsample: {len(decisions)} decisions, "
                  f"seed {args.subsample_seed})")
    elif args.subsample_complement:
        raise SystemExit("--subsample-complement needs --subsample to say what to exclude")
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        decisions = shard(decisions, i, n)
        print(f"  (shard {i} of {n}: {len(decisions)} decisions)")
    print()

    search_cfg = None
    if args.search_depth > 0:
        from raccoon.search.expectimax import SearchConfig
        search_cfg = SearchConfig(
            depth=args.search_depth, k=args.search_k, k2=args.search_k2,
            threshold=args.search_threshold, gate=args.search_gate,
            window_lo=args.window_lo, window_hi=args.window_hi,
        )
        print(f"Search: {search_cfg.tag()}\n")

    all_results: list[dict] = []

    # Score GNUBG engines
    if args.gnubg_plies:
        for ply in args.gnubg_plies:
            print(f"Scoring GNUBG {ply}-ply...")
            result = score_gnubg(
                decisions, ply,
                workers=args.workers if ply >= 2 else 1,
                max_positions=args.max_positions,
                dump_predictions=(
                    os.path.join(args.dump_dir, f"gnubg_{ply}ply.npz")
                    if args.dump_dir else None
                ),
                metric=args.metric,
            )
            all_results.append(result)
            print()

    # Score raccoon checkpoints
    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
    for cp, label in zip(checkpoints, labels):
        print(f"Scoring raccoon: {label} ({cp})...")
        dump_path = (
            os.path.join(args.dump_dir, sanitize_label(label) + ".npz")
            if args.dump_dir else None
        )
        result = score_raccoon(
            decisions, cp, label,
            device_str=args.device,
            max_positions=args.max_positions,
            dump_predictions=dump_path,
            search_cfg=search_cfg,
            metric=args.metric,
            cubeful_rank=args.cubeful_rank,
            cube_jacoby=not args.no_cube_jacoby,
            paired_dump=args.paired_dump,
            error_dump=args.error_dump,
        )
        all_results.append(result)
        print()

    # Score exported ONNX models (what the browser engine runs)
    onnx_models = args.onnx_models or []
    onnx_labels = list(args.onnx_labels or [])
    for path in onnx_models[len(onnx_labels):]:
        onnx_labels.append(Path(path).stem)
    for path, label in zip(onnx_models, onnx_labels):
        print(f"Scoring raccoon (onnx): {label} ({path})...")
        result = score_raccoon(
            decisions, None, label,
            max_positions=args.max_positions,
            onnx_path=path,
            metric=args.metric,
        )
        all_results.append(result)
        print()

    # Print results
    if all_results:
        print_table(all_results)
        if args.output:
            print(f"Saving results to {args.output}/")
            save_results(all_results, args.output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
