"""Is the gnubg binary a stronger opponent than the gnubg-nn package?

Every Raccoon number before exp026 was measured against the **package**: the
distillation teacher, the BGSage benchmark's GNUBG rows, exp019's +0.0129 and
exp020's +0.0472 ppg. exp026 needs the **binary**, because only it can make a
money cube decision. The two are not the same evaluator, so a result against one
is not directly comparable to a result against the other, and the size of that
gap decides whether the back catalogue translates or has to be re-quoted.

**Why they differ.** Same six nets and identical topology — race 214-128-5,
crashed and contact 250-128-5, and three prune nets — but different trained
values from different lineages. The package's weights file declares itself
``GNU Backgammon 1.01``; the binary expects ``1.00``, rejects the file, and
rejects its contents even when the version string is forced. On real positions
their ply-0 probabilities differ by up to about 0.003, which is small but real.

**What this measures.** The two engines playing each other **cubeless** at the
same ply, so the cube model is out of the comparison entirely (the binary is run
with ``set cube use off``) and only the nets are left. The reported ppg is from
the **binary's** point of view: positive means the binary is the stronger
opponent, and therefore that exp026 faces harder opposition than exp019/exp020
did.

**Variance reduction.** The control variate is the package's 0-ply best-play
equity, exactly as in :mod:`raccoon.eval.luck`. Using one player's evaluator as
the control variate is fine and does not favour it: ``h`` has only to be a fixed
function of (pre-roll state, roll) that ignores the move actually played, which
is the same argument that makes the estimator unbiased however weak the
evaluator is.

    python scripts/compare_gnubg_engines.py --games 600 --ply 0 --workers 3 \\
        --exp-dir experiments/exp026-gnubg-cli --tag pilot
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from raccoon.env.game_wrapper import GameWrapper
from raccoon.eval.luck import pre_roll_values
from raccoon.search.mcts import _advance_through_chance


def _shard(games: int, ply: int, cv_ply: int, seed: int, vr: bool) -> tuple:
    """Play ``games`` cubeless binary-vs-package games; results from the binary's POV."""
    from raccoon.eval.gnubg_adapter import board_from_view, pick_move
    from raccoon.eval.gnubg_cli import GnubgCli, actions_reaching, target_board

    np.random.seed(seed)
    cli = GnubgCli(ply=ply, use_cube=False)
    wrapper = GameWrapper()
    pts: list[float] = []
    luck: list[float] = []
    try:
        for g in range(games):
            cli_seat = g % 2                     # alternate, so seats cannot bias
            state = wrapper.new_game()
            total = 0.0
            moves = 0
            while not state.is_terminal() and moves < 2000:
                if state.is_chance_node():
                    if vr:
                        actions, probs, values = pre_roll_values(state, cli_seat, cv_ply)
                    else:
                        outcomes = state.chance_outcomes()
                        actions = [a for a, _ in outcomes]
                        probs = np.array([p for _, p in outcomes], dtype=np.float64)
                    idx = int(np.random.choice(len(actions), p=probs))
                    if vr:
                        total += float(values[idx]) - float(probs @ values)
                    state.apply_action(actions[idx])
                    continue
                if state.current_player() == cli_seat:
                    view = state.board_from_perspective()
                    board = board_from_view(view)
                    notation = cli.best_move(board, view.dice)
                    if not notation:
                        state.apply_action(state.legal_actions()[0])
                    else:
                        for action in actions_reaching(
                            state, target_board(board, notation)
                        ):
                            state.apply_action(action)
                else:
                    state.apply_action(pick_move(state, ply))
                moves += 1
            if not state.is_terminal():
                continue
            pts.append(state.returns()[cli_seat])
            luck.append(total)
        restarts = cli.restarts
    finally:
        cli.close()
    return np.array(pts), np.array(luck), restarts


def summarise(pts: np.ndarray, luck: np.ndarray) -> dict:
    n = len(pts)
    vr = pts - luck
    out = {
        "games": n,
        "raw_ppg": float(pts.mean()),
        "raw_sd": float(pts.std(ddof=1)),
        "raw_ci95": float(1.96 * pts.std(ddof=1) / n**0.5),
        "vr_ppg": float(vr.mean()),
        "vr_sd": float(vr.std(ddof=1)),
        "vr_ci95": float(1.96 * vr.std(ddof=1) / n**0.5),
        "mean_luck": float(luck.mean()),
    }
    if luck.std(ddof=1) > 0:
        out["luck_t"] = float(luck.mean() / (luck.std(ddof=1) / n**0.5))
        out["sd_ratio"] = float(pts.std(ddof=1) / vr.std(ddof=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--ply", type=int, default=0,
                    help="evaluation depth for BOTH engines; 0 isolates the nets")
    ap.add_argument("--cv-ply", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--no-vr", action="store_true")
    ap.add_argument("--seed-base", type=int, default=6000)
    ap.add_argument("--exp-dir", default="")
    ap.add_argument("--tag", default="gnubg_binary_vs_package")
    a = ap.parse_args()

    vr = not a.no_vr
    sizes = [a.games // a.workers + (1 if k < a.games % a.workers else 0)
             for k in range(a.workers)]
    sizes = [s for s in sizes if s > 0]

    started = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        parts = [f.result() for f in [
            ex.submit(_shard, n, a.ply, a.cv_ply, a.seed_base + k, vr)
            for k, n in enumerate(sizes)
        ]]
    pts = np.concatenate([p[0] for p in parts])
    luck = np.concatenate([p[1] for p in parts])
    restarts = sum(p[2] for p in parts)
    elapsed = time.time() - started

    summary = {
        "tag": a.tag,
        "measurement": "cubeless money ppg, gnubg BINARY vs gnubg-nn PACKAGE, "
                       "from the binary's POV",
        "ply": a.ply,
        "cube": False,
        "gnubg_restarts": restarts,
        "elapsed_sec": round(elapsed, 1),
        "sec_per_game_per_worker": round(elapsed * a.workers / len(pts), 3),
        **summarise(pts, luck),
    }

    print(f"[{a.tag}] cubeless, both engines at ply {a.ply}, n={summary['games']}")
    print(f"  raw ppg (binary's POV) = {summary['raw_ppg']:+.4f} "
          f"(95% CI ±{summary['raw_ci95']:.4f})")
    if vr:
        print(f"  VR  ppg               = {summary['vr_ppg']:+.4f} "
              f"(95% CI ±{summary['vr_ci95']:.4f})   "
              f"<- {summary['sd_ratio']:.2f}x tighter")
        print(f"  mean luck {summary['mean_luck']:+.4f} "
              f"(t={summary['luck_t']:+.2f}, should be ~0)")
    print(f"  {elapsed/60:.1f} min, {summary['sec_per_game_per_worker']:.2f} "
          f"s/game/worker, gnubg restarts={restarts}")

    if a.exp_dir:
        exp = Path(a.exp_dir)
        (exp / "logs").mkdir(parents=True, exist_ok=True)
        (exp / "results").mkdir(parents=True, exist_ok=True)
        with (exp / "logs" / "engine_compare_log.jsonl").open("a") as fh:
            fh.write(json.dumps(summary) + "\n")
        record = dict(summary)
        record["game_pts"] = [float(x) for x in pts]
        record["game_luck"] = [round(float(x), 6) for x in luck]
        (exp / "results" / f"{a.tag}.json").write_text(json.dumps(record) + "\n")
        print(f"  wrote {exp / 'results' / f'{a.tag}.json'}")


if __name__ == "__main__":
    main()
