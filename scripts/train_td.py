"""exp010 — TD(λ) self-play training loop (value-only, warm-started).

Each batch: generate 1-ply-greedy self-play games with the current net, compute
forward-view TD(λ) value targets, regress the value head toward them (policy head
left untouched), then every few batches play a 1-ply arena vs the frozen seed and
keep the best checkpoint. Because training is value-only, evaluation uses 1-ply
value play (not MCTS, whose priors would come from the stale policy) — see
docs/plan. Designed to run locally on the iMac (CPU); set --workers for parallel
game generation.

    python scripts/train_td.py --experiment-name exp010-td-pilot \\
        --seed experiments/exp009-ondist-dagger/round_06/checkpoints/pretrained_v2.pt \\
        --games-per-batch 500 --batches 40 --workers 4 --eval-every 2
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# scripts/ is sys.path[0] when run as `python scripts/train_td.py`, so these
# sibling-module imports resolve without packaging gymnastics.
import pipeline_budget  # noqa: E402  (_plateaued reused for the plateau stop)

from raccoon.model.network import RaccoonNet, load_model
from raccoon.train.td_selfplay import gnubg_arena, lambda_returns, play_td_game


def save_ckpt(net: RaccoonNet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": net.state_dict(), "config": net.config,
         "step": -1, "pretrain_info": {"note": "td_selfplay"}},
        path,
    )


def _play_games_worker(ckpt_path, n_games, temperature, seed_base):
    """Worker: load the net (CPU) and play ``n_games`` TD games."""
    torch.set_flush_denormal(True)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    net = load_model(ckpt_path).to(device)
    net.eval()
    rng = np.random.default_rng(seed_base) if temperature > 0 else None
    out = []
    for i in range(n_games):
        np.random.seed(seed_base + i)  # dice RNG (global, used by chance nodes)
        r = play_td_game(net, device, temperature, rng)
        if r is not None:
            out.append(r)
    return out


def generate(net, device, n_games, workers, temperature, gen_ckpt, seed_base):
    """Return a list of (obs, players, values, returns) game trajectories."""
    if workers <= 1:
        rng = np.random.default_rng(seed_base) if temperature > 0 else None
        games = []
        for i in range(n_games):
            np.random.seed(seed_base + i)
            r = play_td_game(net, device, temperature, rng)
            if r is not None:
                games.append(r)
        return games
    # Parallel: snapshot the current net so workers load an identical copy.
    save_ckpt(net, gen_ckpt)
    per = [n_games // workers + (1 if k < n_games % workers else 0)
           for k in range(workers)]
    games = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(_play_games_worker, str(gen_ckpt), per[k],
                      temperature, seed_base + k * 100_000)
            for k in range(workers) if per[k] > 0
        ]
        for f in futs:
            games.extend(f.result())
    return games


def train_value(net, optimizer, X, Y, device, epochs, batch_size):
    """Regress the value head on (obs, td_target). Returns final MSE."""
    net.train()
    Xt = torch.from_numpy(X).float()
    Yt = torch.from_numpy(Y).float()
    n = len(X)
    last = 0.0
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = Xt[idx].to(device)
            yb = Yt[idx].to(device)
            _, v = net(xb)
            loss = F.mse_loss(v.squeeze(-1), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last = float(loss.item())
    return last


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", required=True, help="Warm-start checkpoint.")
    p.add_argument("--experiment-name", default="exp010-td-pilot")
    p.add_argument("--games-per-batch", type=int, default=500)
    p.add_argument("--batches", type=int, default=40)
    p.add_argument("--lam", type=float, default=0.7)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--train-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--eval-games", type=int, default=100)
    p.add_argument("--gnubg-ply", type=int, default=0,
                   help="GNUBG ply for the fixed-reference eval (0 = fast, "
                        "~0.007 ppg weaker than 2-ply).")
    p.add_argument("--patience", type=int, default=0,
                   help="Stop after N evals with no vs-seed improvement (0=off).")
    p.add_argument("--min-delta", type=float, default=0.02)
    p.add_argument("--max-wall-hours", type=float, default=0.0)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.games_per_batch = 6
        args.batches = 2
        args.eval_games = 2
        args.eval_every = 1
        args.workers = min(args.workers, 2)
        print("===== SMOKE MODE =====", flush=True)

    torch.set_flush_denormal(True)  # iMac CPU: avoid denormal slowdown
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  seed={args.seed}", flush=True)

    net = load_model(args.seed).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0)

    exp_dir = Path("experiments") / args.experiment_name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    log_path = exp_dir / "logs" / "td_log.jsonl"
    gen_ckpt = ckpt_dir / "_gen.pt"

    eval_equities: list[float] = []
    best_equity = float("-inf")
    t0 = time.time()

    def log(rec):
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    print(f"config: games/batch={args.games_per_batch} batches={args.batches} "
          f"lam={args.lam} lr={args.lr} temp={args.temperature} "
          f"workers={args.workers} eval_every={args.eval_every}", flush=True)

    for batch in range(1, args.batches + 1):
        bt0 = time.time()
        net.eval()
        games = generate(net, device, args.games_per_batch, args.workers,
                         args.temperature, gen_ckpt, seed_base=batch * 1_000_000)
        if not games:
            print(f"batch {batch}: no games generated — stopping", flush=True)
            break

        X: list[np.ndarray] = []
        Y: list[float] = []
        n_decisions = 0
        for obs, players, values, returns in games:
            g = lambda_returns(players, values, returns, args.lam)
            X.extend(obs)
            Y.extend(g)
            n_decisions += len(obs)
        X_arr = np.stack(X)
        Y_arr = np.array(Y, dtype=np.float32)

        train_loss = train_value(net, optimizer, X_arr, Y_arr, device,
                                 args.train_epochs, args.batch_size)
        save_ckpt(net, ckpt_dir / "latest.pt")

        wall = time.time() - t0
        rec = {"batch": batch, "games": len(games), "decisions": n_decisions,
               "train_value_mse": round(train_loss, 6),
               "batch_sec": round(time.time() - bt0, 1),
               "wall_hours": round(wall / 3600, 3)}

        if batch % args.eval_every == 0 or batch == args.batches:
            net.eval()
            res = gnubg_arena(net, device, args.eval_games,
                              gnubg_ply=args.gnubg_ply, seed=batch)
            eq = res["equity_per_game"]
            eval_equities.append(eq)
            rec[f"eval_vs_gnubg{args.gnubg_ply}ply_ppg"] = round(eq, 4)
            rec["eval_games"] = res["games"]
            if eq > best_equity:
                best_equity = eq
                save_ckpt(net, ckpt_dir / "best.pt")
                rec["new_best"] = True

        log(rec)
        ev = rec.get(f"eval_vs_gnubg{args.gnubg_ply}ply_ppg")
        print(f"batch {batch}/{args.batches}: games={len(games)} "
              f"dec={n_decisions} vmse={rec['train_value_mse']} "
              f"{('vs_gnubg=' + str(ev)) if ev is not None else ''} "
              f"best={best_equity:+.3f} wall={rec['wall_hours']}h", flush=True)

        if args.max_wall_hours > 0 and wall / 3600 > args.max_wall_hours:
            print(f"STOP: wall {wall/3600:.2f}h > --max-wall-hours "
                  f"{args.max_wall_hours}", flush=True)
            break
        if pipeline_budget._plateaued(eval_equities, args.patience, args.min_delta):
            print(f"STOP: vs-seed equity plateaued over last {args.patience} "
                  f"evals (best {best_equity:+.3f})", flush=True)
            break

    print(f"\n===== TD DONE ===== best vs-GNUBG-{args.gnubg_ply}ply = "
          f"{best_equity:+.4f} ppg ({len(eval_equities)} evals)  -> "
          f"{ckpt_dir/'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
