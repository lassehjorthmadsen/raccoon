"""exp011 — value-only distillation of GNUBG-0-ply onto a random-init 10x256 net.

One arm per invocation (``--value-head scalar|outcomes6``), both from random init
so the A/B isolates the target definition. Streams the sharded cache written by
gen_gnubg_selfplay.py, regresses the value head only, and evaluates the net's
0-ply play vs GNUBG-0-ply (raccoon/train/td_selfplay.gnubg_arena) every few
shards, keeping the best checkpoint.

  arm A (scalar):    MSE(value, equity/3)
  arm B (outcomes6): cross-entropy(softmax(6 logits), six-outcome target dist);
                     equity is derived from the softmax at eval/play time.

    python scripts/train_distill.py --cache-dir experiments/exp011-distill/cache \\
        --experiment-name exp011-distill/scalar --value-head scalar --epochs 2
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from raccoon.model.network import RaccoonNet
from raccoon.train.td_selfplay import gnubg_arena


def save_ckpt(net: RaccoonNet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": net.state_dict(), "config": net.config,
                "step": -1, "pretrain_info": {"note": "exp011 distill"}}, path)


def train_on_shard(net, opt, obs, eq, six, head, device, batch_size):
    """One pass over a shard. Value-only; returns the last batch loss."""
    net.train()
    n = len(obs)
    perm = np.random.permutation(n)
    last = 0.0
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        xb = torch.from_numpy(obs[idx].astype(np.float32)).to(device)
        _, vout = net(xb)
        if head == "outcomes6":
            tb = torch.from_numpy(six[idx]).to(device)          # (B, 6) dist
            loss = -(tb * F.log_softmax(vout, dim=-1)).sum(dim=-1).mean()
        else:
            tb = torch.from_numpy(eq[idx]).to(device)           # (B,) equity/3
            loss = F.mse_loss(vout.squeeze(-1), tb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss.item())
    return last


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--experiment-name", required=True)
    p.add_argument("--value-head", choices=["scalar", "outcomes6"], required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--channels", type=int, default=256)
    p.add_argument("--num-blocks", type=int, default=10)
    p.add_argument("--eval-every-shards", type=int, default=6)
    p.add_argument("--eval-games", type=int, default=40)
    p.add_argument("--gnubg-ply", type=int, default=0)
    p.add_argument("--max-wall-hours", type=float, default=0.0)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    torch.set_flush_denormal(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shards = sorted(Path(args.cache_dir).glob("shard_*.npz"))
    if not shards:
        raise SystemExit(f"no shards in {args.cache_dir}")
    if args.smoke:
        shards = shards[:2]
        args.epochs = 1
        args.eval_every_shards = 1
        args.eval_games = 2

    net = RaccoonNet(channels=args.channels, num_blocks=args.num_blocks,
                     value_head=args.value_head).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0)

    exp_dir = Path("experiments") / args.experiment_name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    log_path = exp_dir / "logs" / "distill_log.jsonl"

    def log(rec):
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    print(f"exp011 {args.value_head} [{device}]: {len(shards)} shards x "
          f"{args.epochs} epochs, lr={args.lr}", flush=True)

    best = float("-inf")
    t0 = time.time()
    shard_ctr = 0
    total_shards = len(shards) * args.epochs
    stop = False
    for epoch in range(1, args.epochs + 1):
        order = list(shards)
        random.shuffle(order)
        for sh in order:
            with np.load(sh) as z:
                obs, eq, six = z["observations"], z["equity"], z["outcomes6"]
                loss = train_on_shard(net, opt, obs, eq, six, args.value_head,
                                      device, args.batch_size)
            shard_ctr += 1
            wall = (time.time() - t0) / 3600
            rec = {"epoch": epoch, "shard": shard_ctr, "loss": round(loss, 6),
                   "wall_hours": round(wall, 3)}

            if shard_ctr % args.eval_every_shards == 0 or shard_ctr == total_shards:
                net.eval()
                res = gnubg_arena(net, device, args.eval_games,
                                  gnubg_ply=args.gnubg_ply, seed=shard_ctr)
                eq_ppg = res["equity_per_game"]
                rec[f"eval_vs_gnubg{args.gnubg_ply}ply_ppg"] = round(eq_ppg, 4)
                rec["eval_games"] = res["games"]
                save_ckpt(net, ckpt_dir / "latest.pt")
                if eq_ppg > best:
                    best = eq_ppg
                    save_ckpt(net, ckpt_dir / "best.pt")
                    rec["new_best"] = True

            log(rec)
            ev = rec.get(f"eval_vs_gnubg{args.gnubg_ply}ply_ppg")
            print(f"  ep{epoch} shard {shard_ctr}/{total_shards} loss={rec['loss']} "
                  f"{('gnubg=' + str(ev)) if ev is not None else ''} "
                  f"best={best:+.3f} wall={rec['wall_hours']}h", flush=True)

            if args.max_wall_hours > 0 and wall > args.max_wall_hours:
                print(f"max wall {args.max_wall_hours}h reached — stopping", flush=True)
                stop = True
                break
        if stop:
            break

    save_ckpt(net, ckpt_dir / "latest.pt")
    print(f"\n===== ARM DONE ({args.value_head}) best vs GNUBG-{args.gnubg_ply}ply "
          f"= {best:+.4f} ppg  -> {ckpt_dir/'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
