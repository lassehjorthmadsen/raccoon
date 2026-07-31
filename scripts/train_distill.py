"""exp011 — value-only distillation of GNUBG-0-ply onto a random-init 10x256 net.

One arm per invocation (``--value-head scalar|outcomes6``), both from random init
so the A/B isolates the target definition. Streams the sharded cache written by
gen_gnubg_selfplay.py, regresses the value head only, and evaluates the net's
0-ply play vs GNUBG-0-ply (raccoon/train/td_selfplay.gnubg_arena) every few
shards, keeping the best checkpoint.

  arm A (scalar):    MSE(value, equity/3)
  arm B (outcomes6): cross-entropy(softmax(6 logits), six-outcome target dist);
                     equity is derived from the softmax at eval/play time.

    python scripts/train_distill.py --cache-dir data/distill/0ply/run1 \\
        --experiment-name exp011-distill/scalar --value-head scalar --epochs 2

exp014 reuses this unchanged, only swapping --cache-dir for the 2-ply cache and
adding --holdout-frac so both arms train on the identical 6M positions and hold
out the identical 2M for a fair R^2 comparison (scripts/eval_r2.py):

    python scripts/train_distill.py --cache-dir data/distill/2ply/run1 \\
        --experiment-name exp014-distill/scalar_2ply --value-head scalar --epochs 3 \\
        --holdout-frac 0.25 --split-seed 14

--cache-dir accepts either a single run dir (e.g. data/distill/2ply/run1) or a
ply-level dir spanning multiple runs (e.g. data/distill/2ply) — shard discovery
is recursive (see data/README.md for the run/ply layout and provenance table).
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

from raccoon.data.cache_split import held_out_mask
from raccoon.model.network import RaccoonNet
from raccoon.train.td_selfplay import gnubg_arena


def save_ckpt(net: RaccoonNet, path: Path, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": net.state_dict(), "config": net.config,
               "step": -1, "pretrain_info": {"note": "exp011 distill"}}
    if extra:
        payload.update(extra)          # optimizer_state_dict + train_state for --resume
    torch.save(payload, path)


def epoch_order(shards, epoch, base_seed):
    """Deterministic per-epoch shard order.

    Seeded by (base_seed, epoch) so a resumed run reconstructs the *same* order
    and can skip the shards it already finished — the basis of shard-granular
    preemption recovery. (Pre-resume this was an unseeded random.shuffle, i.e.
    non-reproducible run-to-run anyway, so seeding only adds determinism.)
    """
    order = list(shards)
    random.Random(base_seed * 1_000_003 + epoch).shuffle(order)
    return order


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
    p.add_argument("--holdout-frac", type=float, default=0.0,
                   help="Fraction of each shard excluded from training (exp014: "
                        "0.25). 0 (default) trains on 100%% of the cache, matching "
                        "exp011/exp011b behavior exactly. See raccoon.data.cache_split "
                        "— the same (frac, seed) held out here must be passed to "
                        "eval_r2.py so train/holdout never overlap.")
    p.add_argument("--split-seed", type=int, default=14,
                   help="Seed for --holdout-frac's per-shard split (ignored if "
                        "--holdout-frac is 0).")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", default="",
                   help="'auto' resumes from <ckpt_dir>/latest.pt (shard-granular; "
                        "a no-op if that file is absent, so it's safe as the default "
                        "launch flag for a spot/preemptible run). A path resumes from "
                        "a specific latest-style checkpoint. Default '' = fresh run.")
    p.add_argument("--shuffle-seed", type=int, default=20,
                   help="Base seed for the deterministic per-epoch shard order "
                        "(see epoch_order). Must be held fixed across a resumed run "
                        "so completed shards are skipped correctly.")
    args = p.parse_args()

    torch.set_flush_denormal(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shards = sorted(Path(args.cache_dir).rglob("shard_*.npz"))
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
          f"{args.epochs} epochs, lr={args.lr} holdout_frac={args.holdout_frac} "
          f"split_seed={args.split_seed}", flush=True)

    # ---- resume state (shard-granular preemption recovery; see --resume/epoch_order) ----
    resume_epoch, start_idx = 1, 0
    best = float("-inf")
    cum_wall = 0.0          # training wall accumulated across prior (preempted) sessions
    shard_ctr = 0
    if args.resume:
        rp = ckpt_dir / "latest.pt" if args.resume == "auto" else Path(args.resume)
        if rp.exists():
            ck = torch.load(rp, map_location=device)
            net.load_state_dict(ck["model_state_dict"])
            if ck.get("optimizer_state_dict"):
                opt.load_state_dict(ck["optimizer_state_dict"])
            ts = ck.get("train_state", {})
            ep = int(ts.get("epoch", 1))
            done_in_ep = int(ts.get("shards_done_in_epoch", 0))
            best = float(ts.get("best", float("-inf")))
            cum_wall = float(ts.get("cum_wall_hours", 0.0))
            shard_ctr = int(ts.get("shard_ctr", 0))
            if done_in_ep >= len(shards):       # epoch was fully finished pre-preemption
                resume_epoch, start_idx = ep + 1, 0
            else:
                resume_epoch, start_idx = ep, done_in_ep
            print(f"[resume] from {rp}: epoch {resume_epoch} shard-idx {start_idx} "
                  f"(shard_ctr={shard_ctr}, best={best:+.4f}, cum_wall={cum_wall:.2f}h)",
                  flush=True)
        else:
            print(f"[resume] {rp} absent — starting fresh", flush=True)

    t0 = time.time()
    total_shards = len(shards) * args.epochs
    stop = False
    done_reason = None

    def wall_now():
        return cum_wall + (time.time() - t0) / 3600

    def save_latest(cur_epoch, shards_done):
        # shard-granular resume point: weights + Adam state + exactly where we are.
        save_ckpt(net, ckpt_dir / "latest.pt", extra={
            "optimizer_state_dict": opt.state_dict(),
            "train_state": {"epoch": cur_epoch, "shards_done_in_epoch": shards_done,
                            "shard_ctr": shard_ctr, "best": best,
                            "cum_wall_hours": wall_now()}})

    for epoch in range(resume_epoch, args.epochs + 1):
        order = epoch_order(shards, epoch, args.shuffle_seed)
        idx0 = start_idx if epoch == resume_epoch else 0
        for si in range(idx0, len(order)):
            sh = order[si]
            with np.load(sh) as z:
                obs, eq, six = z["observations"], z["equity"], z["outcomes6"]
                if args.holdout_frac > 0:
                    mask = held_out_mask(sh.name, len(obs), args.holdout_frac,
                                         args.split_seed)
                    train_rows = ~mask
                    obs, eq, six = obs[train_rows], eq[train_rows], six[train_rows]
                loss = train_on_shard(net, opt, obs, eq, six, args.value_head,
                                      device, args.batch_size)
            shard_ctr += 1
            wall = wall_now()
            rec = {"epoch": epoch, "shard": shard_ctr, "loss": round(loss, 6),
                   "wall_hours": round(wall, 3)}

            if shard_ctr % args.eval_every_shards == 0 or shard_ctr == total_shards:
                net.eval()
                res = gnubg_arena(net, device, args.eval_games,
                                  gnubg_ply=args.gnubg_ply, seed=shard_ctr)
                eq_ppg = res["equity_per_game"]
                rec[f"eval_vs_gnubg{args.gnubg_ply}ply_ppg"] = round(eq_ppg, 4)
                rec["eval_games"] = res["games"]
                if eq_ppg > best:
                    best = eq_ppg
                    save_ckpt(net, ckpt_dir / "best.pt")
                    rec["new_best"] = True

            save_latest(epoch, si + 1)   # every shard — cheap vs the ~minutes/shard compute

            log(rec)
            ev = rec.get(f"eval_vs_gnubg{args.gnubg_ply}ply_ppg")
            print(f"  ep{epoch} shard {shard_ctr}/{total_shards} loss={rec['loss']} "
                  f"{('gnubg=' + str(ev)) if ev is not None else ''} "
                  f"best={best:+.3f} wall={rec['wall_hours']}h", flush=True)

            if args.max_wall_hours > 0 and wall > args.max_wall_hours:
                print(f"cumulative max wall {args.max_wall_hours}h reached — stopping",
                      flush=True)
                done_reason = "max_wall"
                stop = True
                break
        if stop:
            break
        # completed a full epoch — distinct per-epoch checkpoint (weights only) so the
        # real best can be picked offline at low noise (exp011b selects at n>=1000, not
        # the noisy n=40 inline eval), then bump latest.pt to the epoch boundary so a
        # preemption right here resumes at epoch+1.
        save_ckpt(net, ckpt_dir / f"ep{epoch}.pt")
        save_latest(epoch, len(order))
        print(f"  saved epoch checkpoint -> {ckpt_dir / f'ep{epoch}.pt'}", flush=True)

    if not stop:
        done_reason = "epochs_complete"
    # DONE sentinel only on clean completion (all epochs, or cumulative max-wall) — a
    # preemption SIGKILLs the process before here, so DONE stays absent and the watchdog
    # relaunches; once DONE exists the watchdog stops restarting (the true cost cap).
    if done_reason in ("epochs_complete", "max_wall"):
        (exp_dir / "DONE").write_text(
            f"reason={done_reason} wall={wall_now():.2f}h best={best:+.4f}\n")
    print(f"\n===== ARM DONE ({args.value_head}) reason={done_reason} "
          f"best vs GNUBG-{args.gnubg_ply}ply = {best:+.4f} ppg  -> {ckpt_dir/'best.pt'}",
          flush=True)


if __name__ == "__main__":
    main()
