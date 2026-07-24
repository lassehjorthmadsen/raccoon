# Generating GNUBG 2-ply labels (for the exp011b → 2-ply fine-tune)

Shared brief between machines. Follow it before committing a long generation run.

## Why

On the iMac we distilled GNUBG-**0-ply** into a fresh 10×256 `RaccoonNet` (experiment
`exp011b`, two arms: scalar equity vs 6-outcome distribution). The winner (`outcomes6/ep3`)
reached **−0.053 ppg vs GNUBG-0-ply** (n=6000, CI ±0.046) — a real improvement over the
truncated exp011 (−0.087) but **short of parity**, and 0-ply distillation can't exceed its
teacher anyway. To go further we distil a **stronger teacher**: fine-tune that net toward
GNUBG-**2-ply** labels, which raises the ceiling from 0-ply-parity toward 2-ply.

This doc specifies how to generate 2-ply labels that the existing training pipeline
(`scripts/train_distill.py`) can consume **unchanged**.

## Pre-flight (do this before generating at volume)

1. **`git pull` and confirm the checkout is current.** Specifically: `raccoon/model/network.py`
   must have `value_head` support, and the encoder must emit **26-channel `(N, 26, 2, 12)`**
   observations. A stale checkout produces labels that don't match the target net's encoder —
   this exact mistake bit our GPU VM (it was 21 commits behind). Verify, don't assume.
2. **Read the ground-truth code, don't trust this doc's field descriptions.** Open
   `scripts/gen_gnubg_selfplay.py` (the 0-ply generator) and `scripts/train_distill.py` (the
   consumer) and match the shard contract exactly. The list below is a summary; the code is
   authoritative (especially the equity *scaling* and the `outcomes6` column order).

## Output format — identical to the 0-ply cache, only the label ply differs

Sharded `.npz`, one file per shard, with the same keys/dtypes/shapes `gen_gnubg_selfplay.py`
writes for 0-ply:

| key | dtype | shape | note |
|-----|-------|-------|------|
| `observations` | float16 | `(N, 26, 2, 12)` | same 26-channel encoder as the 0-ply cache |
| `equity` | float32 | `(N,)` | GNUBG cubeless money equity — **use the same scaling `gen_gnubg_selfplay.py` uses** (check whether it stores raw equity or equity/3) |
| `outcomes6` | float32 | `(N, 6)` | six-outcome distribution in **the same column order** the 0-ply generator uses; rows sum to ~1 |

**The only difference from the 0-ply cache is that the labels are computed at 2-ply, not 0-ply.**
Same encoder, same keys, same dtypes, same shard layout → `train_distill.py` streams it unchanged.

GNUBG's 2-ply evaluation returns the **full** win/gammon/backgammon outcome vector (not just a
scalar), so capture `outcomes6` at 2-ply — the fine-tune will use the 6-outcome head.

## Smoke-test (mandatory before the 3-day run)

1. Generate **one small shard**.
2. Load it and assert: keys/shapes/dtypes match a real 0-ply shard; `outcomes6` rows sum to
   ~1 (±1e-3); `equity` is within the 0-ply cache's range.
3. **Sync that first shard to GCS** (`gs://raccoon-training-lhm/…`) so the iMac session can
   validate the format **before** the long run commits three days of compute.

Only start the full generation once the smoke shard validates.

## Downstream use (so the volume target is right)

The labels feed a **warm-start fine-tune**: resume from the exp011b winner
(`experiments/exp011b-distill/outcomes6/checkpoints/ep3.pt`, in GCS) and fine-tune toward the
2-ply targets. That needs **far fewer labels than the 8M-from-random exp011b run** — on the
order of a few million is plausibly enough. So ~3 days at ~1.3 s/position on a many-core box
(≈3M positions) should suffice; you do **not** need to match 8M.

## Report back

- **Which positions** are you labeling — 0-ply self-play, on-policy from a net, or fresh? This
  changes how we use them: on-policy positions also attack the −0.053 distribution-shift
  residual; re-labeling the *same* positions as the 0-ply cache gives a clean controlled
  comparison (identical positions, 0-ply vs 2-ply teacher — isolates teacher quality).
- **Positions/day** and total target.
- **Where** shards are written, and the GCS sync path.
