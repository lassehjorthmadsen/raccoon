#!/bin/bash
# Finish v5 to 20 epochs, then the standalone strength eval (Stage 5 = the
# doubles-recovery / coverage test on the 10x256 net). Training was fragmented
# by CPU contention: v5 reached epoch 12, v5cont +2 (=>14), v5fix +4 (=>18).
# This resumes the last 2 epochs from v5fix/epoch_004 into the CANONICAL v5 dir
# so the final pretrained_v2.pt + eval logs co-locate there. --checkpoint-every 2
# writes only pretrained_v2.pt (epoch 1: 1%2!=0 skip; epoch 2: final), so the
# existing epoch_001..012 in that dir are not clobbered. OMP spin-fix on so
# desktop contention degrades gracefully. Markers (=====) for log scanning.
set -u
cd /home/lasse/python-projects/raccoon
export OMP_WAIT_POLICY=PASSIVE
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=.venv/bin/python3
ITER=experiments/exp005-6x128-800sims/checkpoints/iter_0447.pt
BASE=experiments/pretrain-gnubg-v5fix-10x256-dbl/checkpoints/epoch_004.pt
OUT=experiments/pretrain-gnubg-v5-10x256-dbl
CACHE=data/bglab/cache/gnubg4ply_cache_dbl.npz
mkdir -p "$OUT/checkpoints" "$OUT/logs"

echo "===== V5EVAL START $(date -u +%F_%T) ====="
echo "resume base=$(basename "$BASE") (training epoch 18); 2 epochs => 20 total; OMP_WAIT_POLICY=$OMP_WAIT_POLICY"

echo "===== STAGE A: finish 2 epochs => 20 total $(date -u +%F_%T) ====="
$PY scripts/pretrain_policy.py --experiment-name pretrain-gnubg-v5-10x256-dbl \
  --base-checkpoint "$BASE" --cache "$CACHE" --epochs 2 --checkpoint-every 2 \
  --lr 1e-3 --value-weight 1.0 \
  --notes "v5 finish: resume from v5fix epoch_004 (train epoch 18), 2 epochs => 20 total, OMP_WAIT_POLICY=PASSIVE"
echo "===== STAGE A DONE $(date -u +%F_%T) ====="

V5="$OUT/checkpoints/pretrained_v2.pt"
if [ ! -f "$V5" ]; then echo "ABORT: $V5 not produced"; exit 1; fi

echo "===== STAGE B: v5 standalone eval (100 games, 100 sims) $(date -u +%F_%T) ====="
echo "----- arena: v5 vs iter_0447 -----"
$PY scripts/evaluate.py --checkpoint1 "$V5" --checkpoint2 "$ITER" \
  --games 100 --simulations 100 || echo "ARENA FAILED"
echo "----- gnubg: v5 vs ply0/world -----"
$PY scripts/eval_gnubg.py --checkpoint "$V5" --games 100 --simulations 100 \
  --gnubg-level world --no-log-games || echo "GNUBG FAILED"
echo "===== STAGE B DONE $(date -u +%F_%T) ====="
echo "===== V5EVAL COMPLETE $(date -u +%F_%T) ====="
