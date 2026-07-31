#!/bin/bash
# Resume v5 with the OpenMP spin-wait fix (PASSIVE => idle threads sleep, not
# busy-spin), so CPU contention from normal desktop use no longer collapses
# throughput. Resumes from v5cont's latest checkpoint to reach 20 total epochs.
set -u
cd /home/lasse/python-projects/raccoon
export OMP_WAIT_POLICY=PASSIVE
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=.venv/bin/python3
ITER=experiments/exp005-6x128-800sims/checkpoints/iter_0447.pt
SRC=experiments/pretrain-gnubg-v5cont-10x256-dbl/checkpoints
FIX=experiments/pretrain-gnubg-v5fix-10x256-dbl
CACHE=data/bglab/cache/gnubg4ply_cache_dbl.npz
mkdir -p "$FIX/checkpoints" "$FIX/logs"

LATEST=$(ls "$SRC"/epoch_*.pt 2>/dev/null | sort -V | tail -1)
if [ -z "$LATEST" ]; then echo "ABORT: no v5cont checkpoint to resume from"; exit 1; fi
NNN=$(basename "$LATEST" .pt | sed 's/epoch_0*//')
REMAIN=$((20 - 12 - NNN))

echo "===== V5FIX START $(date -u +%F_%T) ====="
echo "resume base=$(basename "$LATEST") (training epoch $((12+NNN))); REMAIN=$REMAIN epochs; OMP_WAIT_POLICY=$OMP_WAIT_POLICY"
echo "===== STAGE A: resume $REMAIN epochs with spin-fix $(date -u +%F_%T) ====="
$PY scripts/pretrain_policy.py --experiment-name pretrain-gnubg-v5fix-10x256-dbl \
  --base-checkpoint "$LATEST" --cache "$CACHE" --epochs "$REMAIN" --checkpoint-every 1 \
  --lr 1e-3 --value-weight 1.0 \
  --notes "v5 spin-fix: resume from $(basename "$LATEST") with OMP_WAIT_POLICY=PASSIVE, $REMAIN epochs => 20 total"
echo "===== STAGE A DONE $(date -u +%F_%T) ====="

V5="$FIX/checkpoints/pretrained_v2.pt"
echo "===== STAGE B: v5 standalone eval (100 games, 100 sims) $(date -u +%F_%T) ====="
echo "----- arena: v5 vs iter_0447 -----"
$PY scripts/evaluate.py --checkpoint1 "$V5" --checkpoint2 "$ITER" --games 100 --simulations 100 || echo "ARENA FAILED"
echo "----- gnubg: v5 vs ply0/world -----"
$PY scripts/eval_gnubg.py --checkpoint "$V5" --games 100 --simulations 100 --gnubg-level world --no-log-games || echo "GNUBG FAILED"
echo "===== STAGE B DONE $(date -u +%F_%T) ====="
echo "===== V5FIX COMPLETE $(date -u +%F_%T) ====="
