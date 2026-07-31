#!/bin/bash
# Resume v5 from epoch_012 in a FRESH process (after the stale-process slowdown),
# 8 more epochs => 20 total, then standalone eval. Markers (=====) watched by Monitor.
set -u
cd /home/lasse/python-projects/raccoon
export OMP_NUM_THREADS=4
PY=.venv/bin/python3
ITER=experiments/exp005-6x128-800sims/checkpoints/iter_0447.pt
BASE=experiments/pretrain-gnubg-v5-10x256-dbl/checkpoints/epoch_012.pt
CONT=experiments/pretrain-gnubg-v5cont-10x256-dbl
CACHE=data/bglab/cache/gnubg4ply_cache_dbl.npz
mkdir -p "$CONT/checkpoints" "$CONT/logs"

echo "===== V5CONT START $(date -u +%F_%T) ====="
echo "===== STAGE A: resume from epoch_012, 8 more epochs (fresh process) $(date -u +%F_%T) ====="
$PY scripts/pretrain_policy.py --experiment-name pretrain-gnubg-v5cont-10x256-dbl \
  --base-checkpoint "$BASE" --cache "$CACHE" --epochs 8 --checkpoint-every 1 \
  --lr 1e-3 --value-weight 1.0 \
  --notes "v5 continuation: resume from v5 epoch_012.pt in a fresh process (after stale-process slowdown), 8 more epochs => 20 total"
echo "===== STAGE A DONE $(date -u +%F_%T) ====="

V5="$CONT/checkpoints/pretrained_v2.pt"
echo "===== STAGE B: v5 standalone eval (100 games, 100 sims) $(date -u +%F_%T) ====="
echo "----- arena: v5 vs iter_0447 -----"
$PY scripts/evaluate.py --checkpoint1 "$V5" --checkpoint2 "$ITER" --games 100 --simulations 100 || echo "ARENA FAILED"
echo "----- gnubg: v5 vs ply0/world -----"
$PY scripts/eval_gnubg.py --checkpoint "$V5" --games 100 --simulations 100 --gnubg-level world --no-log-games || echo "GNUBG FAILED"
echo "===== STAGE B DONE $(date -u +%F_%T) ====="
echo "===== V5CONT COMPLETE $(date -u +%F_%T) ====="
