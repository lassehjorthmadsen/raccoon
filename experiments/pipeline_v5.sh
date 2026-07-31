#!/bin/bash
# Hands-off pipeline: v4@800-sim GNUBG eval -> v5 train (20ep) -> v5 eval.
# Stage markers (=====) are watched by a Monitor; chunk markers use -----.
set -u
cd /home/lasse/python-projects/raccoon
export OMP_NUM_THREADS=4
PY=.venv/bin/python3
V4=experiments/pretrain-gnubg-v4-10x256/checkpoints/pretrained_v2.pt
ITER=experiments/exp005-6x128-800sims/checkpoints/iter_0447.pt
V5DIR=experiments/pretrain-gnubg-v5-10x256-dbl
CACHE=data/bglab/cache/gnubg4ply_cache_dbl.npz
mkdir -p "$V5DIR/checkpoints" "$V5DIR/logs"

echo "===== PIPELINE START $(date -u +%F_%T) ====="

echo "===== STAGE 1: v4 @ 800 sims vs GNUBG, 400 games (8 x 50 chunked) $(date -u +%F_%T) ====="
for c in 1 2 3 4 5 6 7 8; do
  echo "----- eval chunk $c/8 (50 games, 800 sims) $(date -u +%F_%T) -----"
  $PY scripts/eval_gnubg.py --checkpoint "$V4" --games 50 --simulations 800 \
    --gnubg-level world --no-log-games || echo "EVAL CHUNK $c FAILED"
done
echo "===== STAGE 1 DONE $(date -u +%F_%T) ====="

echo "===== STAGE 2: build v5 base (10x256 random init) $(date -u +%F_%T) ====="
$PY -c "import torch; from raccoon.model.network import RaccoonNet; torch.manual_seed(42); net=RaccoonNet(channels=256,num_blocks=10); torch.save({'model_state_dict':net.state_dict(),'config':net.config,'step':-1,'pretrain_info':{'note':'v5 base 10x256'}}, '$V5DIR/checkpoints/base_random_10x256.pt'); print('base saved')"

echo "===== STAGE 3: v5 train (20 epochs, doubles cache 164k) $(date -u +%F_%T) ====="
$PY scripts/pretrain_policy.py --experiment-name pretrain-gnubg-v5-10x256-dbl \
  --base-checkpoint "$V5DIR/checkpoints/base_random_10x256.pt" \
  --cache "$CACHE" --epochs 20 --checkpoint-every 1 --lr 1e-3 --value-weight 1.0 \
  --notes "v5: 10x256, doubles-recovered cache (164k, +25k value-only doubles), 20 epochs single-stage from random"
echo "===== STAGE 3 DONE $(date -u +%F_%T) ====="

V5="$V5DIR/checkpoints/pretrained_v2.pt"
echo "===== STAGE 4: v5 standalone eval (100 games, 100 sims) $(date -u +%F_%T) ====="
echo "----- arena: v5 vs iter_0447 -----"
$PY scripts/evaluate.py --checkpoint1 "$V5" --checkpoint2 "$ITER" \
  --games 100 --simulations 100 || echo "ARENA FAILED"
echo "----- gnubg: v5 vs ply0/world -----"
$PY scripts/eval_gnubg.py --checkpoint "$V5" --games 100 --simulations 100 \
  --gnubg-level world --no-log-games || echo "GNUBG FAILED"
echo "===== STAGE 4 DONE $(date -u +%F_%T) ====="

echo "===== PIPELINE COMPLETE $(date -u +%F_%T) ====="
