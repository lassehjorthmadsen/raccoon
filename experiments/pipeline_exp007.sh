#!/bin/bash
# exp007 — the pretraining-pivot validation: self-play fine-tune FROM the v5
# supervised seed (10x256, GNUBG-4ply-distilled). Question: does high-sim
# self-play CLIMB above the seed, or REGRESS it toward the self-play
# signal-quality ceiling (the "regression trap")? Run at 800 sims because a
# strong seed needs strong MCTS targets to improve on its prior (exp005 showed
# 800-sim self-play is the lever; 25-200 sims plateaued/regressed).
#
# CPU threading: each self-play worker self-limits to 1 thread, so 4 workers =
# one game per core. OMP_NUM_THREADS is left UNSET so the main-process SGD uses
# all 4 cores; OMP_WAIT_POLICY=PASSIVE keeps the idle main process from
# busy-spin-collapsing during self-play. Stage markers (=====) for log scans.
set -u
cd /home/lasse/python-projects/raccoon
export OMP_WAIT_POLICY=PASSIVE
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
PY=.venv/bin/python3
SEED=experiments/pretrain-gnubg-v5-10x256-dbl/checkpoints/pretrained_v2.pt
ITER0447=experiments/exp005-6x128-800sims/checkpoints/iter_0447.pt
EXP=exp007-10x256-v5seed-800sims
DIR=experiments/$EXP

echo "===== EXP007 START $(date -u +%F_%T) ====="
echo "seed=$SEED ; 800 sims, 8 games/iter, 50 iters, lr 5e-4, bootstrap 0.5, Dirichlet 0.3, 25k buf, 4 workers"

echo "===== STAGE A: self-play fine-tune (50 iters) $(date -u +%F_%T) ====="
$PY scripts/train.py --experiment-name "$EXP" \
  --resume "$SEED" \
  --iterations 50 --simulations 800 --games-per-iter 8 --training-steps 50 \
  --lr 5e-4 --weight-decay 1e-4 --replay-size 25000 \
  --checkpoint-every 5 --num-workers 4 --virtual-loss 8 \
  --dirichlet-alpha 0.3 --noise-eps 0.25 --value-bootstrap-alpha 0.5 \
  --notes "exp007: self-play fine-tune from v5 (10x256 GNUBG-4ply seed). 800 sims, 8 games/iter, lr 5e-4, bootstrap a=0.5, Dirichlet 0.3, 25k buffer, no LR decay. Tests whether high-sim self-play climbs above or regresses the supervised seed (pretraining-pivot validation)." \
  || echo "TRAIN EXITED NONZERO"
echo "===== STAGE A DONE $(date -u +%F_%T) ====="

LAST=$(ls "$DIR"/checkpoints/iter_*.pt 2>/dev/null | sort -V | tail -1)
if [ -z "$LAST" ]; then echo "ABORT: no exp007 checkpoint produced"; exit 1; fi
echo "final checkpoint = $LAST"

echo "===== STAGE B: strength evals (100 games, 100 sims) $(date -u +%F_%T) ====="
echo "----- arena: exp007-final vs v5 SEED (climb-or-regress test) -----"
$PY scripts/evaluate.py --checkpoint1 "$LAST" --checkpoint2 "$SEED" \
  --games 100 --simulations 100 || echo "ARENA-vs-SEED FAILED"
echo "----- gnubg: exp007-final vs ply0/world (external M6) -----"
$PY scripts/eval_gnubg.py --checkpoint "$LAST" --games 100 --simulations 100 \
  --gnubg-level world --no-log-games || echo "GNUBG FAILED"
echo "----- arena: exp007-final vs iter_0447 (internal bar) -----"
$PY scripts/evaluate.py --checkpoint1 "$LAST" --checkpoint2 "$ITER0447" \
  --games 100 --simulations 100 || echo "ARENA-vs-0447 FAILED"
echo "===== STAGE B DONE $(date -u +%F_%T) ====="
echo "===== EXP007 COMPLETE $(date -u +%F_%T) ====="
