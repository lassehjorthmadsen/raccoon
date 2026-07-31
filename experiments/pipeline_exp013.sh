#!/usr/bin/env bash
# exp013 — TD(λ) value self-play from the exp011b near-parity seed (scalar/ep3).
# Question: does TD from a near-parity seed climb ABOVE 0-ply, or decay to exp010's
# ~-0.3 attractor? Reuses train_td.py (scalar-value TD). iMac CPU, FREE. temp=0 —
# the dice supply exploration (TD-Gammon style), matching exp010 for a clean comparison.
#
# Long, unattended run: no plateau auto-stop (--patience 0) — we decide ride-vs-interrupt
# by hand. Bounded by --max-wall-hours (iMac is free, so the cap is just a backstop).
# Interrupt anytime: `pkill -f train_td.py`. Restart-on-reboot: re-run with SEED=.../latest.pt.
#
# Run detached:
#   nohup setsid experiments/pipeline_exp013.sh > experiments/exp013_run.log 2>&1 &
set -uo pipefail
cd /home/lasse/python-projects/raccoon
export OMP_WAIT_POLICY=PASSIVE   # iMac CPU: avoid spin-collapse under contention
PY=.venv/bin/python3

SEED=${SEED:-experiments/exp011b-distill/scalar/checkpoints/ep3.pt}
EXP=${EXP:-exp013-td-scalar}
WORKERS=${WORKERS:-3}
MAXWALL=${MAXWALL:-24}
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# 1. Seed baseline — the reference the TD trajectory is measured against (error-rate
#    metric is low-variance, so n=1000 pins it tight). Runs before training, no contention.
log "=== SEED BASELINE: eval_errorrate on $SEED (n=1000 vs GNUBG-0-ply) ==="
$PY scripts/eval_errorrate.py --checkpoint "$SEED" --games 1000 --workers "$WORKERS" \
  --ply 0 --label "exp013 seed scalar/ep3" || { echo "SEED EVAL FAILED"; exit 1; }

# 2. TD self-play. temp=0; exp010 hyperparams. --checkpoint-every 5 → a batch_{N}.pt
#    series for the offline error-rate fine-eval (the coarse in-loop eval only flags gross moves).
log "=== TD SELF-PLAY from $SEED (temp=0, lam=0.7, lr=1e-4, 500 games/batch) ==="
$PY scripts/train_td.py --seed "$SEED" --experiment-name "$EXP" \
  --lam 0.7 --lr 1e-4 --games-per-batch 500 --temperature 0.0 \
  --workers "$WORKERS" --batches 500 --eval-every 3 --eval-games 100 \
  --gnubg-ply 0 --checkpoint-every 5 --max-wall-hours "$MAXWALL" --patience 0

log "=== EXP013 DONE (batches exhausted or ${MAXWALL}h wall cap) ==="
