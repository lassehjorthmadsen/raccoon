#!/usr/bin/env bash
# exp012 — benchmark experiment: characterise exp011 arm A (scalar) best.pt.
# No training; large-n evals to (a) confirm 0-ply parity and (b) size the gap to
# the real 2-ply benchmark. Results (committable) -> experiments/exp012-benchmark/logs/.
#
#   (a) n=6000 vs GNUBG-0-ply  — pin parity (95% CI ~+/-0.045)
#   (b) n=500  vs GNUBG-2-ply  — rough gauge of the searchless gap to close
set -u
cd /home/lasse/python-projects/raccoon
export OMP_WAIT_POLICY=PASSIVE
DIR=experiments/exp012-benchmark
mkdir -p "$DIR/logs"
OUT="$DIR/logs/benchmark.log"
CKPT=experiments/exp011-distill/scalar/checkpoints/best.pt
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$OUT"; }

log "START exp012 benchmark of exp011 arm A (scalar) best.pt"
log "(a) n=6000 vs GNUBG-0-ply"
.venv/bin/python3 scripts/eval_gnubg0.py --checkpoint "$CKPT" \
  --games 6000 --workers 3 --ply 0 --label "armA n=6000 0-ply" >> "$OUT" 2>&1
log "(b) n=500 vs GNUBG-2-ply"
.venv/bin/python3 scripts/eval_gnubg0.py --checkpoint "$CKPT" \
  --games 500 --workers 3 --ply 2 --label "armA n=500 2-ply" >> "$OUT" 2>&1
log "EXP012_DONE"
