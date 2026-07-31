#!/bin/bash
# exp010 — TD(λ) self-play pilot, warm-started from the best exp009 checkpoint
# (round_06). Local run on the iMac CPU. Yes/no question: does TD value
# self-play lift the net vs the round_06 seed (1-ply arena)?  If yes, worth
# scaling / a GNUBG confirmation; if flat-or-worse, TD doesn't lift this net.
#
# Smoke first:  SMOKE=1 bash experiments/pipeline_exp010.sh   (~2 min)
# Pilot:        bash experiments/pipeline_exp010.sh           (overnight)
# Tunables via env: SEED EXP GPB BATCHES EVGAMES EVERY WORKERS LAM LR TEMP
#   PATIENCE MAXWALL
set -u
cd "$(dirname "$0")/.."
export OMP_WAIT_POLICY=PASSIVE           # iMac: avoid CPU spin-collapse
PY=.venv/bin/python3

EXP=${EXP:-exp010-td-pilot}
SEED=${SEED:-experiments/exp009-ondist-dagger/round_06/checkpoints/pretrained_v2.pt}
LAM=${LAM:-0.7}; LR=${LR:-1e-4}; TEMP=${TEMP:-0.0}
WORKERS=${WORKERS:-$(( $(nproc) > 2 ? $(nproc) - 1 : 1 ))}   # leave a core free

if [ "${SMOKE:-0}" = "1" ]; then
  GPB=6; BATCHES=2; EVGAMES=2; EVERY=1; WORKERS=2; PATIENCE=0; MAXWALL=0
  echo "===== SMOKE MODE ====="
else
  # ~2.3 h per 500-game batch (generation-bound), so favour information density:
  # small batches with a GNUBG-0-ply eval every batch → a readable, trustworthy
  # strength-vs-GNUBG curve rather than 2 points. Eval is cheap now (0-ply,
  # ~3-5 s/game), so n can be generous. MAXWALL bounds the run; no plateau stop.
  GPB=${GPB:-150}; BATCHES=${BATCHES:-40}; EVGAMES=${EVGAMES:-60}
  EVERY=${EVERY:-1}; PATIENCE=${PATIENCE:-0}; MAXWALL=${MAXWALL:-13}
fi

[ -f "$SEED" ] || { echo "ABORT: missing seed $SEED"; exit 1; }
echo "exp010: seed=$SEED games/batch=$GPB batches=$BATCHES workers=$WORKERS"

$PY scripts/train_td.py --experiment-name "$EXP" --seed "$SEED" \
  --games-per-batch "$GPB" --batches "$BATCHES" --lam "$LAM" --lr "$LR" \
  --temperature "$TEMP" --workers "$WORKERS" --eval-every "$EVERY" \
  --eval-games "$EVGAMES" --gnubg-ply 0 --patience "$PATIENCE" --min-delta 0.02 \
  --max-wall-hours "$MAXWALL"
