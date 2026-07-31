#!/bin/bash
# exp011 — GNUBG-0-ply value distillation at volume, two arms (scalar / outcomes6).
# Generate a GNUBG-self-play cache, then train each arm from random init and eval
# its 0-ply play vs GNUBG-0-ply. Local, value-only. ~48h budget: gen (~4h) +
# arm A (<=MAXWALL) + arm B (<=MAXWALL).
#
# Smoke:  SMOKE=1 bash experiments/pipeline_exp011.sh    (~minutes)
# Full:   bash experiments/pipeline_exp011.sh            (detached; ~1-2 days)
# Tunables via env: EXP CACHE POSITIONS SHARD WORKERS EPOCHS EVGAMES EVERY MAXWALL LR
set -u
cd "$(dirname "$0")/.."
export OMP_WAIT_POLICY=PASSIVE
PY=.venv/bin/python3

EXP=${EXP:-exp011-distill}
CACHE=${CACHE:-data/distill/0ply/run1}
WORKERS=${WORKERS:-$(( $(nproc) > 2 ? $(nproc) - 1 : 2 ))}
LR=${LR:-1e-3}

if [ "${SMOKE:-0}" = "1" ]; then
  POSITIONS=2000; SHARD=500; EPOCHS=1; EVGAMES=2; EVERY=1; MAXWALL=0; SMOKEFLAG="--smoke"
  echo "===== SMOKE MODE ====="
else
  POSITIONS=${POSITIONS:-10000000}; SHARD=${SHARD:-500000}
  EPOCHS=${EPOCHS:-2}; EVGAMES=${EVGAMES:-40}; EVERY=${EVERY:-6}; MAXWALL=${MAXWALL:-18}
  SMOKEFLAG=""
fi

mkdir -p "$CACHE"

if ls "$CACHE"/shard_*.npz >/dev/null 2>&1; then
  echo "cache present in $CACHE — skipping generation"
else
  echo "===== GENERATE $(date -u +%F_%T) ====="
  $PY scripts/gen_gnubg_selfplay.py --out-dir "$CACHE" --positions "$POSITIONS" \
    --shard-size "$SHARD" --workers "$WORKERS" || { echo "GEN FAILED"; exit 1; }
fi

for HEAD in scalar outcomes6; do
  echo "===== TRAIN ARM $HEAD $(date -u +%F_%T) ====="
  $PY scripts/train_distill.py --cache-dir "$CACHE" \
    --experiment-name "$EXP/$HEAD" --value-head "$HEAD" \
    --epochs "$EPOCHS" --lr "$LR" --eval-every-shards "$EVERY" \
    --eval-games "$EVGAMES" --gnubg-ply 0 --max-wall-hours "$MAXWALL" $SMOKEFLAG \
    || { echo "TRAIN $HEAD FAILED"; exit 1; }
done

echo "===== EXP011 COMPLETE $(date -u +%F_%T) ====="
