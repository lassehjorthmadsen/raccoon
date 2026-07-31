#!/bin/bash
# exp014 — does distilling GNUBG-2-ply (vs 0-ply) produce a stronger scalar net?
# Trains the SAME scalar recipe (10x256, 3 epochs, lr=1e-3) twice on the IDENTICAL
# 6M positions — once on the 0-ply cache, once on the 2-ply cache (same shard names,
# byte-identical observations) — holding out the same 2M rows from both via
# --holdout-frac/--split-seed (raccoon.data.cache_split). The held-out R^2 comparison
# (scripts/eval_r2.py) is the primary metric; this script only trains.
#
# Cost controls (same template as pipeline_exp011b.sh — "don't run loose"):
#   - --max-wall-hours per arm (below) → bounded training ceiling
#   - trap EXIT: final GCS sync + STOP THE VM on ANY exit (success / failure / error)
#   - DEAD-MAN TIMER: unconditional guest `shutdown` scheduled at launch — the guard
#     against a HANG (stuck shard/eval) where the trap itself never fires
#   - no auto-restart watchdog; if preempted, restart the affected arm by hand
#
# Smoke:  SMOKE=1 bash experiments/pipeline_exp014.sh    (2 shards, 1 epoch; ~min on T4)
# Full:   bash experiments/pipeline_exp014.sh            (run detached in tmux; ~2-2.5h/arm on T4)
# Keep VM up afterwards (e.g. to inspect): STOPVM=0 bash experiments/pipeline_exp014.sh
# Tunables via env: EXP EPOCHS EVERY EVGAMES MAXWALL LR HOLDOUT SPLITSEED ZONE VM STOPVM
set -u
cd "$(dirname "$0")/.."
export OMP_WAIT_POLICY=PASSIVE
PY=.venv/bin/python3

GCS_EXP_ROOT=${GCS_EXP_ROOT:-gs://raccoon-training-lhm/experiments}
LR=${LR:-1e-3}
HOLDOUT=${HOLDOUT:-0.25}
SPLITSEED=${SPLITSEED:-14}
ZONE=${ZONE:-europe-west1-b}
VM=${VM:-raccoon-gpu}
STOPVM=${STOPVM:-1}

if [ "${SMOKE:-0}" = "1" ]; then
  EXP=${EXP:-exp014-distill-smoke}; EPOCHS=1; EVERY=1; EVGAMES=2; MAXWALL=0; SMOKEFLAG="--smoke"
  STOPVM=0
  echo "===== SMOKE MODE (VM will NOT be stopped) ====="
else
  EXP=${EXP:-exp014-distill}; EPOCHS=${EPOCHS:-3}; EVERY=${EVERY:-18}; EVGAMES=${EVGAMES:-40}
  MAXWALL=${MAXWALL:-5}; SMOKEFLAG=""
fi

# On ANY exit: back everything up to GCS, then stop the VM. Primary cost control —
# a failed/half-run must never leave the VM billing.
finish() {
  local rc=$?
  echo "===== FINISH (rc=$rc) $(date -u +%F_%T) ====="
  gcloud storage rsync "experiments/$EXP/" "$GCS_EXP_ROOT/$EXP/" --recursive \
    2>/dev/null || echo "WARN: final GCS sync failed — sync manually before deleting the VM"
  if [ "$STOPVM" = "1" ]; then
    echo "stopping VM $VM in $ZONE ..."
    gcloud compute instances stop "$VM" --zone="$ZONE" 2>/dev/null \
      || { echo "gcloud stop failed — falling back to guest shutdown"; sudo shutdown -h +1; }
  fi
}
trap finish EXIT

# Dead-man timer: see pipeline_exp011b.sh for rationale. Absolute VM-cost bound;
# a clean finish stops the VM earlier via the trap, making this moot.
DEADMAN_MIN=${DEADMAN_MIN:-900}   # 15h; MAXWALL*2 arms=10h ceiling, generous margin
if [ "$STOPVM" = "1" ] && [ "$DEADMAN_MIN" -gt 0 ]; then
  sudo shutdown -h +"$DEADMAN_MIN" 2>/dev/null \
    && echo "dead-man VM shutdown armed for +${DEADMAN_MIN}min" \
    || echo "WARN: could not arm dead-man shutdown (need passwordless sudo) — watch the VM manually"
fi

# Each arm: (label, run1 ply subdir under data/distill). Same positions in both —
# only the label ply differs (see data/README.md).
declare -A CACHES=( [scalar_0ply]="0ply/run1" [scalar_2ply]="2ply/run1" )

for ARM in scalar_0ply scalar_2ply; do
  CACHE="data/distill/${CACHES[$ARM]}"
  GCS_CACHE="gs://raccoon-training-lhm/data/distill/${CACHES[$ARM]}"

  mkdir -p "$CACHE"
  if ls "$CACHE"/shard_*.npz >/dev/null 2>&1; then
    echo "cache present in $CACHE — skipping GCS pull"
  else
    echo "===== PULL CACHE FROM GCS $(date -u +%F_%T) ($ARM) ====="
    gcloud storage rsync "$GCS_CACHE/" "$CACHE/" --recursive || { echo "CACHE PULL FAILED ($ARM)"; exit 1; }
  fi

  echo "===== TRAIN ARM $ARM $(date -u +%F_%T) ====="
  $PY scripts/train_distill.py --cache-dir "$CACHE" \
    --experiment-name "$EXP/$ARM" --value-head scalar \
    --epochs "$EPOCHS" --lr "$LR" --eval-every-shards "$EVERY" \
    --eval-games "$EVGAMES" --gnubg-ply 0 --max-wall-hours "$MAXWALL" \
    --holdout-frac "$HOLDOUT" --split-seed "$SPLITSEED" $SMOKEFLAG \
    || { echo "TRAIN $ARM FAILED"; exit 1; }
  # prompt per-arm backup (trap also does a final full sync)
  gcloud storage rsync "experiments/$EXP/$ARM/" "$GCS_EXP_ROOT/$EXP/$ARM/" --recursive \
    2>/dev/null || echo "WARN: GCS sync failed for $ARM"
done

echo "===== EXP014 TRAINING COMPLETE $(date -u +%F_%T) ====="
# trap 'finish' runs here: final sync + VM stop.
