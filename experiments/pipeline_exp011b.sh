#!/bin/bash
# exp011b — finish exp011's 0-ply value distillation at FULL epoch length on the T4.
# Fresh from random init (no warm-start), both arms, 3 full epochs. Reuses the exp011
# 0-ply cache (pulled from GCS if absent). train_distill.py now saves a distinct
# ep{N}.pt after each completed epoch → low-noise offline selection on the iMac.
#
# Cost controls ("don't run loose", tolerance ~$20, never ~$200):
#   - --max-wall-hours 8 per arm  → 16h training ceiling (generous; won't strangle a
#     slow-throughput run, but still bounded)
#   - trap EXIT: final GCS sync + STOP THE VM on ANY exit (success / failure / error)
#   - DEAD-MAN TIMER: an unconditional `shutdown` scheduled at launch that halts the VM
#     even if the pipeline HANGS and the trap never fires — the true runaway guard
#   - no auto-restart watchdog; if preempted, restart the affected arm by hand
#
# Smoke:  SMOKE=1 bash experiments/pipeline_exp011b.sh    (2 shards, 1 epoch; ~min on T4)
# Full:   bash experiments/pipeline_exp011b.sh            (run detached in tmux; ~3h on T4)
# Keep VM up afterwards (e.g. to inspect): STOPVM=0 bash experiments/pipeline_exp011b.sh
# Tunables via env: EXP CACHE GCS_CACHE EPOCHS EVERY EVGAMES MAXWALL LR ZONE VM STOPVM
set -u
cd "$(dirname "$0")/.."
export OMP_WAIT_POLICY=PASSIVE
PY=.venv/bin/python3

CACHE=${CACHE:-data/distill/0ply/run1}
GCS_CACHE=${GCS_CACHE:-gs://raccoon-training-lhm/data/distill/0ply/run1}
GCS_EXP_ROOT=${GCS_EXP_ROOT:-gs://raccoon-training-lhm/experiments}
LR=${LR:-1e-3}
ZONE=${ZONE:-europe-west1-b}
VM=${VM:-raccoon-gpu}
STOPVM=${STOPVM:-1}

if [ "${SMOKE:-0}" = "1" ]; then
  EXP=${EXP:-exp011b-distill-smoke}; EPOCHS=1; EVERY=1; EVGAMES=2; MAXWALL=0; SMOKEFLAG="--smoke"
  STOPVM=0
  echo "===== SMOKE MODE (VM will NOT be stopped) ====="
else
  EXP=${EXP:-exp011b-distill}; EPOCHS=${EPOCHS:-3}; EVERY=${EVERY:-18}; EVGAMES=${EVGAMES:-40}
  MAXWALL=${MAXWALL:-8}; SMOKEFLAG=""
fi

# On ANY exit: back everything up to GCS, then stop the VM. This is the primary
# cost control — a failed/half-run must never leave the VM billing.
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

# Dead-man timer: an unconditional guest shutdown scheduled NOW. If the pipeline
# hangs (stuck shard / eval) the EXIT trap never fires and --max-wall-hours never
# trips — this is the only guard against an indefinite runaway. Absolute VM-cost
# bound = DEADMAN_MIN. A clean finish stops the VM earlier via the trap, making
# this moot. Skipped in SMOKE (STOPVM=0). Needs passwordless sudo (default on GCE).
DEADMAN_MIN=${DEADMAN_MIN:-1200}   # 20h; at ~$0.15-0.50/hr that caps VM cost ~$3-10
if [ "$STOPVM" = "1" ] && [ "$DEADMAN_MIN" -gt 0 ]; then
  sudo shutdown -h +"$DEADMAN_MIN" 2>/dev/null \
    && echo "dead-man VM shutdown armed for +${DEADMAN_MIN}min" \
    || echo "WARN: could not arm dead-man shutdown (need passwordless sudo) — watch the VM manually"
fi

# 1. Ensure the 0-ply cache is present (pull from GCS if absent).
mkdir -p "$CACHE"
if ls "$CACHE"/shard_*.npz >/dev/null 2>&1; then
  echo "cache present in $CACHE — skipping GCS pull"
else
  echo "===== PULL CACHE FROM GCS $(date -u +%F_%T) ====="
  gcloud storage rsync "$GCS_CACHE/" "$CACHE/" --recursive || { echo "CACHE PULL FAILED"; exit 1; }
fi

# 2. Train both arms from random init, full epochs. Inline eval is a heartbeat only
#    (n=40, once per epoch); the real best is picked offline at n>=1000 on the iMac.
for HEAD in scalar outcomes6; do
  echo "===== TRAIN ARM $HEAD $(date -u +%F_%T) ====="
  $PY scripts/train_distill.py --cache-dir "$CACHE" \
    --experiment-name "$EXP/$HEAD" --value-head "$HEAD" \
    --epochs "$EPOCHS" --lr "$LR" --eval-every-shards "$EVERY" \
    --eval-games "$EVGAMES" --gnubg-ply 0 --max-wall-hours "$MAXWALL" $SMOKEFLAG \
    || { echo "TRAIN $HEAD FAILED"; exit 1; }
  # prompt per-arm backup (trap also does a final full sync)
  gcloud storage rsync "experiments/$EXP/$HEAD/" "$GCS_EXP_ROOT/$EXP/$HEAD/" --recursive \
    2>/dev/null || echo "WARN: GCS sync failed for $HEAD"
done

echo "===== EXP011B COMPLETE $(date -u +%F_%T) ====="
# trap 'finish' runs here: final sync + VM stop.
