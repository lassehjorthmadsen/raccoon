#!/usr/bin/env bash
# Resume a value-distillation run (train_distill.py) from latest.pt after a spot
# preemption. Runs on the VM. Idempotent — a no-op if training is already up or
# already finished. This is the distillation analogue of resume_training.sh
# (which is welded to self-play train.py: iter_*.pt / training_log.jsonl /
# --iterations, none of which apply here). Leave that one untouched.
#
# Unlike self-play, the exact launch flags don't live in the JSONL — they're
# written to experiments/<name>/resume_params.env by pipeline_exp017.sh and
# sourced here, so every resume relaunches with the identical config.
#
# Usage (called by watch_vm.sh, or by hand on the VM):
#   bash ~/raccoon/scripts/resume_distill.sh [EXPERIMENT_NAME]
set -euo pipefail

EXPNAME="${1:-exp017-distill}"
VM="${VM:-raccoon-gpu}"
ZONE="${ZONE:-europe-west1-b}"
REPO="$HOME/raccoon"
EXP_DIR="$REPO/experiments/$EXPNAME"
GCS_BUCKET="${GCS_BUCKET:-gs://raccoon-training-lhm}"

cd "$REPO"

# 0. Already complete? train_distill.py writes DONE only on a clean finish
#    (all epochs, or the cumulative --max-wall-hours cap). Never relaunch past it.
if [ -f "$EXP_DIR/DONE" ]; then
  echo "DONE present ($EXP_DIR/DONE) — training complete, not relaunching"
  exit 0
fi

# 1. Launch flags — pull the params file from GCS on a fresh disk if needed.
PARAMS="$EXP_DIR/resume_params.env"
if [ ! -f "$PARAMS" ]; then
  mkdir -p "$EXP_DIR"
  gcloud storage cp "$GCS_BUCKET/experiments/$EXPNAME/resume_params.env" "$PARAMS" 2>/dev/null || true
fi
[ -f "$PARAMS" ] || { echo "missing $PARAMS (run the experiment's pipeline_*.sh first)" >&2; exit 1; }
# shellcheck disable=SC1090
source "$PARAMS"   # CACHE_DIR GCS_CACHE VALUE_HEAD EPOCHS LR EVAL_EVERY EVAL_GAMES MAXWALL SHUFFLE_SEED GNUBG_PLY

# 2. Ensure the cache is present (a fresh spot-VM disk has none; a stop/start disk
#    keeps it). CACHE_DIR spans run1+run2+run3 — shard discovery is recursive.
if find "$CACHE_DIR" -name 'shard_*.npz' -print -quit 2>/dev/null | grep -q .; then
  echo "cache present in $CACHE_DIR — skipping GCS pull"
else
  echo "pulling cache $GCS_CACHE -> $CACHE_DIR ($(date -u +%F_%T)) ..."
  mkdir -p "$CACHE_DIR"
  gcloud storage rsync "$GCS_CACHE/" "$CACHE_DIR/" --recursive || { echo "CACHE PULL FAILED" >&2; exit 1; }
fi

# 3. Already training? Idempotent — nothing to do.
if tmux has-session -t train 2>/dev/null; then
  echo "tmux 'train' already running — nothing to do"
  exit 0
fi

# 4. Launch training (--resume auto = continue from latest.pt, or fresh if absent)
#    plus a GCS auto-sync loop. On a clean finish, self-stop the VM (the watchdog
#    also detects DONE in GCS independently and exits).
echo "launching train_distill.py for $EXPNAME ($(date -u +%F_%T))"
tmux new-session -d -s train "
  source .venv/bin/activate &&
  export OMP_WAIT_POLICY=PASSIVE &&
  python scripts/train_distill.py \
    --cache-dir '$CACHE_DIR' \
    --experiment-name '$EXPNAME' \
    --value-head '$VALUE_HEAD' \
    --epochs $EPOCHS --lr $LR \
    --eval-every-shards $EVAL_EVERY --eval-games $EVAL_GAMES \
    --gnubg-ply $GNUBG_PLY --max-wall-hours $MAXWALL \
    --shuffle-seed $SHUFFLE_SEED --resume auto;
  if [ -f '$EXP_DIR/DONE' ]; then
    echo 'DONE — final GCS sync + stopping VM';
    gcloud storage rsync experiments/$EXPNAME/ $GCS_BUCKET/experiments/$EXPNAME/ --recursive || true;
    gcloud compute instances stop $VM --zone=$ZONE 2>/dev/null || sudo shutdown -h now;
  fi;
  exec bash
"

if ! tmux has-session -t sync 2>/dev/null; then
  tmux new-session -d -s sync "
    while true; do
      gcloud storage rsync experiments/$EXPNAME/ $GCS_BUCKET/experiments/$EXPNAME/ --recursive;
      echo \"Synced at \$(date)\";
      sleep 300;
    done
  "
fi

echo "launched tmux sessions: train, sync"
