#!/usr/bin/env bash
# Resume training from the latest checkpoint in an experiment.
# Runs on the VM. Launches training in a detached tmux session named "train"
# and a GCS auto-sync loop in a tmux session named "sync".
#
# Usage: resume_training.sh EXPERIMENT_NAME [ITERATIONS]
#   EXPERIMENT_NAME — e.g. exp001-6x128-200sims
#   ITERATIONS      — remaining iterations to run (default: 500)

set -euo pipefail

EXPNAME="${1:?usage: resume_training.sh EXPERIMENT_NAME [ITERATIONS]}"
ITERATIONS="${2:-500}"

REPO="$HOME/raccoon"
CKPT_DIR="$REPO/experiments/$EXPNAME/checkpoints"
GCS_BUCKET="gs://raccoon-training-lhm"

cd "$REPO"

if ! [ -d "$CKPT_DIR" ]; then
  echo "No checkpoint dir: $CKPT_DIR" >&2
  exit 1
fi

LATEST="$(ls "$CKPT_DIR"/iter_*.pt 2>/dev/null | sort | tail -1 || true)"
if [ -z "$LATEST" ]; then
  echo "No checkpoints found in $CKPT_DIR" >&2
  exit 1
fi

echo "Resuming $EXPNAME from $LATEST for $ITERATIONS iterations"

if tmux has-session -t train 2>/dev/null; then
  echo "tmux session 'train' already running — nothing to do"
  exit 0
fi

tmux new-session -d -s train "
  source .venv/bin/activate &&
  python scripts/train.py \
    --experiment-name '$EXPNAME' \
    --iterations $ITERATIONS \
    --games-per-iter 50 --simulations 200 \
    --training-steps 100 --batch-size 256 \
    --replay-size 500000 --checkpoint-every 1 \
    --resume '$LATEST';
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

echo "Launched tmux sessions: train, sync"
