#!/usr/bin/env bash
# Resume training from the latest checkpoint in an experiment.
# Runs on the VM. Launches training in a detached tmux session named "train"
# and a GCS auto-sync loop in a tmux session named "sync".
#
# Hyperparameters (replay-size, simulations, batch-size, etc.) are read from
# the experiment's training_log.jsonl so the resume uses whatever the
# experiment was originally launched with — no risk of clobbering a tuned
# run with a different default.
#
# Usage: resume_training.sh EXPERIMENT_NAME [ITERATIONS]
#   EXPERIMENT_NAME — e.g. exp001-6x128-200sims
#   ITERATIONS      — remaining iterations to run (default: 500)

set -euo pipefail

EXPNAME="${1:?usage: resume_training.sh EXPERIMENT_NAME [ITERATIONS]}"
ITERATIONS="${2:-500}"

REPO="$HOME/raccoon"
CKPT_DIR="$REPO/experiments/$EXPNAME/checkpoints"
LOG_FILE="$REPO/experiments/$EXPNAME/logs/training_log.jsonl"
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

if ! [ -f "$LOG_FILE" ]; then
  echo "No training_log.jsonl: $LOG_FILE" >&2
  exit 1
fi

# Read hyperparameters from the most recent config entry in the JSONL log.
# Each run writes a {"type": "config", ...} line on startup; we use the
# latest one so partial reruns inherit the same settings.
TRAIN_ARGS="$(python3 - "$LOG_FILE" <<'PYEOF'
import json, sys
config = None
with open(sys.argv[1]) as f:
    for line in f:
        if '"type": "config"' in line:
            config = json.loads(line)
if config is None:
    sys.exit("no config entry found")
t = config["training"]
print(
    f"--games-per-iter {t['games_per_iteration']} "
    f"--simulations {t['num_simulations']} "
    f"--training-steps {t['training_steps_per_iteration']} "
    f"--batch-size {t['batch_size']} "
    f"--replay-size {t['replay_size']}"
)
PYEOF
)"

if [ -z "$TRAIN_ARGS" ]; then
  echo "Could not read training params from $LOG_FILE" >&2
  exit 1
fi

echo "Resuming $EXPNAME from $LATEST for $ITERATIONS iterations"
echo "  training args: $TRAIN_ARGS"

if tmux has-session -t train 2>/dev/null; then
  echo "tmux session 'train' already running — nothing to do"
  exit 0
fi

tmux new-session -d -s train "
  source .venv/bin/activate &&
  python scripts/train.py \
    --experiment-name '$EXPNAME' \
    --iterations $ITERATIONS \
    $TRAIN_ARGS \
    --checkpoint-every 1 \
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
