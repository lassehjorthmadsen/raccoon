#!/usr/bin/env bash
# Resume dataset expansion on the VM after a preemption.
# Runs on the VM. Re-launches the gen40m tmux session if it isn't already alive.
# Both steps (gen_gnubg_selfplay and relabel_2ply) resume safely: the generator
# writes to a fresh out-dir (no collision) and relabel_2ply skips existing shards.

set -euo pipefail

REPO="$HOME/raccoon"
LOG="$REPO/experiments/exp011-distill/expand40m.log"

cd "$REPO"

if tmux has-session -t gen40m 2>/dev/null; then
  echo "tmux session 'gen40m' already running — nothing to do"
  exit 0
fi

echo "=== Resuming expand40m: $(date) ===" | tee -a "$LOG"

tmux new-session -d -s gen40m "
  cd $REPO
  source .venv/bin/activate

  echo '--- gen step: '\$(date) >> $LOG
  python scripts/gen_gnubg_selfplay.py \
      --out-dir experiments/exp011-distill/cache_new2 \
      --positions 24000000 \
      --shard-size 500000 \
      --workers 4 \
      --seed 12345 \
      2>&1 | tee -a $LOG

  echo '--- relabel step: '\$(date) >> $LOG
  python scripts/relabel_2ply.py \
      --in-dir experiments/exp011-distill/cache_new2 \
      --out-dir experiments/exp011-distill/cache_new2_2ply \
      --ply 2 \
      --workers 14 \
      2>&1 | tee -a $LOG

  echo '=== expand40m done: '\$(date) >> $LOG
"

echo "Launched tmux session: gen40m"
