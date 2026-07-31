#!/usr/bin/env bash
# Local watchdog for tmux-run pipelines on the spot VM (generalizes
# watch_vm.sh, which is coupled to resume_training.sh / train.py).
#
# Each tick:
#   1. Check VM status; if preempted/stopped -> start it (retry on stockout).
#   2. Wait for SSH.
#   3. If the pipeline log contains DONE_MARKER -> report, optionally stop the
#      VM (--stop-on-done), and exit.
#   4. Otherwise ensure the tmux session exists, relaunching CMD if not.
#      (The pipelines are resume-safe: completed rounds/artifacts are skipped.)
#
# Usage:
#   watch_pipeline.sh SESSION LOGPATH DONE_MARKER CMD [--stop-on-done]
# Example:
#   scripts/watch_pipeline.sh consol \
#     'experiments/pretrain-consol/logs/pipeline.log' 'CONSOLIDATE COMPLETE' \
#     'BASELINE=1 PLYBENCH=1 bash experiments/pipeline_consolidate.sh >> experiments/pretrain-consol/logs/pipeline.log 2>&1' \
#     --stop-on-done

set -euo pipefail

SESSION="${1:?usage: watch_pipeline.sh SESSION LOGPATH DONE_MARKER CMD [--stop-on-done]}"
LOGPATH="${2:?missing LOGPATH (relative to ~/raccoon on the VM)}"
DONE_MARKER="${3:?missing DONE_MARKER}"
CMD="${4:?missing CMD}"
STOP_ON_DONE="${5:-}"

VM="raccoon-gpu"
ZONE="europe-west1-b"
POLL_INTERVAL=300
STOCKOUT_RETRY=120

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

vm_status() {
  gcloud compute instances describe "$VM" --zone="$ZONE" \
    --format="value(status)" 2>/dev/null || echo "UNKNOWN"
}

start_vm() {
  while true; do
    local out
    if out="$(gcloud compute instances start "$VM" --zone="$ZONE" 2>&1)"; then
      log "VM started"
      return 0
    fi
    if grep -q "ZONE_RESOURCE_POOL_EXHAUSTED\|STOCKOUT" <<<"$out"; then
      log "Stockout in $ZONE — retrying in ${STOCKOUT_RETRY}s"
      sleep "$STOCKOUT_RETRY"
      continue
    fi
    log "Start failed (non-stockout): $out"
    return 1
  done
}

wait_for_ssh() {
  local tries=0
  while (( tries < 30 )); do
    if gcloud compute ssh "$VM" --zone="$ZONE" --command="true" \
         -- -o ConnectTimeout=10 -o StrictHostKeyChecking=no >/dev/null 2>&1; then
      return 0
    fi
    tries=$((tries + 1))
    sleep 10
  done
  return 1
}

log "Watchdog started: session=$SESSION marker='$DONE_MARKER' log=$LOGPATH"
while true; do
  status="$(vm_status)"
  case "$status" in
    RUNNING|TERMINATED|STOPPED|STOPPING)
      if [ "$status" != "RUNNING" ]; then
        log "VM status=$status — bringing it back up"
        start_vm && wait_for_ssh || { log "Could not reach VM"; sleep "$POLL_INTERVAL"; continue; }
      fi
      state="$(gcloud compute ssh "$VM" --zone="$ZONE" --command="
        cd ~/raccoon
        if grep -q '$DONE_MARKER' '$LOGPATH' 2>/dev/null; then echo PIPELINE_DONE
        elif tmux has-session -t '$SESSION' 2>/dev/null; then echo TMUX_UP
        else echo TMUX_DOWN; fi" 2>/dev/null || echo SSH_FAIL)"
      case "$state" in
        *PIPELINE_DONE*)
          log "Pipeline finished ('$DONE_MARKER' found)"
          if [ "$STOP_ON_DONE" = "--stop-on-done" ]; then
            log "Stopping VM"
            gcloud compute instances stop "$VM" --zone="$ZONE" >/dev/null 2>&1 || true
          fi
          log "WATCHDOG_DONE"
          exit 0
          ;;
        *TMUX_UP*)
          log "VM RUNNING, tmux '$SESSION' alive — OK"
          ;;
        *TMUX_DOWN*)
          log "tmux '$SESSION' missing — relaunching pipeline"
          gcloud compute ssh "$VM" --zone="$ZONE" \
            --command="cd ~/raccoon && tmux new-session -d -s '$SESSION' '$CMD'" \
            || log "relaunch failed"
          ;;
        *)
          log "SSH check failed — will retry"
          ;;
      esac
      ;;
    STAGING|PROVISIONING|REPAIRING)
      log "VM transitioning (status=$status) — waiting"
      ;;
    *)
      log "Unexpected status: $status"
      ;;
  esac
  sleep "$POLL_INTERVAL"
done
