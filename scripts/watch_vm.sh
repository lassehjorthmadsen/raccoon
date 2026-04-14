#!/usr/bin/env bash
# Local watchdog: keeps the GCP training VM alive and training running.
#
# Each tick:
#   1. Check VM status.
#   2. If not RUNNING → start it (retry on stockout).
#   3. Wait for SSH to be ready.
#   4. SSH in and run resume_training.sh (idempotent — does nothing if training already up).
#
# Usage: watch_vm.sh EXPERIMENT_NAME [ITERATIONS]

set -euo pipefail

EXPNAME="${1:?usage: watch_vm.sh EXPERIMENT_NAME [ITERATIONS]}"
ITERATIONS="${2:-500}"

VM="raccoon-gpu"
ZONE="europe-west1-b"
POLL_INTERVAL=300      # 5 min between normal polls
STOCKOUT_RETRY=120     # 2 min between start retries on stockout

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

resume_training() {
  log "Launching resume_training.sh on VM"
  gcloud compute ssh "$VM" --zone="$ZONE" \
    --command="bash ~/raccoon/scripts/resume_training.sh '$EXPNAME' $ITERATIONS"
}

log "Watchdog started for $EXPNAME (iterations=$ITERATIONS)"
while true; do
  status="$(vm_status)"
  case "$status" in
    RUNNING)
      log "VM is RUNNING — OK"
      ;;
    TERMINATED|STOPPED|STOPPING)
      log "VM status=$status — bringing it back up"
      if start_vm && wait_for_ssh; then
        resume_training || log "resume_training failed"
      else
        log "Could not reach VM after start"
      fi
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
