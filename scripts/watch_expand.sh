#!/usr/bin/env bash
# Local watchdog: keeps the GCP VM alive during dataset expansion.
# Polls every 5 min; on preemption restarts the VM and re-runs resume_expand.sh
# (which is idempotent — skips existing shards and does nothing if gen40m is alive).
#
# Usage: watch_expand.sh
# Run with systemd-inhibit to prevent the local machine from sleeping:
#   systemd-inhibit --what=sleep:idle --who=watch_expand --why="keep VM alive" \
#     ./scripts/watch_expand.sh

set -euo pipefail

VM="raccoon-gpu"
ZONE="europe-west1-b"
POLL_INTERVAL=300
STOCKOUT_RETRY=120
# Resolve gcloud: works from Git Bash (/c/...) or can be overridden via env
if command -v gcloud &>/dev/null; then
  GCLOUD="gcloud"
elif [ -x "/c/Users/LMDN/google-cloud-sdk/bin/gcloud" ]; then
  GCLOUD="/c/Users/LMDN/google-cloud-sdk/bin/gcloud"
elif [ -x "/mnt/c/Users/LMDN/google-cloud-sdk/bin/gcloud" ]; then
  GCLOUD="/mnt/c/Users/LMDN/google-cloud-sdk/bin/gcloud"
else
  echo "ERROR: gcloud not found" >&2; exit 1
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

vm_status() {
  "$GCLOUD" compute instances describe "$VM" --zone="$ZONE" \
    --format="value(status)" 2>/dev/null || echo "UNKNOWN"
}

start_vm() {
  while true; do
    local out
    if out="$("$GCLOUD" compute instances start "$VM" --zone="$ZONE" 2>&1)"; then
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
    if "$GCLOUD" compute ssh lasse@"$VM" --zone="$ZONE" --command="true" \
         -- -batch >/dev/null 2>&1; then
      return 0
    fi
    tries=$((tries + 1))
    sleep 10
  done
  return 1
}

resume_expand() {
  log "Running resume_expand.sh on VM"
  "$GCLOUD" compute ssh lasse@"$VM" --zone="$ZONE" \
    --command="bash ~/raccoon/scripts/resume_expand.sh" \
    -- -batch
}

log "Watchdog started for dataset expansion"
while true; do
  status="$(vm_status)"
  case "$status" in
    RUNNING)
      log "VM is RUNNING — OK"
      ;;
    TERMINATED|STOPPED|STOPPING)
      log "VM status=$status — bringing it back up"
      if start_vm && wait_for_ssh; then
        resume_expand || log "resume_expand failed"
      else
        log "Could not reach VM after start"
      fi
      ;;
    STAGING|PROVISIONING|REPAIRING)
      log "VM transitioning ($status) — waiting"
      ;;
    *)
      log "Unexpected status: $status"
      ;;
  esac
  sleep "$POLL_INTERVAL"
done
