#!/usr/bin/env bash
# One-shot companion to watch_pipeline.sh for the consol run: wait until the
# watchdog reports the pipeline done (it stops the VM via --stop-on-done),
# then restart the VM, back the experiment up to GCS, fetch checkpoints+logs
# to this machine, and stop the VM again.
#
# caches/ is excluded from both the GCS backup and the local fetch: multi-GB,
# reproducible from the local exp008 caches via scripts/reencode_cache.py +
# scripts/merge_caches.py, and still present on the VM disk regardless.
#
# Run detached:
#   nohup setsid experiments/fetch_consol_on_done.sh \
#     > experiments/fetch_consol.log 2>&1 &
set -uo pipefail
cd /home/lasse/python-projects/raccoon

VM=raccoon-gpu
ZONE=europe-west1-b
EXP=pretrain-consol
WLOG=experiments/watch_consol.log
GCS=gs://raccoon-training-lhm/experiments/$EXP

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
vssh() {
  gcloud compute ssh "$VM" --zone="$ZONE" --command="$1" \
    -- -o ConnectTimeout=15 -o StrictHostKeyChecking=no
}

log "Waiting for WATCHDOG_DONE in $WLOG"
until grep -q "WATCHDOG_DONE" "$WLOG" 2>/dev/null; do sleep 60; done
log "Pipeline complete — bringing VM back up for the fetch"

while true; do
  st=$(gcloud compute instances describe "$VM" --zone="$ZONE" \
    --format='value(status)' 2>/dev/null || echo UNKNOWN)
  case "$st" in
    RUNNING) break ;;
    TERMINATED|STOPPED)
      if out=$(gcloud compute instances start "$VM" --zone="$ZONE" 2>&1); then
        log "VM started"
      elif grep -q "ZONE_RESOURCE_POOL_EXHAUSTED\|STOCKOUT" <<<"$out"; then
        log "Stockout in $ZONE — retrying in 120s"; sleep 120
      else
        log "Start failed: $out — retrying in 300s"; sleep 300
      fi ;;
    *) sleep 20 ;;
  esac
done

tries=0
until vssh true >/dev/null 2>&1; do
  tries=$((tries + 1))
  [ "$tries" -ge 40 ] && { log "FETCH_FAIL: SSH never came up"; exit 1; }
  sleep 10
done
log "SSH up"

log "GCS backup -> $GCS (excluding caches/)"
if vssh "cd ~/raccoon && gsutil -m rsync -r -x 'caches/.*' experiments/$EXP $GCS"; then
  log "GCS backup OK"
else
  log "WARNING: GCS backup failed — continuing with local fetch"
fi

ok=0
for i in 1 2 3; do
  log "Fetching $EXP to local (attempt $i)"
  if vssh "tar czf - -C ~/raccoon/experiments --exclude='$EXP/caches' $EXP" \
      | tar xzf - -C experiments/; then
    ok=1; break
  fi
  sleep 30
done

if [ "$ok" = 1 ] && grep -q "CONSOLIDATE COMPLETE" "experiments/$EXP/logs/pipeline.log"; then
  log "Fetched: $(find experiments/$EXP -type f | wc -l) files," \
      "$(du -sh experiments/$EXP | cut -f1) (caches excluded)"
  log "Stopping VM"
  gcloud compute instances stop "$VM" --zone="$ZONE" >/dev/null 2>&1 && log "VM stopped"
  log "FETCH_DONE"
else
  log "Fetch incomplete — data remains on the VM disk (and GCS if backup OK)"
  log "Stopping VM anyway to save cost"
  gcloud compute instances stop "$VM" --zone="$ZONE" >/dev/null 2>&1
  log "FETCH_FAIL"
  exit 1
fi
