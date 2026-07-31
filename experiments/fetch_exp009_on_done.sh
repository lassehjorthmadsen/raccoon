#!/usr/bin/env bash
# One-shot companion to watch_pipeline.sh for the exp009 run: wait until the
# watchdog reports the pipeline done (it stops the VM via --stop-on-done),
# then restart the VM, back the experiment up to GCS, fetch checkpoints+logs
# to this machine, and stop the VM again.
#
# Unlike the consol fetcher, caches/ ARE included in the GCS backup — exp009's
# on-dist round caches cost hours of GNUBG labeling to regenerate and feed
# future consolidations. They are still excluded from the local fetch.
#
# Run detached:
#   nohup setsid experiments/fetch_exp009_on_done.sh \
#     > experiments/fetch_exp009.log 2>&1 &
set -uo pipefail
cd /home/lasse/python-projects/raccoon

VM=raccoon-gpu
ZONE=europe-west1-b
EXP=exp009-ondist-dagger
WLOG=experiments/watch_exp009.log
GCS=gs://raccoon-training-lhm/experiments/$EXP

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
vssh() {
  gcloud compute ssh "$VM" --zone="$ZONE" --command="$1" \
    -- -o ConnectTimeout=15 -o StrictHostKeyChecking=no
}
transient() {
  grep -qiE "EXHAUSTED|STOCKOUT|ConnectionError|NameResolution|Temporary failure|Max retries" <<<"$1"
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
      elif transient "$out"; then
        log "Transient start failure (stockout/network) — retrying in 120s"; sleep 120
      else
        log "Start failed: $out — retrying in 300s"; sleep 300
      fi ;;
    *) sleep 20 ;;
  esac
done

tries=0
until vssh true >/dev/null 2>&1; do
  tries=$((tries + 1))
  [ "$tries" -ge 60 ] && { log "FETCH_FAIL: SSH never came up"; exit 1; }
  sleep 10
done
log "SSH up"

log "GCS backup -> $GCS (caches included)"
if vssh "cd ~/raccoon && gsutil -m -q rsync -r experiments/$EXP $GCS"; then
  log "GCS backup OK"
else
  log "WARNING: GCS backup failed — continuing with local fetch"
fi

ok=0
for i in 1 2 3 4 5; do
  log "Fetching $EXP to local (attempt $i, caches excluded)"
  if vssh "tar czf - -C ~/raccoon/experiments --exclude='$EXP/caches' $EXP" \
      | tar xzf - -C experiments/; then
    ok=1; break
  fi
  sleep 60
done

if [ "$ok" = 1 ] && grep -q "EXP009 COMPLETE" "experiments/$EXP/logs/pipeline.log"; then
  log "Fetched: $(find experiments/$EXP -type f | wc -l) files," \
      "$(du -sh experiments/$EXP | cut -f1) (caches excluded)"
  log "Stopping VM"
  gcloud compute instances stop "$VM" --zone="$ZONE" >/dev/null 2>&1 && log "VM stopped"
  log "FETCH_DONE"
else
  log "Fetch incomplete — data remains on the VM disk (and GCS if backup OK)"
  log "Stopping VM anyway to save cost"
  for i in 1 2 3; do
    gcloud compute instances stop "$VM" --zone="$ZONE" >/dev/null 2>&1 && break
    log "VM stop failed (attempt $i) — retrying in 60s"; sleep 60
  done
  log "FETCH_FAIL"
  exit 1
fi
