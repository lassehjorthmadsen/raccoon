#!/bin/bash
# Pull the shared datasets that have no other source onto a fresh machine:
# bglab (match archive, private — no download URL, GCS is the only copy) and
# bgsage (the one benchmark file raccoon uses, copied out of the bgsage repo).
#
# data/distill/ is deliberately NOT pulled here — it's 65GB+ across all runs
# and the training pipelines (pipeline_exp011b.sh, pipeline_exp014.sh) already
# pull just the run(s) they need from GCS on demand.
# data/wildbg/ is NOT here either — use `make download-wildbg` (public source).
#
# Usage: ./scripts/pull_data.sh

set -euo pipefail

BUCKET="gs://raccoon-training-lhm/data"

mkdir -p data/bglab data/bgsage

echo "Pulling data/bglab/ from GCS ..."
gcloud storage rsync "$BUCKET/bglab/" data/bglab/ --recursive

echo "Pulling data/bgsage/ from GCS ..."
gcloud storage rsync "$BUCKET/bgsage/" data/bgsage/ --recursive

echo
echo "Done."
du -sh data/bglab data/bgsage
