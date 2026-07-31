#!/bin/bash
# Push local changes to the shared datasets back to GCS — run this after
# regenerating a data/bglab/cache/ artifact (synthesize_gnubg_dataset.py,
# synthesize_policy_dataset.py) or refreshing the data/bglab match archive
# or the data/bgsage benchmark file, so other machines can pull the update.
#
# data/distill/ is NOT here — scripts/resume_expand.sh pushes each new run
# as it's generated. data/wildbg/ is NOT here — it's a straight re-download
# from a public source (make download-wildbg), never pushed.
#
# Usage: ./scripts/push_data.sh

set -euo pipefail

BUCKET="gs://raccoon-training-lhm/data"

echo "Pushing data/bglab/ to GCS ..."
gcloud storage rsync data/bglab/ "$BUCKET/bglab/" --recursive

echo "Pushing data/bgsage/ to GCS ..."
gcloud storage rsync data/bgsage/ "$BUCKET/bgsage/" --recursive

echo
echo "Done."
