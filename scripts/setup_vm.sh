#!/bin/bash
# Runs as root on every VM boot (configured as the raccoon-gpu instance's
# startup-script metadata). Idempotent — safe to re-run.
#
# Purpose: ensure the user's gcloud has an active account so unattended
# `gcloud storage` calls (e.g. from the GCS sync tmux loop) don't silently
# fail with "no active account selected". On a fresh boot disk the default
# compute service account is credentialed but not active until someone runs
# `gcloud config set account` manually.
#
# To install:
#   gcloud compute instances add-metadata raccoon-gpu \
#     --zone=europe-west1-b \
#     --metadata-from-file startup-script=scripts/setup_vm.sh

set -euo pipefail

# Pull the default service account email from the GCE metadata server so
# the script is portable across projects.
SA="$(curl -fs -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email \
  || true)"

if [ -z "$SA" ]; then
  echo "setup_vm.sh: no default service account on metadata server, skipping" >&2
  exit 0
fi

# `runuser -l` ensures the right HOME so gcloud writes to ~lasse/.config.
runuser -l lasse -c "gcloud config set account '$SA'" >/dev/null

echo "setup_vm.sh: active gcloud account set to $SA for user lasse"

# Load NVIDIA kernel modules. After a kernel upgrade the modules may not be
# present for the new kernel version; install them if missing, then load.
KVER="$(uname -r)"
if ! modinfo nvidia >/dev/null 2>&1; then
  echo "setup_vm.sh: NVIDIA module missing for kernel $KVER — installing"
  apt-get install -y "linux-modules-nvidia-580-server-open-${KVER}" >/dev/null 2>&1 || \
    echo "setup_vm.sh: WARNING — could not install NVIDIA modules for $KVER" >&2
fi
modprobe nvidia     2>/dev/null || echo "setup_vm.sh: WARNING — modprobe nvidia failed" >&2
modprobe nvidia-uvm 2>/dev/null || true
echo "setup_vm.sh: NVIDIA modules loaded ($(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown'))"
