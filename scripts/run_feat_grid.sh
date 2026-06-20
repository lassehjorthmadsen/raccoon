#!/usr/bin/env bash
# Stage 6 ablation grid: do handcrafted encoder features earn their place,
# once correctly scaled? Two scaling fixes compared head-to-head:
#   Fix-N (norm): rescale handcrafted channels in the encoder  (--feature-norm)
#   Fix-B (inbn): BatchNorm over raw inputs inside the network (--input-bn)
# across the feature grid {base, pip, blots, anchors, contact, all}, 2 seeds.
#
# The raw (broken) baseline already lives in pretrain-feat-{base,pip}-s{0,1};
# Fix-N's base == raw base (normalize is a no-op on the base channels), so it
# is reused rather than rerun. Idempotent: a run whose pretrained.pt exists is
# skipped, so the script can be re-launched after an interrupt.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python3
COMMON="--max-positions 50000 --epochs 15 --channels 128 --num-blocks 6"

run() {
  local name=$1; shift
  if [ -f "experiments/$name/checkpoints/pretrained.pt" ]; then
    echo "[skip $(date +%H:%M:%S)] $name (already done)"
    return
  fi
  echo "=== [$(date +%H:%M:%S)] $name ==="
  OMP_WAIT_POLICY=PASSIVE $PY scripts/pretrain.py \
    --experiment-name "$name" $COMMON "$@"
}

for S in 0 1; do
  # Fix-N (encoder normalize): base == raw base, so skip it here.
  for G in pip blots anchors contact all; do
    run "pretrain-feat-norm-$G-s$S" --features "$G" --feature-norm --seed "$S"
  done
  # Fix-B (input BatchNorm): include base (input-BN on the clean baseline).
  run "pretrain-feat-inbn-base-s$S" --features --input-bn --seed "$S"
  for G in pip blots anchors contact all; do
    run "pretrain-feat-inbn-$G-s$S" --features "$G" --input-bn --seed "$S"
  done
done

echo "=== ALL DONE $(date) ==="
