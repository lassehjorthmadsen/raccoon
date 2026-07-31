#!/usr/bin/env bash
# exp017 offline scoring — the PRIMARY metric. Scores every epoch checkpoint of the
# 40M 2-ply run on the BGSage money benchmark (checker PR, n=14,693; plus rollout /
# 3T / 3P equity R²/MSE), alongside the exp014 8M anchor, to build the per-epoch curve
# for post-hoc selection. Zero play cost; run on the iMac (CPU).
#
# The teacher (GNUBG-2-ply, PR 0.56 / rollout-MSE 0.0019) is NOT re-scored here — it's
# a fixed reference from exp015 and 2-ply scoring is expensive. Uncomment the --gnubg-ply
# line only if you want to reproduce it.
#
# Usage: bash experiments/score_exp017.sh            (scores whatever ep*.pt exist)
#        PULL=1 bash experiments/score_exp017.sh     (rsync checkpoints from GCS first)
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_WAIT_POLICY=PASSIVE

EXPNAME="${EXPNAME:-exp017-distill}"
GCS_BUCKET="${GCS_BUCKET:-gs://raccoon-training-lhm}"
CKPT_DIR="experiments/$EXPNAME/checkpoints"
OUT_DIR="experiments/exp017-benchmark/results"
ANCHOR="experiments/exp014-distill/scalar_2ply/checkpoints/ep3.pt"

if [ "${PULL:-0}" = "1" ]; then
  echo "pulling checkpoints from GCS ..."
  mkdir -p "$CKPT_DIR"
  gcloud storage rsync "$GCS_BUCKET/experiments/$EXPNAME/checkpoints/" "$CKPT_DIR/" --recursive
fi

mapfile -t EPS < <(ls "$CKPT_DIR"/ep*.pt 2>/dev/null | sort -V)
[ "${#EPS[@]}" -gt 0 ] || { echo "no ep*.pt in $CKPT_DIR (run PULL=1 first?)" >&2; exit 1; }

ARGS=()
for cp in "${EPS[@]}"; do
  ARGS+=(--checkpoint "$cp" --engine-label "exp017-$(basename "$cp" .pt)")
done
# anchor: the exp014 8M 2-ply net (already known PR 2.16 / rollout-MSE 0.0044)
[ -f "$ANCHOR" ] && ARGS+=(--checkpoint "$ANCHOR" --engine-label "exp014-8M-ep3")

mkdir -p "$OUT_DIR"
echo "scoring ${#EPS[@]} epoch checkpoints (+anchor) on BGSage -> $OUT_DIR"
.venv/bin/python3 scripts/eval_benchmark_pr.py \
  "${ARGS[@]}" \
  --device cpu \
  --output "$OUT_DIR"
  # --gnubg-ply 2   # uncomment to re-score the teacher (slow)

echo "done — see $OUT_DIR (per-decision JSON per engine label)"
