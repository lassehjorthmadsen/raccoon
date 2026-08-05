#!/usr/bin/env bash
# exp018 offline scoring — the PRIMARY metric. Scores every epoch checkpoint of the
# 40M 2-ply outcomes6 run on the BGSage money benchmark (checker PR, n=14,693; plus
# rollout / 3T / 3P equity R²/MSE) to build the per-epoch curve for post-hoc
# selection. Zero play cost; run on the iMac (CPU). ~2h for 24 epochs.
#
# exp017's per-epoch scalar results are NOT re-scored here — they are already in
# experiments/exp017-benchmark/results/ from the same script and the same benchmark,
# so exp018 ep_k and exp017 ep_k are directly comparable as they stand. Likewise the
# exp014 8M anchor and the GNUBG references live in exp015's/exp017's results.
#
# Usage:
#   bash experiments/score_exp018.sh          score whatever ep*.pt exist locally
#   PULL=1 bash experiments/score_exp018.sh   rsync checkpoints from GCS first
#   PAIR=12 bash experiments/score_exp018.sh  paired head A/B at one epoch (below)
#
# PAIR=<n> mode. The per-epoch JSONs give point estimates but no CI on the head
# difference. This mode re-scores exp018 ep<n> and exp017 ep<n> with --dump-dir and
# runs the decision- and game-clustered bootstrap of exp016_paired_mse.py over the
# pair. Clustering matters: the ~149k rollout candidates are 2-773 per decision and
# are not independent draws. Remember the pre-registered asymmetry when reading the
# MSE delta — it is the *scalar* arm's own training loss, so it favours exp017 at
# equal strength; outcomes6 winning it is strong, scalar winning it is ambiguous.
# PR in the same output is the objective-neutral read.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_WAIT_POLICY=PASSIVE   # CPU: default busy-spin collapses under contention

EXPNAME="${EXPNAME:-exp018-distill}"
GCS_BUCKET="${GCS_BUCKET:-gs://raccoon-training-lhm}"
CKPT_DIR="experiments/$EXPNAME/checkpoints"
OUT_DIR="experiments/exp018-benchmark/results"
DUMP_DIR="experiments/exp018-benchmark/dumps"
SCALAR_CKPT_DIR="experiments/exp017-distill/checkpoints"

if [ "${PULL:-0}" = "1" ]; then
  echo "pulling checkpoints from GCS ..."
  mkdir -p "$CKPT_DIR"
  gcloud storage rsync "$GCS_BUCKET/experiments/$EXPNAME/checkpoints/" "$CKPT_DIR/" --recursive
fi

# ---- PAIR=<n>: paired head A/B at a single epoch -----------------------------
if [ -n "${PAIR:-}" ]; then
  A="$CKPT_DIR/ep${PAIR}.pt"                 # exp018, outcomes6
  B="$SCALAR_CKPT_DIR/ep${PAIR}.pt"          # exp017, scalar
  for f in "$A" "$B"; do
    [ -f "$f" ] || { echo "missing $f (PULL=1 first? exp017 only reached ep14)" >&2; exit 1; }
  done
  mkdir -p "$DUMP_DIR"
  # Labels here are lowercase/underscored so sanitize_label() in
  # eval_benchmark_pr.py is the identity and the .npz names below are predictable
  # (it lowercases and rewrites " ()/-.," to "_", so a hyphen would not survive).
  LA="exp018_outcomes6_ep$PAIR"
  LB="exp017_scalar_ep$PAIR"
  echo "dumping per-candidate predictions for ep$PAIR (both arms) -> $DUMP_DIR"
  .venv/bin/python3 scripts/eval_benchmark_pr.py \
    --checkpoint "$A" --engine-label "$LA" \
    --checkpoint "$B" --engine-label "$LB" \
    --device cpu \
    --dump-dir "$DUMP_DIR"

  echo "paired cluster bootstrap (decision + game level) ..."
  .venv/bin/python3 scripts/exp016_paired_mse.py \
    --dump-a "$DUMP_DIR/$LA.npz" --label-a "outcomes6 (exp018)" \
    --dump-b "$DUMP_DIR/$LB.npz" --label-b "scalar (exp017)" \
    --panel "value head at 40M, ep$PAIR" \
    --output "$OUT_DIR/paired_head_ep$PAIR.json"
  echo "done — see $OUT_DIR/paired_head_ep$PAIR.json"
  exit 0
fi

# ---- default: per-epoch curve ------------------------------------------------
mapfile -t EPS < <(ls "$CKPT_DIR"/ep*.pt 2>/dev/null | sort -V)
[ "${#EPS[@]}" -gt 0 ] || { echo "no ep*.pt in $CKPT_DIR (run PULL=1 first?)" >&2; exit 1; }

ARGS=()
for cp in "${EPS[@]}"; do
  ARGS+=(--checkpoint "$cp" --engine-label "exp018-$(basename "$cp" .pt)")
done

mkdir -p "$OUT_DIR"
echo "scoring ${#EPS[@]} epoch checkpoints on BGSage -> $OUT_DIR"
.venv/bin/python3 scripts/eval_benchmark_pr.py \
  "${ARGS[@]}" \
  --device cpu \
  --output "$OUT_DIR"

echo "done — see $OUT_DIR (per-decision JSON per engine label)"
echo "next: PAIR=<epoch> bash experiments/score_exp018.sh   # CI on the head difference"
