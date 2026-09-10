#!/usr/bin/env bash
# exp029 -- is the remaining gap training scale, or training method?
#
# WHY THIS RUN EXISTS.
# exp028 (docs/speed.qmd#exp028) distilled the 43M GNU Backgammon 2-ply labels
# into an MLP [512,512,256,256] over 201 de-broadcast inputs -- 0.563 M
# multiply-accumulates against ep22's 285 M -- and reached PR 1.368 on the full
# BGSage benchmark (n = 14,693), against ep22's 0.950.
#
# Capacity and inference cost are MATCHED with PureTD (562k parameters against
# 565k, 0.561 M MACs against 0.563 M). Training compute is NOT: 200M self-play
# games over ~65 h on a 64-core workstation with a GPU, against roughly 38
# core-hours per exp028 arm with no GPU. Two orders of magnitude. So the
# remaining ~0.42 PR is either the METHOD (on-policy TD with exact Bellman
# backups over all 21 rolls) or simply SCALE, and exp028's write-up refused to
# pick one. This run separates them for one arm's cost, before anyone commits to
# an RL pipeline that would take weeks.
#
# HYPOTHESIS (one).
# Running exp028's combined configuration for 100 epochs instead of 20 -- same
# architecture, inputs, data, head and schedule family, cosine annealed over the
# longer horizon -- materially improves PR over the 20-epoch arm.
#
# THE AXIS THIS CAN AND CANNOT VARY. data/distill/2ply holds 86 shards / 43M
# labels and there is no more, so the only scale axis available here is passes
# over fixed data, not fresh data. That is what the supporting metric is for: it
# separates "not trained enough" from "43M labels is the ceiling".
#
# PRIMARY METRIC, fixed before the run.
# PR on the BGSage money benchmark, FULL n = 14,693, no subsampling, scored by
# scripts/eval_benchmark_pr.py. The headline number is the PAIRED difference
# between combined100 at epoch 100 and exp028's combined20 at epoch 20, on
# identical decisions, 95% interval bootstrapped over the 500 generating games
# (scripts/exp028_summary.py::clustered_ci, imported rather than reimplemented).
# Paired resolution is about 0.09 PR; differences below it are reported "not
# resolved", which is a statement about the instrument, not a null result.
#
# SUPPORTING, demarcated: rollout-tier MSE / R^2 on ~149k candidate positions
# that were never trained on, from the same scoring pass at no extra cost. It is
# what the distillation loss actually optimises. A PR curve that flattens while
# MSE also flattens points at the label set; a PR curve that flattens while MSE
# still falls points at move ranking rather than fit.
#
# PRE-REGISTERED DECISION RULE. Baseline combined20 e20 = 1.368; ep22 = 0.950.
#
#   paired gain >= 0.27   scale is the dominant explanation -> keep scaling, and
#                         get more labels, before any RL work
#   0.09 to 0.27          partial -> report the extrapolation of epochs needed
#                         for the rest, decide then
#   < 0.09, not resolved  training scale on fixed labels is exhausted -> the
#                         remaining difference is method or the 43M-label
#                         ceiling, and PureTD-style TD becomes the next
#                         CANDIDATE, not the next assumption
#
# Also recorded: where the curve's own long-interval steps (e20->e50, e50->e100,
# e90->e100) stop resolving. exp028 showed adjacent-epoch comparisons read "not
# resolved" three times in a row on a curve that was genuinely still descending,
# so only long intervals can see this.
#
# CONFIGURATION. Identical to combined20 except --epochs, reconstructed from
# experiments/exp028-mlp/logs/train_combined20.log, whose first line reads
#   [arch] mlp hidden [512, 512, 256, 256] in=201, 0.911M params (0.563M deployable)
# --cache-dir is the ply-level directory, spanning run1+run2+run3 = 86 shards =
# all 43M labels, which is what that log's "86 shards x 20 epochs" says
# combined20 read. Cosine T_max is len(shards) * epochs, so the schedule anneals
# over 8,600 shard steps -- which is why this is a FRESH run and not a resume of
# combined20. --eval-every-shards is set past the end of the run so the noisy
# inline 40-game arena fires once; selection happens offline on the epoch
# checkpoints, as exp011b established.
#
# COST, measured from combined20: 7.68 h / 20 epochs = 0.384 h/epoch, so ~38 h
# of training plus ~1.8 h of scoring (100 checkpoints x ~65 s at full n) on this
# iMac. Scoring runs AFTER training, never interleaved: a benchmark pass would
# otherwise compete with training for the four cores and corrupt the wall-clock
# figure the cost line above depends on.
#
# SCOPE -- explicitly NOT in this experiment.
#   * No architecture change. Same widths, same inputs, same head.
#   * No new or relabelled training data.
#   * No head-to-head play, no search, no cube.
#   * No mid-run schedule change. A cosine run whose horizon moves is not a
#     measurement of anything.
#
# OUTPUTS.
#   experiments/exp029-longrun/combined100/  checkpoints (gitignored) + logs
#   experiments/exp029-longrun/dumps/        per-decision errors (gitignored)
#   experiments/exp029-longrun/scratch/      per-epoch benchmark JSON (gitignored)
#   experiments/exp029-longrun/results/      exp029_summary.json, committed
#   Write-up: docs/speed.qmd, a new #exp029 section.

set -euo pipefail
cd "$(dirname "$0")/.."

EXP=experiments/exp029-longrun
ARM=combined100
EPOCHS=${EPOCHS:-100}
CACHE=data/distill/2ply
PY=.venv/bin/python3
MAX_ATTEMPTS=${MAX_ATTEMPTS:-8}

mkdir -p "$EXP/logs" "$EXP/results" "$EXP/dumps" "$EXP/scratch"
export OMP_WAIT_POLICY=PASSIVE

# ---- train -------------------------------------------------------------------
# Relaunch loop, not a bare invocation: a 38 h run on a desktop gets interrupted.
# --resume auto is a no-op on a fresh start and picks up at shard granularity
# afterwards. The DONE sentinel is written only on clean completion, so it is the
# loop's exit condition and the true cost cap.
attempt=0
while [ ! -f "$EXP/$ARM/DONE" ]; do
    attempt=$((attempt + 1))
    if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
        echo "exp029: $MAX_ATTEMPTS attempts without DONE -- stopping" >&2
        exit 1
    fi
    echo "=== training $ARM (attempt $attempt) ==="
    $PY scripts/train_distill.py \
        --cache-dir "$CACHE" \
        --experiment-name "exp029-longrun/$ARM" \
        --value-head scalar --lr-schedule cosine \
        --trunk mlp --hidden 512,512,256,256 --hidden-layers 4 \
        --mlp-input debroadcast --features "" \
        --epochs "$EPOCHS" --eval-every-shards 8600 --eval-games 40 \
        --resume auto 2>&1 | tee -a "$EXP/logs/train_$ARM.log" || true
done
echo "TRAIN_DONE" >> "$EXP/logs/arm.progress"

# ---- score every epoch on the full benchmark ---------------------------------
# Every epoch, full n, no subsampling. epochs.progress makes an interrupted
# scoring pass resumable without rescoring what is already dumped.
for ep in $(seq 1 "$EPOCHS"); do
    ckpt="$EXP/$ARM/checkpoints/ep$ep.pt"
    [ -f "$ckpt" ] || { echo "missing $ckpt -- skipping"; continue; }
    grep -qx "ep$ep scored" "$EXP/logs/epochs.progress" 2>/dev/null && continue
    echo "=== scoring ep$ep ==="
    $PY scripts/eval_benchmark_pr.py \
        --checkpoint "$ckpt" \
        --engine-label "${ARM}_e$ep" \
        --error-dump "$EXP/dumps/${ARM}_e$ep.npz" \
        --output "$EXP/scratch" \
        2>&1 | tee -a "$EXP/logs/curve_$ARM.log"
    echo "ep$ep scored" >> "$EXP/logs/epochs.progress"
done
echo "EPOCH_CURVE_DONE" >> "$EXP/logs/epochs.progress"

# The final epoch is the deliverable, so its result JSON is a conclusion and goes
# in results/ alongside the summary; the other 99 stay in the gitignored scratch.
cp "$EXP/scratch/${ARM}_e${EPOCHS}.json" "$EXP/results/$ARM.json" 2>/dev/null || true

# ---- consolidate -------------------------------------------------------------
$PY scripts/exp029_summary.py --exp-dir "$EXP" --epochs "$EPOCHS" \
    2>&1 | tee "$EXP/logs/summary.log"
