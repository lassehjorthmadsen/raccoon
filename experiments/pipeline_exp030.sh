#!/usr/bin/env bash
# exp030 -- does capacity still bind, under the recipe exp028 fixed?
#
# WHY THIS RUN EXISTS.
# exp029 ran exp028's best small-MLP configuration five times longer and found
# nothing: PR 1.368 -> 1.296, paired +0.072 [-0.013, +0.156], below the 0.09
# resolution, with rollout-tier fit flat (MSE 0.00219 -> 0.00215). The network is
# not undertrained.
#
# Nor is it short of labels, and that took no new measurement -- it was already
# in the repo. ep22 was distilled from THESE SAME 86 shards (pipeline_exp018.sh
# sets CACHE_DIR=data/distill/2ply) and reached PR 0.950. So 43M positions carry
# enough signal for 0.950; a network 500x larger extracted it. "Not enough
# distinct positions" would have to explain why a SMALLER network needs MORE data
# than a larger one to reach the same error, which is backwards -- small networks
# plateau early because of capacity, not sample size.
#
# That leaves two explanations for the 0.35 PR between the small MLP and ep22:
# CAPACITY (0.563 M multiply-accumulates cannot represent this function below
# ~1.3) or METHOD (regressing a fixed teacher over GNU Backgammon's self-play
# positions, rather than PureTD's on-policy TD with exact Bellman backups over
# all 21 rolls). Method also absorbs the surviving form of the data argument:
# with capacity this tight, WHICH positions the network spends it on matters more
# than how many there are.
#
# Capacity is much the cheaper of the two to test, and it has never been measured
# under the fixed recipe. exp028's capacity curve -- 5.40, 2.99, 2.31, 1.93 as
# the network grew -- was still descending at the largest arm AND ran entirely
# under the constant learning rate that the same experiment then showed leaves
# about 0.2 PR unclaimed. One wider arm under the fixed recipe answers the binary
# question; an RL pipeline would take weeks to answer the other one.
#
# HYPOTHESIS (one).
# Doubling the width of exp028's combined arm, everything else held fixed,
# materially improves PR -- i.e. capacity is still binding at 0.563 M MACs.
#
# THE ARM, and why only one. Width doubles; depth, inputs, head, schedule, data
# and epoch budget do not.
#
#     1x  [512,512,256,256]     0.562 M MACs   PR 1.368   already measured (exp028 combined20)
#     2x  [1024,1024,512,512]   2.041 M MACs   THIS RUN
#     4x  [2048,2048,1024,1024] 7.752 M MACs   ~112 h on this machine -- not run
#
# Training here is compute-bound, not loader-bound: exp029 measured 0.408 h/epoch
# at 0.563 M MACs, and the arithmetic for 43M positions at ~50 GFLOPS effective
# predicts ~0.42 h. So wall time scales with MACs, the 2x arm costs ~1.5 h/epoch
# (~30 h for 20 epochs), and the 4x arm costs ~112 h, which is why it is not in
# this run. If the 2x arm descends steeply, 4x is worth a GPU VM; if it plateaus
# at ~1.3, the question is already answered and 4x would add nothing.
#
# PRIMARY METRIC.
# PR on the BGSage money benchmark, FULL n = 14,693, no subsampling. The headline
# is the PAIRED difference against exp028's combined20 at epoch 20 -- the same
# recipe at the same 20-epoch budget, differing only in width -- on identical
# decisions, 95% interval bootstrapped over the 500 generating games.
#
# SECONDARY, and stated up front so it cannot look like a metric switch later:
# the same paired comparison against exp029's combined100 at epoch 100, i.e. the
# 1x network at its best under any budget. The primary is matched-budget; the
# secondary guards against "you only beat a 1x net that had not finished".
#
# SUPPORTING, demarcated: rollout-tier MSE / R^2 on ~149k candidate positions
# never trained on, and MEASURED evaluation throughput for the wider net
# (scripts/measure_eval_speed.py --checkpoint). The whole point of the small-MLP
# line is accuracy against speed, so a capacity result that ignores what the
# capacity costs to evaluate would be answering half the question.
#
# PRE-REGISTERED DECISION RULE. Baseline combined20 e20 = 1.368; ep22 = 0.950.
#
#   paired gain >= 0.27   capacity binds hard -> the 562k-parameter class is the
#                         limit, the 4x arm is worth a GPU VM, and PureTD's
#                         report of 0.94 at this capacity is the claim to
#                         scrutinise (their inputs, their measurement scale)
#   0.09 to 0.27          capacity binds partially -> report the slope, and what
#                         width the extrapolation says 0.95 would need
#   < 0.09, not resolved  capacity is NOT the limit at 0.563 M MACs under this
#                         recipe -> method or position distribution is, and RL
#                         becomes worth its weeks rather than an assumption
#
# SCOPE -- explicitly NOT in this experiment.
#   * No depth change, no input change, no head change, no schedule change.
#     One variable.
#   * No new or relabelled data. Same 86 shards.
#   * No head-to-head play, no search, no cube.
#   * The 4x arm. It is a decision this run's result makes, not part of it.
#
# OUTPUTS.
#   experiments/exp030-capacity/width2x/     checkpoints (gitignored) + logs
#   experiments/exp030-capacity/dumps/       per-decision errors (gitignored)
#   experiments/exp030-capacity/scratch/     per-epoch benchmark JSON (gitignored)
#   experiments/exp030-capacity/results/     exp030_summary.json, committed
#   Write-up: docs/speed.qmd, extending the #exp029 section's open question.

set -euo pipefail
cd "$(dirname "$0")/.."

EXP=experiments/exp030-capacity
ARM=width2x
EPOCHS=${EPOCHS:-20}
CACHE=data/distill/2ply
PY=.venv/bin/python3
MAX_ATTEMPTS=${MAX_ATTEMPTS:-8}

mkdir -p "$EXP/logs" "$EXP/results" "$EXP/dumps" "$EXP/scratch"
export OMP_WAIT_POLICY=PASSIVE

# ---- train -------------------------------------------------------------------
attempt=0
while [ ! -f "$EXP/$ARM/DONE" ]; do
    attempt=$((attempt + 1))
    if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
        echo "exp030: $MAX_ATTEMPTS attempts without DONE -- stopping" >&2
        exit 1
    fi
    echo "=== training $ARM (attempt $attempt) ==="
    $PY scripts/train_distill.py \
        --cache-dir "$CACHE" \
        --experiment-name "exp030-capacity/$ARM" \
        --value-head scalar --lr-schedule cosine \
        --trunk mlp --hidden 1024,1024,512,512 --hidden-layers 4 \
        --mlp-input debroadcast --features "" \
        --epochs "$EPOCHS" --eval-every-shards 1720 --eval-games 40 \
        --resume auto 2>&1 | tee -a "$EXP/logs/train_$ARM.log" || true
done
echo "TRAIN_DONE" >> "$EXP/logs/arm.progress"

# ---- score every epoch on the full benchmark ---------------------------------
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
cp "$EXP/scratch/${ARM}_e${EPOCHS}.json" "$EXP/results/$ARM.json" 2>/dev/null || true

# ---- what the extra capacity costs to evaluate -------------------------------
# Accuracy without its price is half the question on this page.
$PY scripts/measure_eval_speed.py \
    --checkpoint "$EXP/$ARM/checkpoints/ep${EPOCHS}.pt" \
    --output "$EXP/results" 2>&1 | tee "$EXP/logs/eval_speed.log"

# ---- consolidate -------------------------------------------------------------
$PY scripts/exp030_summary.py --exp-dir "$EXP" --epochs "$EPOCHS" \
    2>&1 | tee "$EXP/logs/summary.log"
