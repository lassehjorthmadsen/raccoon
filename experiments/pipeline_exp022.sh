#!/usr/bin/env bash
# exp022 — does it pay to make search width depend on how much contact a position has?
#
# WHY THIS RUN EXISTS.
# exp021 measured search at a *fixed* width and found the gain wildly uneven
# across position types (paired, n=2,000 subsample, seed 21):
#     anchoring  1.482 -> 0.981   gain 0.501
#     attacking  0.869 -> 0.401   gain 0.469
#     priming    0.873 -> 0.667   gain 0.206
#     purerace   0.101 -> 0.067   gain 0.034
#     racing     1.429 -> 1.390   gain 0.039   <- 20% of decisions, ~nothing gained
# So we pay full search cost on racing for a 3% improvement. If those positions can
# be recognised at play time, that budget is better spent elsewhere.
#
# THE SIGNAL, and why it is the only one available.
# The benchmark's game_plan labels CANNOT be used: they are annotations in the data
# file, not something an engine has over the board. gnubg_nn.classify() is available
# at play time but useless here -- it lumps racing/attacking/priming/anchoring into
# one "contact" bucket (89-97%), separating only purerace, which is already both
# cheap and near-perfect (PR 0.067). Our own encoder's contact feature does separate
# them (racing 0.27 mean vs 0.46-0.70 for the tactical plans), it is pip-scale and
# continuous -- exactly 1.0 at the opening, 0.0 in a pure race -- and it is now
# callable directly via raccoon/env/encoder.py::contact_pips.
#
# WHICH KNOB, measured rather than assumed. A candidate must be inside the top-k AND
# within the equity window. On the committed exp018/ep22 dump the window is what
# binds: at threshold 0.16 it admits a median of 5 candidates and <=8 in 72% of
# decisions, so k=8 is inert in most positions. exp022 therefore scales the WINDOW
# with contact and leaves k as a cap:
#     window(contact) = w0 + (w1 - w0) * clip(contact, 0, 1)
# It degenerates gracefully: as the window closes only the static-best move clears
# it, so "spend nothing on races" is the same dial, not a special case.
#
# PRIMARY METRIC. PR on the SAME seed-21 n=2,000 subsample every exp021 row used,
# read against evals_per_decision -- already recorded in each result JSON, so
# "at equal compute" is a column, not machinery. A win is either lower PR at
# comparable cost, or comparable PR at materially lower cost.
#
# THE ROWS, and why these endpoints. (w0, w1) are NOT fitted against PR. Survivor
# counts under any window rule are computable offline from the committed dump
# (scripts/exp022_calibrate_window.py), and survivors drive cost, so the endpoints
# are chosen to match the fixed arm's cost BY CONSTRUCTION:
#     fixed threshold 0.16          -> 8.42 mean survivors   (the exp021 baseline)
#     row A  w0=0.02 w1=0.36        -> 8.63 mean survivors   (1.02x, iso-cost)
#     row B  w0=0.00 w1=0.24        -> 5.31 mean survivors   (0.63x, cheaper)
# Row A is the STEEPEST of the four iso-cost pairs the calibration found (an 18x
# window ratio between race and full contact), chosen deliberately: it reallocates
# the most, so it is the strongest test of the hypothesis rather than a hedge.
#
# HARNESS-CORRECTNESS CONTROL: with --window-lo unset the search must reproduce the
# fixed-window values bit for bit (tests/test_expectimax.py::
# test_fixed_window_is_unchanged_bit_for_bit), and the encoder refactor that exposed
# contact_pips was verified bit-exact over 2,000 positions.
#
# SCOPE NOTE, ON THE RECORD. One hypothesis: width as a function of contact. NOT in
# scope -- head-to-head play vs GNUBG 2-ply, the policy-head filter, adapting depth
# rather than width, and learned allocation policies. Local CPU only, no GCP spend.
#
# COST, from exp021's measured 4.25 s/dec at the fixed window: row A ~5 h, row B
# ~3 h on the n=2,000 subsample.
#
# SCHEDULING. Runs nothing until exp021 is off the shared iMac -- CPU contention
# would corrupt the sec_per_decision figures both experiments rely on.
#
# OUTPUTS. experiments/exp022-contact/{logs,results}/, committed with the write-up
# (docs/search.qmd, section exp022), which computes its table from results/.

set -euo pipefail
cd "$(dirname "$0")/.."

EXP="${EXP:-experiments/exp022-contact}"
CKPT="${CKPT:-experiments/exp018-distill/checkpoints/ep22.pt}"
BENCH="${BENCH:-data/bgsage/money_benchmark/benchmark.json.gz}"
SUB="${SUB:-2000}"
SUB_SEED="${SUB_SEED:-21}"
PY="${PY:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)}"

export OMP_WAIT_POLICY=PASSIVE

mkdir -p "$EXP/logs" "$EXP/results"
LOG="$EXP/logs/exp022.log"
run() { echo "=== $* ===" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }

echo "exp022 started $(date -Is) on $(hostname)" | tee -a "$LOG"

row() {  # w0 w1 label
    run "$PY" -u scripts/eval_benchmark_pr.py \
        --benchmark "$BENCH" --checkpoint "$CKPT" \
        --engine-label "ep22_$3" \
        --search-depth 1 --search-k 16 --search-gate 0.08 \
        --window-lo "$1" --window-hi "$2" \
        --subsample "$SUB" --subsample-seed "$SUB_SEED" \
        --output "$EXP/results"
}

row 0.02 0.36 "smooth_isocost"
row 0.00 0.24 "smooth_cheap"

echo "exp022 finished $(date -Is)" | tee -a "$LOG"
echo "Compare against exp021's fixed row (PR 0.734, 1650 evals/dec) on PR *and* cost." | tee -a "$LOG"
