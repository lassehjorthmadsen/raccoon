#!/usr/bin/env bash
# exp019 — does exp018/ep22 beat GNUBG-0-ply in head-to-head cubeless money play?
#
# WHY THIS RUN EXISTS. exp018/ep22 is the project's best net and the BGSage static
# benchmark already puts it PAST GNUBG-0-ply (PR 0.950 at n=14,693; rollout MSE 0.00190
# at n~149k). But no head-to-head ppg has ever been measured for it — the exp018 write-up
# says so outright ("a raw-ppg confirmation at matched search is still outstanding").
#
# The obstacle is resolution, not compute discipline. Raw cubeless-money ppg has a
# per-game SD ~1.8, so the project's standard n=6000 buys only +/-0.046 while the effect
# in play is plausibly ~+0.05 — a ~2-sigma test at best (~55% power). Getting the raw CI
# to +/-0.015 would need n~55,000 games, ~30 h x 3 workers.
#
# So this experiment first builds the tool the project pre-registered as future work back
# in commit e1940a5 (see scripts/eval_errorrate.py and raccoon/train/td_selfplay.py):
# a proper variance-reduced rollout with a SELF-CONSISTENT control variate, the technique
# XG, GNUBG and BGSage all use. At every pre-roll state, before the dice are sampled,
#
#     h(s,d) = GNUBG 0-ply equity after best play for roll d, signed to the net's POV
#     m(s)   = sum_d p(d) h(s,d)
#     luck   = h(s, d_actual) - m(s)          ->   vr = raw_result - sum_t luck
#
# E[luck | history] = 0 by construction, so the correction is a mean-zero martingale and
# cannot move the expectation HOWEVER BAD the evaluator is. The rule that makes it work,
# and the one the dropped "error-rate ppg" proxy broke: h must not depend on the move
# actually played. We always use GNUBG's 0-ply best-play value for whoever is on roll,
# whatever either side actually does. See raccoon/eval/luck.py.
#
# PRIMARY METRIC (fixed up front): variance-reduced ppg vs GNUBG-0-ply, cubeless money,
# seats alternated, n = 6000 games, 95% CI from the empirical SD of the per-game VR
# values. ONE arm — ep22 only. ep22 was selected on BGSage PR, an independent metric on
# independent data, so this is a clean confirmation with no winner's curse. No epoch
# sweep on ppg, and no re-selection on this metric.
#
# SUPPORTING (demarcated, never the headline): raw ppg from the same games (must agree —
# it is the project's existing unbiased ruler); mean luck and its t-stat; the SD ratio and
# its square as an equivalent-games multiplier; corr(raw, luck); and beta_hat, the
# regression-optimal control-variate coefficient, as a calibration read only (the
# estimator stays at beta = 1, which is exactly unbiased; a beta fitted on the same sample
# is not).
#
# POWER. Raw at n=6000 gives +/-0.046: a +0.05 effect is 2.1 sigma, ~55% power. The
# write-up reports the ACHIEVED SD ratio and recomputes power from it rather than assuming
# one up front. A 3x SD reduction would give +/-0.015 (>6 sigma).
#
# SCOPE NOTE, ON THE RECORD. An external unbiasedness control — re-measuring
# exp011b-distill/outcomes6/ep3, the one checkpoint with a published raw answer
# (-0.053 +/- 0.046, experiments/exp012b-benchmark/logs/exp012b.log) — was considered and
# DELIBERATELY DROPPED to save an overnight run. It would have been the strongest evidence
# available: a VR estimate landing inside a known interval with a ~3x tighter CI. What
# remains is solid but all internal (the exact zero-mean unit test, mean luck with its
# t-stat, the two demo arms agreeing). The write-up says so rather than overclaiming, and
# this control is the first follow-up if the headline result surprises.
#
# METHODOLOGY DEMO (feeds the figure in docs/variance_reduction.qmd): 20 independent
# matches of 50 games with VR off, then 20 more with VR on, different dice in each set.
# Two genuinely independent samples of one process — legitimate because the control
# variate touches no RNG, so VR on/off from the same seed plays identical games
# (asserted by tests/test_luck.py::test_vr_identity_and_no_perturbation_of_play).
# The demo carries the PICTURE; the n=6000 run carries the NUMBER (a 20-point SD has ~16%
# relative SE, so the ratio from the demo alone would only be good to ~+/-25%).
#
# COST. ~2 s/game/worker measured; 6000 games on 3 iMac workers ~3.5 h, demo ~25 min.
# Local CPU only — no GPU, no GCP spend. Variance reduction itself adds ~5%: the 21-roll
# sweep costs ~2.6 ms per pre-roll state via gnubg's native move generator, against
# ~260 ms for the equivalent loop over candidate_equities.
#
# OUTPUTS. experiments/exp019-vr/{logs,results}/ — both committed alongside the write-up
# in the same commit, since docs/variance_reduction.qmd computes its tables and figures
# from results/*.json at render time.
set -euo pipefail
cd "$(dirname "$0")/.."

EXP_DIR="${EXP_DIR:-experiments/exp019-vr}"
CKPT="${CKPT:-experiments/exp018-distill/checkpoints/ep22.pt}"
WORKERS="${WORKERS:-3}"
GAMES="${GAMES:-6000}"
BLOCKS="${BLOCKS:-20}"
BLOCK_SIZE="${BLOCK_SIZE:-50}"
PLY="${PLY:-0}"

# Prefer the project venv (system python3 has neither torch nor gnubg-nn).
PY="${PY:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)}"

# Park idle OpenMP threads rather than busy-spinning (~12x collapse under contention).
export OMP_WAIT_POLICY=PASSIVE

mkdir -p "$EXP_DIR/logs"
LOG="$EXP_DIR/logs/exp019.log"

run() { echo "=== $* ===" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }

echo "exp019 started $(date -Is)" | tee -a "$LOG"

# 1. Demo, VR off — 20 independent 50-game matches. Seeds 2000..2019.
run "$PY" scripts/eval_gnubg_vr.py --checkpoint "$CKPT" \
    --blocks "$BLOCKS" --block-size "$BLOCK_SIZE" --workers "$WORKERS" \
    --ply "$PLY" --no-vr --seed-base 2000 --exp-dir "$EXP_DIR" --tag demo_raw

# 2. Demo, VR on — 20 more, disjoint seeds 3000..3019 so the dice differ.
run "$PY" scripts/eval_gnubg_vr.py --checkpoint "$CKPT" \
    --blocks "$BLOCKS" --block-size "$BLOCK_SIZE" --workers "$WORKERS" \
    --ply "$PLY" --seed-base 3000 --exp-dir "$EXP_DIR" --tag demo_vr

# 3. Headline — n=6000, the pre-registered primary metric.
run "$PY" scripts/eval_gnubg_vr.py --checkpoint "$CKPT" \
    --games "$GAMES" --workers "$WORKERS" \
    --ply "$PLY" --seed-base 1000 --exp-dir "$EXP_DIR" --tag ep22_vr

echo "exp019 finished $(date -Is)" | tee -a "$LOG"
