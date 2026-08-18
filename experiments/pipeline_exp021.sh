#!/usr/bin/env bash
# exp021 — does multi-ply search close Raccoon's remaining gap to GNUBG?
#
# WHY THIS RUN EXISTS.
# exp017 took distillation to the point where more 2-ply data stopped helping
# (PR 2.16 -> 0.93 at 40M positions, then a plateau) and concluded the residual
# gap to the teacher is *search, not data*. exp019/exp020 then showed the net
# already beats GNUBG-0-ply over the board (+0.0478 +/- 0.0034 VR ppg, n=12,000)
# while playing pure 0-ply itself. What has never been tried in this project is
# giving the network lookahead. docs/search.qmd is the design study; this is its
# experiment.
#
# THE MECHANISM, confirmed in code rather than hypothesised.
#   raccoon/search/expectimax.py    full-width expectation over the 21 rolls,
#                                   greedy opponent reply, filter chain k -> k2,
#                                   level-batched dedup'd evaluation.
#   gnubg_nn.moves()                whole-turn board-native move generation. The
#                                   benchmark stores raw boards and OpenSpiel
#                                   cannot build a state from one, so the search
#                                   could not use state.legal_actions().
#   tests/test_expectimax.py        gnubg's enumeration == OpenSpiel's, position
#                                   for position, over 180 random states.
#
# PRIMARY METRIC, fixed up front.
# PR on the BGSage money benchmark, **full n=14,693**, both engines at 1-ply
# (GNUBG numbering). Reference points on the identical benchmark:
#     ep22 0-ply      0.950     (experiments/exp018-benchmark/results/)
#     GNUBG 0-ply     2.145
#     GNUBG 2-ply     0.56
# GNUBG 1-ply is the matched-depth opponent and is UNMEASURED — stage 2 measures
# it before any Raccoon search config is scored, so the bar is set independently
# of our own result.
# SE(PR) from the exp018/ep22 per-decision error distribution: +/-0.032 at
# n=14,693, +/-0.087 at n=2,000. The subsample therefore resolves config gaps of
# ~0.25 PR and no finer; close calls go to full n by construction.
#
# SELECTION RULE, fixed up front: configs are swept on ONE random 2,000-decision
# subsample (--subsample 2000 --subsample-seed 21, identical for every config and
# for GNUBG). The winner is then RE-SCORED at full n and that number is the
# result. A subsample winner is never reported as the headline (winner's curse).
#
# THE GATE, and why it is not a free parameter. A decision whose static top-2 gap
# is large cannot change under search, so the search is skipped there. The cost of
# this is bounded *in advance* from the committed exp018/ep22 dump — it is not an
# empirical hope:
#     gate 0.05 -> 63.5% of decisions searched, at most 0.027 PR sacrificed
#     gate 0.08 -> 73.7% searched,              at most 0.002 PR sacrificed
# Stage 3 sweeps 0.05 vs 0.08 anyway, to confirm the bound behaves as derived.
#
# HARNESS-CORRECTNESS CONTROL: --search-depth 0 through the new code path must
# reproduce PR 0.9498 EXACTLY, the committed exp018/ep22 number. Anything else
# means the board conversion, encoding or sign convention moved, and no deeper
# result can be trusted. Stage 1 is that check and it gates everything after it.
#
# SCOPE NOTE, ON THE RECORD.
#   * Head-to-head play vs GNUBG-2-ply is NOT part of exp021 (it needs ~14e9
#     evaluations at n=6,000). It is exp022.
#   * Local CPU only, no GCP spend. That is what caps the primary metric at 1-ply:
#     2-ply full-n is ~6 days of unattended compute, run once at the end for the
#     selected config if stage 3 justifies it.
#   * The policy-head move filter designed in docs/search.qmd is NOT implemented
#     here. It needs an action->board mapping the board-native path does not have
#     (benchmark candidates are boards with no OpenSpiel action index). Deferred
#     to exp022, where play runs on OpenSpiel states and the mapping is free.
#     The value filter is what GNUBG and BGSage actually ship.
#   * The full-width opponent arm is likewise deferred: ~13x the cost of greedy at
#     depth 2, which local compute cannot absorb.
#
# COST, measured (stage 0), on lasse-iMac14-2, ep22 at ~400 boards/s:
#   see experiments/exp021-search/logs/stage0_spike.log for the table this run
#   was budgeted from.
#
# OUTPUTS. experiments/exp021-search/{logs,results}/ — both committed with the
# write-up (docs/search.qmd, section exp021), since the qmd computes its tables
# from results/ at render time.

set -euo pipefail
cd "$(dirname "$0")/.."

EXP="${EXP:-experiments/exp021-search}"
CKPT="${CKPT:-experiments/exp018-distill/checkpoints/ep22.pt}"
BENCH="${BENCH:-data/bgsage/money_benchmark/benchmark.json.gz}"
SUB="${SUB:-2000}"
SUB_SEED="${SUB_SEED:-21}"
WORKERS="${WORKERS:-4}"
PY="${PY:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)}"

export OMP_WAIT_POLICY=PASSIVE

mkdir -p "$EXP/logs" "$EXP/results"
LOG="$EXP/logs/exp021.log"
run() { echo "=== $* ===" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }

echo "exp021 started $(date -Is) on $(hostname)" | tee -a "$LOG"

# --- stage 1: harness-correctness control (gates everything below) -----------
echo "=== stage 1: depth-0 regression, must reproduce PR 0.9498 ===" | tee -a "$LOG"
run "$PY" scripts/eval_benchmark_pr.py \
    --benchmark "$BENCH" --checkpoint "$CKPT" --engine-label "ep22_depth0_control" \
    --search-depth 0 --output "$EXP/results"

# --- stage 2: the matched-depth bar, measured before our own configs ---------
echo "=== stage 2: GNUBG 1-ply reference (full n) ===" | tee -a "$LOG"
run "$PY" scripts/eval_benchmark_pr.py \
    --benchmark "$BENCH" --gnubg-ply 1 --workers "$WORKERS" \
    --output "$EXP/results"

# --- stage 3: sweep on the fixed subsample ----------------------------------
echo "=== stage 3: config sweep (subsample n=$SUB, seed $SUB_SEED) ===" | tee -a "$LOG"
sweep() {  # depth k k2 gate
    run "$PY" scripts/eval_benchmark_pr.py \
        --benchmark "$BENCH" --checkpoint "$CKPT" \
        --engine-label "ep22_d$1_k$2_k2$3_g$4" \
        --search-depth "$1" --search-k "$2" --search-k2 "$3" --search-gate "$4" \
        --subsample "$SUB" --subsample-seed "$SUB_SEED" \
        --output "$EXP/results"
}
sweep 1 4 2 0.08
sweep 1 8 2 0.08
sweep 1 8 2 0.05
sweep 1 16 2 0.08
sweep 2 8 2 0.08

echo "exp021 sweep finished $(date -Is)" | tee -a "$LOG"
echo "Select the best config, then run stage 4 (full n) explicitly:" | tee -a "$LOG"
echo "  $PY scripts/eval_benchmark_pr.py --benchmark $BENCH --checkpoint $CKPT \\" | tee -a "$LOG"
echo "      --engine-label ep22_<tag> --search-depth D --search-k K --search-k2 K2 \\" | tee -a "$LOG"
echo "      --search-gate G --output $EXP/results --dump-dir $EXP/dumps" | tee -a "$LOG"
