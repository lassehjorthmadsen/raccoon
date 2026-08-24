#!/usr/bin/env bash
# exp024 — is Janowski's cube-life index mis-set, or is the model wrong?
#
# WHY THIS RUN EXISTS.
# Raccoon has no cube. The shipped net (exp018/ep22) already emits the six-outcome
# distribution the cube needs, but nothing turned that into a double or a take.
# The standard conversion is Rick Janowski's cube-life-index model, which GNUBG
# and XG both ship. docs/cube.qmd is the design study; this is its experiment.
#
# THE ENABLING DISCOVERY. The BGSage money benchmark already on disk carries
# 2,196 kind=="cube" entries with cubeful equity_nd/equity_dt/equity_dp and
# reference double/take actions — 2,842 scorable sub-decisions.
# scripts/eval_benchmark_pr.py discards all of it at load time (it filters to
# kind=="checker"). So the cube can be measured today, at zero play cost, on the
# same benchmark every other Raccoon number is quoted on.
#
# THE MECHANISM, in code rather than hypothesised.
#   raccoon/cube/janowski.py            the model. Piecewise-linear live-cube
#                                       variant (GNUBG's MoneyLive), which is
#                                       what the engines ship — NOT the paper's
#                                       Appendix-1 closed form. The two agree on
#                                       an owned cube and differ on a centred one
#                                       under Jacoby.
#   raccoon/eval/cube_benchmark.py      BGSage's own PR formulas, ported from
#                                       bgsage/scripts/benchmark_money.py:_score_cube
#                                       so the numbers stay comparable.
#   raccoon/model/network.py            value_probs6(), the six-outcome accessor
#                                       the cube needs. value_equity() had been
#                                       collapsing the distribution away.
#   tests/test_janowski.py              pins the paper's own worked example and
#                                       tables, plus the Jacoby branch structure.
#
# PRIMARY METRIC.
# Cube PR on the BGSage money benchmark, restricted to the 700 positions whose
# reference equities were ACTUALLY ROLLED OUT -- n=1,018 cube sub-decisions --
# fed the benchmark's own reference probabilities so the net's evaluation error
# is out of the comparison. Baseline: x = 0.68 -> 5.406 +/- 0.609. Variants are
# compared to it PAIRED.
#
# WHY NOT ALL 2,196 POSITIONS. `eval_level` records how each reference was made:
#     Rollout (700)  1,296 trials PLAYED TO COMPLETION with the cube live, VR on;
#                    an observed average carrying a standard error (~+/-0.005).
#     3-ply (1,496)  a cubeful 3-ply search that calls cl2cf_money -- Janowski --
#                    at its own leaves (cpp/src/cube.cpp:670).
# Scoring a Janowski variant against the second group is close to scoring it
# against itself, and it shows: the published model measures PR 0.649 there
# against 5.406 on the rolled-out set, and refitting the index buys +0.117 +/-
# 0.210 (nothing) against +3.049 +/- 0.641. Pooling the two -- which an earlier
# version of this experiment did -- dilutes the real effect with 1,824 near-
# circular, near-null decisions and understates the model's error fourfold.
#
# NOTE tier is NOT the same split and must not be used for this. Tier is assigned
# by how close a decision is, so it confounds difficulty with provenance (3P
# ~ 3-ply, 3T+rollout ~ Rollout). Split on eval_level.
#
# WHAT THE MEASURED REFERENCES STILL ARE NOT. The payoff is empirical, but the
# cube decisions made DURING each rollout come from a 3-ply bot that also bottoms
# out in Janowski. So they measure equity under Janowski-quality cube play, not
# optimal cube play. Policy error can only lower realised equity, and its cost is
# second-order near a decision threshold, so this understates what a better model
# could reach rather than overstating it. Also: each position is rolled out in
# its own cube state, so equity_nd gives E_O *or* E_C and equity_dt gives 2*E_U --
# E_O and E_C are never measured for the same position, which is exactly why the
# ordering violation below could not have been caught by the data.
#
# FIT AND CONFIRM. Fitting an index on the decisions we score is overfitting. The
# fitted variants are fitted on a 50% split of GAMES (the `seed` field, 500 distinct
# games) and confirmed on the held-out half; splitting on positions would leak,
# since several cube decisions can come from one game. Selection happens on the
# confirm half.
#
# FITTING ON THE REPORTED METRIC, deliberately. The variants are fitted by minimising
# cube PR, not equity RMSE. Only the ND/DT gap NEAR THE DECISION BOUNDARY moves a
# decision, so an index that fits the equities better can choose worse; selecting
# on one estimator and reporting another is a bias this project has been bitten by
# twice (see the exp023 pruned-move trap). PR is piecewise constant in the
# parameters, so the search is a grid, and multi-parameter variants use coordinate
# descent — a coordinate-wise minimum, not a certified global one.
#
# VARIANTS.
#   baseline         x = 0.68 everywhere (GNUBG's published contact constant)
#   race_split       x = 0.60 in race plans, 0.68 elsewhere (GNUBG's documented split)
#   refit_global     one re-fitted index
#   refit_per_state  re-fitted per cube state (ND-centred / ND-owned / DT)
#   dead / live      x = 0 and x = 1, the envelope the model interpolates between
#   offset           x = 0.68 plus a fitted additive ND shift — the direct test of
#                    whether the residual is a flat shift the index could absorb
#
# COHERENCE GATE, applied before PR. For one position, E_O >= E_C >= E_U must
# hold: owning the cube cannot be worth less than it being centred, which cannot
# be worth less than the opponent owning it. Nothing in the fitting enforces
# this, because each entry is EITHER centred OR owned, so the three lines are
# fitted on disjoint position sets. A variant that breaks the ordering is
# disqualified regardless of PR. refit_per_state fails it at 38.9% and is
# rejected; the single-index variants sit at the ~3% Jacoby-asymmetry floor.
#
# COMPLETION (stage B). The selected variant is applied UNCHANGED (no per-engine
# re-fit, which would confound cube modelling with evaluation error) to
# exp018/ep22's own six-outcome vector, with GNUBG-0-ply's probabilities through
# the published constant as the reference point.
#
# OUTPUT LAYOUT.
#   experiments/exp024-cube/results/<source>_::_<variant>.json   one per variant; carries
#       the per-sub-decision error vector so any later paired comparison —
#       including across probability sources, which one run cannot do — works off
#       committed results instead of a re-run.
#   experiments/exp024-cube/results/cube_blind_floor.json    the motivating
#       measurement: what perfect cubeless checker play costs on the cubeful
#       metric. Reported, not chased; its experiment is pre-registered separately.
#   experiments/exp024-cube/results/cube_equity_fit.json      each of the three
#       Janowski equity formulas scored against 42,636 measured rollout equities,
#       fitted independently. This is what identifies E_C rather than the index
#       as the broken component.
#   experiments/exp024-cube/results/residual_diagnostics.json where the
#       no-double residual lives, and the three tests that rule out the Jacoby
#       rule as its cause. Best-fit indices there minimise equity RMSE and are
#       diagnostics only — the variants above are fitted on cube PR.
#   experiments/exp024-cube/results/summary.json             rebuilt from every
#       variant file in the directory, so stage A and stage B both show up.
#   experiments/exp024-cube/logs/                            raw stdout.
#
# Runtime: about 20 s for stage A, a couple of minutes for stage B. This is a
# laptop-scale experiment; no GPU, no VM.

set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS=experiments/exp024-cube/results
LOGS=experiments/exp024-cube/logs
CKPT=experiments/exp018-distill/checkpoints/ep22.pt
mkdir -p "$RESULTS" "$LOGS"

export OMP_WAIT_POLICY=PASSIVE   # CPU: the default busy-spin collapses under load

echo "=== Motivation: what cube-blind checker play costs ==="
python3 scripts/cube_blind_floor.py --output "$RESULTS" \
  2>&1 | tee "$LOGS/cube_blind_floor.log"

echo "=== The three equity formulas vs measured rollout equities ==="
# The primary evidence. The benchmark's CHECKER decisions carry 42,636 candidates
# rolled out to completion, each with a measured cubeful equity next to the
# cubeless probabilities for the same position, and the parent decision records
# where the cube sat -- so each of E_O, E_C, E_U can be tested on its own against
# measurement. That is 42x the rolled-out cube positions and it separates the
# three formulas, which cube PR cannot: a cube decision only ever compares two
# locations at once.
python3 scripts/cube_equity_fit.py --output "$RESULTS" \
  2>&1 | tee "$LOGS/cube_equity_fit.log"

echo "=== Spot-check sheet: worst-fit positions with GNU Position IDs ==="
# A 0.23-point error in a formula GNUBG and XG have shipped for decades should be
# independently checkable. gnubg_nn refuses money play, so this emits position
# IDs for checking in the real GUI instead.
python3 scripts/cube_spot_check.py --n 2 --output "$RESULTS" \
  2>&1 | tee "$LOGS/cube_spot_check.log"

echo "=== Diagnosis: where the residual lives, and whether Jacoby explains it ==="
python3 scripts/cube_residual_diagnostics.py --output "$RESULTS" \
  2>&1 | tee "$LOGS/residual_diagnostics.log"

echo "=== Stage A: the cube model in isolation (reference probabilities) ==="
python3 scripts/eval_cube_pr.py --probs reference --variants all \
  --output "$RESULTS" 2>&1 | tee "$LOGS/stage_a_reference.log"

echo "=== Stage B: exp018/ep22's own evaluations ==="
python3 scripts/eval_cube_pr.py --probs checkpoint --checkpoint "$CKPT" \
  --variants baseline --engine-label ep22 --output "$RESULTS" \
  2>&1 | tee "$LOGS/stage_b_ep22.log"

# The selected variant, applied with the parameters stage A settled on. That is
# the single re-fitted index, NOT the per-cube-state fit: fitting an index per
# cube state scores better on PR but breaks E_O >= E_C >= E_U on 39% of
# positions, and PR cannot see that because each scored decision involves only
# one cube location.
python3 scripts/eval_cube_pr.py --probs checkpoint --checkpoint "$CKPT" \
  --custom-name selected --x-nd-centered 0.800 --x-nd-player 0.800 --x-dt 0.800 \
  --variants selected --engine-label ep22 --output "$RESULTS" \
  2>&1 | tee -a "$LOGS/stage_b_ep22.log"

echo "=== Stage B reference point: GNUBG 0-ply through the published constant ==="
python3 scripts/eval_cube_pr.py --probs gnubg --gnubg-ply 0 \
  --variants baseline --engine-label gnubg-0ply --output "$RESULTS" \
  2>&1 | tee "$LOGS/stage_b_gnubg.log"

echo "=== Contrast: the same variants on search-derived references ==="
# Not a result -- the demonstration that references produced by a search which
# uses Janowski at its leaves cannot judge a Janowski variant.
python3 scripts/eval_cube_pr.py --probs reference --references all \
  --variants baseline,refit_global --engine-label "all-refs" \
  --output "$RESULTS" 2>&1 | tee "$LOGS/contrast_all_references.log"

echo "Done. Results in $RESULTS"
