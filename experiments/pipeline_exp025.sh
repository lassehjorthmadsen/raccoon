#!/usr/bin/env bash
# exp025 — is cube-aware checker ranking worth anything?
#
# WHY THIS RUN EXISTS.
# goal.md targets money game *without* Jacoby. Every play result so far
# satisfies that only degenerately: Raccoon plays cubeless, and a game with no
# cube has no Jacoby to suppress. exp024 settled the cube MODEL — piecewise
# Janowski, x = 0.68 contact / 0.60 race, W and L floored at 1, the [-L, W]
# bound as a test (docs/cube.qmd). What was missing was everything downstream:
# nothing in any play loop referenced raccoon/cube/janowski.py. exp025 ships the
# cube into move selection and measures what that alone buys.
#
# THE PRIZE, ALREADY MEASURED.
# experiments/exp024-cube/results/cube_blind_floor.json: a PERFECT CUBELESS
# player throws away PR 0.336 +/- 0.042 when its moves are judged cubefully,
# with cubeless-best and cubeful-best differing on 992 of 14,693 decisions
# (6.8%). It concentrates where the cube is alive — 0.553 centred against 0.145
# when the opponent owns it — and is nearly absent in pure races (0.046). That
# is a floor no amount of extra cubeless accuracy reaches, and cube-aware
# ranking is what claims it. This experiment asks how much of it ep22 actually
# claims, given its own evaluation error.
#
# HYPOTHESIS (one).
# Ranking candidate moves by Janowski cubeful equity instead of cubeless equity
# reduces the cubeful move-selection error of the shipped net (exp018/ep22,
# outcomes6).
#
# PRIMARY METRIC.
# Cubeful PR on all 14,693 BGSage checker decisions — mean cubeful equity thrown
# away per decision, times 500 — cubeless-ranked against cubeful-ranked, PAIRED
# on the same decisions with the same net, 95% interval bootstrapped over the
# 500 generating games.
#
# WHY THE HEAD-TO-HEAD IS NOT IN THIS EXPERIMENT — read this before adding it.
# The obvious completion is "and it beats GNUBG at cubeful money". It is not
# here because there is no honest opponent to hand yet:
#
#   * gnubg-nn, the Python package every other Raccoon number is measured
#     against, CANNOT make a money cube decision at all —
#     evaluate_cube_decision raises "RuntimeError: Not implemented for money"
#     (verified on 1.1.0a6; the binding covers match play only), and its
#     cubeful_rollout binding is broken outright ("More keyword list entries (4)
#     than format specifiers (3)"). It also ranks checker plays CUBELESSLY, so
#     it is a stand-in on both halves of a cubeful game.
#   * Bolting our own Janowski layer onto its probabilities — which is what
#     raccoon/eval/cube_arena.py does — gives a symmetric match in which BOTH
#     sides share one cube model. Real GNU Backgammon does not use the closed
#     form as its cube engine; it uses it as a leaf evaluator inside a CUBEFUL
#     SEARCH that makes cube decisions at every node (docs/cube.qmd, "Still
#     open"). So that opponent is weaker than real GNUBG precisely on the cube,
#     and a win against it would not be the goal.md criterion.
#
# The real opponent exists — /usr/games/gnubg 1.07 answers money cube decisions
# and cube-aware checker play, at ~71 ms per 2-ply decision — but wiring it up
# is its own piece of work. It becomes exp026. BGSage (../bgsage, a money
# cube_action API behind an unbuilt C++ extension) is the stronger target and
# reuses the same harness, but "beats GNUBG" and "beats BGSage" are two
# questions, so it gets its own experiment rather than riding along.
#
# SUPPORTING, DEMARCATED — NOT part of the hypothesis and not a headline.
# results/pilot_cv_*.json measure the CUBEFUL VARIANCE-REDUCED ESTIMATOR itself
# (raccoon/eval/luck.py, raccoon/eval/cube_arena.py) over 600 games against the
# stand-in opponent: with the cube live the control variate stays unbiased
# (luck t = +0.26), and a cubeful h beats a cube-scaled cubeless one, sd_vr
# 0.434 against 0.606 (7.32x vs 5.24x SD reduction, beta_hat 1.017). That is a
# property of the estimator, which exp026 reuses; the ppg those games implied
# was deliberately never read and is not reported anywhere. Do not quote these
# as a strength result.
#
# PAIRING IS NOT OPTIONAL. The two arms agree on the great majority of
# decisions, so their per-decision errors are identical almost everywhere and an
# unpaired comparison of two PR numbers has an interval far wider than the
# effect. --paired-dump takes both errors off the SAME six-outcome forward pass
# (for an outcomes6 head, value_equity IS cubeless_equity of its own
# distribution), so the arms are paired by construction. Step 1 is the
# standalone cubeless arm, kept precisely to check the derived one reproduces it.
#
# JACOBY, THE ONE PLACE THIS EXPERIMENT IS NOT THE TARGET GAME.
# The benchmark's 500 generating games were played WITH Jacoby, so its cubeful
# references price a game in which gammons are worthless while the cube is
# centred. Scoring against them therefore models the same rule (cube_jacoby on);
# raccoon/eval/cube_arena.py plays with it off, per goal.md. The
# `owned (Jacoby-clean)` breakdown bounds what that costs — and note where the
# measured gain lands before quoting it.
#
# SCOPE — explicitly NOT in this experiment.
#   * No head-to-head play of any kind. See above; it is exp026.
#   * No beavers.
#   * No cube in raccoon/cli/play.py or raccoon/protocol/rgp.py.
#   * No match play.
#   * No re-tuning of x. exp024's non-goals stand.
#
# COST. Two full-benchmark passes of the 10x256 net on CPU, ~30 min each on
# lasse-iMac14-2, plus seconds for the paired analysis. No GPU, no GCP spend.
#
# OUTPUTS.
#   experiments/exp025-cube-ranking/logs/     raw stdout per step
#   experiments/exp025-cube-ranking/results/  idempotent per-engine JSON plus
#                                             exp025_paired_pr.json, the
#                                             conclusion
#   experiments/exp025-cube-ranking/dumps/    per-decision paired errors
#                                             (gitignored)
#   Write-up: docs/cube.qmd, a new #exp025 section.

set -euo pipefail
cd "$(dirname "$0")/.."

EXP=experiments/exp025-cube-ranking
RESULTS="$EXP/results"
LOGS="$EXP/logs"
CKPT=experiments/exp018-distill/checkpoints/ep22.pt
PY=.venv/bin/python3

mkdir -p "$RESULTS" "$LOGS" "$EXP/dumps"
export OMP_WAIT_POLICY=PASSIVE

echo "=== [1/3] benchmark, cubeless ranking, cubeful metric ==="
# Also the cross-check on step 2's derived cubeless arm: the two must agree.
$PY scripts/eval_benchmark_pr.py --checkpoint "$CKPT" --metric cubeful \
    --engine-label "ep22 cubeless-rank" --output "$RESULTS" \
    2>&1 | tee "$LOGS/bench_cubeless_rank.log"

echo "=== [2/3] benchmark, cubeful ranking, cubeful metric, paired dump ==="
$PY scripts/eval_benchmark_pr.py --checkpoint "$CKPT" --metric cubeful --cubeful-rank \
    --engine-label "ep22 cubeful-rank" --output "$RESULTS" \
    --paired-dump "$EXP/dumps/ep22_paired.npz" \
    2>&1 | tee "$LOGS/bench_cubeful_rank.log"

echo "=== [3/3] the paired PR gain, bootstrapped over games ==="
$PY scripts/exp025_paired_pr.py --dump "$EXP/dumps/ep22_paired.npz" \
    --output "$RESULTS" 2>&1 | tee "$LOGS/paired_pr.log"
