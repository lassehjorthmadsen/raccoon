#!/usr/bin/env bash
# exp026 — does Raccoon beat REAL GNU Backgammon at non-Jacoby cubeful money?
#
# WHY THIS RUN EXISTS.
# goal.md's success criterion is a positive average win against GNUBG at world
# class settings over 1000+ games, with the CI clear of zero. Nothing so far has
# tested it. Every previous play result was CUBELESS, which satisfies the
# non-Jacoby framing only degenerately, and exp025 shipped the cube but scored it
# on a static benchmark rather than in play. This is the first time the stated
# criterion is actually put to a real opponent.
#
# WHY IT COULD NOT BE PART OF exp025.
# gnubg-nn, the Python package every Raccoon number before this was measured
# against, CANNOT make a money cube decision: evaluate_cube_decision raises
# "RuntimeError: Not implemented for money" (the binding covers match play only)
# and its cubeful_rollout binding is broken outright. It also ranks checker plays
# CUBELESSLY. Bolting our own Janowski layer onto its probabilities — what
# raccoon/eval/cube_arena.py's stand-in does — yields a match in which BOTH sides
# share one cube model, while real GNU Backgammon uses the closed form only as a
# leaf evaluator inside a cubeful SEARCH that decides the cube at every node.
# That opponent is weaker than the real thing on exactly the dimension a cube
# experiment tests, so a win against it would not be the goal.md criterion.
# exp025 therefore reported no head-to-head at all.
#
# THE OPPONENT, AND WHY IT IS HONEST NOW.
# raccoon/eval/gnubg_cli.py drives /usr/games/gnubg 1.07.001 as a decision
# oracle. GNUBG answers "given this position, this cube and these dice, what do
# you do?" and nothing more, so Raccoon keeps driving the game and rolling the
# dice — which is what preserves the variance-reduced estimator, since that has
# to intercept every pre-roll state.
#
# Positions go in BY NAME, not by field order: `set board <PositionID>` with
# GNUBG's own encoder (via gnubg_nn.position_id), then set turn / set dice /
# set cube value / set cube owner. The alternative, a FIBS `board:` line over a
# socket, is ~50 unlabelled integers whose layout we could not obtain — every
# reachable copy of the spec is truncated, and guessing it produced
# self-contradictory cube answers. Named commands have no field order to get
# wrong, and are marginally faster besides (~62 ms vs ~84 ms per 2-ply decision).
#
# HARNESS-CORRECTNESS CONTROLS, all pinned in tests/test_gnubg_cli.py.
#   * Position ID round trip, EXACT. Compares an encoding against itself, so a
#     perspective or seating bug surfaces as a mismatched ID rather than as a
#     plausible-looking equity. This is the control that matters.
#   * Every move GNUBG returns is parsed into a target board and then REQUIRED to
#     match one of the mover's legal turn outcomes (actions_reaching). A notation
#     slip stops the run instead of quietly playing a different game for
#     thousands of positions. Validated over whole games: 105 turns, zero
#     mismatches, including bar entries, hits, bear-offs, doubles and dances.
#   * Book opening plays at 2-ply, which no GNUBG build disputes.
#   * A guard that GNUBG's cubeful search and our closed form do NOT price the
#     cube identically — if they did, the opponent swap would be measuring
#     nothing.
#
# THE TWO GNUBGs ARE DIFFERENT NETS, AND IT DOES NOT MATTER — MEASURED.
# The binary and the package ship different weights: same six-net topology (race
# 214-128-5, crashed and contact 250-128-5, three prune nets), 101,966 weights
# against 102,004 float32, but different trained values from different lineages.
# The package's file declares "GNU Backgammon 1.01"; the binary expects "1.00",
# rejects it, and rejects its contents even when the version string is forced.
# Their ply-0 probabilities differ by up to ~0.003.
#
# So they were played against each other, cubeless, at ply 0 — the exact setting
# exp019 and exp020 used:
#     binary vs package = -0.0026 +/- 0.0047 ppg (n=6,000, variance-reduced)
#     results/binary_vs_package_0ply.json
# Statistically indistinguishable, and an order of magnitude below exp020's
# +0.0472 edge. The back catalogue therefore translates without a correction
# term, provided the +/-0.005 caveat travels with it. NOT measured at ply 2,
# where this experiment runs; search over near-identical nets should converge
# further, but that is reasoning, not measurement.
#
# HYPOTHESIS (one).
# The shipped net (exp018/ep22) with the Janowski cube layer wins at non-Jacoby
# cubeful money against real GNU Backgammon at 2-ply.
#
# PRIMARY METRIC.
# Variance-reduced cubeful ppg from Raccoon's POV, seats alternating, 95% CI.
# n is FIXED FROM THE PILOT below before the headline run and not revisited.
#
# THE ASYMMETRY, ON THE RECORD.
# GNUBG decides the cube inside a 2-ply search; Raccoon uses a static closed form
# at the root. RACCOON IS THE HANDICAPPED SIDE. That is deliberate — it is
# Raccoon as it actually is, against GNUBG as it actually is, which is what
# goal.md asks. It also predicts the follow-up: if the result is negative, the
# first thing to try is cubeful search on our side, not a better cube constant.
# Do not "fix" the asymmetry by weakening the opponent.
#
# VARIANCE REDUCTION.
# Built and validated in exp025: each luck term scaled by the cube value at its
# own roll, h a cubeful equity. Unbiased with a live cube because h stays a
# deterministic function of (pre-roll state, roll), the cube state is part of
# that state, and cube decisions are not chance events so they add no luck terms.
# The control variate remains gnubg-nn's 0-ply best play whoever the opponent is:
# h has only to be a FIXED function, not the opponent's own evaluator, and
# keeping it fixed lets a match against the binary and one against the stand-in
# be read on one ruler. cv=cubeful was chosen in exp025 (sd_vr 0.434 vs 0.606)
# and is not re-litigated here.
#
# POWER, and what the pilot said.
# A 600-game pilot at GNUBG-2-ply whose ONLY readout is sd_raw and sd_vr; its ppg
# was not looked at and is reported nowhere.
#     sd_raw 2.907   sd_vr 0.433   sd_ratio 6.71   beta_hat 1.034   luck t -0.56
#     zero gnubg restarts over 600 games; 5.48 s/game/worker
#     results/pilot_sd.json
# sd_vr lands on 0.433, within a thousandth of the 0.434 exp025 measured against
# the stand-in -- the estimator does not care which opponent it is watching,
# which is what makes the two experiments readable on one ruler. Then
#     n = (1.96 * 0.433 / 0.01)^2 = 7,206   (~3.7 h at 3 workers)
# for delta = 0.01 ppg, matching the precision exp019/exp020 reached cubelessly
# (+/-0.0053, +/-0.0027) so the numbers sit in one table. n is FIXED at 7,206 and
# is not revisited after the run.
#
# COST. ~2 s/game wall at 3 workers (the binary is ~3x cheaper than the exp025
# stand-in, because GNUBG's own move filtering beats our brute-force candidate
# loop). CPU only; no GPU, no GCP spend.
#
# SCOPE — explicitly NOT in this experiment.
#   * No BGSage. It reuses this harness but "beats GNUBG" and "beats BGSage" are
#     two questions; it gets its own experiment.
#   * No beavers.
#   * No cubeful search for Raccoon. That is the follow-up this experiment sizes.
#   * No match play, no re-tuning of x.
#   * No re-running exp025's benchmark arms; that result stands on its own.
#
# OUTPUTS.
#   experiments/exp026-gnubg-cli/logs/     raw stdout, plus cube_eval_log.jsonl
#   experiments/exp026-gnubg-cli/results/  idempotent JSON with per-game arrays
#   Write-up: docs/cube.qmd, a new #exp026 section.

set -euo pipefail
cd "$(dirname "$0")/.."

EXP=experiments/exp026-gnubg-cli
RESULTS="$EXP/results"
LOGS="$EXP/logs"
CKPT=experiments/exp018-distill/checkpoints/ep22.pt
PY=.venv/bin/python3
WORKERS=3

# Fixed from the pilot: n = (1.96 * 0.433 / 0.01)^2 = 7,206, a multiple of
# WORKERS. Do not change after the run.
GAMES=${GAMES:-7206}

mkdir -p "$RESULTS" "$LOGS"
export OMP_WAIT_POLICY=PASSIVE

# --- Supporting: are the two GNUBGs the same opponent? -------------------------
# Not part of the hypothesis. It exists so a number against the binary can be
# quoted next to every number ever taken against the package.

echo "=== [1/3] binary vs package, cubeless, ply 0 ==="
$PY scripts/compare_gnubg_engines.py --games 6000 --ply 0 --workers "$WORKERS" \
    --seed-base 6000 --exp-dir "$EXP" --tag binary_vs_package_0ply \
    2>&1 | tee "$LOGS/engine_compare_0ply.log"

# --- Pilot: SD only. Do NOT read the ppg. --------------------------------------

echo "=== [2/3] pilot, 600 games vs real gnubg 2-ply ==="
$PY scripts/eval_gnubg_cube.py --checkpoint "$CKPT" --games 600 --ply 2 \
    --workers "$WORKERS" --cv cubeful --seed-base 7000 \
    --exp-dir "$EXP" --tag pilot_sd 2>&1 | tee "$LOGS/pilot_sd.log"

if [ "$GAMES" -eq 0 ]; then
    echo
    echo "Pilot done. Read sd_vr from $RESULTS/pilot_sd.json, compute"
    echo "n = (1.96 * sd_vr / 0.01)^2, write it into GAMES here and into the"
    echo "POWER section above, then re-run with that GAMES set."
    exit 0
fi

# --- Headline: the primary metric ----------------------------------------------

echo "=== [3/3] headline, $GAMES games vs real gnubg 2-ply ==="
$PY scripts/eval_gnubg_cube.py --checkpoint "$CKPT" --games "$GAMES" --ply 2 \
    --workers "$WORKERS" --cv cubeful --seed-base 8000 \
    --exp-dir "$EXP" --tag ep22_vs_gnubg_cli_2ply \
    2>&1 | tee "$LOGS/headline_ep22_2ply.log"
