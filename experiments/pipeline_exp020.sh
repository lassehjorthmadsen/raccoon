#!/usr/bin/env bash
# exp020 — is Raccoon's two-step doubles execution a strength bug, and does fixing it
#          collect the ppg that exp019 said the benchmark was promising?
#
# WHY THIS RUN EXISTS. exp019 measured exp018/ep22 at +0.0129 +/- 0.0053 VR ppg vs
# GNUBG-0-ply (n=6000) and left one thing open: the BGSage benchmark predicts +0.067 ppg
# from ep22's PR advantage, but play delivered +0.013 — a ~20 sigma shortfall of 0.054 ppg.
# The write-up named two candidate explanations and committed to testing the specific one.
#
# THE MECHANISM, confirmed in code rather than hypothesised. OpenSpiel splits a doubles
# turn into two consecutive decisions by the same player, two of the four half-moves each.
#   - GNUBG's side (raccoon/eval/gnubg_adapter.py::candidate_equities) RECURSES:
#     my_equity = max(eq for _, eq in candidate_equities(child, ply)). It scores each
#     half-1 action by its best half-2 continuation — the turn is optimised jointly.
#   - Raccoon's side (raccoon/train/lookahead.py::child_values) did NOT. It ranked half-1
#     by the static value of the INTERMEDIATE position, which encode_pre_roll hands over
#     with dice=None and mid_doubles=False — telling the value head "you are about to roll"
#     about a position where the mover still owes two half-moves of a known die. Then it
#     picked half-2 greedily.
# So on 1/6 of turns the two engines were not playing by the same rule, and only one of
# them was handicapped. The value head cannot evaluate those intermediate positions: the
# 40M-position exp017/exp018 shards contain none, because scripts/gen_gnubg_selfplay.py
# emits one record per turn at the start-of-turn pre-roll and skips mid-doubles states.
#
# A 24-game probe (67 two-step doubles turns) put the concession at ~+0.009/turn with a
# 16% disagreement rate — the right order for the 0.054 shortfall, but n=67 settles nothing.
#
# PRIMARY METRIC (the completion): variance-reduced ppg vs GNUBG-0-ply, cubeless money,
# seats alternated, n = 6000, for the engine with joint doubles enumeration — against the
# pre-registered exp019 baseline +0.0129 +/- 0.0053 on the identical metric. Same seed base
# (1000+) as exp019 for partial pairing; the CI is computed UNPAIRED (conservative).
# SE(gain) ~ 0.0038, so a 0.024 ppg gain resolves at ~6 sigma.
#
# STAGE-1 GATE METRIC (mechanism, its own n): mean equity concession per net two-step
# doubles turn, concession = eq_gnubg(final_joint) - eq_gnubg(final_greedy), both finals
# from the SAME net, judged by GNUBG at 2-ply, over n = 1500 games (~4200 two-step turns).
# Priced at 0-ply too as a free cross-check. The main line plays the GREEDY arm, so the
# sample is the exp019 distribution and the joint arm is a per-turn counterfactual.
#
# GATE RULE, fixed up front: proceed to stage 2 only if the stage-1 ppg cost estimate is
# >= 0.01 AND its CI excludes 0. Otherwise stop, write up the null, and move to the other
# exp019 explanation (the benchmark's positions are not this matchup's positions).
#
# HARNESS-CORRECTNESS CONTROL: the same machinery runs on NON-DOUBLES turns, where the turn
# is a single OpenSpiel action so both arms enumerate the identical leaf set and must agree
# exactly. Any non-zero concession there is a harness bug — bad resting signature, sign
# error, mis-judged terminal — not a strength difference. Smoke run: 711/711 turns, exactly
# 0.0. tests/test_doubles.py pins the same property at the child_values level.
#
# INVARIANT, stated rather than re-measured: BGSage PR is UNCHANGED at 0.95 (ep22's own
# score; the 0.93 quoted on the site's front page is exp017-ep12). The weights do
# not change and scripts/eval_benchmark_pr.py scores complete post-move boards without
# touching child_values. Any ppg gain is therefore pure execution — which is the claim.
#
# SUPPORTING (demarcated, never the headline): disagreement rate; the zero-inflated
# concession distribution; the 0-ply cross-check; two-step turns per game; paths per turn;
# and the wall-clock cost of the fix (measured at 1.14x — leaf dedup by encoded position
# collapses ~550 ordered half-move paths onto ~58 distinct boards).
#
# COST. Stage 1 ~35 min, stage 2 ~90 min, 3 iMac workers, CPU only, no GCP spend.
set -euo pipefail

PY="${PY:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)}"
CKPT="${CKPT:-experiments/exp018-distill/checkpoints/ep22.pt}"
EXP="${EXP:-experiments/exp020-doubles}"
WORKERS="${WORKERS:-3}"

echo "=== stage 1: concession per doubles turn (gate) ==="
"$PY" scripts/eval_doubles.py \
    --checkpoint "$CKPT" --games 1500 --workers "$WORKERS" \
    --ply 0 --judge-ply 2 --seed-base 5000 \
    --exp-dir "$EXP" --tag ep22_concession

echo "=== stage 2: VR ppg at n=6000 with joint doubles (primary) ==="
"$PY" scripts/eval_gnubg_vr.py \
    --checkpoint "$CKPT" --games 6000 --workers "$WORKERS" \
    --ply 0 --cv-ply 0 --seed-base 1000 \
    --exp-dir "$EXP" --tag ep22_vr_joint
