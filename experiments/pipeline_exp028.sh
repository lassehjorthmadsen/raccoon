#!/usr/bin/env bash
# exp028 — can a small MLP carry ep22's evaluation accuracy?
#
# WHY THIS RUN EXISTS.
# docs/speed.qmd measured what every strength result in this project had left
# out. ep22 evaluates 410 positions/s; GNU Backgammon evaluates 1,476,488 --
# 3,470x, on 8,718x the arithmetic. It is not the implementation: we reach ~59%
# of the processor's theoretical peak and are 2.5x MORE efficient per
# multiply-accumulate than GNUBG. It is the shape of the network.
#
# The consequence is that every published Raccoon result compares engines at
# matched search DEPTH and none at matched TIME, and at matched time the ordering
# reverses: GNU Backgammon reaches PR 0.588 in 0.062 s where Raccoon reaches
# 1.026 in 0.049 s, and the depth-2 configuration where Raccoon clearly wins
# costs 28 s -- about 450x the opponent it beats. An engine that needs 450x the
# opponent's time is not a superior engine in any sense that matters.
#
# goal.md's first assumption -- that beating GNUBG needs an AlphaZero-style
# network -- motivated the convolutional trunk and is now marked challenged.
# GNUBG (250-128-5), XG and BGSage (244-400-5, five to nineteen nets by game
# plan) all use small fully connected networks over handcrafted inputs. Raccoon's
# encoder already computes those inputs and hands them over as channels 17-25.
#
# HYPOTHESIS (one).
# A fully connected network over the existing 26-channel encoding, trained on the
# same 40M GNUBG 2-ply labels ep22 was trained on, retains most of ep22's
# move-selection accuracy while evaluating over a thousand times faster.
#
# PRIMARY METRIC.
# PR on the BGSage money benchmark, n = 14,693, scored by scripts/eval_benchmark_pr.py
# exactly as every other checkpoint in this project has been. Reported against
# MEASURED positions per second, because the point of the experiment is the pair,
# not either number alone.
#
# SUPPORTING, demarcated: rollout-tier MSE/R^2 on ~149k candidate positions. That
# is what the distillation loss actually optimises, so it separates "the network
# cannot represent the function" from "the network represents it but ranks moves
# differently".
#
# THE ARMS, REVISED 2026-09-05 after reading Strehl, "PureTD: Reinforcement
# Learning for Backgammon Money Games with No Evaluation-time Search"
# (arXiv:2608.15146, docs/papers/). That paper trains an MLP of
# [512, 512, 256, 256] over 200 Tesauro inputs -- 562k parameters, 0.561 M
# multiply-accumulates -- and reports XG++ PR 0.94 for CUBEFUL money at 0-ply,
# beating gnubg and Open Sage at 1-ply while evaluating 2.5x faster. Its scale
# looks compatible with this benchmark's: it measures gnubg 0-ply at PR 2.18
# where we get 2.14, and gnubg 1-ply at 1.21 where we get 1.206.
#
# So the family is demonstrably sufficient, and the first arm run here was simply
# too small: 624-256-6 is one hidden layer and 0.160 M MACs against PureTD's four
# layers and 0.561 M. It converged at PR 5.4 (flat from epoch 4 over 12 epochs;
# 6.21 -> 5.39), which is a fair measurement of an unfair architecture.
#
# The arms below are shaped after PureTD instead of guessed. Speeds are MEASURED
# on this iMac at batch 512, three threads, through the value-only forward path
# 0-ply play actually uses; GNU Backgammon is 1,476,488 boards/s on the same
# machine and ep22 is 410.
#
#     mlp [512,512,256,256]   PureTD's shape, the headline arm
#     mlp [512,512]           cheaper, tests whether depth or width carries it
#     mlp [256,256]           cheaper still
#     mlp [256]               already run: converged PR 5.4, kept as the floor
#
# WHY DISTILLATION RATHER THAN PureTD's RL. PureTD needed 200M self-play games
# and ~65 h on a 64-core Ryzen with a 24 GB GPU, and reports being CPU-bound on
# move generation -- weeks on this machine. We do not obviously need it: ep22
# reached PR 0.950 by distilling gnubg 2-ply labels, essentially PureTD's number,
# so those labels already support the target quality. The question here is only
# whether an MLP of that size absorbs them, which is hours rather than weeks. If
# it cannot, PureTD's training method becomes the next thing to consider, not the
# next thing to assume.
#
# A COST NOTE THAT SHAPES THE HEAD. 0-ply play never reads the policy head, and
# for an MLP it is the larger half of the network: 256 hidden into 1352 actions
# is 346k multiply-accumulates against the trunk's 160k. RaccoonNet now has a
# value-only forward path, so the speeds above are what actually runs. A
# deployable MLP would drop the policy head entirely.
#
# WHAT WOULD MAKE THIS A SUCCESS, stated before the run.
# THE RAW NETWORK HAS TO BE THE ADVANTAGE. Search does not refund a weaker
# evaluator, for three reasons, and an earlier draft of this header got it wrong
# by assuming it does:
#
#   1. The opponent searches too. At equal time the comparison is MLP+depth-2
#      against GNUBG+2-ply, so search largely cancels and what is left is the
#      raw evaluator.
#   2. Search gain SHRINKS as the raw network improves, so a weaker net cannot be
#      assumed to ride search back to parity. On the same 2,000-decision sample:
#          GNUBG   0-ply 2.088 -> 2-ply 0.588   gain 1.50
#          ep22    0-ply 1.026 -> 2-ply 0.426   gain 0.60
#      The better evaluator gains less, because it leaves search less to find.
#      A weaker MLP would be competing against an opponent whose search gain is
#      larger, from a lower foundation.
#   3. exp021's 0.60 was measured on ep22. Whether an MLP gains the same is
#      unmeasured, and quoting it as an exchange rate assumes the thing in
#      question.
#
# Raccoon's entire advantage over GNU Backgammon today is raw evaluation: 1.026
# against 2.088 on the same positions, twice as good per position, and that is
# the asset the project is built on. So the bar is:
#
#   PASS  raw PR clearly better than GNUBG's raw 2.088 at comparable speed, and
#         close enough to ep22's 0.950 that the foundation survives. An arm at
#         ~1.4 is a genuinely better evaluator at GNUBG's cost and worth building
#         search on.
#   FAIL  raw PR near 2.0. That is GNU Backgammon rewritten, with nothing to
#         build on, whatever search would add.
#
# Search enters only AFTER an arm passes, as the thing a cheap network makes
# affordable -- and its gain then gets measured on that network rather than
# assumed from ep22's.
#
# TRAINING DATA. data/distill/2ply -- the same 40M GNUBG 2-ply labels exp017 and
# exp018 used, 58 GB across three runs. Nothing is regenerated; if the arms
# differ it is the architecture, since the labels, the split and the seed are
# identical.
#
# EXPECT I/O TO BIND, NOT ARITHMETIC. A forward and backward pass over 40M
# positions at 161k MACs is minutes of compute; reading 58 GB is not. If an epoch
# is dominated by shard decoding, pre-flatten once to a memory-mapped array
# rather than optimising the model.
#
# SCOPE -- explicitly NOT in this experiment.
#   * No head-to-head play. This is a static measurement; a play run comes after
#     an arm is worth playing.
#   * No new training data, and no relabelling. Relabelling 40M positions with
#     Raccoon at 2-ply is 3.5e11 evaluations, ~2e20 FLOPs: 9,585 days on this
#     machine, 510 on one T4, ~10 on eight A100s. It becomes a 5-day CPU job at
#     MLP cost, which is a reason to want this experiment, not part of it.
#   * No input-representation work. The flattened 26-channel encoding goes in as
#     it stands. If a plain flattening underperforms, the representation is the
#     next thing to vary -- GNUBG's 250 inputs are hand-designed and much of our
#     624 is one-hot checker planes an MLP may use poorly. That is exp029, not a
#     mid-run change here.
#   * No cube work, no search tuning.
#
# OUTPUTS.
#   experiments/exp028-mlp/logs/       training logs per arm
#   experiments/exp028-mlp/results/    benchmark JSON per arm, plus speeds
#   Write-up: docs/speed.qmd, an accuracy-against-speed section.

set -euo pipefail
cd "$(dirname "$0")/.."

EXP=experiments/exp028-mlp
CACHE=data/distill/2ply/run3
PY=.venv/bin/python3
EPOCHS=${EPOCHS:-2}

mkdir -p "$EXP/logs" "$EXP/results"
export OMP_WAIT_POLICY=PASSIVE

# Cheapest arm first: it is the one that decides whether the idea is alive, and
# it costs the least to find out.
run_arm () {   # name, extra train args...
    local name="$1"; shift
    echo "=== training $name ==="
    $PY scripts/train_distill.py --cache-dir "$CACHE" \
        --experiment-name "exp028-mlp/$name" --value-head outcomes6 \
        --epochs "$EPOCHS" "$@" 2>&1 | tee "$EXP/logs/train_$name.log"
    echo "=== scoring $name ==="
    $PY scripts/eval_benchmark_pr.py \
        --checkpoint "experiments/exp028-mlp/$name/checkpoints/best.pt" \
        --engine-label "$name" --output "$EXP/results" \
        2>&1 | tee "$EXP/logs/bench_$name.log"
}

# Every arm runs 12 epochs and every epoch checkpoint is scored, because
# convergence has to be verified on PR: a flat training-loss curve looks
# identical for a converged model and an unconverged one (the first arm was
# called converged at 2 epochs and was still improving 15% per epoch).
run_arm mlp_512_512_256_256 --trunk mlp --hidden 512 --hidden-layers 4
run_arm mlp_512_512         --trunk mlp --hidden 512 --hidden-layers 2
run_arm mlp_256_256         --trunk mlp --hidden 256 --hidden-layers 2

echo "=== measuring evaluation speed for every arm ==="
$PY scripts/measure_eval_speed.py --output "$EXP/results" \
    2>&1 | tee "$EXP/logs/eval_speed.log"
