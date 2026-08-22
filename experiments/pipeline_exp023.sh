#!/usr/bin/env bash
# exp023 — does a position-dependent search rule beat one fixed setting at matched compute?
#
# WHY THIS RUN EXISTS, AND WHY IT IS NOT PART OF exp022.
# exp022 asked whether scaling the equity window linearly with contact beats a fixed
# window at equal cost, and answered no. Replaying its stored values afterwards showed
# why, and produced two hypotheses it could not itself test:
#   1. a move must clear the equity window AND the top-k cap, and the CAP is the stronger
#      filter -- so a position-dependent rule has to move both together;
#   2. search is worth ~nothing at BOTH ends of the contact range (a pure race has no
#      interaction left; a near-opening position the net already judges well, PR 0.227
#      unchanged by any extra search), so effort should be a BUMP in contact, not a ramp.
# Both were derived from exp022's 2,000-decision sample, so that sample can no longer test
# them. exp023 is the confirmation on positions the tuning never saw.
#
# HYPOTHESIS (one). At matched compute, a bump-shaped position-dependent effort rule scores
# a lower PR than the best single fixed setting.
#
# PRE-REGISTERED, FROZEN BEFORE THE RUN -- these are the exp022 in-sample fits and are NOT
# to be retuned on the holdout:
#     effort(c) = tanh(c / 0.03) * tanh((1 - c) / 0.25)
#     window(c) = 0.02 + 0.08 * effort(c)      cap(c) = round(2 + 14 * effort(c))
#     overall effort multiplier chosen to land at ~1,000 evaluations per decision
# The FLAT COMPARATOR is fixed by rule, not by search-after-the-fact: the single (window,
# cap) whose measured cost on the holdout is closest to the bump's. Picking the best flat
# setting afterwards would reintroduce exactly the selection this experiment exists to
# remove.
#
# PRIMARY METRIC. PR on the BGSage money benchmark, on the 12,693 checker decisions NOT in
# exp022's seed-21 sample (--subsample 2000 --subsample-seed 21 --subsample-complement,
# provably disjoint). Paired: both rules score the identical positions, so the margin is
# measured decision by decision.
#
# POWER, computed before committing the machine. On exp022's sample the effect is 0.061 PR
# with a paired SD of 2.24, so 80% power needs (1.96+0.84)^2 * 2.24^2 / 0.061^2 = 10,704
# decisions. The holdout has 12,693 -- just enough. n=6,000 would give ~50% power, whose
# likely outcome is another ambiguous null; that option was considered and rejected.
#
# HOW ONE RUN SCORES EVERY RULE. At depth 1 a move's searched value does not depend on
# which other moves were searched, so any rule inside a measured run's window and cap can
# be replayed exactly from its dump -- validated in exp022, where the replay reproduced
# both measured arms' PR and evaluation counts to the digit. So this pipeline runs ONE
# reference config, (window 0.10, cap 16, gate 0.08), which contains every frozen rule, and
# the analysis replays the bump, the flat comparator and any later rule off its dumps at no
# extra compute. The gate must match exp022's 0.08 or the replay is not comparable.
#
# COST. 1,720 evaluations per decision measured, ~4.8 s/decision on this iMac -> ~17 h for
# 12,693 decisions. Split into 4 shards so partial results land on disk as they finish;
# shards stride rather than slice, so each is representative and they recombine exactly.
#
# SCOPE. NOT in scope: retuning the bump's constants on the holdout, depth allocation,
# head-to-head play. Local CPU only, no GCP spend.
#
# OUTPUTS. experiments/exp023-holdout/{logs,results}/, committed with the write-up
# (docs/search.qmd, section exp023). dumps/ is gitignored; the analysis writes its
# conclusion to results/.

set -euo pipefail
# Long runs are launched from a snapshot outside the repo -- bash reads a script
# incrementally, so editing the tracked file mid-run corrupts it (that killed
# exp021's sweep). A snapshot cannot find the repo from its own path, so REPO
# overrides it.
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"

EXP="${EXP:-experiments/exp023-holdout}"
CKPT="${CKPT:-experiments/exp018-distill/checkpoints/ep22.pt}"
BENCH="${BENCH:-data/bgsage/money_benchmark/benchmark.json.gz}"
SHARDS="${SHARDS:-4}"
PY="${PY:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)}"

export OMP_WAIT_POLICY=PASSIVE

mkdir -p "$EXP/logs" "$EXP/results" "$EXP/dumps"
LOG="$EXP/logs/exp023.log"
echo "exp023 started $(date -Is) on $(hostname), $SHARDS shards" | tee -a "$LOG"

for ((i = 0; i < SHARDS; i++)); do
    echo "=== shard $i/$SHARDS starting $(date -Is) ===" | tee -a "$LOG"
    "$PY" -u scripts/eval_benchmark_pr.py \
        --benchmark "$BENCH" --checkpoint "$CKPT" \
        --engine-label "ep22_ref_w010_k16_shard${i}" \
        --search-depth 1 --search-k 16 --search-k2 2 \
        --search-threshold 0.10 --search-gate 0.08 \
        --subsample 2000 --subsample-seed 21 --subsample-complement \
        --shard "$i/$SHARDS" \
        --output "$EXP/results" --dump-dir "$EXP/dumps" 2>&1 | tee -a "$LOG"
done

echo "exp023 reference run finished $(date -Is)" | tee -a "$LOG"
echo "Next: scripts/exp023_confirm.py replays the frozen bump and its flat comparator." | tee -a "$LOG"
