#!/usr/bin/env bash
# exp018 — at 40M 2-ply labels, does the six-outcome value head match the scalar
# head, and does the curve keep improving past exp017's wall-clock cutoff?
#
# Identical to exp017 in every respect except the value head (outcomes6, not scalar)
# and the epoch budget (24, not 20-capped-at-14). Same 40M corpus, same 10x256
# recipe, same lr, same shard order — see SHUFFLE_SEED below.
#
# WHY THIS RUN EXISTS. exp017's deliverable is scalar-headed: it emits a cubeless
# equity and nothing else, so it cannot feed cube decisions, which need the
# win/gammon/backgammon split. The 2-ply shards already carry valid `outcomes6`
# labels alongside `equity`, so the cube-capable net costs no new data — only GPU
# time. That makes this run the deliverable *and* two pre-registered readings:
#
#   Q1 (head)   — does outcomes6 match scalar at 40M? exp018 ep_k vs exp017 ep_k,
#                 k <= 14, both already scored on the same benchmark.
#   Q2 (epochs) — exp017 stopped on its 72h wall cap with PR still descending
#                 (-0.0082/epoch over ep6-14, t=-4.0), not on convergence. Does
#                 PR at ep24 beat ep14? Fit predicts ~0.09 PR against an observed
#                 epoch-to-epoch scatter of SD ~0.016, so this is powered.
#
# PRIMARY METRIC (fixed up front): BGSage checker PR (n=14,693), scored OFFLINE
# per epoch by score_exp018.sh. Rollout-tier MSE (n~149k candidates) is SECONDARY
# for Q1 and read asymmetrically, because it is literally the *scalar* arm's
# training loss and so favours scalar at equal strength:
#     outcomes6 wins MSE  -> strong evidence (it won despite the handicap)
#     scalar    wins MSE  -> ambiguous (the handicap alone could explain it)
# PR is the objective-neutral tiebreak and carries the strength claim either way.
# Cluster the MSE bootstrap at decision and game level (exp016_paired_mse.py does
# this): the ~149k candidates are 2-773 per decision and are not independent.
#
# ATTRIBUTION CAVEAT, on the record: train_distill.py does not seed model init, and
# this is one run per arm, so only a Q1 difference clearly exceeding the ~0.016 PR
# scatter reads as a head effect. A small gap (exp016 saw 2.62 vs 2.75 at 8M) would
# not be separable from init noise.
#
# NO HOLDOUT, deliberately. Q1 needs an exact 40M-vs-40M comparison, and per-epoch
# BGSage scoring is already an overfitting monitor on *external* data — strictly
# better than an internal holdout for this purpose, since overfitting would show up
# as the deliverable metric turning back up.
#
# COST CONTROLS (spot preemptions WILL happen — watchdog-driven, as exp017 was):
#   - --epochs 24                      -> the intended binding constraint (~109h at
#                                         exp017's measured 86 shards x ~190s/shard).
#   - cumulative --max-wall-hours 200  -> insurance only, NOT the intended stop.
#                                         exp017 set this to 72 and the wall bound
#                                         first at ep14 — the truncation Q2 exists to
#                                         undo. Expected ~kr.150, worst case ~kr.280.
#   - n1-standard-4                    -> exp017 ran n1-standard-16 and was never CPU
#                                         bound (shards are uncompressed npz, so the
#                                         loader is I/O + memcpy). See resize below.
#   - shard-granular --resume auto     -> a preemption loses ~minutes, not a ~4.5h epoch.
#   - DONE sentinel + watchdog         -> stops the VM and exits once epochs/cap hit.
#
# This script only BOOTSTRAPS: it writes the resume-params file and pushes it to GCS.
# All heavy lifting (48G cache pull, training, resume) is done on the VM by
# resume_distill.sh, launched + kept alive by watch_vm.sh. Run it once, locally.
set -euo pipefail
cd "$(dirname "$0")/.."

EXPNAME="${EXPNAME:-exp018-distill}"
GCS_BUCKET="${GCS_BUCKET:-gs://raccoon-training-lhm}"
ZONE="${ZONE:-europe-west1-b}"
VM="${VM:-raccoon-gpu}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-4}"
EXP_DIR="experiments/$EXPNAME"
PARAMS="$EXP_DIR/resume_params.env"

mkdir -p "$EXP_DIR"
cat > "$PARAMS" <<EOF
# exp018 launch config — sourced by scripts/resume_distill.sh on every (re)launch.
EXPNAME=$EXPNAME
CACHE_DIR=data/distill/2ply
GCS_CACHE=$GCS_BUCKET/data/distill/2ply
VALUE_HEAD=outcomes6
EPOCHS=24
LR=1e-3
EVAL_EVERY=40
EVAL_GAMES=40
MAXWALL=200
# SHUFFLE_SEED must stay 20: epoch_order() is deterministic in (seed, epoch), so
# reusing exp017's seed means exp018 sees the identical shards in the identical
# order at every epoch. That is what makes Q1 a matched A/B rather than two
# unrelated runs — do not "refresh" it.
SHUFFLE_SEED=20
GNUBG_PLY=0
EOF

echo "wrote $PARAMS:"; sed 's/^/    /' "$PARAMS"

gcloud storage cp "$PARAMS" "$GCS_BUCKET/experiments/$EXPNAME/resume_params.env" \
  && echo "pushed params to GCS" \
  || { echo "WARN: GCS push failed — resume_distill.sh will need the file present on the VM"; }

CURRENT_MT="$(gcloud compute instances describe "$VM" --zone="$ZONE" \
                --format='value(machineType.basename())' 2>/dev/null || echo unknown)"

cat <<EOF

===== exp018 bootstrap complete =====
VM machine type is currently: $CURRENT_MT (target: $MACHINE_TYPE)
EOF

if [ "$CURRENT_MT" != "$MACHINE_TYPE" ] && [ "$CURRENT_MT" != "unknown" ]; then
cat <<EOF
Resize it before launching (the VM must be TERMINATED; this run does not need 16 vCPU):
    gcloud compute instances set-machine-type $VM --zone=$ZONE --machine-type=$MACHINE_TYPE
EOF
fi

cat <<EOF
Make sure the VM has this code:
    git push                         # from here
    (on VM) cd ~/raccoon && git pull

Then start the watchdog LOCALLY (brings the VM up, launches training, auto-resumes on
preemption, and stops the VM + exits once DONE):

    nohup systemd-inhibit --what=sleep:idle --who=watch_exp018 --why="exp018 spot watchdog" \\
      env RESUME_SCRIPT=scripts/resume_distill.sh \\
      bash scripts/watch_vm.sh $EXPNAME > /tmp/watch_exp018.log 2>&1 &

Monitor:
    tail -f /tmp/watch_exp018.log                         # watchdog (local)
    gcloud compute ssh $VM --zone=$ZONE -- 'tmux capture-pane -pt train | tail -30'

When DONE appears, score every epoch offline (~2h on the iMac CPU for 24 epochs):
    PULL=1 bash experiments/score_exp018.sh
EOF
