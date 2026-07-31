#!/usr/bin/env bash
# exp017 — does scaling 2-ply value distillation 8M -> 40M (many epochs) lower
# checker-play error on the BGSage benchmark, and fix the pure-race deficit?
#
# ONE arm, all 2-ply (0-ply is settled-worse; not re-run). Trains the exp014 scalar
# recipe (10x256) on the full 40M cache (data/distill/2ply = run1+run2+run3) for up
# to 20 epochs, checkpointing + logging every shard. Anchor for comparison is the
# already-BGSage-scored exp014 ep3 (PR 2.16 / rollout-MSE 0.0044) vs teacher
# GNUBG-2-ply (0.56 / 0.0019).
#
# PRIMARY METRIC (fixed up front): BGSage checker PR (n=14,693), scored OFFLINE per
# epoch by score_exp017.sh. Rollout-tier MSE (n~149k) and the pure-race sub-PR are the
# leading signals. Per-epoch scoring is for post-hoc SELECTION, not early-stopping —
# slow progress is expected at near-parity and is not a stop signal.
#
# COST CONTROLS (spot preemptions WILL happen — this run is watchdog-driven):
#   - cumulative --max-wall-hours 72  → hard cap on *useful* compute across restarts
#     (train_distill.py tracks wall in the checkpoint; ~kr.129 worst case @ ~$0.26/hr).
#   - --epochs 20                      → secondary cap (wall binds first at 40M).
#   - shard-granular --resume auto     → a preemption loses ~minutes, not a ~3.75h epoch.
#   - DONE sentinel + watchdog         → once the cap/epochs are hit, the watchdog stops
#                                        the VM and exits (no dead-man needed; a restart
#                                        would reset a dead-man timer anyway).
#
# This script only BOOTSTRAPS: it writes the resume-params file and pushes it to GCS.
# All heavy lifting (48G cache pull, training, resume) is done on the VM by
# resume_distill.sh, launched + kept alive by watch_vm.sh. Run it once, locally.
set -euo pipefail
cd "$(dirname "$0")/.."

EXPNAME="${EXPNAME:-exp017-distill}"
GCS_BUCKET="${GCS_BUCKET:-gs://raccoon-training-lhm}"
EXP_DIR="experiments/$EXPNAME"
PARAMS="$EXP_DIR/resume_params.env"

mkdir -p "$EXP_DIR"
cat > "$PARAMS" <<EOF
# exp017 launch config — sourced by scripts/resume_distill.sh on every (re)launch.
EXPNAME=$EXPNAME
CACHE_DIR=data/distill/2ply
GCS_CACHE=$GCS_BUCKET/data/distill/2ply
VALUE_HEAD=scalar
EPOCHS=20
LR=1e-3
EVAL_EVERY=40
EVAL_GAMES=40
MAXWALL=72
SHUFFLE_SEED=20
GNUBG_PLY=0
EOF

echo "wrote $PARAMS:"; sed 's/^/    /' "$PARAMS"

gcloud storage cp "$PARAMS" "$GCS_BUCKET/experiments/$EXPNAME/resume_params.env" \
  && echo "pushed params to GCS" \
  || { echo "WARN: GCS push failed — resume_distill.sh will need the file present on the VM"; }

cat <<EOF

===== exp017 bootstrap complete =====
Before launching, make sure the VM has this code:
    git push                         # from here
    (on VM) cd ~/raccoon && git pull # picks up train_distill.py resume + resume_distill.sh

Then start the watchdog LOCALLY (brings the VM up, launches training, auto-resumes on
preemption, and stops the VM + exits once DONE):

    nohup systemd-inhibit --what=sleep:idle --who=watch_exp017 --why="exp017 spot watchdog" \\
      env RESUME_SCRIPT=scripts/resume_distill.sh \\
      bash scripts/watch_vm.sh $EXPNAME > /tmp/watch_exp017.log 2>&1 &

Monitor:
    tail -f /tmp/watch_exp017.log                         # watchdog (local)
    gcloud compute ssh raccoon-gpu --zone=europe-west1-b -- 'tmux capture-pane -pt train | tail -30'

When DONE appears, score every epoch offline:
    bash experiments/score_exp017.sh
EOF
