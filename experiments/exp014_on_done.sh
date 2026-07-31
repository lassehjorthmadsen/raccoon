#!/usr/bin/env bash
# exp014 — R^2 fit-quality benchmark, run once pipeline_exp014.sh's VM training finishes.
# Waits for the VM to self-stop (TERMINATED), pulls exp014-distill checkpoints from GCS,
# then compares held-out R^2 for BOTH arms (0-ply-trained vs 2-ply-trained) on the SAME
# held-out positions — the primary metric (scripts/eval_r2.py). All local/free (CPU
# forward passes over the cache, no gnubg_nn needed for the primary comparison).
#
# Mirrors the *_on_done.sh watcher style (see experiments/exp012b_on_done.sh).
#
# Run detached:
#   nohup setsid experiments/exp014_on_done.sh \
#     > experiments/exp014_on_done.log 2>&1 &
set -uo pipefail
cd /home/lasse/python-projects/raccoon
export OMP_WAIT_POLICY=PASSIVE
PY=.venv/bin/python3

VM=raccoon-gpu
ZONE=europe-west1-b
SRC_EXP=exp014-distill
GCS_SRC=gs://raccoon-training-lhm/experiments/$SRC_EXP
OUT=experiments/exp014-benchmark
HOLDOUT=${HOLDOUT:-0.25}
SPLITSEED=${SPLITSEED:-14}
SAMPLE_TRAIN=${SAMPLE_TRAIN:-200000}
SUPPORT_HOLDOUT=${SUPPORT_HOLDOUT:-200000}  # cap for the per-epoch curve (step 4) —
                                             # not the headline (step 5, always full)
PPG_SANITY_GAMES=${PPG_SANITY_GAMES:-0}   # 0=off; set e.g. 2000 to also run the optional ppg sanity check
WORKERS=${WORKERS:-3}

CACHE0=data/distill/0ply/run1
CACHE2=data/distill/2ply/run1

mkdir -p "$OUT/logs"
LOG="$OUT/logs/exp014.log"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# 1. Wait for the VM to self-stop. pipeline_exp014.sh stops the VM (trap) only after a
#    clean finish or the dead-man; either way, TERMINATED means the run is over.
log "waiting for VM $VM to reach TERMINATED (pipeline_exp014.sh self-stops on completion)"
while true; do
  st=$(gcloud compute instances describe "$VM" --zone="$ZONE" --format='value(status)' 2>/dev/null || echo UNKNOWN)
  [ "$st" = TERMINATED ] && break
  sleep 120
done
log "VM is TERMINATED — exp014 training run is over"

# 2. Pull exp014-distill checkpoints + logs from GCS (additive rsync, no deletes).
log "pulling $SRC_EXP from GCS"
mkdir -p "experiments/$SRC_EXP"
gsutil -m -q rsync -r "$GCS_SRC" "experiments/$SRC_EXP" 2>>"$LOG" || log "WARN: GCS pull had errors"

# 3. Completion guard: a clean 3-epoch run leaves ep3.pt for BOTH arms. Missing =>
#    preempted/truncated => do NOT auto-benchmark a partial net; alert instead.
c0="experiments/$SRC_EXP/scalar_0ply/checkpoints/ep3.pt"
c2="experiments/$SRC_EXP/scalar_2ply/checkpoints/ep3.pt"
if [[ ! -f "$c0" || ! -f "$c2" ]]; then
  log "EXP014_ABORT: exp014 did not finish 3 epochs on both arms (preempted/truncated?)."
  log "  scalar_0ply/ep3.pt: $([ -f "$c0" ] && echo yes || echo MISSING)   scalar_2ply/ep3.pt: $([ -f "$c2" ] && echo yes || echo MISSING)"
  log "  Inspect experiments/$SRC_EXP before benchmarking."
  exit 1
fi

# 4. Supporting per-epoch curve: own-label held-out R^2 at ep1/ep2/ep3 for both arms
#    (monotonicity / overfitting-shape check, NOT the headline selector). Capped via
#    --max-holdout — the real 10x256 net makes a full ~2M-row pass take well over an
#    hour each; 200k rows is already precise enough for a supporting curve (see
#    scripts/eval_r2.py docstring). The headline (step 5) stays at full precision.
log "=== SUPPORTING: own-label held-out R^2 per epoch (capped at $SUPPORT_HOLDOUT rows) ==="
for ep in 1 2 3; do
  e0="experiments/$SRC_EXP/scalar_0ply/checkpoints/ep${ep}.pt"
  e2="experiments/$SRC_EXP/scalar_2ply/checkpoints/ep${ep}.pt"
  [ -f "$e0" ] && $PY scripts/eval_r2.py --checkpoint "$e0" --cache-dir "$CACHE0" \
    --holdout-frac "$HOLDOUT" --split-seed "$SPLITSEED" --max-holdout "$SUPPORT_HOLDOUT" \
    --label "0ply/ep$ep" 2>&1 | tee -a "$LOG"
  [ -f "$e2" ] && $PY scripts/eval_r2.py --checkpoint "$e2" --cache-dir "$CACHE2" \
    --holdout-frac "$HOLDOUT" --split-seed "$SPLITSEED" --max-holdout "$SUPPORT_HOLDOUT" \
    --label "2ply/ep$ep" 2>&1 | tee -a "$LOG"
done

# 5. PRIMARY: full 2x2 + train-sample (overfitting check) on the ep3 winners.
log "=== PRIMARY: exp014 2x2 R^2 table + overfitting check (ep3, n_holdout~2M, sample_train=$SAMPLE_TRAIN) ==="
log "--- 0-ply-trained net (scalar_0ply/ep3) ---"
$PY scripts/eval_r2.py --checkpoint "$c0" --cache-dir "$CACHE0" --cross-cache-dir "$CACHE2" \
  --holdout-frac "$HOLDOUT" --split-seed "$SPLITSEED" --sample-train "$SAMPLE_TRAIN" \
  --label "0ply/ep3 FINAL" 2>&1 | tee -a "$LOG"
log "--- 2-ply-trained net (scalar_2ply/ep3) ---"
$PY scripts/eval_r2.py --checkpoint "$c2" --cache-dir "$CACHE2" --cross-cache-dir "$CACHE0" \
  --holdout-frac "$HOLDOUT" --split-seed "$SPLITSEED" --sample-train "$SAMPLE_TRAIN" \
  --label "2ply/ep3 FINAL" 2>&1 | tee -a "$LOG"

# 6. Optional ppg sanity (off by default; set PPG_SANITY_GAMES to enable) — a guard
#    that the 2-ply-trained net isn't silently broken, NOT part of the hypothesis.
if [ "$PPG_SANITY_GAMES" -gt 0 ]; then
  log "=== OPTIONAL ppg sanity, n=$PPG_SANITY_GAMES vs GNUBG-0-ply ==="
  $PY scripts/eval_gnubg0.py --checkpoint "$c0" --games "$PPG_SANITY_GAMES" --workers "$WORKERS" --ply 0 --label "0ply/ep3 sanity" 2>&1 | tee -a "$LOG"
  $PY scripts/eval_gnubg0.py --checkpoint "$c2" --games "$PPG_SANITY_GAMES" --workers "$WORKERS" --ply 0 --label "2ply/ep3 sanity" 2>&1 | tee -a "$LOG"
fi

log "EXP014_DONE — see $LOG for the R^2 table (own/cross/train per arm)"
