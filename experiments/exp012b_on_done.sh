#!/usr/bin/env bash
# exp012b — auto-benchmark exp011b's winner once the T4 run finishes.
# Waits for the VM to self-stop (TERMINATED), pulls exp011b's checkpoints from GCS,
# runs an n=1000 0-ply sweep over the per-epoch checkpoints to pick the winner, then
# benchmarks that winner: n=6000 vs GNUBG-0-ply + n=1000 vs GNUBG-2-ply (= exp012b).
#
# All evals run LOCALLY on the iMac (free); the VM is NOT restarted. Mirrors the
# *_on_done.sh watcher style. exp012b is the completed-net parallel to exp012
# (which benchmarked exp011's truncated net at -0.087 0-ply / -0.124 2-ply).
#
# Run detached:
#   nohup setsid experiments/exp012b_on_done.sh \
#     > experiments/exp012b_on_done.log 2>&1 &
set -uo pipefail
cd /home/lasse/python-projects/raccoon
export OMP_WAIT_POLICY=PASSIVE          # avoid CPU spin-collapse under contention
PY=.venv/bin/python3

VM=raccoon-gpu
ZONE=europe-west1-b
SRC_EXP=exp011b-distill                  # the training run being benchmarked
GCS_SRC=gs://raccoon-training-lhm/experiments/$SRC_EXP
OUT=experiments/exp012b-benchmark
SWEEP_GAMES=${SWEEP_GAMES:-1000}         # pick the winner (CI ~±0.11)
CONFIRM_GAMES=${CONFIRM_GAMES:-6000}     # definitive 0-ply number (CI ~±0.046), matches exp012(a)
PLY2_GAMES=${PLY2_GAMES:-1000}           # 2-ply gap, tighter than exp012's n=500
WORKERS=${WORKERS:-3}

mkdir -p "$OUT/logs"
LOG="$OUT/logs/exp012b.log"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
# extract the signed ppg from an eval_gnubg0 result line ("... +0.0150 ppg vs ...")
parse_ppg() { grep -oE '[-+][0-9]+\.[0-9]+ ppg' | grep -oE '[-+][0-9]+\.[0-9]+' | head -1; }

# 1. Wait for the VM to self-stop. exp011b's pipeline stops the VM (trap) only after a
#    clean finish or the dead-man; either way, TERMINATED means the run is over.
log "waiting for VM $VM to reach TERMINATED (exp011b self-stops on completion)"
while true; do
  st=$(gcloud compute instances describe "$VM" --zone="$ZONE" --format='value(status)' 2>/dev/null || echo UNKNOWN)
  [ "$st" = TERMINATED ] && break
  sleep 120
done
log "VM is TERMINATED — exp011b run is over"

# 2. Pull exp011b checkpoints + logs from GCS (additive rsync, no deletes).
#    gsutil rsync requires the destination dir to already exist — create it first.
log "pulling $SRC_EXP from GCS"
mkdir -p "experiments/$SRC_EXP"
gsutil -m -q rsync -r "$GCS_SRC" "experiments/$SRC_EXP" 2>>"$LOG" || log "WARN: GCS pull had errors"

# 3. Completion guard: a clean 3-epoch run leaves ep3.pt for BOTH arms. Missing =>
#    preempted/truncated => do NOT auto-benchmark a partial net; alert instead.
s3="experiments/$SRC_EXP/scalar/checkpoints/ep3.pt"
o3="experiments/$SRC_EXP/outcomes6/checkpoints/ep3.pt"
if [[ ! -f "$s3" || ! -f "$o3" ]]; then
  log "EXP012B_ABORT: exp011b did not finish 3 epochs on both arms (preempted/truncated?)."
  log "  scalar/ep3.pt: $([ -f "$s3" ] && echo yes || echo MISSING)   outcomes6/ep3.pt: $([ -f "$o3" ] && echo yes || echo MISSING)"
  log "  Inspect experiments/$SRC_EXP before benchmarking. Not touching gnubg."
  exit 1
fi

# 4. Sweep the per-epoch checkpoints vs 0-ply; log the epoch curve; pick the winner.
shopt -s nullglob
cands=(experiments/$SRC_EXP/scalar/checkpoints/ep*.pt experiments/$SRC_EXP/outcomes6/checkpoints/ep*.pt)
log "=== SWEEP n=$SWEEP_GAMES vs GNUBG-0-ply over ${#cands[@]} epoch checkpoints ==="
best_ppg=-999; best_ckpt=""
for ck in "${cands[@]}"; do
  out=$($PY scripts/eval_gnubg0.py --checkpoint "$ck" --games "$SWEEP_GAMES" --workers "$WORKERS" --ply 0 --label "sweep" 2>>"$LOG")
  ppg=$(printf '%s\n' "$out" | parse_ppg)
  log "  ${ck#experiments/$SRC_EXP/} -> ${ppg:-PARSE_FAIL} ppg (0-ply, n=$SWEEP_GAMES)"
  if [ -n "$ppg" ] && awk "BEGIN{exit !($ppg > $best_ppg)}"; then best_ppg=$ppg; best_ckpt=$ck; fi
done
[ -z "$best_ckpt" ] && { log "EXP012B_ABORT: sweep produced no parseable ppg"; exit 1; }
log "WINNER: ${best_ckpt#experiments/$SRC_EXP/}  (sweep 0-ply ppg=$best_ppg)"

# 5. Benchmark the winner = exp012b.
log "=== exp012b(a): n=$CONFIRM_GAMES vs GNUBG-0-ply on winner ==="
$PY scripts/eval_gnubg0.py --checkpoint "$best_ckpt" --games "$CONFIRM_GAMES" --workers "$WORKERS" --ply 0 --label "exp012b winner 0-ply" 2>&1 | tee -a "$LOG"
log "=== exp012b(b): n=$PLY2_GAMES vs GNUBG-2-ply on winner ==="
$PY scripts/eval_gnubg0.py --checkpoint "$best_ckpt" --games "$PLY2_GAMES" --workers "$WORKERS" --ply 2 --label "exp012b winner 2-ply" 2>&1 | tee -a "$LOG"

log "EXP012B_DONE  winner=${best_ckpt#experiments/$SRC_EXP/}  (compare to exp011 control: -0.087 0-ply / -0.124 2-ply)"
