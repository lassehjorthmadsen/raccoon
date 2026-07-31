#!/bin/bash
# exp008 — on-distribution GNUBG expert iteration (DAgger). Warm-start the v5
# supervised seed (10x256, GNUBG-4ply distilled) and loop:
#   synth on-dist GNUBG labels (the net's OWN play) -> merge with the 4-ply
#   archive + all prior on-dist rounds -> pretrain_policy (warm) -> eval.
# Pure supervised distillation throughout: continuous GNUBG money-equity value
# targets + GNUBG soft-policy targets, so the exp007 self-play "regression trap"
# cannot recur, while on-distribution states fill the coverage gap behind the
# -1.59 ppg plateau. Each round checkpoints + saves its cache, so the run is
# safe to kill at any time. Stage markers (=====) for log scans; chunks (-----).
#
# Threading: synth is single-process (gnubg_nn serial). Training uses all cores;
# OMP_WAIT_POLICY=PASSIVE stops the idle main process busy-spin-collapsing.
#
# Smoke check first:   SMOKE=1 bash experiments/pipeline_exp008.sh
# Full hands-off run:  bash experiments/pipeline_exp008.sh
# Tunable via env: ROUNDS MAXDEC MAXMIN EPOCHS LR TEMP EVGAMES EVSIMS GNUBG_EVERY
set -u
cd /home/lasse/python-projects/raccoon
export OMP_WAIT_POLICY=PASSIVE
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
PY=.venv/bin/python3

SEED=experiments/pretrain-gnubg-v5-10x256-dbl/checkpoints/pretrained_v2.pt
ARCHIVE=data/bglab/cache/gnubg4ply_cache_dbl.npz
ITER0447=experiments/exp005-6x128-800sims/checkpoints/iter_0447.pt
EXP=exp008-ondist-10x256-2ply
DIR=experiments/$EXP
mkdir -p "$DIR/caches" "$DIR/logs"

PLY=2
if [ "${SMOKE:-0}" = "1" ]; then
  ROUNDS=1; MAXDEC=200; MAXMIN=10; EPOCHS=1; LR=3e-4
  TEMP=1.0; EVGAMES=10; EVSIMS=25; GNUBG_EVERY=1; WINDOW=0
  echo "===== SMOKE MODE (mini round; archive skipped in merge) ====="
else
  ROUNDS=${ROUNDS:-15}; MAXDEC=${MAXDEC:-60000}; MAXMIN=${MAXMIN:-600}
  EPOCHS=${EPOCHS:-4}; LR=${LR:-3e-4}; TEMP=${TEMP:-1.0}; WINDOW=${WINDOW:-4}
  EVGAMES=${EVGAMES:-100}; EVSIMS=${EVSIMS:-50}; GNUBG_EVERY=${GNUBG_EVERY:-4}
fi
# WINDOW>0 trains each round on the archive + the last WINDOW on-dist rounds
# (bounds round time; recency-weighted DAgger). WINDOW=0 aggregates all rounds.

echo "===== EXP008 START $(date -u +%F_%T) ====="
echo "seed=$SEED ply=$PLY rounds=$ROUNDS maxdec=$MAXDEC maxmin=$MAXMIN epochs=$EPOCHS lr=$LR temp=$TEMP"
for f in "$SEED" "$ARCHIVE" "$ITER0447"; do
  [ -f "$f" ] || { echo "ABORT: missing required file $f"; exit 1; }
done

CUR="$SEED"
for r in $(seq 1 "$ROUNDS"); do
  RR=$(printf "%02d" "$r")
  CACHE="$DIR/caches/ondist_round_$RR.npz"
  COMBINED="$DIR/caches/combined_round_$RR.npz"
  NEXT="$DIR/round_$RR/checkpoints/pretrained_v2.pt"

  # Resume support: skip a fully-finished round, and reuse an already-completed
  # synth cache (an existing $CACHE is always complete — synth writes atomically).
  if [ -f "$NEXT" ]; then
    echo "===== ROUND $RR already complete ($NEXT) — skipping $(date -u +%F_%T) ====="
    CUR="$NEXT"; continue
  fi

  if [ -s "$CACHE" ]; then
    echo "===== ROUND $RR SYNTH — reusing existing cache $CACHE $(date -u +%F_%T) ====="
  else
    echo "===== ROUND $RR SYNTH (net=$CUR, ply=$PLY) $(date -u +%F_%T) ====="
    $PY scripts/synthesize_ondist_dataset.py --net "$CUR" --out "$CACHE" \
      --ply "$PLY" --max-decisions "$MAXDEC" --max-minutes "$MAXMIN" \
      --temperature "$TEMP" --seed "$r" \
      || { echo "SYNTH ROUND $RR FAILED"; break; }
  fi

  echo "===== ROUND $RR MERGE $(date -u +%F_%T) ====="
  if [ "${SMOKE:-0}" = "1" ]; then
    MERGE_INPUTS="$CACHE"
  else
    ONDIST=$(ls "$DIR"/caches/ondist_round_*.npz 2>/dev/null | sort -V)
    if [ "$WINDOW" -gt 0 ]; then ONDIST=$(echo "$ONDIST" | tail -n "$WINDOW"); fi
    MERGE_INPUTS="$ARCHIVE $ONDIST"
  fi
  $PY scripts/merge_caches.py --out "$COMBINED" $MERGE_INPUTS \
    || { echo "MERGE ROUND $RR FAILED"; break; }

  echo "===== ROUND $RR TRAIN (warm from $CUR) $(date -u +%F_%T) ====="
  $PY scripts/pretrain_policy.py --experiment-name "$EXP/round_$RR" \
    --base-checkpoint "$CUR" --cache "$COMBINED" \
    --epochs "$EPOCHS" --lr "$LR" --value-weight 1.0 --checkpoint-every 1 \
    --notes "exp008 round $r: warm from $CUR; archive + ondist(1..$r) @ ${PLY}-ply DAgger" \
    || { echo "TRAIN ROUND $RR FAILED"; break; }
  [ -f "$NEXT" ] || { echo "ABORT: round $RR produced no checkpoint"; break; }

  echo "===== ROUND $RR EVAL $(date -u +%F_%T) ====="
  echo "----- arena: round_$RR vs v5 SEED (climb-or-regress; exp007 test) -----"
  $PY scripts/evaluate.py --checkpoint1 "$NEXT" --checkpoint2 "$SEED" \
    --games "$EVGAMES" --simulations "$EVSIMS" || echo "ARENA-vs-SEED FAILED"
  if [ $((r % GNUBG_EVERY)) -eq 0 ] || [ "$r" -eq "$ROUNDS" ]; then
    echo "----- arena: round_$RR vs iter_0447 (internal bar) -----"
    $PY scripts/evaluate.py --checkpoint1 "$NEXT" --checkpoint2 "$ITER0447" \
      --games "$EVGAMES" --simulations "$EVSIMS" || echo "ARENA-vs-0447 FAILED"
    echo "----- gnubg: round_$RR vs world/2-ply (external M6) -----"
    $PY scripts/eval_gnubg.py --checkpoint "$NEXT" --games "$EVGAMES" \
      --simulations "$EVSIMS" --gnubg-level world --no-log-games || echo "GNUBG FAILED"
  fi

  CUR="$NEXT"
  echo "===== ROUND $RR DONE (next=$CUR) $(date -u +%F_%T) ====="
done
echo "===== EXP008 COMPLETE $(date -u +%F_%T) ====="
