#!/bin/bash
# exp009 — on-distribution GNUBG DAgger, round 2 of the loop that produced the
# project's best result (exp008: −1.65 → −1.19 ppg vs GNUBG 2-ply). Same
# structure — synth on-dist labels with the current net, merge with the 4-ply
# archive + a recency window of on-dist caches, fine-tune warm, eval — plus
# the four upgrades motivated by the exp008 post-mortem:
#   1. PARALLEL SYNTH: gnubg-nn is serial per process, so N worker processes
#      (distinct seeds) shard each round's labeling; shards are merged into the
#      round cache. exp008's 10 h synth cap becomes <1 h on a 16-vCPU VM.
#   2. --doubles-policy: doubles half-move policy labels are computed anyway
#      by candidate_equities and were discarded in exp008 (26% of examples,
#      0.85 ppg/game measured leak). Now emitted.
#   3. --select-best + EPOCHS=2: every exp008 round overfit past epoch 1
#      (val top-1 53%→48% within a round); ship the best-val epoch instead of
#      the last.
#   4. Bigger round budget (MAXDEC=120k vs 60k) now that synth is cheap.
#
# COST CONTROL (added after exp009 overran its estimate ~2–4×): each round, before
# any expensive work, scripts/pipeline_budget.py projects the total cost from the
# measured per-round time and stops the loop CLEANLY (at a round boundary, leaving
# a complete resumable checkpoint) if either a budget/wall cap would be crossed or
# the GNUBG score has plateaued. Enforcement is on by default; a stop still emits
# EXP009 COMPLETE so the watchdog/fetcher shut the VM down normally. To resume a
# budget-stopped run, raise MAX_BUDGET_DKK and re-run — completed rounds skip.
#
# Seed: winner of experiments/pipeline_consolidate.sh (SEED_CKPT env),
# default exp008 round_15. ONDIST_BACKFILL pre-fills the recency window with
# exp008's last rounds so early exp009 rounds train on a full-size mix.
# NOTE: backfill caches must match the seed's encoder width — use the
# re-encoded 26ch caches from pipeline_consolidate.sh for a 26ch seed.
#
# Smoke check first:   SMOKE=1 bash experiments/pipeline_exp009.sh
# Full hands-off run:  bash experiments/pipeline_exp009.sh
# Tunables via env: SEED_CKPT ARCHIVE ONDIST_BACKFILL ROUNDS WORKERS MAXDEC
#   MAXMIN EPOCHS LR TEMP WINDOW EVGAMES EVSIMS GNUBG_EVERY MILESTONE_EVERY
#   RATE_DKK_PER_HR MAX_BUDGET_DKK MAX_WALL_HOURS PLATEAU_PATIENCE
#   PLATEAU_MIN_DELTA CALIB_HOURS_PER_ROUND
set -u
cd "$(dirname "$0")/.."
export OMP_WAIT_POLICY=PASSIVE
PY=.venv/bin/python3

EXP=${EXP:-exp009-ondist-dagger}
DIR=experiments/$EXP
SEED_CKPT=${SEED_CKPT:-experiments/exp008-ondist-10x256-2ply/round_15/checkpoints/pretrained_v2.pt}
ARCHIVE=${ARCHIVE:-data/bglab/cache/gnubg4ply_cache_dbl.npz}
ONDIST_BACKFILL=${ONDIST_BACKFILL:-$(ls experiments/exp008-ondist-10x256-2ply/caches/ondist_round_*.npz 2>/dev/null | sort -V | tail -n 4 | tr '\n' ' ')}
ITER0447=experiments/exp005-6x128-800sims/checkpoints/iter_0447.pt
STATE="$DIR/state"
mkdir -p "$DIR/caches" "$DIR/logs" "$STATE"

PLY=${PLY:-2}
if [ "${SMOKE:-0}" = "1" ]; then
  ROUNDS=1; WORKERS=2; MAXDEC=100; MAXMIN=5; EPOCHS=1; LR=3e-4; TEMP=1.0
  WINDOW=4; EVGAMES=4; EVSIMS=10; GNUBG_EVERY=1; MILESTONE_EVERY=99
  GNUBG_CHUNKS=1; GNUBG_CHUNK_GAMES=2; ARENA_CHUNKS=1; ONDIST_BACKFILL=""
  # smoke runs must never trip a cost stop — disable all enforcement
  RATE_DKK_PER_HR=5.0; MAX_BUDGET_DKK=0; MAX_WALL_HOURS=0
  PLATEAU_PATIENCE=0; PLATEAU_MIN_DELTA=0.05; CALIB_HOURS_PER_ROUND=0.1
  echo "===== SMOKE MODE (1 mini round, 2 workers, no backfill, no cost caps) ====="
else
  ROUNDS=${ROUNDS:-25}
  WORKERS=${WORKERS:-$(( $(nproc) > 4 ? $(nproc) - 2 : 2 ))}
  MAXDEC=${MAXDEC:-120000}; MAXMIN=${MAXMIN:-600}
  EPOCHS=${EPOCHS:-2}; LR=${LR:-3e-4}; TEMP=${TEMP:-1.0}; WINDOW=${WINDOW:-4}
  EVGAMES=${EVGAMES:-100}; EVSIMS=${EVSIMS:-50}
  GNUBG_EVERY=${GNUBG_EVERY:-2}; MILESTONE_EVERY=${MILESTONE_EVERY:-6}
  GNUBG_CHUNKS=${GNUBG_CHUNKS:-2}; GNUBG_CHUNK_GAMES=${GNUBG_CHUNK_GAMES:-50}
  ARENA_CHUNKS=${ARENA_CHUNKS:-2}
  # cost control — conservative rate (spot discounts shrink in a busy zone);
  # kr.800 backstop sits under the ~kr.1000/mo bill; plateau stops after 4
  # GNUBG evals with no improvement. Set MAX_BUDGET_DKK=0 to disable the cap.
  RATE_DKK_PER_HR=${RATE_DKK_PER_HR:-5.0}
  MAX_BUDGET_DKK=${MAX_BUDGET_DKK:-800}
  MAX_WALL_HOURS=${MAX_WALL_HOURS:-0}
  PLATEAU_PATIENCE=${PLATEAU_PATIENCE:-4}
  PLATEAU_MIN_DELTA=${PLATEAU_MIN_DELTA:-0.05}
  CALIB_HOURS_PER_ROUND=${CALIB_HOURS_PER_ROUND:-9.0}
fi

echo "===== EXP009 START $(date -u +%F_%T) ====="
echo "seed=$SEED_CKPT ply=$PLY rounds=$ROUNDS workers=$WORKERS maxdec=$MAXDEC"
echo "epochs=$EPOCHS lr=$LR window=$WINDOW backfill=[$ONDIST_BACKFILL]"
echo "cost: rate=kr.$RATE_DKK_PER_HR/h max_budget=kr.$MAX_BUDGET_DKK max_wall=${MAX_WALL_HOURS}h plateau_patience=$PLATEAU_PATIENCE"
for f in "$SEED_CKPT" "$ARCHIVE"; do
  [ -f "$f" ] || { echo "ABORT: missing required file $f"; exit 1; }
done

CUR="$SEED_CKPT"
for r in $(seq 1 "$ROUNDS"); do
  RR=$(printf "%02d" "$r")
  CACHE="$DIR/caches/ondist_round_$RR.npz"
  COMBINED="$DIR/caches/combined_round_$RR.npz"
  NEXT="$DIR/round_$RR/checkpoints/pretrained_v2.pt"

  if [ -f "$NEXT" ]; then
    echo "===== ROUND $RR already complete — skipping $(date -u +%F_%T) ====="
    CUR="$NEXT"; continue
  fi

  # Cost gate: project total, stop cleanly before starting an unaffordable or
  # plateaued round. Fail-open (continue on helper error) so a budget-tool bug
  # never kills a healthy run.
  $PY scripts/pipeline_budget.py check --state "$STATE" \
    --completed $((r - 1)) --total "$ROUNDS" --rate "$RATE_DKK_PER_HR" \
    --max-budget "$MAX_BUDGET_DKK" --max-wall "$MAX_WALL_HOURS" \
    --patience "$PLATEAU_PATIENCE" --min-delta "$PLATEAU_MIN_DELTA" \
    --calib-hours-per-round "$CALIB_HOURS_PER_ROUND"
  STOP_RC=$?
  if [ "$STOP_RC" -eq 10 ]; then
    echo "===== EXP009 STOPPING EARLY at round $RR (see [budget] above) $(date -u +%F_%T) ====="
    break
  elif [ "$STOP_RC" -ne 0 ]; then
    echo "WARNING: budget check errored (rc=$STOP_RC) — continuing"
  fi

  ROUND_T0=$(date +%s)

  if [ -s "$CACHE" ]; then
    echo "===== ROUND $RR SYNTH — reusing existing cache $CACHE $(date -u +%F_%T) ====="
  else
    echo "===== ROUND $RR SYNTH (net=$CUR, ply=$PLY, $WORKERS workers) $(date -u +%F_%T) ====="
    PER_W=$(( (MAXDEC + WORKERS - 1) / WORKERS ))
    for k in $(seq 1 "$WORKERS"); do
      SHARD="$DIR/caches/ondist_round_$RR.w$k.npz"
      [ -s "$SHARD" ] && continue
      OMP_NUM_THREADS=1 $PY scripts/synthesize_ondist_dataset.py \
        --net "$CUR" --out "$SHARD" --ply "$PLY" --doubles-policy \
        --max-decisions "$PER_W" --max-minutes "$MAXMIN" \
        --temperature "$TEMP" --seed $((r * 1000 + k)) \
        > "$DIR/logs/synth_r${RR}_w${k}.log" 2>&1 &
    done
    wait
    SHARDS=$(ls "$DIR"/caches/ondist_round_$RR.w*.npz 2>/dev/null | sort -V | tr '\n' ' ')
    [ -n "$SHARDS" ] || { echo "SYNTH ROUND $RR FAILED (no shards)"; break; }
    $PY scripts/merge_caches.py --out "$CACHE" $SHARDS \
      || { echo "SHARD MERGE ROUND $RR FAILED"; break; }
    rm -f "$DIR"/caches/ondist_round_$RR.w*.npz
  fi

  echo "===== ROUND $RR MERGE $(date -u +%F_%T) ====="
  POOL="$ONDIST_BACKFILL $(ls "$DIR"/caches/ondist_round_*.npz 2>/dev/null | sort -V | tr '\n' ' ')"
  if [ "$WINDOW" -gt 0 ]; then
    ONDIST=$(echo $POOL | tr ' ' '\n' | grep -v '^$' | tail -n "$WINDOW" | tr '\n' ' ')
  else
    ONDIST="$POOL"
  fi
  $PY scripts/merge_caches.py --out "$COMBINED" "$ARCHIVE" $ONDIST \
    || { echo "MERGE ROUND $RR FAILED"; break; }

  echo "===== ROUND $RR TRAIN (warm from $CUR) $(date -u +%F_%T) ====="
  $PY scripts/pretrain_policy.py --experiment-name "$EXP/round_$RR" \
    --base-checkpoint "$CUR" --cache "$COMBINED" \
    --epochs "$EPOCHS" --lr "$LR" --value-weight 1.0 \
    --checkpoint-every 1 --select-best \
    --notes "exp009 round $r: warm from $CUR; archive + window($WINDOW) on-dist @ ${PLY}-ply DAgger, doubles-policy" \
    || { echo "TRAIN ROUND $RR FAILED"; break; }
  [ -f "$NEXT" ] || { echo "ABORT: round $RR produced no checkpoint"; break; }
  rm -f "$COMBINED"

  echo "===== ROUND $RR EVAL $(date -u +%F_%T) ====="
  echo "----- arena: round_$RR vs SEED ($ARENA_CHUNKS x $((EVGAMES / ARENA_CHUNKS)) games) -----"
  for c in $(seq 1 "$ARENA_CHUNKS"); do
    OMP_NUM_THREADS=3 $PY scripts/evaluate.py --checkpoint1 "$NEXT" \
      --checkpoint2 "$SEED_CKPT" --games $((EVGAMES / ARENA_CHUNKS)) \
      --simulations "$EVSIMS" > "$DIR/logs/arena_r${RR}_c${c}.log" 2>&1 &
  done
  wait
  grep -h "Result\|equity" "$DIR"/logs/arena_r${RR}_c*.log | head -8
  if [ $((r % GNUBG_EVERY)) -eq 0 ] || [ "$r" -eq "$ROUNDS" ]; then
    CHUNKS=$GNUBG_CHUNKS
    if [ $((r % MILESTONE_EVERY)) -eq 0 ] || [ "$r" -eq "$ROUNDS" ]; then
      CHUNKS=$((GNUBG_CHUNKS * 2))
    fi
    echo "----- gnubg: round_$RR vs world/2-ply ($CHUNKS x $GNUBG_CHUNK_GAMES games) -----"
    for c in $(seq 1 "$CHUNKS"); do
      OMP_NUM_THREADS=3 $PY scripts/eval_gnubg.py --checkpoint "$NEXT" \
        --games "$GNUBG_CHUNK_GAMES" --simulations "$EVSIMS" \
        --gnubg-level world --no-log-games \
        > "$DIR/logs/gnubg_r${RR}_c${c}.log" 2>&1 &
    done
    wait
    grep -h "Wins:\|Equity/game" "$DIR"/logs/gnubg_r${RR}_c*.log
    # record pooled equity (mean over chunks) for plateau detection
    EQ=$(grep -h "Equity/game" "$DIR"/logs/gnubg_r${RR}_c*.log 2>/dev/null \
         | awk '{s+=$2; n++} END{if (n>0) printf "%.4f", s/n}')
    if [ -n "$EQ" ]; then
      $PY scripts/pipeline_budget.py record-eval --state "$STATE" \
        --round "$r" --equity "$EQ" || true
    fi
  fi

  $PY scripts/pipeline_budget.py record-round --state "$STATE" \
    --seconds $(( $(date +%s) - ROUND_T0 )) || true

  CUR="$NEXT"
  echo "===== ROUND $RR DONE (next=$CUR) $(date -u +%F_%T) ====="
done

if [ "${SMOKE:-0}" != "1" ] && [ "$CUR" != "$SEED_CKPT" ] && [ -f "$ITER0447" ] && [ -f "$CUR" ]; then
  echo "===== FINAL: vs iter_0447 (docs continuity) $(date -u +%F_%T) ====="
  $PY scripts/evaluate.py --checkpoint1 "$CUR" --checkpoint2 "$ITER0447" \
    --games "$EVGAMES" --simulations "$EVSIMS" || echo "ARENA-vs-0447 FAILED"
fi
echo "===== EXP009 COMPLETE $(date -u +%F_%T) ====="
