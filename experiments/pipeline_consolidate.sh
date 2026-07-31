#!/bin/bash
# Phase A consolidation (pre-exp009): train 10x256 FROM SCRATCH on ALL
# accumulated GNUBG labels (4-ply archive 164k + exp008's 15 on-dist rounds
# 680k ~= 844k), as an A/B of the encoder:
#   arm 17ch — the production base-only encoder (v5 / exp008 lineage)
#   arm 26ch — the Stage-6-validated Fix-N handcrafted-feature encoder
#              (caches upgraded losslessly via scripts/reencode_cache.py)
# Each arm is evaluated vs the v5 seed, vs exp008 round_15 (the incumbent
# exp009 seed), and vs GNUBG 2-ply (4x50-game chunks, n=200). Winner (if it
# beats round_15) becomes the exp009 warm-start seed.
#
# Extra stages, off by default:
#   BASELINE=1  — also benchmark round_15 vs GNUBG at n=200 (fair comparator)
#   SWEEP_CKPT=path  — sims sweep {1,25,50,100} x 100 games vs GNUBG
#   PLYBENCH=1  — price 3-ply/4-ply labeling (scripts/bench_gnubg_ply.py)
#
# Smoke check first:   SMOKE=1 bash experiments/pipeline_consolidate.sh
# Full run:            bash experiments/pipeline_consolidate.sh
set -u
cd "$(dirname "$0")/.."
export OMP_WAIT_POLICY=PASSIVE
PY=.venv/bin/python3

SEED_V5=experiments/pretrain-gnubg-v5-10x256-dbl/checkpoints/pretrained_v2.pt
ROUND15=experiments/exp008-ondist-10x256-2ply/round_15/checkpoints/pretrained_v2.pt
ARCHIVE=data/bglab/cache/gnubg4ply_cache_dbl.npz
ONDIST_DIR=experiments/exp008-ondist-10x256-2ply/caches
EXP=pretrain-consol
DIR=experiments/$EXP
mkdir -p "$DIR/caches/26ch" "$DIR/logs"

EPOCHS=${EPOCHS:-12}; LR=${LR:-1e-3}; EVGAMES=${EVGAMES:-100}; EVSIMS=${EVSIMS:-50}
GNUBG_CHUNKS=${GNUBG_CHUNKS:-4}; GNUBG_CHUNK_GAMES=${GNUBG_CHUNK_GAMES:-50}
ARENA_CHUNKS=${ARENA_CHUNKS:-2}
if [ "${SMOKE:-0}" = "1" ]; then
  EPOCHS=1; EVGAMES=4; EVSIMS=10; GNUBG_CHUNKS=1; GNUBG_CHUNK_GAMES=2
  ARENA_CHUNKS=1
  echo "===== SMOKE MODE (2k-row slices, 1 epoch, mini evals) ====="
fi

arena () {
  local label=$1 ckpt=$2 opp=$3 lg
  echo "----- arena: $label ($ARENA_CHUNKS x $((EVGAMES / ARENA_CHUNKS)) games) -----"
  for c in $(seq 1 "$ARENA_CHUNKS"); do
    lg="$DIR/logs/arena_${label// /_}_c${c}.log"
    # resume-safe: a chunk log with a result line survived a preemption
    if grep -q "P1 wins" "$lg" 2>/dev/null; then
      echo "  chunk $c already done — skipping"
      continue
    fi
    OMP_NUM_THREADS=3 $PY scripts/evaluate.py --checkpoint1 "$ckpt" \
      --checkpoint2 "$opp" --games $((EVGAMES / ARENA_CHUNKS)) \
      --simulations "$EVSIMS" > "$lg" 2>&1 &
  done
  wait
  grep -h "P1 wins" "$DIR"/logs/arena_${label// /_}_c*.log
}

echo "===== CONSOLIDATE START $(date -u +%F_%T) ====="
for f in "$SEED_V5" "$ROUND15" "$ARCHIVE"; do
  [ -f "$f" ] || { echo "ABORT: missing required file $f"; exit 1; }
done

if [ "${SMOKE:-0}" = "1" ]; then
  SRC17="$DIR/caches/smoke_slice.npz"
  [ -f "$SRC17" ] || $PY - "$ARCHIVE" "$ONDIST_DIR/ondist_round_15.npz" "$SRC17" <<'EOF'
import sys
import numpy as np
srcs, out = sys.argv[1:3], sys.argv[3]
import json
parts = {}
for s in sys.argv[1:3]:
    with np.load(s, allow_pickle=True) as z:
        for k in ("observations", "policy_actions", "policy_probs", "value_targets"):
            parts.setdefault(k, []).append(z[k][:1000])
np.savez_compressed(out, meta=np.array(json.dumps({"source": "smoke_slice"})),
                    **{k: np.concatenate(v) for k, v in parts.items()})
print(f"wrote {out}")
EOF
  SRCS17="$SRC17"
else
  SRCS17="$ARCHIVE $(ls "$ONDIST_DIR"/ondist_round_*.npz | sort -V | tr '\n' ' ')"
fi

echo "===== STAGE 1: merge 17ch cache $(date -u +%F_%T) ====="
C17="$DIR/caches/combined_17ch.npz"
[ -f "$C17" ] || $PY scripts/merge_caches.py --out "$C17" $SRCS17 \
  || { echo "MERGE17 FAILED"; exit 1; }

echo "===== STAGE 2: re-encode to 26ch $(date -u +%F_%T) ====="
SRCS26=""
for src in $SRCS17; do
  stem=$(basename "$src" .npz)
  out="$DIR/caches/26ch/${stem}_26ch.npz"
  [ -f "$out" ] || $PY scripts/reencode_cache.py --out-dir "$DIR/caches/26ch" --verify "$src" \
    || { echo "REENCODE $src FAILED"; exit 1; }
  SRCS26="$SRCS26 $out"
done

echo "===== STAGE 3: merge 26ch cache $(date -u +%F_%T) ====="
C26="$DIR/caches/combined_26ch.npz"
[ -f "$C26" ] || $PY scripts/merge_caches.py --out "$C26" $SRCS26 \
  || { echo "MERGE26 FAILED"; exit 1; }

echo "===== STAGE 4: random-init bases $(date -u +%F_%T) ====="
B17="$DIR/checkpoints/base_random_17ch.pt"
B26="$DIR/checkpoints/base_random_26ch.pt"
mkdir -p "$DIR/checkpoints"
[ -f "$B17" ] || $PY - "$B17" <<'EOF'
import sys
import torch
from raccoon.env.encoder import FEATURE_GROUPS
from raccoon.model.network import RaccoonNet
torch.manual_seed(42)
net = RaccoonNet(channels=256, num_blocks=10,
                 feature_channels=list(FEATURE_GROUPS["base"]))
torch.save({"model_state_dict": net.state_dict(), "config": net.config,
            "step": -1, "pretrain_info": {"note": "consol base 10x256 17ch"}},
           sys.argv[1])
print(f"saved {sys.argv[1]} in_channels={net.config['in_channels']}")
EOF
[ -f "$B26" ] || $PY - "$B26" <<'EOF'
import sys
import torch
from raccoon.model.network import RaccoonNet
torch.manual_seed(42)
net = RaccoonNet(channels=256, num_blocks=10)
torch.save({"model_state_dict": net.state_dict(), "config": net.config,
            "step": -1, "pretrain_info": {"note": "consol base 10x256 26ch"}},
           sys.argv[1])
print(f"saved {sys.argv[1]} in_channels={net.config['in_channels']}")
EOF

train_arm () {
  local arm=$1 base=$2 cache=$3
  local ckpt="experiments/$EXP/$arm/checkpoints/pretrained_v2.pt"
  echo "===== STAGE 5: train $arm $(date -u +%F_%T) ====="
  if [ -f "$ckpt" ]; then
    echo "  $ckpt exists — skipping train"
  else
    $PY scripts/pretrain_policy.py --experiment-name "$EXP/$arm" \
      --base-checkpoint "$base" --cache "$cache" \
      --epochs "$EPOCHS" --lr "$LR" --value-weight 1.0 \
      --checkpoint-every 2 --select-best \
      --notes "consolidation from scratch on archive+ondist(1..15), $arm encoder" \
      || { echo "TRAIN $arm FAILED"; return 1; }
  fi
  echo "===== STAGE 6: eval $arm $(date -u +%F_%T) ====="
  arena "${arm}_vs_v5seed" "$ckpt" "$SEED_V5"
  arena "${arm}_vs_round15" "$ckpt" "$ROUND15"
  echo "----- gnubg: $arm vs world/2-ply, $GNUBG_CHUNKS x $GNUBG_CHUNK_GAMES games -----"
  for c in $(seq 1 "$GNUBG_CHUNKS"); do
    if grep -q "Equity/game" "$DIR/logs/gnubg_${arm}_chunk${c}.log" 2>/dev/null; then
      echo "  chunk $c already done — skipping"
      continue
    fi
    OMP_NUM_THREADS=3 $PY scripts/eval_gnubg.py --checkpoint "$ckpt" \
      --games "$GNUBG_CHUNK_GAMES" --simulations "$EVSIMS" \
      --gnubg-level world --no-log-games \
      > "$DIR/logs/gnubg_${arm}_chunk${c}.log" 2>&1 &
  done
  wait
  grep -h "Equity/game\|Wins:" "$DIR"/logs/gnubg_${arm}_chunk*.log
}

train_arm 17ch "$B17" "$C17" || exit 1
train_arm 26ch "$B26" "$C26" || exit 1

if [ "${BASELINE:-0}" = "1" ]; then
  echo "===== STAGE 7: round_15 GNUBG baseline n=$((GNUBG_CHUNKS*GNUBG_CHUNK_GAMES)) $(date -u +%F_%T) ====="
  for c in $(seq 1 "$GNUBG_CHUNKS"); do
    if grep -q "Equity/game" "$DIR/logs/gnubg_round15_chunk${c}.log" 2>/dev/null; then
      echo "  chunk $c already done — skipping"
      continue
    fi
    OMP_NUM_THREADS=3 $PY scripts/eval_gnubg.py --checkpoint "$ROUND15" \
      --games "$GNUBG_CHUNK_GAMES" --simulations "$EVSIMS" \
      --gnubg-level world --no-log-games \
      > "$DIR/logs/gnubg_round15_chunk${c}.log" 2>&1 &
  done
  wait
  grep -h "Equity/game\|Wins:" "$DIR"/logs/gnubg_round15_chunk*.log
fi

if [ -n "${SWEEP_CKPT:-}" ]; then
  echo "===== STAGE 8: sims sweep on $SWEEP_CKPT $(date -u +%F_%T) ====="
  for sims in 1 25 50 100; do
    echo "----- sweep: $sims sims, 100 games -----"
    $PY scripts/eval_gnubg.py --checkpoint "$SWEEP_CKPT" --games 100 \
      --simulations "$sims" --gnubg-level world --no-log-games \
      || echo "SWEEP sims=$sims FAILED"
  done
fi

if [ "${PLYBENCH:-0}" = "1" ]; then
  echo "===== STAGE 9: gnubg ply pricing $(date -u +%F_%T) ====="
  $PY scripts/bench_gnubg_ply.py --net "$ROUND15" --positions 30 --plies 0 2 3 4 \
    || echo "PLYBENCH FAILED"
fi

echo "===== CONSOLIDATE COMPLETE $(date -u +%F_%T) ====="
