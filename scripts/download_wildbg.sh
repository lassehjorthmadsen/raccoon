#!/bin/bash
# Download wildbg-training rollout-labeled positions for supervised pretraining.
#
# Pulls the latest clean folders (0021 = contact, 0022 = race) from
# https://github.com/carsten-wenderdel/wildbg-training (CC0). Earlier folders
# 0009/0010/0012/0015 have known encoding bugs and are intentionally skipped.
#
# Usage: ./scripts/download_wildbg.sh

set -euo pipefail

BASE_URL="https://raw.githubusercontent.com/carsten-wenderdel/wildbg-training/main/data"
DEST="${DEST:-data/wildbg}"

mkdir -p "$DEST/0021" "$DEST/0022"

echo "Downloading wildbg-training data into $DEST/ ..."
curl -fL --progress-bar "$BASE_URL/0021/contact.csv" -o "$DEST/0021/contact.csv"
curl -fL --progress-bar "$BASE_URL/0022/race.csv"    -o "$DEST/0022/race.csv"

echo
echo "Done. Row counts (incl. header):"
wc -l "$DEST/0021/contact.csv" "$DEST/0022/race.csv"
