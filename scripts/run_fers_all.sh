#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
for c in eval_birds terrain_ir_canopy terrain_ir_birds; do
  echo "######## FERS track: $c ########"
  rm -rf "$REPO/radar/fers_track/$c"
  CLIP="data/clips/$c" DWELLS=90 bash "$REPO/scripts/run_fers_track.sh"
done
echo "FERSALLDONE"
