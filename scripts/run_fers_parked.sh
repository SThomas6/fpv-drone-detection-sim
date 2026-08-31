#!/usr/bin/env bash
# The same flight with the vehicle parked three different ways.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
for y in 40 70; do
  echo "######## array yaw ${y} deg ########"
  rm -rf "$REPO/radar/fers_track/terrain_ir_sweep_yaw$y"
  CLIP=data/clips/terrain_ir_sweep DWELLS=40 ARRAY_YAW=$y \
    OUT_SUFFIX="_yaw$y" bash "$REPO/scripts/run_fers_track.sh"
done
echo "PARKEDDONE"
