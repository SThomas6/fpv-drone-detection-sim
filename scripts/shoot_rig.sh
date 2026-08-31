#!/usr/bin/env bash
# Photograph the car rig from three angles.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export GZ_SIM_RESOURCE_PATH="$REPO/simulator/models:$REPO/simulator/worlds"
OUT="$REPO/docs/rig"
mkdir -p "$OUT"
pkill -9 -f "[g]z sim" 2>/dev/null || true; sleep 2
gz sim -s -r --headless-rendering "$REPO/simulator/worlds/rig_showcase.sdf" \
  > /opt/gz_rig.log 2>&1 &
for _ in $(seq 1 45); do
  sleep 4
  gz topic -l 2>/dev/null | grep -q "/rig/view0" && break
done
sleep 8
for i in 0 1 2; do
  python3 scripts/grab_inspect.py --out "$OUT/rig_view$i.png" \
    --topic "/rig/view$i" --wait 40 || echo "view$i failed"
done
pkill -9 -f "[g]z sim" 2>/dev/null || true
echo "RIGSHOTSDONE"
