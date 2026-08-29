#!/usr/bin/env bash
# Capture the 20 m - 1 km range-performance clip (thermal + visible).
#   wsl -d Ubuntu-24.04 -u root -e bash scripts/run_longrange_clip.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export GZ_SIM_RESOURCE_PATH="$REPO/simulator/models:$REPO/simulator/worlds"
WORLD="detection_world_terrain"          # NAME, not filename (see handoff)
SDF="$REPO/simulator/worlds/detection_world_terrain_ir.sdf"
[ -e data/clips/range_1km ] && { echo "Refusing to overwrite data/clips/range_1km" >&2; exit 2; }
pkill -9 -f "[g]z sim" 2>/dev/null || true; sleep 2
gz sim -s -r --headless-rendering "$SDF" > /tmp/gz_range.log 2>&1 &
GZPID=$!
for _ in $(seq 1 60); do
  sleep 4
  gz topic -l 2>/dev/null | grep -q "detection_station/camera" && break
done
sleep 6
python3 scripts/drive_scene.py --world "$WORLD" --profile longsweep \
  --seed 51 --birds 4 --bird-seed 411 --thermal > /tmp/drive_range.log 2>&1 &
DRPID=$!
sleep 18
python3 scripts/capture_dataset.py --seconds 300 --interval 0.5 \
  --world "$WORLD" --thermal --verify-below-v 250 \
  --occlusion-manifest "$REPO/simulator/worlds/terrain_manifest.json" \
  --occlusion-heightmap "$REPO/simulator/worlds/terrain_assets/heightmap.npy" \
  --out data/clips/range_1km \
  --note "range performance: 20 m - 1 km longsweep seed 51, 4 birds"
kill "$DRPID" 2>/dev/null || true
kill "$GZPID" 2>/dev/null || true
pkill -9 -f "[g]z sim" 2>/dev/null || true
echo "=== range clip complete ==="
