#!/usr/bin/env bash
# Three things never tested: several drones at once, MOVING vegetation, and
# what the rig actually looks like.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export GZ_SIM_RESOURCE_PATH="$REPO/simulator/models:$REPO/simulator/worlds"
WORLD="detection_world_terrain"
MANIFEST="$REPO/simulator/worlds/terrain_manifest.json"
HEIGHTMAP="$REPO/simulator/worlds/terrain_assets/heightmap.npy"
W="$REPO/simulator/worlds"

for t in data/clips/nosway_canopy; do
  [ -e "$t" ] && { echo "Refusing to overwrite $t" >&2; exit 2; }
done

boot() {  # sdf logname
  pkill -9 -f "[g]z sim" 2>/dev/null || true; sleep 3
  gz sim -s -r --headless-rendering "$1" > "/tmp/gz_$2.log" 2>&1 &
  for _ in $(seq 1 60); do
    sleep 4
    gz topic -l 2>/dev/null | grep -q "detection_station/camera" && return 0
  done
  echo "world $2 never became ready" >&2; tail -5 "/tmp/gz_$2.log" >&2; exit 3
}

# ---- 3. wind-driven canopy, one drone ----
echo "=== nosway_canopy (control) ==="
boot "$W/detection_world_terrain_ir.sdf" sway
sleep 6
python3 scripts/drive_scene.py --world "$WORLD" --profile canopy --seed 73 \
  --birds 4 --bird-seed 733 \
  --thermal > /tmp/drive_nosway.log 2>&1 &
DR=$!
sleep 18
python3 scripts/capture_dataset.py --seconds 200 --interval 0.2 \
  --world "$WORLD" --thermal --verify-below-v 250 \
  --occlusion-manifest "$MANIFEST" --occlusion-heightmap "$HEIGHTMAP" \
  --out data/clips/nosway_canopy \
  --note "CONTROL: no sway, 1 drone + 4 birds, seed 73"
kill "$DR" 2>/dev/null || true
pkill -9 -f "[g]z sim" 2>/dev/null || true
echo "=== multi + sway captures complete ==="
