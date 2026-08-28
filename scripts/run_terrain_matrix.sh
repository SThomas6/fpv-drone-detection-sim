#!/usr/bin/env bash
# Terrain-background capture matrix (PX4-free, WSL2/Linux).
# Each clip gets a fresh server so worlds are clean and seeds reproducible.
#
#   wsl -d Ubuntu-24.04 -u root -e bash scripts/run_terrain_matrix.sh
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export GZ_SIM_RESOURCE_PATH="$REPO/simulator/models:$REPO/simulator/worlds"
WORLD=detection_world_terrain
SDF="$REPO/simulator/worlds/$WORLD.sdf"

capture () { # name profile birds bird_seed seconds extra...
  local name=$1 profile=$2 birds=$3 bseed=$4 secs=$5; shift 5
  echo "=== clip $name (profile=$profile birds=$birds seed=$bseed ${secs}s) ==="
  rm -rf "data/clips/$name"
  gz sim -s -r --headless-rendering "$SDF" > "/tmp/gz_$name.log" 2>&1 &
  local GZPID=$!
  sleep 25
  local DRPID=""
  if [ "$profile" != "none" ] || [ "$birds" != "0" ]; then
    python3 scripts/drive_scene.py --world $WORLD --profile "$profile" \
      --birds "$birds" --bird-seed "$bseed" > "/tmp/drive_$name.log" 2>&1 &
    DRPID=$!
    sleep 18
  fi
  python3 scripts/capture_dataset.py --seconds "$secs" --interval 0.5 \
    --world $WORLD --verify-below-v 250 --out "data/clips/$name" \
    --note "terrain matrix: $profile, $birds birds seed $bseed" "$@"
  echo "clip $name exit: $?"
  [ -n "$DRPID" ] && kill $DRPID 2>/dev/null
  kill $GZPID 2>/dev/null
  sleep 4
}

capture terrain_baseline mission 0 0   150
capture terrain_birds    mission 8 42  300
capture terrain_canopy   canopy  6 13  300
capture terrain_sweep    sweep   0 0   200
capture terrain_only     none    0 0   150 --no-target

echo "=== matrix complete ==="
for d in terrain_baseline terrain_birds terrain_canopy terrain_sweep terrain_only; do
  n=$(ls "data/clips/$d/frames" 2>/dev/null | wc -l)
  echo "$d: $n frames"
done
