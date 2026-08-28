#!/usr/bin/env bash
# Terrain TRAINING clips — different trajectory and bird seeds from the eval
# matrix (seeds 0/42/13), so the eval clips stay strictly held out.
# train: mission seed 5 / birds 99, canopy seed 6 / birds 101
# selection (checkpoint sweep target): canopy seed 8 / birds 105
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export GZ_SIM_RESOURCE_PATH="$REPO/simulator/models:$REPO/simulator/worlds"
WORLD=detection_world_terrain
SDF="$REPO/simulator/worlds/$WORLD.sdf"

capture () { # name profile traj_seed birds bird_seed seconds
  local name=$1 profile=$2 tseed=$3 birds=$4 bseed=$5 secs=$6
  echo "=== clip $name ==="
  rm -rf "data/clips/$name"
  gz sim -s -r --headless-rendering "$SDF" > "/tmp/gz_$name.log" 2>&1 &
  local GZPID=$!
  sleep 25
  python3 scripts/drive_scene.py --world $WORLD --profile "$profile" \
    --seed "$tseed" --birds "$birds" --bird-seed "$bseed" \
    > "/tmp/drive_$name.log" 2>&1 &
  local DRPID=$!
  sleep 18
  python3 scripts/capture_dataset.py --seconds "$secs" --interval 0.5 \
    --world $WORLD --verify-below-v 250 --out "data/clips/$name" \
    --note "terrain training clip: $profile seed $tseed, $birds birds seed $bseed"
  echo "clip $name exit: $?"
  kill $DRPID $GZPID 2>/dev/null
  sleep 4
}

capture train_terrain_mission mission 5 8 99  300
capture train_terrain_canopy  canopy  6 8 101 300
capture sel_terrain_canopy    canopy  8 6 105 200

echo "=== training clips complete ==="
