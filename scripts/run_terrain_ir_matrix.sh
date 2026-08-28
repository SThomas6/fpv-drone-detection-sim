#!/usr/bin/env bash
# EO/IR capture matrix: same scene seeds as run_terrain_matrix.sh, thermal
# world (station thermal camera + honest temperatures), RGB + L16 IR frames.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export GZ_SIM_RESOURCE_PATH="$REPO/simulator/models:$REPO/simulator/worlds"
WORLD=detection_world_terrain
SDF="$REPO/simulator/worlds/detection_world_terrain_ir.sdf"

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
      --birds "$birds" --bird-seed "$bseed" --thermal \
      > "/tmp/drive_$name.log" 2>&1 &
    DRPID=$!
    sleep 18
  fi
  python3 scripts/capture_dataset.py --seconds "$secs" --interval 0.5 \
    --world $WORLD --verify-below-v 250 --thermal \
    --out "data/clips/$name" \
    --note "EO/IR matrix: $profile, $birds birds seed $bseed" "$@"
  echo "clip $name exit: $?"
  [ -n "$DRPID" ] && kill $DRPID 2>/dev/null
  kill $GZPID 2>/dev/null
  sleep 4
}

capture terrain_ir_birds  mission 8 42  300
capture terrain_ir_canopy canopy  6 13  300
capture terrain_ir_sweep  sweep   0 0   200
capture terrain_ir_only   none    0 0   150 --no-target

echo "=== EO/IR matrix complete ==="
for d in terrain_ir_birds terrain_ir_canopy terrain_ir_sweep terrain_ir_only; do
  n=$(ls "data/clips/$d/frames" 2>/dev/null | wc -l)
  m=$(ls "data/clips/$d/frames_ir" 2>/dev/null | wc -l)
  echo "$d: $n rgb / $m ir frames"
done
