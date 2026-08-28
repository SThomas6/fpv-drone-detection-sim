#!/usr/bin/env bash
# Capture thermal TRAINING clips without changing any held-out terrain/IR
# benchmark. Requires WSL2 Ubuntu-24.04 + Gazebo Harmonic.
#
# The world, assets and manifest are deliberately distinct from the original
# terrain_ir matrix. New trajectory/bird seeds (17/211 and 19/213) preserve
# the train/eval split: held-out IR seeds are 0, 13 and 42.
#
#   wsl -d Ubuntu-24.04 -u root -e bash scripts/run_thermal_train_clips.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export GZ_SIM_RESOURCE_PATH="$REPO/simulator/models:$REPO/simulator/worlds"

WORLD="detection_world_terrain_train_ir"
SDF="$REPO/simulator/worlds/${WORLD}.sdf"
ASSETS="$REPO/simulator/worlds/terrain_assets_train_ir"
MANIFEST="$REPO/simulator/worlds/terrain_manifest_train_ir.json"

# Captures are expensive. Fail rather than silently replacing an existing
# dataset; delete/replace only after deliberate inspection.
for target in data/clips/train_ir_sweep data/clips/train_ir_canopy; do
  if [ -e "$target" ]; then
    echo "Refusing to overwrite existing $target" >&2
    exit 2
  fi
done

if [ -e "$SDF" ] && [ -d "$ASSETS" ] && [ -f "$MANIFEST" ]; then
  echo "Reusing existing isolated thermal training world"
elif [ -e "$SDF" ] || [ -e "$ASSETS" ] || [ -e "$MANIFEST" ]; then
  echo "Refusing a partial isolated world; inspect/remove it deliberately" >&2
  exit 2
else
  python3 scripts/gen_terrain_world.py --seed 17 --thermal \
    --world-name "$WORLD" --out "${WORLD}.sdf" \
    --assets-dir "terrain_assets_train_ir" \
    --manifest "terrain_manifest_train_ir.json"
fi

capture() { # name profile traj_seed birds bird_seed seconds
  local name=$1 profile=$2 tseed=$3 birds=$4 bseed=$5 secs=$6
  echo "=== thermal training clip $name ==="
  gz sim -s -r --headless-rendering "$SDF" > "/tmp/gz_${name}.log" 2>&1 &
  local gzpid=$!
  local drpid=""
  cleanup() { [ -n "$drpid" ] && kill "$drpid" 2>/dev/null || true; kill "$gzpid" 2>/dev/null || true; }
  trap cleanup RETURN
  sleep 25
  python3 scripts/drive_scene.py --world "$WORLD" --profile "$profile" \
    --seed "$tseed" --birds "$birds" --bird-seed "$bseed" --thermal \
    > "/tmp/drive_${name}.log" 2>&1 &
  drpid=$!
  sleep 18
  python3 scripts/capture_dataset.py --seconds "$secs" --interval 0.5 \
    --world "$WORLD" --thermal --verify-below-v 250 \
    --occlusion-manifest "$MANIFEST" --occlusion-heightmap "$ASSETS/heightmap.npy" \
    --out "data/clips/$name" \
    --note "thermal TRAINING: $profile trajectory seed $tseed, $birds birds seed $bseed; isolated world seed 17"
  trap - RETURN
  cleanup
  sleep 4
}

# Long-range supplies thermal-carried small-target tracks; canopy supplies
# thermal+bird negatives under the hardest background. Both are training-only.
capture train_ir_sweep  sweep  17 8 211 300
capture train_ir_canopy canopy 19 8 213 300

echo "=== thermal training captures complete ==="
