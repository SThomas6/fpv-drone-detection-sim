#!/usr/bin/env bash
# Capture 15 Hz clips - the wingbeat campaign's new data.
#
# Why 15 Hz: the station camera's SDF update_rate is 15, and the sim's birds
# flap at 2.5-7.0 Hz (simulator/birds.py). At the historic 0.5 s capture
# interval (2 Hz) that flapping is beyond Nyquist and invisible, which is why
# the classifier could never use the cleanest optical discriminator fielded
# counter-UAS systems exploit. 1/15 s sampling puts Nyquist at 7.5 Hz - just
# above the whole flap band.
#
# Four clips: eval + train for the two alarm-price scenarios (canopy, sky).
# Eval clips use the EXISTING benchmark worlds (unmodified - a capture does
# not write to a world) with NEW seeds; the canopy train clip uses the
# isolated train world from run_thermal_train_clips.sh. No seed collides
# with any existing clip (eval: 0/13/42, train: 5/6/9/10/17/19).
#
#   wsl -d Ubuntu-24.04 -u root -e bash scripts/run_highrate_clips.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export GZ_SIM_RESOURCE_PATH="$REPO/simulator/models:$REPO/simulator/worlds"

INTERVAL=0.0667     # 15 Hz
SECONDS_CAP=150     # 2250 frames per clip

for target in data/clips/hr_eval_canopy data/clips/hr_eval_sky \
              data/clips/hr_train_canopy data/clips/hr_train_sky; do
  if [ -e "$target" ]; then
    echo "Refusing to overwrite existing $target" >&2
    exit 2
  fi
done

capture() { # name world sdf profile tseed birds bseed thermal(0/1) manifest heightmap
  local name=$1 world=$2 sdf=$3 profile=$4 tseed=$5 birds=$6 bseed=$7 thermal=$8
  local manifest=${9:-} heightmap=${10:-}
  echo "=== 15 Hz clip $name (world $world, seed $tseed, $birds birds) ==="
  gz sim -s -r --headless-rendering "$sdf" > "/tmp/gz_${name}.log" 2>&1 &
  local gzpid=$!
  local drpid=""
  cleanup() {
    [ -n "$drpid" ] && kill "$drpid" 2>/dev/null || true
    # gz spawns server children the wrapper PID does not own - kill the tree,
    # or the NEXT capture finds a zombie server on the same topics and every
    # spawn fails (hit once: two zombie servers, drive_scene spawn FAILED)
    kill "$gzpid" 2>/dev/null || true
    pkill -9 -f "[g]z sim" 2>/dev/null || true   # [g] so pkill never matches itself
    sleep 2
  }
  trap cleanup RETURN
  sleep 25
  local thermal_flag=""
  [ "$thermal" = "1" ] && thermal_flag="--thermal"
  python3 scripts/drive_scene.py --world "$world" --profile "$profile" \
    --seed "$tseed" --birds "$birds" --bird-seed "$bseed" $thermal_flag \
    > "/tmp/drive_${name}.log" 2>&1 &
  drpid=$!
  sleep 18
  # bash array so the repo path's SPACE ("drone sim") survives word-splitting
  local occ=()
  [ -n "$manifest" ] && occ=(--occlusion-manifest "$manifest" --occlusion-heightmap "$heightmap")
  python3 scripts/capture_dataset.py --seconds "$SECONDS_CAP" --interval "$INTERVAL" \
    --world "$world" $thermal_flag --verify-below-v 250 "${occ[@]}" \
    --out "data/clips/$name" \
    --note "15 Hz wingbeat campaign: $profile seed $tseed, $birds birds seed $bseed"
  trap - RETURN
  cleanup
  sleep 4
}

# Eval pair - benchmark worlds, fresh seeds.
capture hr_eval_canopy detection_world_terrain_ir \
  "$REPO/simulator/worlds/detection_world_terrain_ir.sdf" canopy 31 6 313 1 \
  "$REPO/simulator/worlds/terrain_manifest.json" \
  "$REPO/simulator/worlds/terrain_assets/heightmap.npy"
capture hr_eval_sky detection_world \
  "$REPO/simulator/worlds/detection_world.sdf" mission 32 8 314 0

# Train pair - isolated terrain train world; sky world is stateless.
capture hr_train_canopy detection_world_terrain_train_ir \
  "$REPO/simulator/worlds/detection_world_terrain_train_ir.sdf" canopy 33 8 315 1 \
  "$REPO/simulator/worlds/terrain_manifest_train_ir.json" \
  "$REPO/simulator/worlds/terrain_assets_train_ir/heightmap.npy"
capture hr_train_sky detection_world \
  "$REPO/simulator/worlds/detection_world.sdf" mission 34 8 316 0

echo "=== 15 Hz captures complete ==="
