#!/usr/bin/env bash
# Long-range optics + high-rate captures: the two experiments that the
# existing clips physically cannot answer.
#
#  range_tele       6 deg telephoto, boresight fly-out 300 m -> 1.35 km.
#                   At 1400 m a 0.34 m airframe is 2.97 px in this lens
#                   versus 0.27 px in the 60 deg wide camera - the whole
#                   claim that longer glass beats the sub-pixel wall.
#  range_tele_wide  IDENTICAL flight, 60 deg wide camera. The control:
#                   without it a telephoto number is unfalsifiable.
#  range_1km_hr     longsweep seed 51 (the SAME flight as range_1km) at
#                   15 Hz instead of 2 Hz. Track-before-detect scored only
#                   11.5% past 550 m because a 12-frame stack spanned 6 s
#                   of manoeuvring; at 15 Hz the same stack spans 0.8 s.
#
# Both new worlds move the camera <far> clip from 1000 m to 2000 m. That is
# not cosmetic: beyond the far plane nothing renders at all, so every
# past-1 km measurement in this project was capped by the renderer, not by
# the optics.
#
#   wsl -d Ubuntu-24.04 -u root -e bash scripts/run_telephoto_clips.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export GZ_SIM_RESOURCE_PATH="$REPO/simulator/models:$REPO/simulator/worlds"

WORLD="detection_world_terrain"        # world NAME, not the SDF filename
MANIFEST="$REPO/simulator/worlds/terrain_manifest.json"
HEIGHTMAP="$REPO/simulator/worlds/terrain_assets/heightmap.npy"

for t in data/clips/range_tele_hr; do
  [ -e "$t" ] && { echo "Refusing to overwrite existing $t" >&2; exit 2; }
done

capture() { # name sdf profile interval seconds hfov seed birds birdseed
  local name=$1 sdf=$2 profile=$3 interval=$4 secs=$5 hfov=$6
  local seed=$7 birds=$8 bseed=$9
  echo "=== $name ($profile, ${interval}s interval, hfov $hfov) ==="
  pkill -9 -f "[g]z sim" 2>/dev/null || true    # [g] so pkill never matches itself
  sleep 3
  gz sim -s -r --headless-rendering "$sdf" > "/tmp/gz_${name}.log" 2>&1 &
  local gzpid=$!
  local drpid=""
  cleanup() {
    [ -n "$drpid" ] && kill "$drpid" 2>/dev/null || true
    kill "$gzpid" 2>/dev/null || true
    pkill -9 -f "[g]z sim" 2>/dev/null || true  # gz owns children the wrapper does not
    sleep 3
  }
  trap cleanup RETURN
  # Poll for the world, never sleep a fixed time: cold WSL + software
  # rendering can take a heightmap+thermal world past any fixed wait, and
  # driving the scene early makes every spawn FAIL.
  local ready=0
  for _ in $(seq 1 60); do
    sleep 4
    if gz topic -l 2>/dev/null | grep -q "detection_station/camera"; then ready=1; break; fi
  done
  if [ "$ready" -ne 1 ]; then
    echo "world never became ready; gz log tail:" >&2
    tail -5 "/tmp/gz_${name}.log" >&2
    exit 3
  fi
  sleep 6
  # STATION_HFOV must match the SDF or every projected label is wrong.
  STATION_HFOV="$hfov" python3 scripts/drive_scene.py --world "$WORLD" \
    --profile "$profile" --seed "$seed" --birds "$birds" --bird-seed "$bseed" \
    --thermal > "/tmp/drive_${name}.log" 2>&1 &
  drpid=$!
  sleep 18
  # bash array so the repo path's SPACE ("drone sim") survives word-splitting
  local occ=(--occlusion-manifest "$MANIFEST" --occlusion-heightmap "$HEIGHTMAP")
  STATION_HFOV="$hfov" python3 scripts/capture_dataset.py \
    --seconds "$secs" --interval "$interval" --world "$WORLD" --thermal \
    --verify-below-v 250 "${occ[@]}" --out "data/clips/$name" \
    --note "$profile seed $seed, ${birds} birds seed ${bseed}, hfov ${hfov} rad, far clip 2000 m"
  trap - RETURN
  cleanup
  sleep 4
}

W="$REPO/simulator/worlds"
capture range_tele_hr    "$W/detection_world_terrain_tele.sdf" telesweep 0.0667 300 0.10472 61 4 611

echo "=== telephoto + high-rate captures complete ==="
