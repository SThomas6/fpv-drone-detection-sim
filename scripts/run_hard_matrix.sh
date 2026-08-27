#!/usr/bin/env bash
#
# Hard-conditions test matrix: birds as distractors, and difficult lighting.
#
#   ./scripts/run_hard_matrix.sh [scenario ...]     # default: all
#
# Scenarios:
#   birds_only     drone parked out of view, birds flying — pure false-positive
#                  test. Any detection here is the detector calling a bird a drone.
#   birds_drone    range sweep WITH birds — precision under distraction
#   backlit        crossing into a low sun (worst light)
#   dusk           crossing in dim, low, warm light
#   backlit_birds  both at once
#
# NOTE: no associative arrays — macOS ships bash 3.2, where `declare -A` fails
# and every lookup silently collapses to index 0.

set -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

ALL="birds_only birds_drone backlit dusk backlit_birds"
SCENARIOS="$*"
[ -z "$SCENARIOS" ] && SCENARIOS="$ALL"

# scenario -> "profile birds light seconds"
config_for() {
  case "$1" in
    birds_only)    echo "no_drone 10 midday 130" ;;
    birds_drone)   echo "range_sweep 10 midday 380" ;;
    backlit)       echo "crossing 0 backlit 240" ;;
    dusk)          echo "crossing 0 dusk 240" ;;
    backlit_birds) echo "crossing 10 backlit 240" ;;
    *)             echo "" ;;
  esac
}

note_for() {
  case "$1" in
    birds_only)    echo "birds flying, drone parked out of view - false positives on birds" ;;
    birds_drone)   echo "range sweep 8-250m with 10 birds as distractors" ;;
    backlit)       echo "lateral crossings into a low sun" ;;
    dusk)          echo "lateral crossings in dim warm dusk light" ;;
    backlit_birds) echo "lateral crossings into a low sun, with 10 birds" ;;
  esac
}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate dronesim || { echo "conda env dronesim missing"; exit 1; }

for s in $SCENARIOS; do
  cfg="$(config_for "$s")"
  if [ -z "$cfg" ]; then echo "unknown scenario: $s"; continue; fi
  set -- $cfg
  profile="$1"; birds="$2"; light="$3"; secs="$4"

  echo ""
  echo "=============================================================="
  echo "scenario: $s  (profile=$profile birds=$birds light=$light ${secs}s)"
  echo "=============================================================="
  ./scripts/stop_sim.sh > /dev/null 2>&1
  ./scripts/start_sim.sh --no-fly --birds "$birds" --light "$light" \
      > "logs/launch_${s}.log" 2>&1 || { echo "FAILED to start sim"; continue; }

  python scripts/fly_test_profile.py --profile "$profile" --no-land \
      > "logs/profile_${s}.log" 2>&1 &
  fly_pid=$!
  sleep 30

  python scripts/capture_dataset.py --seconds "$secs" --interval 0.4 \
      --out "data/clips/${s}" --note "$(note_for "$s")" 2>&1 | tail -6

  wait $fly_pid 2>/dev/null
done

./scripts/stop_sim.sh > /dev/null 2>&1
echo ""
echo "hard matrix capture complete."
