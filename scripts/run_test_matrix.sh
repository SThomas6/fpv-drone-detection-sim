#!/usr/bin/env bash
#
# Phase 2 test matrix: fly each profile, record a labelled clip for it.
# The simulator is restarted per profile so every clip starts from a known
# state (disarmed, on the ground at home).
#
#   ./scripts/run_test_matrix.sh [profile ...]      # default: all
#
# Evaluate afterwards with:
#   python camera/evaluate.py run     --clip data/clips/<name> --mode full
#   python camera/evaluate.py analyze --clip data/clips/<name> --mode full

set -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

PROFILES="$*"
[ -z "$PROFILES" ] && PROFILES="crossing low_altitude no_drone"

# No associative arrays: macOS ships bash 3.2, where `declare -A` fails and
# every lookup silently collapses to index 0 (one config for all profiles).
seconds_for() {
  case "$1" in
    range_sweep) echo 400 ;; crossing) echo 260 ;;
    low_altitude) echo 220 ;; no_drone) echo 120 ;; *) echo 200 ;;
  esac
}
note_for() {
  case "$1" in
    range_sweep) echo "controlled range sweep 8-250m, clear daylight, sky background" ;;
    crossing) echo "lateral traverses across the field of view at 40/80/120m" ;;
    low_altitude) echo "low flight, drone seen against terrain rather than sky" ;;
    no_drone) echo "target parked out of view - false-positive baseline" ;;
  esac
}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate dronesim || { echo "conda env dronesim missing"; exit 1; }

for p in $PROFILES; do
  echo ""
  echo "=============================================================="
  echo "profile: $p"
  echo "=============================================================="
  ./scripts/stop_sim.sh > /dev/null 2>&1
  ./scripts/start_sim.sh --no-fly > "logs/launch_${p}.log" 2>&1 || {
    echo "FAILED to start sim for $p"; continue; }

  python scripts/fly_test_profile.py --profile "$p" --no-land \
      > "logs/profile_${p}.log" 2>&1 &
  fly_pid=$!
  sleep 30    # let it arm, climb and reach the first setpoint

  python scripts/capture_dataset.py \
      --seconds "$(seconds_for "$p")" --interval 0.4 \
      --out "data/clips/${p}" --note "$(note_for "$p")" 2>&1 | tail -6

  wait $fly_pid 2>/dev/null
  tail -2 "logs/profile_${p}.log"
done

./scripts/stop_sim.sh > /dev/null 2>&1
echo ""
echo "test matrix capture complete. Clips:"
ls -d data/clips/*/ 2>/dev/null
