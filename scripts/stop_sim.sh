#!/usr/bin/env bash
# Stop every simulator component started by start_sim.sh.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIDFILE="$ROOT/.sim_pids"

if [ -f "$PIDFILE" ]; then
  while read -r pid name; do
    [ -n "${pid:-}" ] && kill "$pid" 2>/dev/null && echo "[stop_sim] stopped $name ($pid)"
  done < "$PIDFILE"
  rm -f "$PIDFILE"
fi

# belt and braces: catch children the PID file misses (make -> px4, etc.)
pkill -f "gz sim" 2>/dev/null
pkill -f "bin/px4" 2>/dev/null
pkill -f "parameter_bridge" 2>/dev/null
pkill -f "fly_mission.py" 2>/dev/null
pkill -f "pose_mirror.py" 2>/dev/null
pkill -f "birds.py" 2>/dev/null
sleep 1
echo "[stop_sim] done."
