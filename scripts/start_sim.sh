#!/usr/bin/env bash
#
# One-command launcher for the anti-drone detection simulator (Phase 1).
#
#   ./scripts/start_sim.sh [--gui] [--mode auto|gz|sih] [--no-fly]
#
#   --gui      also open the Gazebo GUI (a separate process on macOS;
#              headless is the default because of the 8 GB RAM budget)
#   --mode     gz  = PX4 SITL with physics inside Gazebo (x500 model)
#              sih = PX4 SIH internal physics + pose-mirrored visual drone
#              auto (default) = gz if that PX4 build exists, else sih
#   --no-fly   start everything but don't launch the autonomous mission
#   --birds N  spawn N birds flying through the field of view (false-positive
#              pressure: a gull at 100 m is a BIGGER target than the drone)
#   --light P  lighting preset: midday|morning|backlit|dusk|overcast
#
# Starts: Gazebo server (detection_world) -> PX4 SITL -> ros_gz bridge
#         -> autonomous mission. Logs land in logs/, PIDs in .sim_pids.
# Stop everything with scripts/stop_sim.sh.

# NOTE: no `set -u` — robostack's conda activate scripts reference unset vars.
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# PX4 lives OUTSIDE the project dir: its build system cannot handle the space
# in "FPV drone system". $ROOT/PX4-Autopilot is a symlink to this real path.
PX4_DIR="$(cd "$ROOT/PX4-Autopilot" && pwd -P)"
LOGS="$ROOT/logs"
PIDFILE="$ROOT/.sim_pids"
WORLD_NAME="detection_world"
WORLD_FILE="$ROOT/simulator/worlds/detection_world.sdf"
# Where the target drone spawns, relative to the detection station (ENU).
SPAWN_E=50; SPAWN_N=0; SPAWN_U=0.15

GUI=0; MODE=auto; FLY=1; BIRDS=0; LIGHT=""; BIRD_SEED=7
while [ $# -gt 0 ]; do
  case "$1" in
    --gui) GUI=1 ;;
    --no-fly) FLY=0 ;;
    --mode) shift; MODE="$1" ;;
    --birds) shift; BIRDS="$1" ;;
    --bird-seed) shift; BIRD_SEED="$1" ;;
    --light) shift; LIGHT="$1" ;;
    gz|sih|auto) MODE="$1" ;;
    *) ;;
  esac
  shift
done

mkdir -p "$LOGS"
: > "$PIDFILE"

log() { echo "[start_sim] $*"; }
die() { echo "[start_sim] FATAL: $*" >&2; "$ROOT/scripts/stop_sim.sh" >/dev/null 2>&1; exit 1; }

# ---- conda env ------------------------------------------------------------
source "$HOME/miniconda3/etc/profile.d/conda.sh" || die "miniconda not found"
conda activate dronesim || die "conda env 'dronesim' missing — see README installation"

export GZ_SIM_RESOURCE_PATH="$ROOT/simulator/worlds:$ROOT/simulator/models:$PX4_DIR/Tools/simulation/gz/models:$PX4_DIR/Tools/simulation/gz/worlds${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"

# ---- resolve mode ---------------------------------------------------------
if [ "$MODE" = "auto" ]; then
  if [ -x "$PX4_DIR/build/px4_sitl_default/bin/px4" ]; then MODE=gz
  elif [ -x "$PX4_DIR/build/px4_sitl_sih/bin/px4" ]; then MODE=sih
  else die "no PX4 build found — run the build first (see README)"; fi
fi
log "mode: $MODE"

# ---- 1. Gazebo server -----------------------------------------------------
log "starting Gazebo server ($WORLD_NAME, headless)..."
gz sim -v 3 -r -s "$WORLD_FILE" > "$LOGS/gz_server.log" 2>&1 &
echo "$! gz-server" >> "$PIDFILE"

# wait for the world clock topic => server is up; then camera topic => sensors up
for i in $(seq 1 60); do
  gz topic -l 2>/dev/null | grep -q "/world/$WORLD_NAME/clock" && break
  [ "$i" = 60 ] && die "Gazebo server did not come up (logs/gz_server.log)"
  sleep 1
done
log "Gazebo server up."
for i in $(seq 1 30); do
  gz topic -l 2>/dev/null | grep -q "/detection_station/camera/image" && break
  [ "$i" = 30 ] && log "WARNING: camera topic not visible yet (continuing; check logs/gz_server.log)"
  sleep 1
done
log "camera sensor publishing."

if [ "$GUI" = 1 ]; then
  log "starting Gazebo GUI (separate process on macOS)..."
  gz sim -g > "$LOGS/gz_gui.log" 2>&1 &
  echo "$! gz-gui" >> "$PIDFILE"
fi

# ---- 2. PX4 SITL ----------------------------------------------------------
# PX4 runs as the bare binary in daemon mode (-d): without a terminal the
# interactive pxh console redraws its prompt in a tight loop and writes
# hundreds of MB of escape codes into the log.
if [ "$MODE" = "gz" ]; then
  log "starting PX4 SITL (gz_x500 attaches to the running world)..."
  ( cd "$PX4_DIR/build/px4_sitl_default/rootfs" && \
    PX4_SIM_MODEL=gz_x500 PX4_GZ_STANDALONE=1 PX4_GZ_WORLD="$WORLD_NAME" \
    PX4_GZ_MODEL_POSE="$SPAWN_E,$SPAWN_N,$SPAWN_U,0,0,3.14159" \
    ../bin/px4 -d ) > "$LOGS/px4.log" 2>&1 &
  echo "$! px4" >> "$PIDFILE"
else
  log "starting PX4 SIH (internal physics) + pose mirror into Gazebo..."
  ( cd "$PX4_DIR/build/px4_sitl_sih/rootfs" && \
    PX4_SIM_MODEL=sihsim_quadx ../bin/px4 -d ) > "$LOGS/px4.log" 2>&1 &
  echo "$! px4" >> "$PIDFILE"

  # visual stand-in drone that the station camera will see
  sleep 2
  gz service -s "/world/$WORLD_NAME/create" \
    --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --timeout 5000 \
    --req "sdf_filename: \"$ROOT/simulator/models/target_drone/model.sdf\", name: \"target_drone\", pose: {position: {x: $SPAWN_E, y: $SPAWN_N, z: $SPAWN_U}}" \
    >> "$LOGS/gz_server.log" 2>&1 || log "WARNING: target_drone spawn service call failed"

  python "$ROOT/simulator/pose_mirror.py" \
    --world "$WORLD_NAME" --model target_drone \
    --origin-e "$SPAWN_E" --origin-n "$SPAWN_N" --origin-u "$SPAWN_U" \
    > "$LOGS/pose_mirror.log" 2>&1 &
  echo "$! pose-mirror" >> "$PIDFILE"
fi

# wait for PX4 to accept MAVLink (mavsdk connects in fly/test scripts)
for i in $(seq 1 90); do
  grep -q "Startup script returned successfully\|Ready for takeoff" "$LOGS/px4.log" 2>/dev/null && break
  [ "$i" = 90 ] && log "WARNING: PX4 startup marker not seen after 90s (check logs/px4.log)"
  sleep 1
done
log "PX4 running."

# ---- 2b. environment: birds and lighting ----------------------------------
if [ -n "$LIGHT" ]; then
  python "$ROOT/simulator/lighting.py" --preset "$LIGHT" --world "$WORLD_NAME" \
    >> "$LOGS/gz_server.log" 2>&1 && log "lighting preset: $LIGHT"
fi

if [ "$BIRDS" -gt 0 ] 2>/dev/null; then
  log "spawning $BIRDS birds..."
  python "$ROOT/simulator/birds.py" --world "$WORLD_NAME" --count "$BIRDS" \
    --seed "$BIRD_SEED" > "$LOGS/birds.log" 2>&1 &
  echo "$! birds" >> "$PIDFILE"
fi

# ---- 3. ros_gz bridge -----------------------------------------------------
log "starting ros_gz bridge (Gazebo topics -> ROS 2)..."
ros2 run ros_gz_bridge parameter_bridge --ros-args \
  -p config_file:="$ROOT/simulator/config/ros_gz_bridge.yaml" \
  > "$LOGS/bridge.log" 2>&1 &
echo "$! ros-gz-bridge" >> "$PIDFILE"

# ---- 4. autonomous mission ------------------------------------------------
if [ "$FLY" = 1 ]; then
  log "starting autonomous mission (fly_mission.py, loops until stopped)..."
  python "$ROOT/scripts/fly_mission.py" --loop > "$LOGS/mission.log" 2>&1 &
  echo "$! mission" >> "$PIDFILE"
fi

echo ""
log "all components launched."
echo "  Gazebo topics : gz topic -l"
echo "  ROS 2 topics  : ros2 topic list   (camera: /detection_station/camera/image)"
echo "  PX4 log       : tail -f logs/px4.log"
echo "  Smoke test    : python tests/test_phase1.py"
echo "  Stop all      : scripts/stop_sim.sh"
