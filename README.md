# Anti-Drone Detection Simulator

Virtual anti-drone detection system, entirely in simulation. A simulated FPV
drone flies in a Gazebo world; a fixed ground detection station observes it
with independent sensors (camera now; microphone and radar in later phases),
with everything ultimately fused into one detection/tracking system.

Phase order (strict): **Simulator → Camera → Microphone → Radar → Fusion.**
Current status: **Phases 1 and 2 complete.** The camera sensor detects and
tracks the drone end to end; results in
[docs/phase2-results.md](docs/phase2-results.md). Phase 3 (acoustic) not started.

## Architecture (Phase 1)

```
┌────────────────────────┐    MAVLink UDP     ┌──────────────────────────┐
│ PX4 SITL v1.18.0-beta2 │◄──14540 offboard───│ scripts/fly_mission.py   │
│ SIH mode: flight       │                    │ (MAVSDK autonomous       │
│ dynamics INSIDE PX4    │───14550 GCS──────► │  flight pattern)         │
└────────────────────────┘        │           └──────────────────────────┘
                                  ▼
                     ┌────────────────────────┐  set_pose service (30 Hz)
                     │ simulator/pose_mirror  │────────────┐
                     │ (NED→ENU conversion)   │            ▼
                     └────────────────────────┘  ┌───────────────────────┐
                                                 │ Gazebo Harmonic 8.10  │
                                                 │ detection_world:      │
                                                 │  · ground station     │
                                                 │  · camera 720p@15Hz   │
                                                 │  · target_drone model │
                                                 └──────────┬────────────┘
                                                            │ gz-transport
                                              ┌─────────────┴─────────────┐
                                              ▼                           ▼
                                   ┌────────────────────┐   ┌────────────────────┐
                                   │ ros_gz bridge      │   │ Python direct      │
                                   │ → ROS 2 topics     │   │ (gz.transport13)   │
                                   └────────────────────┘   └────────────────────┘
```

**Why SIH mode?** PX4's Gazebo bridge is Linux-first; on macOS the supported,
CI-verified path is SIH (Simulation-In-Hardware: PX4 simulates the vehicle
dynamics internally). Gazebo is used as the *sensor world*: `pose_mirror.py`
streams PX4's pose into Gazebo at 30 Hz so the station camera sees a
correctly-moving drone. The alternative full `gz_x500` mode (physics inside
Gazebo) is scaffolded in `start_sim.sh --mode gz` and can be enabled later by
building `px4_sitl_default` against the conda Gazebo libraries.

## What is installed where

| Component | Version | Location | How |
|---|---|---|---|
| PX4-Autopilot | v1.18.0-beta2 (shallow clone) | `~/Projects/PX4-Autopilot` (symlinked into project — PX4's build system can't handle the space in "FPV drone system") | git |
| PX4 python venv | — | `~/Projects/PX4-Autopilot/.venv` | `python -m venv` |
| Gazebo Harmonic | gz-sim 8.10.0 | conda env `dronesim` | conda-forge (binary — Homebrew has no macOS 26 bottles and would compile for hours) |
| ROS 2 Jazzy (ros-base) + ros_gz bridge | ros-gz 1.0.x | conda env `dronesim` | robostack-jazzy channel |
| gz Python bindings | gz-transport13 / gz-msgs10 | conda env `dronesim` | conda-forge |
| MAVSDK-Python | 3.x | conda env `dronesim` | pip |
| cmake / ninja / ccache | — | Homebrew | brew |

Recreate the conda env from scratch:

```bash
conda create -y -n dronesim --override-channels --strict-channel-priority \
  -c robostack-jazzy -c conda-forge \
  python=3.12 ros-jazzy-ros-base ros-jazzy-ros-gz-sim ros-jazzy-ros-gz-bridge \
  ros-jazzy-ros-gz-image gz-transport13-python gz-msgs10-python gz-tools2
conda run -n dronesim pip install mavsdk
```

Rebuild PX4 (SIH target):

```bash
cd ~/Projects/PX4-Autopilot && source .venv/bin/activate && make px4_sitl_sih
```

## Running

Start everything (Gazebo server headless → PX4 → pose mirror → ROS bridge →
autonomous mission):

```bash
./scripts/start_sim.sh
```

Options: `--gui` (also open the Gazebo GUI — a separate process on macOS, and
documented-unstable there, so treat it as a debugging tool), `--no-fly` (don't
start the mission), `--mode gz|sih|auto`.

Stop everything:

```bash
./scripts/stop_sim.sh
```

Smoke-test a running sim (5 checks: gz topics, camera via gz-transport,
camera via ROS 2, PX4 telemetry, actual movement):

```bash
conda activate dronesim && python tests/test_phase1.py
```

Detection regression test (offline, needs recorded clips but no simulator):

```bash
conda activate dronesim && python tests/test_phase2.py
```

Logs land in `logs/` (`gz_server.log`, `px4.log`, `pose_mirror.log`,
`bridge.log`, `mission.log`).

## How the drone is controlled

`scripts/fly_mission.py` connects over MAVSDK to PX4's offboard API link
(UDP 14540), arms, takes off to 15 m, then loops an offboard position pattern
(N ±18 m, E −20..+30 m, alt 8-30 m relative to home — i.e. 30-80 m from the
station, inside the camera's view). Edit `PATTERN` in that file to change the
flight path. MAVLink port map: 14540 = mission, 14550 = pose mirror (50 Hz
position+attitude), 14030 = tests/ad-hoc tools.

## How sensor data is accessed (Phase 2+ starts here)

ROS 2 topics (via `ros_gz` bridge, config in `simulator/config/ros_gz_bridge.yaml`):

- `/detection_station/camera/image` — `sensor_msgs/Image`, 1280x720 RGB @ 15 Hz
- `/detection_station/camera/camera_info` — `sensor_msgs/CameraInfo`
- `/world/detection_world/dynamic_pose/info` — ground-truth poses (TFMessage)
- `/clock`

```bash
conda activate dronesim
ros2 topic list
ros2 topic hz /detection_station/camera/image
```

Or subscribe directly from Python without ROS (lower latency, same data):

```python
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
node = Node()
node.subscribe(Image, "/detection_station/camera/image", callback)
```

Microphone and radar have **mount poses** on the station model
(`mic_array_link`, `radar_link` in `simulator/worlds/detection_world.sdf`).
Gazebo simulates neither acoustics nor radar; Phases 3/4 will synthesize those
signals from ground-truth kinematics relative to these mounts (see
`audio/README.md`, `radar/README.md`).

## Camera detection (Phase 2)

Run detection against the live simulator (start the sim first):

```bash
conda activate dronesim && python camera/detect_live.py --track
```

Output is per frame:

```text
Target ID: 1
Position: (428, 197)
Velocity: (38.4, 23.7) px/s
Confidence: 0.54
```

Without `--track` it reports detection only (yes/no, confidence, bounding box,
centre). `--sahi` switches to sliced inference — better on very distant
targets, far slower. `--jsonl out.jsonl` records results for later analysis.

Headline measured performance (full detail in
[docs/phase2-results.md](docs/phase2-results.md)):

- 100% recall out to ~200 m, where the drone is under 2 px wide
- zero false positives across 300 frames of empty sky
- 13–17 FPS full-frame on this Mac, against a 15 Hz camera
- tracking holds one ID with zero switches; velocity within ~9% when moving

**Read those numbers with suspicion.** They were measured against empty sky.
With birds in the air the same detector raises ~660 false alarms a minute,
because it cannot tell a bird from a drone — see "Birds" below. The simulated
sky is also uniform and noise-free, which makes a tiny dark speck far easier to
find than it would be against real cloud, haze and sensor noise.

### Birds and lighting

Lighting turned out not to matter (100% recall backlit and at dusk). Birds
matter enormously. Run the hard scenarios with:

```bash
./scripts/start_sim.sh --birds 8 --light backlit
./scripts/run_hard_matrix.sh          # capture bird + lighting clips
```

Because appearance is nearly gone at a few pixels, birds are rejected on **how
they move** rather than how they look — a multirotor hovers and flies straight
legs, a bird wheels and flaps:

```bash
python camera/detect_live.py --classify        # motion-gated alarms
```

That removes 84-91% of bird false alarms on held-out data, at the cost of ~30%
of drone track-frames. It is a large improvement, not a fix; details and the
recommended next step (a two-class drone+bird detector) are in
[docs/phase2-results.md](docs/phase2-results.md).

### Recording and evaluating

```bash
./scripts/run_test_matrix.sh                     # fly profiles, record labelled clips
python camera/evaluate.py run     --clip data/clips/baseline --mode full
python camera/evaluate.py analyze --clip data/clips/baseline --mode full
python camera/evaluate_tracking.py --clip data/clips/tracking
python camera/visualize.py --clip data/clips/baseline --only fp   # eyeball failures
```

Labels are free: the simulator knows exactly where the drone is, so
`capture_dataset.py` writes ground-truth boxes alongside each frame — and
verifies them against the rendered pixels, because an early version of this
pipeline silently produced mislabelled data (see
[docs/detection-notes.md](docs/detection-notes.md)).

## Project layout

```
simulator/   world, models, pose mirror, camera geometry, bridge config
camera/      detector, tracker, live CLI, evaluation (Phase 2)
audio/       Phase 3 (not started)
radar/       Phase 4 (not started)
fusion/      Phase 5 (not started)
scripts/     start/stop sim, flight profiles, dataset capture, test matrix
tests/       test_phase1.py (simulator), test_phase2.py (detection)
docs/        model-selection.md, phase2-results.md, detection-notes.md
data/clips/  recorded labelled clips (git-ignored)
PX4-Autopilot -> ~/Projects/PX4-Autopilot (symlink)
```

## Models / datasets in use

`sapoepsilon/yolov11s-drone-detector` (YOLOv11s drone fine-tune, 19 MB, pulled
from Hugging Face on first run and cached) — **used as-is, no fine-tuning
needed for simulated imagery**. Optional SAHI sliced inference. Tracking is our
own centroid/Kalman tracker, because every IoU-based tracker fragments identity
on targets this small.

Licensing: treat the model as **AGPL-3.0** (it is a fine-tune of Ultralytics
weights and the runtime is AGPL). Fine for personal, non-distributed R&D; see
[docs/model-selection.md](docs/model-selection.md) if this is ever published.

## Known limitations

- **SIH mode**: target drone visual pose is mirrored at 30 Hz from 50 Hz
  telemetry — smooth enough for 15 Hz camera, but physics does not act on the
  drone *in Gazebo* (no collisions with world objects). Fine for detection work.
- **macOS specifics**: Gazebo server and GUI must run as separate processes;
  the GUI is unstable on macOS — the system is designed to run headless.
  A benign `Unable to load Ogre Plugin` error appears in gz logs (a non-Metal
  plugin); rendering itself works via Metal (verified pixel-level).
- **8 GB RAM**: keep the GUI closed during experiments; camera kept at
  720p@15Hz for the same reason.
- **Disk**: ~4-6 GB free after install. `~/Projects/PX4-Autopilot/build` and
  `conda clean --all` are the reclaim points if space runs out.
- PX4 is v1.18.0-**beta2** — stable v1.17.0 cannot build its Gazebo bridge on
  current macOS/Homebrew (protobuf breakage) and its macOS setup script
  installs deprecated no-op formulae, so the beta is deliberate.

## Next milestone (Phase 3 — awaiting approval)

Acoustic detection in `audio/`. Gazebo does not simulate sound, so this phase
synthesizes microphone audio from simulator state (target distance, radial
velocity, rotor regime) relative to the station's `mic_array_link`, then builds
the spectrogram → feature → detection pipeline against it, keeping the sensor
independent of the camera as the plan requires.

Worth doing at some point, but not blocking Phase 3:

- Spawn birds or other distractors — bird/drone discrimination is completely
  untested, and is the classic false-positive source for real systems.
- Vary lighting and add cloud/haze to the world, to close some of the gap
  between these optimistic synthetic results and reality.
- Multi-target scenarios; the tracker has only ever seen one drone.
