#!/usr/bin/env python3
"""Drive the target drone (scripted trajectory) and birds without PX4.

Gazebo is purely the renderer in this project — PX4 only ever supplied poses
(see simulator/pose_mirror.py). This replays a kinematically plausible
trajectory instead: accel-capped waypoint legs with hovers, exactly the flight
envelope of the original missions (30-80 m from the station, 8-30 m altitude),
plus profiles that put the drone against the terrain world's treeline.

Poses are computed against SIM time (the world clock), not wall time, so
software rendering that runs below real time cannot skew pixels against
labels — the pose the camera renders is the pose the labels record.

    python3 scripts/drive_scene.py --world detection_world_terrain \
        --profile mission --birds 8 --bird-seed 42 --seconds 320
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from gz.msgs10.boolean_pb2 import Boolean  # noqa: E402
from gz.msgs10.entity_factory_pb2 import EntityFactory  # noqa: E402
from gz.msgs10.pose_pb2 import Pose  # noqa: E402
from gz.msgs10.pose_v_pb2 import Pose_V  # noqa: E402
from gz.msgs10.world_stats_pb2 import WorldStatistics  # noqa: E402
from gz.transport13 import Node  # noqa: E402

from simulator.birds import Bird, bird_sdf, euler_to_quat  # noqa: E402

MODEL_SDF = Path(__file__).resolve().parents[1] / "simulator" / "models" / \
    "target_drone" / "model.sdf"
UPDATE_HZ = 30.0
ACCEL_CAP = 4.0          # m/s^2 — a small quad can pull more; stay modest
HOVER_JITTER = 0.12      # metres of slow station-keeping wander


def profile_waypoints(name: str, rng: random.Random):
    """(x, y, z, speed_mps, hover_s) legs. Speeds/altitudes match the original
    PX4 mission envelope so the motion classifier sees familiar dynamics."""
    if name == "mission":
        pts = [(35, -20, 12, 6, 0), (55, 15, 18, 8, 0), (75, -5, 25, 9, 5),
               (60, -30, 15, 7, 0), (40, 25, 10, 6, 0), (70, 20, 28, 10, 5),
               (50, 0, 20, 8, 0), (32, -10, 9, 5, 0)]
    elif name == "canopy":
        # low legs: from the station camera these sit against the treeline
        pts = [(45, -25, 8, 6, 0), (70, 10, 11, 7, 0), (90, -15, 13, 8, 4),
               (60, 25, 7, 6, 0), (85, 0, 10, 9, 0), (50, -5, 12, 6, 4),
               (75, -30, 9, 7, 0), (42, 15, 8, 5, 0)]
    elif name == "sweep":
        pts = [(30, 0, 14, 8, 2), (120, 10, 18, 11, 0), (250, -10, 22, 12, 3),
               (150, 5, 16, 11, 0), (60, -5, 12, 8, 0), (35, 0, 10, 6, 2)]
    elif name == "longsweep":
        # 20 m to 1 km and back: the range-performance profile. Beyond ~300 m
        # the drone is sub-pixel to the wide camera and inaudible to the
        # array, so this leg is what actually measures where each sensor
        # stops working rather than assuming it.
        # Altitudes climb steeply with range: the terrain heightmap is 800 m
        # across with peaks tens of metres high, and a shallow climb puts the
        # far legs BEHIND the ridge - measured, 448 of 600 frames came back
        # occluded on the first attempt. A real surveillance drone at 1 km is
        # high anyway. Lateral offsets stay small so it holds the 60 deg FOV.
        pts = [(20, 0, 10, 5, 3), (60, 3, 22, 8, 2), (120, -4, 38, 10, 2),
               (200, 4, 55, 11, 2), (320, -5, 75, 12, 2),
               (480, 5, 95, 13, 2), (650, -4, 115, 14, 2),
               (820, 4, 135, 14, 2), (1000, 0, 155, 14, 4),
               (700, -4, 120, 14, 0), (400, 4, 85, 12, 0),
               (150, -3, 45, 10, 0), (40, 0, 14, 6, 2)]
    elif name == "telesweep":
        # 300 m to 1.35 km along the TELEPHOTO's boresight: the cued-optics
        # profile. A 6 deg lens sees a 3.4 deg vertical band, so a fly-out
        # that climbs at any other angle leaves the frame within seconds and
        # measures nothing. The station camera sits 2.5 m up pitched 8.6 deg
        # (0.15 rad) nose-up, so holding z = 2.5 + tan(0.15)*x keeps the
        # target on boresight - which is exactly what a pan-tilt mount does
        # once it is slewed onto a track. Lateral offset is scaled to range
        # (~0.7 deg) so the target still crosses the frame and the motion
        # channel has something to work with, without falling out the side.
        pts = [(300, 0, 48, 12, 2), (420, 5, 66, 13, 0), (540, -6, 84, 13, 0),
               (660, 8, 102, 14, 0), (780, -9, 120, 14, 0),
               (900, 11, 138, 14, 2), (1050, -12, 161, 14, 0),
               (1200, 14, 184, 14, 0), (1350, -16, 206, 14, 2),
               (1200, 14, 184, 14, 0), (1000, -12, 154, 14, 0),
               (800, 9, 123, 14, 0), (600, -7, 93, 13, 0), (400, 0, 63, 12, 2)]
    elif name == "skysweep":
        # Same 300 m -> 1.35 km fly-out as telesweep, but held on the
        # 13 deg boresight of the sky world instead of 8.6 deg. The
        # terrain ridge tops out at ~10.2 deg elevation (120 m high,
        # 650 m out), so this profile keeps the target ABOVE it and
        # therefore against SKY - the realistic geometry for a drone
        # at a kilometre, and a far easier background than rock.
        # Requires STATION_PITCH=-0.2269 and the tele_sky world.
        pts = [(300, 0, 72, 12, 2), (420, 5, 99, 12, 0),
               (540, -6, 127, 13, 0), (660, 8, 155, 14, 0),
               (780, -9, 183, 14, 0), (900, 11, 210, 14, 2),
               (1050, -12, 245, 14, 0), (1200, 14, 280, 14, 0),
               (1350, -16, 314, 14, 2), (1200, 14, 280, 14, 0),
               (1000, -12, 233, 14, 0), (800, 9, 187, 14, 0),
               (600, -7, 141, 13, 0), (400, 0, 95, 12, 2)]
    elif name == "crosssweep":
        # Pure CROSSING passes at fixed ranges - the worst case for
        # the tracker, and the one skysweep does not test. Radial
        # flight barely moves the image (skysweep median 1.5
        # px/frame); a target crossing at 41.7 m/s at 500 m through
        # the 6 deg lens moves 1018 px/s = 68 px/frame at 15 Hz.
        # Shuttles +/-50 m across the boresight at 400/600/900/1200 m,
        # each leg long enough to actually reach speed at accel 15.
        # Use with the tele_sky world and STATION_PITCH=-0.2269.
        pts = [(400, -50, 95, 14, 0), (400, 50, 95, 14, 0),
               (600, -50, 141, 14, 0), (600, 50, 141, 14, 0),
               (900, -50, 210, 14, 0), (900, 50, 210, 14, 0),
               (1200, -50, 280, 14, 0), (1200, 50, 280, 14, 0),
               (900, -50, 210, 14, 0), (900, 50, 210, 14, 0),
               (600, -50, 141, 14, 0), (600, 50, 141, 14, 0),
               (400, -50, 95, 14, 0), (400, 50, 95, 14, 0)]
    else:
        raise SystemExit(f"unknown profile {name}")
    # jitter waypoints a little so seeds differ
    return [(x + rng.uniform(-3, 3), y + rng.uniform(-3, 3),
             max(6.0, z + rng.uniform(-2, 2)), s, h) for x, y, z, s, h in pts]


class Trajectory:
    """Dense accel-capped path through waypoints, sampled by sim time."""

    def __init__(self, waypoints, rng: random.Random, dt=0.02,
                 accel: float = ACCEL_CAP):
        self.dt = dt
        pos = []
        p = np.array(waypoints[0][:3], dtype=float)
        v = 0.0
        for (x, y, z, speed, hover) in waypoints[1:] + [waypoints[0]]:
            q = np.array([x, y, z], dtype=float)
            leg = q - p
            dist = float(np.linalg.norm(leg))
            u = leg / max(dist, 1e-6)
            s = 0.0
            while s < dist:
                # trapezoidal speed: accel-capped ramp up/down
                v = min(speed, v + accel * dt,
                        math.sqrt(max(0.5, 2 * accel * (dist - s))))
                s += v * dt
                pos.append(p + u * min(s, dist))
            p = q
            for _ in range(int(hover / dt)):
                v = 0.0
                pos.append(p.copy())
        self.path = np.array(pos)
        self.total = len(pos) * dt
        self.rng = rng

    def pose_at(self, t: float):
        i = int((t % self.total) / self.dt)
        i = min(i, len(self.path) - 2)
        p = self.path[i]
        nxt = self.path[min(i + 5, len(self.path) - 1)]
        d = nxt - p
        # slow station-keeping wander, visible during hovers
        j = HOVER_JITTER * np.array([math.sin(0.5 * t), math.cos(0.37 * t),
                                     0.5 * math.sin(0.23 * t)])
        yaw = math.atan2(d[1], d[0]) if np.linalg.norm(d[:2]) > 0.05 else 0.0
        return p + j, (0.0, 0.0, yaw)


class SimClock:
    """Sim time from the ~5 Hz stats topic, extrapolated between messages.

    Deliberately NOT the high-rate /clock topic: in the Python bindings a
    250 Hz subscription starves blocking service requests on the same
    process (spawns and pose batches all time out), which froze the scene
    in the first version of this script.
    """

    def __init__(self, node: Node, world: str):
        self.t = 0.0
        self.rtf = 1.0
        self.wall = time.time()
        self._lock = threading.Lock()
        node.subscribe(WorldStatistics, f"/world/{world}/stats", self._cb)

    def _cb(self, msg: WorldStatistics):
        with self._lock:
            self.t = msg.sim_time.sec + msg.sim_time.nsec * 1e-9
            if msg.real_time_factor > 0.01:
                self.rtf = msg.real_time_factor
            self.wall = time.time()

    def now(self) -> float:
        with self._lock:
            return self.t + (time.time() - self.wall) * self.rtf


def spawn(node: Node, world: str, name: str, sdf: str, pos):
    req = EntityFactory()
    req.sdf = sdf
    req.pose.position.x, req.pose.position.y, req.pose.position.z = pos
    req.pose.orientation.w = 1.0
    ok, rep = node.request(f"/world/{world}/create", req, EntityFactory,
                           Boolean, 15000)
    print(f"[scene] spawn {name}: {'ok' if ok and rep.data else 'FAILED'}",
          flush=True)


def fill_pose(msg: Pose, name: str, pos, rpy):
    msg.name = name
    msg.position.x, msg.position.y, msg.position.z = pos
    qw, qx, qy, qz = euler_to_quat(*rpy)
    msg.orientation.w, msg.orientation.x = qw, qx
    msg.orientation.y, msg.orientation.z = qy, qz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="detection_world_terrain")
    ap.add_argument("--profile", default="mission",
                    choices=["mission", "canopy", "sweep", "longsweep",
                             "telesweep", "skysweep",
                             "crosssweep", "none"])
    ap.add_argument("--birds", type=int, default=0)
    ap.add_argument("--bird-seed", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed-scale", type=float, default=1.0,
                    help="multiply every waypoint speed. 3.0 turns the 14 m/s "
                         "profiles into ~150 kph, the threat speed this "
                         "system has to hold track on")
    ap.add_argument("--accel", type=float, default=ACCEL_CAP,
                    help="accel cap m/s^2; the 4.0 default needs 217 m to "
                         "reach 42 m/s, so a fast profile must raise it")
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = forever")
    ap.add_argument("--thermal", action="store_true",
                    help="give birds their honest LWIR surface temperature "
                         "(ambient+4K torso; wings stay ambient)")
    args = ap.parse_args()

    node = Node()
    rng = random.Random(args.seed)

    # Spawn BEFORE any subscription exists (see SimClock docstring).
    traj = None
    if args.profile != "none":
        wps = [(x, y, z, s_ * args.speed_scale, h)
               for x, y, z, s_, h in
               profile_waypoints(args.profile, rng)]
        traj = Trajectory(wps, rng, accel=args.accel)
        spawn(node, args.world, "target_drone", MODEL_SDF.read_text(),
              traj.pose_at(0.0)[0])
    birds = []
    if args.birds:
        brng = random.Random(args.bird_seed)
        birds = [Bird(i, brng) for i in range(args.birds)]
        for b in birds:
            spawn(node, args.world, b.name,
                  bird_sdf(b.name, b.span, b.body, b.grey,
                           thermal_k=292.0 if args.thermal else None),
                  b.pose_at(0.0)[0])

    clock = SimClock(node, args.world)
    deadline = time.time() + 20
    while clock.now() == 0.0 and time.time() < deadline:
        time.sleep(0.1)
    t0 = clock.now()
    print(f"[scene] driving on sim time (t0={t0:.2f}s, profile="
          f"{args.profile}, {len(birds)} birds)", flush=True)

    # One batched set_pose_vector request per tick: under software rendering
    # the server answers services slowly, and per-entity 100 ms requests all
    # time out (the frozen-scene failure this replaced).
    svc = f"/world/{args.world}/set_pose_vector"
    wall0 = time.time()
    sent = ok_n = 0
    while True:
        t = clock.now() - t0
        if args.seconds and t > args.seconds:
            break
        req = Pose_V()
        if traj is not None:
            pos, rpy = traj.pose_at(t)
            fill_pose(req.pose.add(), "target_drone", pos, rpy)
        for b in birds:
            (x, y, z), (roll, pitch, yaw) = b.pose_at(t)
            fill_pose(req.pose.add(), b.name, (x, y, z), (roll, pitch, yaw))
        ok, _ = node.request(svc, req, Pose_V, Boolean, 800)
        sent += 1
        ok_n += 1 if ok else 0
        if sent % 300 == 0:
            print(f"[scene] t={t:.0f}s sim, pose batches ok "
                  f"{ok_n}/{sent}", flush=True)
        time.sleep(1.0 / UPDATE_HZ)
    print(f"[scene] done after {time.time() - wall0:.0f}s wall / {t:.0f}s sim; "
          f"pose batches ok {ok_n}/{sent}", flush=True)


if __name__ == "__main__":
    main()
