#!/usr/bin/env python3
"""Mirror the PX4 SIH drone's pose into Gazebo (SIH fallback mode).

In SIH mode PX4 simulates the flight dynamics internally, so Gazebo never
sees the drone move. This node subscribes to PX4 telemetry over MAVSDK and
continuously re-poses the visual `target_drone` model in the Gazebo world,
so the detection station's camera (and later radar/audio models) observe a
correctly moving target.

Frames: PX4 telemetry is NED (north, east, down) relative to home; the
Gazebo world is ENU. Home maps to (--origin-e, --origin-n, --origin-u).
"""

import argparse
import asyncio
import math
import sys

from mavsdk import System

from gz.transport13 import Node
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.boolean_pb2 import Boolean

UPDATE_HZ = 30.0


def quat_mul(a, b):
    """Hamilton product, quaternions as (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


# ENU<-NED frame rotation and FRD->FLU body flip (both are 180-deg-class fixed
# rotations; combined they convert PX4's NED/FRD attitude to Gazebo's ENU/FLU).
S2 = math.sqrt(0.5)
Q_ENU_NED = (0.0, S2, S2, 0.0)
Q_FRD_FLU = (0.0, 1.0, 0.0, 0.0)


def ned_to_enu_pose(n, e, d, q_ned):
    pos = (e, n, -d)
    q = quat_mul(quat_mul(Q_ENU_NED, q_ned), Q_FRD_FLU)
    return pos, q


class Mirror:
    def __init__(self, world, model, origin_enu):
        self.world = world
        self.model = model
        self.origin = origin_enu
        self.node = Node()
        self.service = f"/world/{world}/set_pose"
        self.pos_ned = (0.0, 0.0, 0.0)
        self.q_ned = (1.0, 0.0, 0.0, 0.0)
        self.updates_sent = 0

    def push(self):
        (e0, n0, u0) = self.origin
        (px, py, pz), (qw, qx, qy, qz) = ned_to_enu_pose(*self.pos_ned, self.q_ned)
        req = Pose()
        req.name = self.model
        req.position.x = e0 + px
        req.position.y = n0 + py
        req.position.z = u0 + pz
        req.orientation.w = qw
        req.orientation.x = qx
        req.orientation.y = qy
        req.orientation.z = qz
        ok, _resp = self.node.request(self.service, req, Pose, Boolean, 200)
        if ok:
            self.updates_sent += 1
        return ok


async def run(args):
    mirror = Mirror(args.world, args.model, (args.origin_e, args.origin_n, args.origin_u))

    drone = System()
    print(f"[pose_mirror] connecting to PX4 on {args.mav_url} ...", flush=True)
    await drone.connect(system_address=args.mav_url)
    async for state in drone.core.connection_state():
        if state.is_connected:
            break
    print("[pose_mirror] PX4 connected.", flush=True)

    async def track_position():
        async for pv in drone.telemetry.position_velocity_ned():
            p = pv.position
            mirror.pos_ned = (p.north_m, p.east_m, p.down_m)

    async def track_attitude():
        async for q in drone.telemetry.attitude_quaternion():
            mirror.q_ned = (q.w, q.x, q.y, q.z)

    async def push_loop():
        n_fail = 0
        while True:
            ok = await asyncio.get_event_loop().run_in_executor(None, mirror.push)
            n_fail = 0 if ok else n_fail + 1
            if n_fail in (25, 250):
                print(f"[pose_mirror] WARNING: {n_fail} consecutive set_pose failures", flush=True)
            if mirror.updates_sent == 1:
                print("[pose_mirror] first pose pushed to Gazebo.", flush=True)
            await asyncio.sleep(1.0 / UPDATE_HZ)

    await asyncio.gather(track_position(), track_attitude(), push_loop())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="detection_world")
    ap.add_argument("--model", default="target_drone")
    ap.add_argument("--origin-e", type=float, default=50.0)
    ap.add_argument("--origin-n", type=float, default=0.0)
    ap.add_argument("--origin-u", type=float, default=0.15)
    # GCS broadcast link: LOCAL_POSITION_NED + ATTITUDE_QUATERNION at 50 Hz,
    # and it leaves 14540 (offboard API link) free for fly_mission.py.
    ap.add_argument("--mav-url", default="udpin://0.0.0.0:14550")
    args = ap.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
