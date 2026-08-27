#!/usr/bin/env python3
"""Phase 1 smoke test: is the whole simulator stack actually up?

Run AFTER scripts/start_sim.sh, inside the dronesim conda env:

    conda activate dronesim && python tests/test_phase1.py

Checks:
  1. Gazebo server up (world clock + camera topics listed)
  2. Camera frames arrive in Python via gz-transport (offscreen rendering works)
  3. Camera frames arrive on ROS 2 via the ros_gz bridge (rclpy)
  4. PX4 telemetry reachable via MAVSDK (onboard-payload stream on 14030 —
     14540 is held by fly_mission.py and 14550 by pose_mirror.py)
  5. The drone is actually moving (PX4 position AND the Gazebo ground-truth
     pose of the target drone both displace over a 6 s window)

Exit code = number of failed checks. --no-fly skips the movement check
(for when start_sim.sh was run with --no-fly).
"""

import argparse
import asyncio
import math
import subprocess
import sys
import threading
import time

WORLD = "detection_world"
CAM_TOPIC = "/detection_station/camera/image"
RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    return ok


# ---- 1. gz topics ---------------------------------------------------------

def check_gz_topics():
    print("[1/5] Gazebo server topics", flush=True)
    try:
        out = subprocess.run(["gz", "topic", "-l"], capture_output=True, text=True,
                             timeout=20).stdout
    except Exception as e:  # noqa: BLE001
        return report("gz topic -l", False, str(e))
    report(f"world clock topic (/world/{WORLD}/clock)", f"/world/{WORLD}/clock" in out)
    return report(f"camera topic ({CAM_TOPIC})", CAM_TOPIC in out)


# ---- 2. camera frames via gz-transport ------------------------------------

def check_gz_camera(timeout_s=20.0):
    print("[2/5] Camera frames via gz-transport (offscreen render)", flush=True)
    from gz.msgs10.image_pb2 import Image
    from gz.transport13 import Node

    got = {}
    evt = threading.Event()

    def cb(msg: Image):
        got["wh"] = (msg.width, msg.height)
        evt.set()

    node = Node()
    if not node.subscribe(Image, CAM_TOPIC, cb):
        return report("subscribe camera topic", False)
    if not evt.wait(timeout_s):
        return report("camera frame received", False, f"nothing in {timeout_s:.0f}s")
    w, h = got["wh"]
    return report("camera frame received", (w, h) == (1280, 720), f"{w}x{h}")


# ---- 3. camera frames via ROS 2 -------------------------------------------

def check_ros_camera(timeout_s=20.0):
    print("[3/5] Camera frames via ROS 2 (ros_gz bridge + rclpy)", flush=True)
    try:
        import rclpy
        from sensor_msgs.msg import Image
    except ImportError as e:
        return report("rclpy import", False, str(e))

    rclpy.init()
    node = rclpy.create_node("phase1_smoke_test")
    got = {}

    def cb(msg):
        got["wh"] = (msg.width, msg.height)

    node.create_subscription(Image, CAM_TOPIC, cb, 1)
    deadline = time.time() + timeout_s
    while time.time() < deadline and "wh" not in got:
        rclpy.spin_once(node, timeout_sec=0.5)
    node.destroy_node()
    rclpy.shutdown()
    if "wh" not in got:
        return report("ROS image received", False, f"nothing in {timeout_s:.0f}s")
    return report("ROS image received", got["wh"] == (1280, 720), f"{got['wh'][0]}x{got['wh'][1]}")


# ---- 4 & 5. PX4 telemetry + movement --------------------------------------

async def sample_px4_position(drone):
    async for pos in drone.telemetry.position():
        return (pos.latitude_deg, pos.longitude_deg, pos.absolute_altitude_m)


def gz_target_position():
    """Ground-truth XYZ of target_drone from the world pose topic (one shot)."""
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.transport13 import Node

    got = {}
    evt = threading.Event()

    def cb(msg: Pose_V):
        for p in msg.pose:
            if p.name == "target_drone" or p.name.startswith("x500"):
                got["xyz"] = (p.position.x, p.position.y, p.position.z)
                evt.set()

    node = Node()
    node.subscribe(Pose_V, f"/world/{WORLD}/dynamic_pose/info", cb)
    evt.wait(10.0)
    return got.get("xyz")


def check_px4_and_movement(expect_flying: bool):
    print("[4/5] PX4 telemetry via MAVSDK", flush=True)
    from mavsdk import System

    async def go():
        drone = System()
        await drone.connect(system_address="udpin://0.0.0.0:14030")
        try:
            await asyncio.wait_for(_wait_conn(drone), timeout=30)
        except asyncio.TimeoutError:
            report("PX4 MAVLink heartbeat", False, "no connection in 30s")
            return
        report("PX4 MAVLink heartbeat", True)
        p1 = await asyncio.wait_for(sample_px4_position(drone), timeout=20)
        report("PX4 position telemetry", p1 is not None,
               f"alt {p1[2]:.1f} m" if p1 else "")

        print("[5/5] Movement (6 s displacement window)", flush=True)
        if not expect_flying:
            print("  [SKIP] --no-fly given", flush=True)
            return
        g1 = gz_target_position()
        await asyncio.sleep(6)
        p2 = await asyncio.wait_for(sample_px4_position(drone), timeout=20)
        g2 = gz_target_position()
        if p1 and p2:
            dist = math.hypot((p2[0] - p1[0]) * 111_320,
                              (p2[1] - p1[1]) * 76_000) + abs(p2[2] - p1[2])
            report("PX4 reports movement", dist > 1.5, f"~{dist:.1f} m in 6 s")
        else:
            report("PX4 reports movement", False, "position sampling failed")
        if g1 and g2:
            gd = math.dist(g1, g2)
            report("Gazebo target moved (ground truth)", gd > 1.5, f"{gd:.1f} m in 6 s")
        else:
            report("Gazebo target moved (ground truth)", False,
                   "target pose not seen on dynamic_pose/info")

    async def _wait_conn(drone):
        async for st in drone.core.connection_state():
            if st.is_connected:
                return

    asyncio.run(go())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fly", action="store_true",
                    help="skip the movement check")
    args = ap.parse_args()

    print("=== Phase 1 smoke test ===", flush=True)
    check_gz_topics()
    check_gz_camera()
    check_ros_camera()
    check_px4_and_movement(expect_flying=not args.no_fly)

    fails = RESULTS.count(False)
    print(f"\n=== {len(RESULTS) - fails}/{len(RESULTS)} checks passed ===", flush=True)
    sys.exit(fails)


if __name__ == "__main__":
    main()
