#!/usr/bin/env python3
"""Capture one station-camera frame and mark the target drone's true position.

Grabs a frame from the Gazebo camera topic plus the drone's ground-truth pose,
projects that pose into the image with the camera's known intrinsics/extrinsics,
draws a box there, and saves a PNG. Useful as visual proof the camera sees the
drone, and later to sanity-check Phase 2 detections against ground truth.

    conda activate dronesim && python scripts/capture_frame.py --out frame.png
"""

import argparse
import math
import threading

import numpy as np
from PIL import Image as PILImage, ImageDraw

from gz.msgs10.image_pb2 import Image
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node

WORLD = "detection_world"
CAM_TOPIC = "/detection_station/camera/image"
POSE_TOPIC = f"/world/{WORLD}/dynamic_pose/info"

# Must match simulator/worlds/detection_world.sdf (camera_link + sensor).
CAM_POS = np.array([0.1, 0.0, 2.5])
CAM_PITCH = -0.15
W, H, HFOV = 1280, 720, 1.047


def project(p_world):
    """World point -> pixel (u, v), or None if behind the camera."""
    th = CAM_PITCH
    ry = np.array([[math.cos(th), 0, math.sin(th)],
                   [0, 1, 0],
                   [-math.sin(th), 0, math.cos(th)]])
    d = ry.T @ (np.asarray(p_world) - CAM_POS)   # camera frame: +X fwd, +Y left, +Z up
    if d[0] <= 0.5:
        return None
    fx = (W / 2) / math.tan(HFOV / 2)
    u = W / 2 - fx * d[1] / d[0]
    v = H / 2 - fx * d[2] / d[0]
    return u, v, float(np.linalg.norm(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/frame.png")
    ap.add_argument("--target", default="target_drone",
                    help="model name (x500-based names also matched)")
    args = ap.parse_args()

    got = {}
    f_evt, p_evt = threading.Event(), threading.Event()

    def img_cb(msg: Image):
        if "img" not in got:
            got["img"] = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3).copy()
            f_evt.set()

    def pose_cb(msg: Pose_V):
        for p in msg.pose:
            if p.name == args.target or p.name.startswith("x500"):
                got["pos"] = (p.position.x, p.position.y, p.position.z)
                p_evt.set()

    node = Node()
    node.subscribe(Image, CAM_TOPIC, img_cb)
    node.subscribe(Pose_V, POSE_TOPIC, pose_cb)
    if not (f_evt.wait(15) and p_evt.wait(15)):
        raise SystemExit("timed out waiting for camera frame / target pose")

    img = PILImage.fromarray(got["img"])
    px = project(got["pos"])
    x, y, z = got["pos"]
    if px:
        u, v, rng = px
        r = max(8.0, 900.0 / rng)   # box shrinks with range, floor of 8 px
        draw = ImageDraw.Draw(img)
        draw.rectangle([u - r, v - r, u + r, v + r], outline=(255, 40, 40), width=2)
        draw.text((u + r + 4, v - r),
                  f"truth ({x:.0f},{y:.0f},{z:.0f})m rng {rng:.0f}m", fill=(255, 40, 40))
        print(f"target at world ({x:.1f}, {y:.1f}, {z:.1f}) -> pixel ({u:.0f}, {v:.0f}), range {rng:.0f} m")
        if not (0 <= u < W and 0 <= v < H):
            print("NOTE: target currently outside the camera frame")
    else:
        print(f"target at world ({x:.1f}, {y:.1f}, {z:.1f}) is behind the camera")
    img.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
