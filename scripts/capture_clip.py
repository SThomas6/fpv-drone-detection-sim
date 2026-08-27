#!/usr/bin/env python3
"""Record a clip from the detection-station camera while the drone flies.

Samples the camera topic at a fixed interval, draws the drone's ground-truth
position on each frame, and writes an animated GIF plus the raw PNG frames.

Used as visual proof the simulator works, and in Phase 2 as the recorder for
the detection test matrix (clips at different ranges/backgrounds/lighting).

    conda activate dronesim
    python scripts/capture_clip.py --seconds 40 --out logs/clip
"""

import argparse
import math
import os
import threading
import time

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

_latest = {}
_lock = threading.Lock()


def project(p_world):
    """World point -> (u, v, range) in pixels, or None if behind the camera."""
    th = CAM_PITCH
    ry = np.array([[math.cos(th), 0, math.sin(th)],
                   [0, 1, 0],
                   [-math.sin(th), 0, math.cos(th)]])
    d = ry.T @ (np.asarray(p_world) - CAM_POS)   # camera frame: +X fwd, +Y left, +Z up
    if d[0] <= 0.5:
        return None
    fx = (W / 2) / math.tan(HFOV / 2)
    return W / 2 - fx * d[1] / d[0], H / 2 - fx * d[2] / d[0], float(np.linalg.norm(d))


def img_cb(msg: Image):
    with _lock:
        _latest["img"] = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3).copy()


def pose_cb(msg: Pose_V):
    for p in msg.pose:
        if p.name == "target_drone" or p.name.startswith("x500"):
            with _lock:
                _latest["pos"] = (p.position.x, p.position.y, p.position.z)


def annotate(frame, pos, scale):
    img = PILImage.fromarray(frame)
    px = project(pos)
    if px:
        u, v, rng = px
        r = max(7.0, 900.0 / rng)
        d = ImageDraw.Draw(img)
        d.rectangle([u - r, v - r, u + r, v + r], outline=(255, 40, 40), width=3)
        d.text((u + r + 6, v - r - 4), f"{rng:.0f} m", fill=(255, 40, 40))
    if scale != 1.0:
        img = img.resize((int(W * scale), int(H * scale)), PILImage.LANCZOS)
    return img, (px[2] if px else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between samples")
    ap.add_argument("--scale", type=float, default=0.5, help="output size vs 1280x720")
    ap.add_argument("--out", default="logs/clip", help="output dir (GIF written alongside)")
    args = ap.parse_args()

    node = Node()
    node.subscribe(Image, CAM_TOPIC, img_cb)
    node.subscribe(Pose_V, POSE_TOPIC, pose_cb)

    deadline = time.time() + 20
    while time.time() < deadline:
        with _lock:
            if "img" in _latest and "pos" in _latest:
                break
        time.sleep(0.2)
    else:
        raise SystemExit("timed out waiting for camera frame / target pose — is the sim running?")

    os.makedirs(args.out, exist_ok=True)
    frames, ranges = [], []
    n = int(args.seconds / args.interval)
    for i in range(n):
        with _lock:
            frame, pos = _latest.get("img"), _latest.get("pos")
        if frame is not None and pos is not None:
            img, rng = annotate(frame, pos, args.scale)
            img.save(os.path.join(args.out, f"frame_{i:03d}.png"))
            frames.append(img)
            if rng:
                ranges.append(rng)
                print(f"  frame {i + 1}/{n}: target at {rng:.0f} m", flush=True)
        time.sleep(args.interval)

    if not frames:
        raise SystemExit("no frames captured")
    gif = args.out.rstrip("/") + ".gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:],
                   duration=int(args.interval * 1000), loop=0)
    print(f"\nsaved {len(frames)} frames to {args.out}/ and {gif}")
    if ranges:
        print(f"target range over clip: {min(ranges):.0f} m to {max(ranges):.0f} m")


if __name__ == "__main__":
    main()
