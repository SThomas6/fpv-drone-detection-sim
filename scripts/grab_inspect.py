#!/usr/bin/env python3
"""Grab one frame from the third-person /inspect/image camera.

There has never been a picture of the rig itself - every image in this
project is the view FROM the station, so nothing shows what the station is.
This world adds a camera looking back at it.

    python3 scripts/grab_inspect.py --out rig.png
"""
import argparse
import threading
import time

from gz.msgs10.image_pb2 import Image
from gz.transport13 import Node
import numpy as np
from PIL import Image as PILImage

_lock = threading.Lock()
_latest = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--topic", default="/inspect/image")
    ap.add_argument("--wait", type=float, default=40.0)
    args = ap.parse_args()

    def cb(msg: Image):
        with _lock:
            _latest["img"] = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3).copy()

    node = Node()
    node.subscribe(Image, args.topic, cb)
    deadline = time.time() + args.wait
    while time.time() < deadline:
        with _lock:
            if "img" in _latest:
                break
        time.sleep(0.2)
    with _lock:
        img = _latest.get("img")
    if img is None:
        raise SystemExit(f"no frame on {args.topic} after {args.wait:.0f}s")
    PILImage.fromarray(img).save(args.out)
    print(f"wrote {args.out} {img.shape[1]}x{img.shape[0]}")


if __name__ == "__main__":
    main()
