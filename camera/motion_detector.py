#!/usr/bin/env python3
"""Moving-object detection for a STATIC station camera.

The station camera never moves, and every target of interest does. A rolling
median background model makes a moving 4-pixel speck stand out against static
canopy or rock that defeats the appearance detector — the Drone-vs-Bird
literature's classically under-exploited lever. This is a candidate GENERATOR:
it cannot tell drone from bird (both move); classification stays with the
track-level machinery.

Honesty guards: Gaussian pixel noise is injected before differencing (a real
sensor is not noise-free), and results from the sway-free simulator are an
UPPER BOUND — real foliage motion will cost false candidates that the travel
gate and motion classifier must absorb.

    python camera/motion_detector.py run --clip data/clips/terrain_canopy
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera.ir_detector import _blobs  # noqa: E402

BG_WINDOW = 9            # frames in the rolling median background
NOISE_DN = 2.0           # injected sensor noise, digital numbers
DIFF_THRESH = 14.0       # background difference threshold, DN
MIN_AREA = 2             # pixels
MAX_AREA = 2500          # larger movers = camera bump / scene change, drop
MAX_CANDIDATES = 40      # per frame; above this the scene model has failed


def _global_shift(a, b):
    """Whole-frame translation (dx, dy) from a to b, by phase correlation.

    Cheap, robust to the low-contrast structure of cloud, and does not need
    features. Used to keep the background model registered to a MOVING sky.
    """
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    fa = np.fft.rfft2((a - a.mean()) * win)
    fb = np.fft.rfft2((b - b.mean()) * win)
    cross = fa.conj() * fb
    mag = np.abs(cross)
    mag[mag < 1e-12] = 1e-12
    corr = np.fft.irfft2(cross / mag, s=a.shape)
    peak = int(np.argmax(corr))
    py, px = divmod(peak, w)
    if py > h // 2:
        py -= h
    if px > w // 2:
        px -= w
    return float(px), float(py)


def detect_stream(frames, thresh=DIFF_THRESH, rng=None,
                  scene_motion=True, flood_limit=MAX_CANDIDATES):
    """Yield per-frame candidate lists from an iterable of RGB arrays.

    scene_motion: register the background model to whole-frame motion before
    differencing. A rolling-median background assumes a STATIC scene; drifting
    cloud violates that and every cloud edge becomes "a mover" - measured,
    8,055 alarms/min on a broken-cloud clip, which made the channel unusable
    in exactly the weather where the cameras need help most.

    flood_limit: if a frame still yields more candidates than this, the scene
    model is not describing the scene (fast cloud, camera bump, rain streaks).
    Emitting hundreds of movers is worse than emitting none, so the frame is
    reported EMPTY and flagged. Silence is an honest answer; a flood is not.
    """
    rng = rng or np.random.default_rng(0)
    window: deque = deque(maxlen=BG_WINDOW)
    prev = None
    for frame in frames:
        grey = frame.astype(np.float32).mean(axis=2)
        grey += rng.normal(0.0, NOISE_DN, grey.shape).astype(np.float32)
        if scene_motion and prev is not None and window:
            dx, dy = _global_shift(prev, grey)
            if abs(dx) >= 1.0 or abs(dy) >= 1.0:
                # roll the whole background history into the new frame's
                # reference so the median is compared like-for-like
                for i in range(len(window)):
                    window[i] = np.roll(np.roll(window[i], int(round(dy)),
                                                axis=0),
                                        int(round(dx)), axis=1)
        prev = grey
        dets = []
        if len(window) >= BG_WINDOW // 2 + 1:
            bg = np.median(np.stack(window), axis=0)
            diff = np.abs(grey - bg)
            mask = diff > thresh
            # cv2 connected components when available: the pure-python
            # flood fill is fine on 720p sim frames and takes tens of
            # minutes per 1080p real clip. Identical semantics.
            try:
                import cv2
                n, lab, stats, cent = cv2.connectedComponentsWithStats(
                    mask.astype(np.uint8), connectivity=8)
                for i in range(1, n):
                    area = int(stats[i, cv2.CC_STAT_AREA])
                    if not (MIN_AREA <= area <= MAX_AREA):
                        continue
                    x = int(stats[i, cv2.CC_STAT_LEFT])
                    y = int(stats[i, cv2.CC_STAT_TOP])
                    w = int(stats[i, cv2.CC_STAT_WIDTH])
                    h = int(stats[i, cv2.CC_STAT_HEIGHT])
                    cx, cy = float(cent[i][0]), float(cent[i][1])
                    strength = float(diff[y:y + h, x:x + w].max())
                    dets.append({
                        "xyxy": [cx - w / 2, cy - h / 2,
                                 cx + w / 2, cy + h / 2],
                        "conf": round(min(1.0, strength / 80.0), 4),
                        "cls": "mover",
                        "area_px": area,
                    })
            except ImportError:
                for blob in _blobs(mask):
                    if not (MIN_AREA <= len(blob) <= MAX_AREA):
                        continue
                    arr = np.array(blob)
                    cy, cx = arr[:, 0].mean(), arr[:, 1].mean()
                    h = np.ptp(arr[:, 0]) + 1
                    w = np.ptp(arr[:, 1]) + 1
                    strength = float(diff[arr[:, 0], arr[:, 1]].max())
                    dets.append({
                        "xyxy": [cx - w / 2, cy - h / 2,
                                 cx + w / 2, cy + h / 2],
                        "conf": round(min(1.0, strength / 80.0), 4),
                        "cls": "mover",
                        "area_px": len(blob),
                    })
        window.append(grey)
        if len(dets) > flood_limit:
            # scene model has lost the scene - say nothing rather than flood
            dets = []
        yield dets


def run(clip: Path, thresh: float, scene_motion: bool = True,
        flood_limit: int = MAX_CANDIDATES):
    frames_dir = clip / "frames"
    files = sorted(frames_dir.glob("*.png"))
    out = clip / "detections_motion.jsonl"
    n = 0
    def frame_iter():
        for f in files:
            yield np.array(Image.open(f).convert("RGB"))
    with open(out, "w") as fh:
        for f, dets in zip(files, detect_stream(frame_iter(), thresh,
                                                scene_motion=scene_motion,
                                                flood_limit=flood_limit)):
            fh.write(json.dumps({"frame": f.name, "detections": dets}) + "\n")
            n += 1
            if n % 100 == 0:
                print(f"  {n} frames", flush=True)
    print(f"wrote {out} ({n} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run"])
    ap.add_argument("--clip", required=True)
    ap.add_argument("--thresh", type=float, default=DIFF_THRESH)
    ap.add_argument("--no-scene-motion", dest="scene_motion",
                    action="store_false",
                    help="disable global-motion registration of the "
                         "background model (the pre-2026-08-29 behaviour)")
    ap.add_argument("--flood-limit", type=int, default=MAX_CANDIDATES)
    args = ap.parse_args()
    run(Path(args.clip), args.thresh, args.scene_motion, args.flood_limit)


if __name__ == "__main__":
    main()
