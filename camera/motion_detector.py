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

from camera.clutter_map import StaticClutterSuppressor  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera.ir_detector import _blobs  # noqa: E402

BG_WINDOW = 9            # frames in the rolling median background
NOISE_DN = 2.0           # injected sensor noise, digital numbers
DIFF_THRESH = 14.0       # background difference threshold, DN
MIN_AREA = 2             # pixels
MAX_AREA = 2500          # larger movers = camera bump / scene change, drop
MAX_CANDIDATES = 40      # per frame; above this the scene model has failed
STORM_PIXELS = 4000      # changed pixels that mean 'the scene itself moved'


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


class _Blob:
    """Adapter so a mover dict can be fed to StaticClutterSuppressor."""
    __slots__ = ("d",)

    def __init__(self, d):
        self.d = d

    @property
    def centre(self):
        x1, y1, x2, y2 = self.d["xyxy"]
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def width(self):
        return self.d["xyxy"][2] - self.d["xyxy"][0]


def detect_stream(frames, thresh=DIFF_THRESH, rng=None,
                  scene_motion=True, flood_limit=MAX_CANDIDATES,
                  fast_mover="auto", noise_k=6.0, clutter=False,
                  flood_mode="blank", dt=0.2,
                  clutter_radius=20.0, clutter_extent=14.0,
                  clutter_persist=10):
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
    # Foliage scale, not real-footage scale. The suppressor's 60 px
    # default anchor radius was tuned for building/foliage blur in
    # 1080p real clips; here it swallows the drone, because a drone
    # passing within 60 px of a swaying crown lands inside that
    # crown's anchor and gets muted with it (measured: everything
    # suppressed, 0% drone). A swaying crown at 44-92 m subtends
    # ~8-16 px of oscillation, so the anchor must be that tight.
    supp = (StaticClutterSuppressor(radius=clutter_radius,
                                    max_extent=clutter_extent,
                                    min_persist=clutter_persist)
            if clutter else None)
    t_now = 0.0
    window: deque = deque(maxlen=BG_WINDOW)
    prev = None
    # Sub-pixel drift accumulator. Cloud in these clips drifts ~0.35 px per
    # frame: a naive "roll if >= 1 px" test NEVER fires, the shift silently
    # accumulates across the 9-frame window, and the flood guard then has to
    # blank the frame. Carry the fraction and roll when it reaches a pixel.
    acc_x = acc_y = 0.0
    for frame in frames:
        grey = frame.astype(np.float32).mean(axis=2)
        grey += rng.normal(0.0, NOISE_DN, grey.shape).astype(np.float32)
        if scene_motion and prev is not None and window:
            dx, dy = _global_shift(prev, grey)
            if abs(dx) < 1.0 and abs(dy) < 1.0:
                # phase correlation is integer-valued; a sub-pixel drift
                # shows up as a long run of zeros with occasional ones.
                # Estimate the trend from the frame difference instead.
                acc_x += dx
                acc_y += dy
                dx = float(int(acc_x))
                dy = float(int(acc_y))
                acc_x -= dx
                acc_y -= dy
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
            # NOISE-ADAPTIVE THRESHOLD. A fixed 14 DN threshold assumes a
            # daylight noise floor. At night a high-gain camera's read+shot
            # noise alone exceeds it, every pixel "changes", and the flood
            # guard blanks the frame (measured: motion 83% day -> 5% moonlit).
            # Scale the threshold to the frame's own measured noise instead:
            # the median absolute difference IS the noise floor when most of
            # the scene is static, so k*MAD keeps the false rate roughly
            # constant across light levels. Never goes below the daylight
            # value, so clear-day behaviour is untouched.
            mad = float(np.median(diff))
            eff_thresh = max(thresh, noise_k * mad)
            if fast_mover == "auto":
                # Adaptive, because neither mode is right everywhere:
                #   standard  - sees a HOVERING drone (measured: clear-sky
                #               canopy 83% vs 66% under fast-mover, because
                #               a hovering target has little frame-to-frame
                #               motion)
                #   fast      - survives DRIFTING CLOUD (measured: cloud 2%
                #               vs 65%, and the detection ceiling 68% -> 92%)
                # Decide per frame on the evidence: if the plain background
                # difference is producing a candidate storm, the background
                # model is no longer describing the scene, so trust motion
                # speed instead of background difference.
                storm = int((diff > eff_thresh).sum())
                use_fast = storm > STORM_PIXELS
            else:
                use_fast = bool(fast_mover)
            if use_fast:
                # SPEED is what separates a drone from cloud. The median
                # background says "different from the last 4.5 s", which a
                # slowly evolving sky also satisfies. Requiring the pixel to
                # ALSO differ from the immediately preceding (registered)
                # frame demands motion on a ~0.5 s timescale: a drone crosses
                # several pixels in that time, cloud crosses a fraction of
                # one. Cheap, and it needs no cloud model.
                fast = np.abs(grey - window[-1])
                diff = np.minimum(diff, fast)
            mask = diff > eff_thresh
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
                    # Cloud EDGES differ from aircraft in shape: a mis-
                    # registered cloud rim is a long thin ribbon, an aircraft
                    # is compact. Reject extreme elongation and very sparse
                    # fills - both are edge artefacts, neither is a drone.
                    if max(w, h) > 3 * max(min(w, h), 1) and max(w, h) > 6:
                        continue
                    if area < 0.25 * w * h and area > 6:
                        continue
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

        # WIND. Swaying vegetation fires at one place forever and never
        # travels; a drone travels. That is exactly what the static-clutter
        # map decides, and it runs BEFORE the flood cap so foliage is removed
        # rather than crowding the drone out of the budget. Measured on
        # data/clips/sway_canopy: 40 swaying crowns produce ~77 movers/frame
        # without it, and the frame is then discarded whole.
        if supp is not None:
            kept, _ = supp.step([_Blob(d) for d in dets], t_now)
            dets = [b.d for b in kept]

        if len(dets) > flood_limit:
            if flood_mode == "blank":
                # MEASURED CORRECT, and not merely historic. Under wind the
                # channel does go blind (drone 65% -> 2% at this stage), but
                # end-to-end the fused system barely notices - camera and
                # thermal carry it, 57% coverage against 54% with no wind at
                # all - whereas emitting the flood instead drops fused
                # coverage to 20%, because ~39 movers/frame spawn rival
                # tracks that steal association from the drone's own. Silence
                # really is the better answer; the comment was right.
                dets = []
            else:
                # A cliff is not a safety guard. Blanking on ONE mover past
                # the limit took a frame from 41 detections to zero and
                # silently blinded the channel under wind - drone 65% -> 2%
                # while the ALARM RATE IMPROVED, so no metric flagged it.
                # Degrade instead: keep the strongest and mark the frame, so
                # a truncated frame is never read as a quiet one.
                dets = sorted(dets, key=lambda d: -d["conf"])[:flood_limit]
                for d in dets:
                    d["truncated"] = True
        t_now += dt
        yield dets


def run(clip: Path, thresh: float, scene_motion: bool = True,
        flood_limit: int = MAX_CANDIDATES, fast_mover="auto",
        noise_k: float = 6.0, clutter: bool = False,
        flood_mode: str = "blank",
        clutter_radius: float = 20.0, clutter_extent: float = 14.0,
        clutter_persist: int = 10):
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
                                                flood_limit=flood_limit,
                                                fast_mover=fast_mover,
                                                noise_k=noise_k,
                                                clutter=clutter,
                                                flood_mode=flood_mode,
                                                clutter_radius=clutter_radius,
                                                clutter_extent=clutter_extent,
                                                clutter_persist=clutter_persist)):
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
    ap.add_argument("--clutter", action="store_true",
                    help="suppress movers that keep firing at ONE place and "
                         "never travel - swaying foliage. Measured: 40 wind-"
                         "driven crowns produce ~77 movers/frame and blank "
                         "the whole frame without this")
    ap.add_argument("--clutter-radius", type=float, default=20.0)
    ap.add_argument("--clutter-extent", type=float, default=14.0)
    ap.add_argument("--clutter-persist", type=int, default=10)
    ap.add_argument("--flood-mode", default="blank",
                    choices=["blank", "truncate"],
                    help="what to do above --flood-limit. BLANK is correct "
                         "and stays the default: measured end-to-end under "
                         "wind, blanking gives 57% fused coverage against "
                         "20% for truncate, because a flood of movers spawns "
                         "rival tracks that disrupt the drone's own. Use "
                         "truncate as a DIAGNOSTIC - it makes a flood "
                         "visible instead of silent - not in deployment")
    ap.add_argument("--noise-k", type=float, default=6.0,
                    help="threshold = max(--thresh, k x median |difference|); "
                         "scales the detector to the frame's own noise floor")
    ap.add_argument("--fast-mover", default="auto",
                    choices=["auto", "on", "off"],
                    help="short-baseline motion requirement. auto (default): per-frame, engaged only when the background model is storming - keeps hovering targets in calm scenes and survives drifting cloud")
    args = ap.parse_args()
    run(Path(args.clip), args.thresh, args.scene_motion, args.flood_limit,
        {"auto": "auto", "on": True, "off": False}[args.fast_mover],
        args.noise_k, clutter=args.clutter,
        flood_mode=args.flood_mode,
        clutter_radius=args.clutter_radius,
        clutter_extent=args.clutter_extent,
        clutter_persist=args.clutter_persist)


if __name__ == "__main__":
    main()
