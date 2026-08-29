#!/usr/bin/env python3
"""Motion blur at 150 kph: how short does the exposure have to be?

Gazebo renders every frame with an instantaneous shutter, so no clip in this
project contains motion blur at all. That is harmless at the 14 m/s the
profiles used to fly and dishonest at 41.7 m/s (150 kph), because blur length
scales with angular rate AND with focal length - and the telephoto's focal
length is 12212 px.

    blur_px = (v_tangential / range) * focal_px * exposure_s

A 0.34 m drone crossing at 41.7 m/s at 500 m through the 6 deg lens moves
1018 px/s. At 1/60 s that smears it over 17 px - the target is 8 px wide, so
its contrast is spread over twice its own length and largely destroyed. At
1/1000 s it is 1 px and nothing happens. Exposure is therefore a first-class
design parameter, and it trades directly against light: 1/1000 s is four
stops down from 1/60 s, which is exactly the budget night operation does not
have.

Modelled by convolving a directional line kernel over a patch around the
target. On a UNIFORM background (sky) that is exact, because blurring a
constant is a no-op and only the target smears. On textured terrain it also
blurs the clutter, which flatters detection slightly - so the rock numbers
here are an optimistic bound, and are labelled as such.

    python scripts/motion_blur.py --clip data/clips/range_tele_sky
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera.evaluate import is_hit  # noqa: E402
from scripts.point_target_detect import detect  # noqa: E402

EXPOSURES = [("1/2000", 1 / 2000), ("1/1000", 1 / 1000), ("1/500", 1 / 500),
             ("1/250", 1 / 250), ("1/125", 1 / 125), ("1/60", 1 / 60)]


def line_kernel(length_px: float, angle_deg: float) -> np.ndarray:
    """Normalised line PSF: what a point becomes while the shutter is open."""
    n = max(1, int(round(length_px)))
    if n <= 1:
        return None
    k = np.zeros((n, n), np.float32)
    cv2.line(k, (0, n // 2), (n - 1, n // 2), 1.0, 1)
    m = cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, m, (n, n))
    tot = k.sum()
    return k / tot if tot > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--speed", type=float, default=41.7,
                    help="tangential speed, m/s (41.7 = 150 kph)")
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--pad", type=int, default=40,
                    help="half-size of the blurred patch around the target")
    ap.add_argument("--k", type=float, default=6.0)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="sensor noise sigma in DN added before detection. "
                         "The RENDERED sky has sigma 0.00 - a flat colour with "
                         "no grain, haze or gradient - so a blurred target "
                         "stays infinitely detectable against it. Real "
                         "imagery does not, and this is the knob that says "
                         "how much that mattered")
    args = ap.parse_args()

    clip = Path(args.clip)
    meta = json.loads((clip / "meta.json").read_text())
    fx = float(meta.get("fx", 1108.77))
    recs = [json.loads(l) for l in open(clip / "labels.jsonl")
            if json.loads(l).get("visible")][::args.step]
    print(f"\n{clip.name}: fx={fx:.0f}, {len(recs)} frames, "
          f"tangential {args.speed:.1f} m/s ({args.speed*3.6:.0f} kph), noise sigma {args.noise:.1f} DN")
    print(f"{'exposure':>9} {'blur @500m':>11} {'blur @1km':>10} "
          f"{'detected':>9} {'vs sharp':>9}")

    rng = np.random.default_rng(5)
    base = None
    for name, sec in [("sharp", 0.0)] + EXPOSURES:
        hits = tot = 0
        for rec in recs:
            im = cv2.imread(str(clip / "frames" / rec["frame"]),
                            cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            r = rec.get("range_m") or 1e9
            blur = (args.speed / r) * fx * sec
            gx = int((rec["bbox"][0] + rec["bbox"][2]) / 2)
            gy = int((rec["bbox"][1] + rec["bbox"][3]) / 2)
            if blur >= 2.0:
                k = line_kernel(blur, 0.0)     # horizontal crossing motion
                if k is not None:
                    y0 = max(0, gy - args.pad); y1 = min(im.shape[0], gy + args.pad)
                    x0 = max(0, gx - args.pad); x1 = min(im.shape[1], gx + args.pad)
                    im = im.copy()
                    im[y0:y1, x0:x1] = cv2.filter2D(
                        im[y0:y1, x0:x1], -1, k, borderType=cv2.BORDER_REFLECT)
            if args.noise > 0:
                im = np.clip(im.astype(np.float32)
                             + rng.normal(0, args.noise, im.shape),
                             0, 255).astype(np.uint8)
            dets = detect(im, [9, 21], args.k, "dark", 30)
            tot += 1
            hits += any(is_hit(d["xyxy"], rec) for d in dets)
        rate = hits / max(tot, 1)
        if base is None:
            base = rate
        b500 = (args.speed / 500) * fx * sec
        b1k = (args.speed / 1000) * fx * sec
        print(f"{name:>9} {b500:>9.1f}px {b1k:>8.1f}px {rate:>8.0%} "
              f"{rate - base:>+8.0%}", flush=True)


if __name__ == "__main__":
    main()
