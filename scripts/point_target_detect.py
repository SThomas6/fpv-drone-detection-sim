#!/usr/bin/env python3
"""Single-frame point-target detection for the TELEPHOTO channel.

Measured on data/clips/range_tele: a 6 deg lens keeps a 0.34 m airframe
1.3-4.5x above the local background texture all the way to 1400 m, while
the 60 deg station lens leaves it 10-100x BELOW it. The signal is there.
The trained detector still scored 0% past 700 m - because it is a YOLO
model whose training set contains no 6 deg imagery at all, so a 4 px blob
on 11x-magnified rock is out of distribution. That is a data problem being
mistaken for a sensor problem.

A point target has no shape to learn - it is a few pixels darker than the
local background - so the right detector for this channel is the same one
the thermal channel already uses: estimate the local background, keep the
residual, threshold on the LOCAL noise. Two properties matter at 6 deg:

  * magnification resolves the clutter. Terrain texture measured 8.7-9.5 DN
    in the telephoto against 62-64 DN in the wide camera, because each rock
    feature is now spread over ~11x11 px instead of sitting inside one. The
    clutter stops looking like point targets, which is what actually kills
    small-target detection at wide FOV.
  * the target is DARK on a bright background here (airframe diffuse 0.08),
    so the residual is taken as background-minus-image; --polarity both
    keeps the bright-target case for other scenes.

    python scripts/point_target_detect.py --clip data/clips/range_tele
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera.evaluate import is_hit  # noqa: E402


def _score_at_scale(g: np.ndarray, bg_px: int, polarity: str):
    """Normalised point-target response at ONE background scale."""
    # median background: robust to the target itself, unlike a box blur
    bg = cv2.medianBlur(g.astype(np.uint8), bg_px).astype(np.float32)
    dark, bright = bg - g, g - bg
    res = dark if polarity == "dark" else (
        bright if polarity == "bright" else np.maximum(dark, bright))
    # LOCAL noise, so a textured hillside raises its own threshold instead of
    # a global sigma being set by whichever part of the frame is busiest
    mean = cv2.blur(res, (65, 65))
    sq = cv2.blur(res * res, (65, 65))
    sigma = np.sqrt(np.maximum(sq - mean * mean, 1e-6))
    return (res - mean) / sigma


def detect(gray: np.ndarray, bg_px, k: float, polarity: str,
           max_dets: int):
    """Local-contrast point targets, across several background scales.

    One scale is not enough, and the failure is asymmetric. A median window
    only rejects the target if the target occupies well under half of it, so
    a 9 px window absorbs an 11 px target and the residual collapses. That
    produced a table where 300-400 m scored WORSE (80%) than 400-550 m (97%)
    - a detection rate rising with range, which is a tuning artefact, not
    physics. Scoring at several window sizes and keeping the strongest
    response covers the whole span from a 3 px speck to a 15 px airframe.
    """
    g = gray.astype(np.float32)
    scales = [bg_px] if isinstance(bg_px, int) else list(bg_px)
    score = _score_at_scale(g, scales[0], polarity)
    for s in scales[1:]:
        np.maximum(score, _score_at_scale(g, s, polarity), out=score)
    mask = (score > k).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area > 400:                 # a cloud edge or ridge line, not a point
            continue
        cx, cy = cent[i]
        # Confidence is the blob's PEAK score, not the score at its centroid.
        # A blob need not contain its own centroid - an annulus around a
        # bright core does not - so the centroid pixel can sit below
        # threshold, and was measured at -0.05 on a real detection. Any
        # downstream `conf >= 0` filter then silently discarded a correct
        # detection: it cost 3-12 percentage points per range bin before the
        # cause was found.
        sub = score[y:y + h, x:x + w][lab[y:y + h, x:x + w] == i]
        out.append({"xyxy": [cx - w / 2 - 2, cy - h / 2 - 2,
                             cx + w / 2 + 2, cy + h / 2 + 2],
                    "conf": float(sub.max()) if sub.size else float(k),
                    "cls": "point"})
    out.sort(key=lambda d: -d["conf"])
    return out[:max_dets]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--bg-px", type=int, nargs="+", default=[9, 21],
                    help="median background windows (odd). Several scales: a "
                         "window the target fills stops rejecting it, so one "
                         "size cannot span 3 px specks and 15 px airframes")
    ap.add_argument("--k", type=float, default=6.0,
                    help="threshold in local sigmas")
    ap.add_argument("--polarity", default="dark",
                    choices=["dark", "bright", "both"])
    ap.add_argument("--max-dets", type=int, default=30,
                    help="per-frame cap: the alarm budget a cued channel gets")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--alarm-budget", type=float, default=3.0,
                    help="false alarms/frame above which a bin's detection "
                         "rate is flagged as clutter coincidence, not signal")
    ap.add_argument("--frames-subdir", default="frames",
                    help="which image stream to run on; frames_zoom is the "
                         "telephoto view of a dual-camera clip")
    ap.add_argument("--labels", default="labels.jsonl",
                    help="label file matching --frames-subdir's pixel frame")
    ap.add_argument("--out-name", default="point",
                    help="writes detections_<out-name>.jsonl")
    ap.add_argument("--save", action="store_true",
                    help="write detections_point.jsonl into the clip")
    args = ap.parse_args()

    clip = Path(args.clip)
    recs = [json.loads(l) for l in open(clip / args.labels)]
    BINS = [(300, 450), (450, 550), (550, 700), (700, 850), (850, 1000),
            (1000, 1150), (1150, 1300), (1300, 1500)]
    rows = {b: [0, 0, 0.0, 0.0] for b in BINS}     # n, hits, alarms, px
    fh = (open(clip / f"detections_{args.out_name}.jsonl", "w")
          if args.save else None)
    for rec in recs[::args.step]:
        im = np.array(Image.open(
            clip / args.frames_subdir / rec["frame"]).convert("L"))
        dets = detect(im, args.bg_px, args.k, args.polarity, args.max_dets)
        if fh:
            fh.write(json.dumps({"frame": rec["frame"],
                                 "detections": dets}) + "\n")
        if not rec.get("visible"):
            continue
        b = next((b for b in BINS
                  if b[0] <= (rec.get("range_m") or 0) < b[1]), None)
        if b is None:
            continue
        row = rows[b]
        row[0] += 1
        hit = [d for d in dets if is_hit(d["xyxy"], rec)]
        row[1] += bool(hit)
        row[2] += len(dets) - len(hit)
        row[3] += rec["bbox"][2] - rec["bbox"][0]
    if fh:
        fh.close()
    print(f"\n{clip.name}: point-target detector, k={args.k} sigma, "
          f"bg {args.bg_px} px, polarity {args.polarity}")
    print(f"{'range':>12} {'frames':>7} {'target px':>10} {'detected':>9} "
          f"{'false/frame':>12}  note")
    saturated = False
    for b in BINS:
        n, hits, alarms, px = rows[b]
        if not n:
            continue
        # A "detection rate" measured while the alarm cap is full is not a
        # detection rate: with tens of clutter points per frame, one lands
        # inside the is_hit tolerance by coincidence. Measured on the WIDE
        # camera this produced a rate that ROSE with range (0% at 550-700 m,
        # 100% at 850-1000 m) - impossible, and the giveaway. Flag it rather
        # than let it into a fused table.
        bad = alarms / n > args.alarm_budget
        saturated = saturated or bad
        note = "CLUTTER-SATURATED - rate not meaningful" if bad else ""
        print(f"{f'{b[0]}-{b[1]} m':>12} {n:>7} {px / n:>9.2f}p "
              f"{hits / n:>8.0%} {alarms / n:>12.1f}  {note}")
    if saturated:
        print(f"\nWARNING: bins above {args.alarm_budget} false alarms/frame "
              f"are clutter, not signal.\nThis detector assumes the target "
              f"rises above the local background; where it\ndoes not, the cap "
              f"fills with terrain texture and any hit is coincidence.\n"
              f"Do not fuse those rows.")


if __name__ == "__main__":
    main()
