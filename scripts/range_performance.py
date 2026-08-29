#!/usr/bin/env python3
"""How often is the drone found, at what distance, by which sensor?

The question this answers: "what percentage of the time is the drone found
at different distances, with all the senses and capabilities combined?"

Reports, per range bin, the fraction of drone-visible frames in which each
sensor sees the target, plus the two numbers that matter operationally:

  ANY SENSOR   - the detection ceiling: something saw it this frame
  TRACKED      - the system holds a confirmed track on it (tracking coasts
                 through frames where every sensor blinked, so this can and
                 does exceed the per-frame ceiling)

Sensors, and what each row means honestly:
  wide RGB   - the 1280x720 station camera, native scale
  tiled RGB  - the same camera under SAHI tiling, the closest thing in the
               cached data to a ZOOM look (it infers at ~2x pixel scale)
  thermal    - the LWIR camera's hot-spot detector
  motion     - background-subtraction movers (static camera only)
  acoustic   - the mic array's bearing cue (range-limited by physics)
  radar      - the micro-Doppler model. NOTE: in the deployed architecture
               the radar is CUED, not scanning, so this row is a capability
               ceiling ("would it see it if pointed there"), not a
               free-running detection rate.

    python scripts/range_performance.py --clip data/clips/range_1km
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera.detector import Detection, merge_close  # noqa: E402
from camera.evaluate import is_hit  # noqa: E402
from camera.fuse_eval import fuse_measurements  # noqa: E402
from camera.tracking import CentroidTracker  # noqa: E402

BINS = [(0, 30), (30, 60), (60, 100), (100, 150), (150, 200), (200, 300),
        (300, 400), (400, 550), (550, 700), (700, 850), (850, 1100)]


def load(clip: Path, name: str):
    p = clip / f"detections_{name}.jsonl"
    if not p.exists():
        return None
    return {r["frame"]: r["detections"]
            for r in (json.loads(l) for l in open(p))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--conf", type=float, default=0.10)
    args = ap.parse_args()
    clip = Path(args.clip)

    labels = [json.loads(l) for l in open(clip / "labels.jsonl")]
    streams = {n: load(clip, n) for n in
               ("full", "sahi", "ir", "motion", "acoustic", "radar")}
    have = [n for n, v in streams.items() if v]
    print(f"\n{clip.name}: streams present = {', '.join(have)}")

    # tracked: replay the fused passive pipeline once
    tracker = CentroidTracker(class_consistent=True,
                              suppress_spawn_near_coasting=True)
    tracked_frames = set()
    for rec in labels:
        f = rec["frame"]
        rgb = [Detection(*d["xyxy"], confidence=d["conf"],
                         cls_name=d.get("cls", "drone"))
               for d in (streams["full"] or {}).get(f, [])
               if d["conf"] >= 0.05]
        aux = [Detection(*d["xyxy"], confidence=d["conf"], cls_name="hotspot")
               for d in (streams["ir"] or {}).get(f, []) if d["conf"] >= 0.2]
        aux += [Detection(*d["xyxy"], confidence=d["conf"], cls_name="mover")
                for d in (streams["motion"] or {}).get(f, [])
                if d["conf"] >= 0.10]
        dets, _ = fuse_measurements(merge_close(rgb), aux)
        for tr in tracker.update(dets, timestamp=rec["t"]):
            if rec.get("visible") and is_hit(tr.box, rec):
                tracked_frames.add(f)

    rows = {b: {k: 0 for k in
                ("n", "full", "sahi", "ir", "motion", "acoustic", "radar",
                 "any", "tracked")} for b in BINS}
    for rec in labels:
        if not rec.get("visible"):
            continue
        r = rec.get("range_m")
        if r is None:
            continue
        b = next((b for b in BINS if b[0] <= r < b[1]), None)
        if b is None:
            continue
        row = rows[b]
        row["n"] += 1
        seen_any = False
        for name in ("full", "sahi", "ir", "motion", "radar"):
            st = streams[name]
            if not st:
                continue
            thr = args.conf if name in ("full", "sahi") else 0.0
            hit = any(d["conf"] >= thr and is_hit(d["xyxy"], rec)
                      for d in st.get(rec["frame"], []))
            row[name] += hit
            if name != "radar":
                seen_any = seen_any or hit
        st = streams["acoustic"]
        if st:
            # acoustic is a bearing: it "sees" the drone if a cue's column
            # lines up with the target's column
            cx = (rec["bbox"][0] + rec["bbox"][2]) / 2
            hit = any(abs((d["xyxy"][0] + d["xyxy"][2]) / 2 - cx) <= 140
                      for d in st.get(rec["frame"], []))
            row["acoustic"] += hit
            seen_any = seen_any or hit
        row["any"] += seen_any
        row["tracked"] += rec["frame"] in tracked_frames

    hdr = (f"{'range':>12} {'frames':>7} {'wideRGB':>8} {'tiled':>7} "
           f"{'thermal':>8} {'motion':>7} {'acoustic':>9} {'radar*':>7} "
           f"{'ANY':>6} {'TRACKED':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for b in BINS:
        row = rows[b]
        if row["n"] == 0:
            continue
        def pc(k):
            return f"{100 * row[k] / row['n']:.0f}%"
        print(f"{f'{b[0]}-{b[1]} m':>12} {row['n']:>7} {pc('full'):>8} "
              f"{pc('sahi'):>7} {pc('ir'):>8} {pc('motion'):>7} "
              f"{pc('acoustic'):>9} {pc('radar'):>7} {pc('any'):>6} "
              f"{pc('tracked'):>8}")
    print("\n*radar is CUED in the deployed design, not scanning: that column "
          "is what it would see if pointed there, not a free-running rate.")


if __name__ == "__main__":
    main()
