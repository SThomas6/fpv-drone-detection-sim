#!/usr/bin/env python3
"""Whole-system detection rate versus range, every sensor combined.

The question: at each distance, how often does the system actually have the
drone? Per-sensor tables answer "can this channel see it"; this answers
"does the system know it is there", which is the number that matters.

Two clips are needed, and the reason is physical rather than procedural: a
6 deg telephoto and a 60 deg wide camera cannot be the same camera, so the
telephoto had to be flown separately (data/clips/range_tele, telesweep) from
the wide/thermal/motion/acoustic suite (data/clips/range_1km, longsweep).
Within a clip, sensors are combined HONESTLY - a per-frame OR, frame by
frame. Across the two clips they cannot be, because there is no frame
correspondence, so the two are combined per RANGE BIN with max() rather than
an independence assumption that would invent coverage nobody measured.
Where they overlap that is conservative: it credits the system with the
better channel, never with both at once.

Columns:
  wideRGB/tiled  the 60 deg station camera, native and SAHI-tiled
  thermal        LWIR hot-spot detector
  motion         background-subtraction movers
  point          local-contrast point-target detector on the WIDE camera
  acoustic       mic-array bearing, scored as a bearing (angular gate)
  PCL            passive radar bearing, scored angularly like acoustic
  RANGED         the track carries a MEASURED slant range this frame - only
                 PCL supplies it, and it is what an effector hand-off needs
  TELE           6 deg telephoto + point-target detector (separate flight)
  ANY            per-frame OR of every channel on the wide clip
  TRACKED        the fused tracker holds it (coasts through blinks, so this
                 can exceed ANY - that is the point of tracking)
  SYSTEM         max(TRACKED, TELE)

Radar is deliberately absent: it is CUED in this design and emits nothing
until the passive stack is already confident, so it cannot contribute to a
detection rate - it confirms one.

    python scripts/system_range_table.py --clip data/clips/range_1km \
        --tele-clip data/clips/range_tele
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera.detector import Detection, merge_close  # noqa: E402
from camera.evaluate import is_hit  # noqa: E402
from camera.fuse_eval import fuse_measurements  # noqa: E402
from camera.tracking import CentroidTracker  # noqa: E402

BINS = [(0, 30), (30, 60), (60, 100), (100, 150), (150, 200), (200, 300),
        (300, 400), (400, 550), (550, 700), (700, 850), (850, 1000),
        (1000, 1150), (1150, 1300), (1300, 1500)]
SENSORS = ("full", "sahi", "ir", "motion", "point")


def load(clip: Path, name: str):
    p = clip / f"detections_{name}.jsonl"
    if not p.exists():
        return None
    return {r["frame"]: r["detections"]
            for r in (json.loads(l) for l in open(p))}


def bin_of(r):
    return next((b for b in BINS if b[0] <= r < b[1]), None)


def per_bin_rates(clip: Path, conf: float):
    """Detection rate per range bin for every stream present on one clip."""
    labels = [json.loads(l) for l in open(clip / "labels.jsonl")]
    meta = json.loads((clip / "meta.json").read_text())
    fx = float(meta.get("fx", 1108.77))
    streams = {n: load(clip, n)
               for n in SENSORS + ("acoustic", "pcl")}

    # tracked: replay the fused passive pipeline once, exactly as deployed
    tracker = CentroidTracker(class_consistent=True,
                              suppress_spawn_near_coasting=True)
    tracked, ranged = set(), set()
    for rec in labels:
        f = rec["frame"]
        rgb = [Detection(*d["xyxy"], confidence=d["conf"],
                         cls_name=d.get("cls", "drone"))
               for d in (streams["full"] or {}).get(f, []) if d["conf"] >= 0.05]
        aux = [Detection(*d["xyxy"], confidence=d["conf"], cls_name="hotspot")
               for d in (streams["ir"] or {}).get(f, []) if d["conf"] >= 0.2]
        aux += [Detection(*d["xyxy"], confidence=d["conf"], cls_name="mover")
                for d in (streams["motion"] or {}).get(f, [])
                if d["conf"] >= 0.10]
        dets, _ = fuse_measurements(merge_close(rgb), aux)
        live = tracker.update(dets, timestamp=rec["t"])
        # predict -> camera update -> bearing update is the required order
        cues = [((d["xyxy"][0] + d["xyxy"][2]) / 2,
                 max(4.0, fx * math.tan(math.radians(
                     d.get("bearing_sigma_deg", 3.0)))),
                 d.get("range_m", 0.0), d.get("sigma_range_m", 20.0))
                for d in (streams["pcl"] or {}).get(f, [])]
        if cues:
            tracker.fuse_pcl(cues)
        for tr in live:
            if rec.get("visible") and is_hit(tr.box, rec):
                tracked.add(f)
                if tr.range_m is not None and tr.range_age == 0:
                    ranged.add(f)

    rows = {b: {k: 0 for k in
                SENSORS + ("acoustic", "pcl", "n", "any",
                           "tracked", "ranged")}
            for b in BINS}
    for rec in labels:
        if not rec.get("visible") or rec.get("range_m") is None:
            continue
        b = bin_of(rec["range_m"])
        if b is None:
            continue
        row = rows[b]
        row["n"] += 1
        seen = False
        for name in SENSORS:
            st = streams[name]
            if not st:
                continue
            thr = conf if name in ("full", "sahi") else 0.0
            hit = any(d["conf"] >= thr and is_hit(d["xyxy"], rec)
                      for d in st.get(rec["frame"], []))
            row[name] += hit
            seen = seen or hit
        if streams["pcl"]:
            cx = (rec["bbox"][0] + rec["bbox"][2]) / 2
            tol = fx * math.tan(math.radians(7.0))
            hit = any(abs((d["xyxy"][0] + d["xyxy"][2]) / 2 - cx) <= tol
                      for d in streams["pcl"].get(rec["frame"], []))
            row["pcl"] += hit
            seen = seen or hit
        if streams["acoustic"]:
            # a bearing is an ANGLE; gating it in pixels silently changes the
            # tolerance by 11x between the wide and telephoto lenses
            cx = (rec["bbox"][0] + rec["bbox"][2]) / 2
            tol = fx * math.tan(math.radians(7.0))
            hit = any(abs((d["xyxy"][0] + d["xyxy"][2]) / 2 - cx) <= tol
                      for d in streams["acoustic"].get(rec["frame"], []))
            row["acoustic"] += hit
            seen = seen or hit
        row["any"] += seen
        row["tracked"] += rec["frame"] in tracked
        row["ranged"] += rec["frame"] in ranged
    return rows, [n for n in SENSORS + ("acoustic", "pcl")
                  if streams[n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--tele-clip", default=None)
    ap.add_argument("--conf", type=float, default=0.10)
    args = ap.parse_args()

    rows, present = per_bin_rates(Path(args.clip), args.conf)
    tele = {}
    if args.tele_clip:
        trows, _ = per_bin_rates(Path(args.tele_clip), args.conf)
        for b, r in trows.items():
            if r["n"]:
                tele[b] = (r["point"] / r["n"], r["n"])

    print(f"\nwide/thermal/motion/acoustic: {Path(args.clip).name}"
          f"   ({', '.join(present)})")
    if args.tele_clip:
        print(f"6 deg telephoto:              {Path(args.tele_clip).name}"
              f"   (separate flight, combined per range bin)")
    hdr = (f"{'range':>12} {'frames':>7} {'wideRGB':>8} {'tiled':>7} "
           f"{'thermal':>8} {'motion':>7} {'point':>6} {'acoustic':>9} "
           f"{'PCL':>5} {'ANY':>5} {'TRACKED':>8} {'RANGED':>7} "
           f"{'TELE':>6} {'SYSTEM':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for b in BINS:
        row = rows[b]
        t = tele.get(b)
        if row["n"] == 0 and t is None:
            continue
        n = row["n"]

        def pc(k):
            return f"{100 * row[k] / n:.0f}%" if n else "-"
        tp = f"{100 * t[0]:.0f}%" if t else "-"
        best = max(row["tracked"] / n if n else 0.0, t[0] if t else 0.0)
        print(f"{f'{b[0]}-{b[1]} m':>12} {n if n else (t[1] if t else 0):>7} "
              f"{pc('full'):>8} {pc('sahi'):>7} {pc('ir'):>8} {pc('motion'):>7} "
              f"{pc('point'):>6} {pc('acoustic'):>9} {pc('pcl'):>5} "
              f"{pc('any'):>5} {pc('tracked'):>8} {pc('ranged'):>7} "
              f"{tp:>6} {100 * best:>6.0f}%")
    print("\nSYSTEM = max(TRACKED, TELE): the telephoto flew a different "
          "sortie, so its\nbin rate is combined conservatively rather than "
          "OR-ed frame by frame.")
    print("Radar is cued-only by design and confirms rather than detects, "
          "so it is not a column.")


if __name__ == "__main__":
    main()
