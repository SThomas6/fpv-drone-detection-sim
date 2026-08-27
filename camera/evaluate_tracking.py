#!/usr/bin/env python3
"""Score the tracker: ID stability, continuity and velocity accuracy.

Replays a clip through DroneDetector.track() in order and compares against the
simulator's ground truth.

    python camera/evaluate_tracking.py --clip data/clips/tracking

Reported:
  · tracking recall vs detection-only recall (does temporal persistence help?)
  · number of distinct IDs used for the one real drone, and ID switches
  · time to first confirmed track after the drone becomes visible
  · velocity error against ground-truth image-plane motion

Note: ultralytics' track() emits only CONFIRMED tracks, so a track coasting
through a missed detection is not visible in its output; ID continuity is the
observable proxy for that.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera.detector import DroneDetector  # noqa: E402
from camera.evaluate import is_hit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="data/clips/tracking")
    ap.add_argument("--device", default=None, help="cuda|mps|cpu (default: auto)")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tracker", default="centroid",
                    choices=["centroid", "bytetrack"],
                    help="centroid = distance-gated Kalman (camera/tracking.py); "
                         "bytetrack = ultralytics' IoU tracker")
    args = ap.parse_args()

    clip = Path(args.clip)
    labels = [json.loads(l) for l in open(clip / "labels.jsonl")]
    if args.limit:
        labels = labels[:args.limit]

    det = DroneDetector(device=args.device, conf=args.conf, track_conf=args.conf)
    det.warmup()
    det.reset_tracks()

    centroid = None
    if args.tracker == "centroid":
        from camera.tracking import CentroidTracker
        centroid = CentroidTracker()

    rows = []
    for rec in labels:
        frame = np.array(Image.open(clip / "frames" / rec["frame"]).convert("RGB"))
        if centroid is not None:
            tracks = centroid.update(det.detect(frame), timestamp=rec["t"])
        else:
            tracks = det.track(frame, timestamp=rec["t"])
        matched = None
        if rec["visible"]:
            for t in tracks:
                box = t.box if hasattr(t, "box") else t.detection.xyxy
                if is_hit(box, rec):
                    # Snapshot: CentroidTracker mutates its track objects in
                    # place, so holding the reference would read final state.
                    matched = {"track_id": t.track_id,
                               "velocity": tuple(t.velocity), "box": list(box)}
                    break
        rows.append({"rec": rec,
                     "track_ids": [t.track_id for t in tracks],
                     "matched": matched})
        if len(rows) % 100 == 0:
            print(f"  {len(rows)}/{len(labels)} frames", flush=True)

    visible = [r for r in rows if r["rec"]["visible"]]
    tracked = [r for r in visible if r["matched"]]
    recall = len(tracked) / len(visible) if visible else 0.0

    ids = [r["matched"]["track_id"] for r in tracked]
    switches = sum(1 for a, b in zip(ids, ids[1:]) if a != b)
    spurious = {i for r in rows for i in r["track_ids"]} - set(ids)

    # time from the drone first being visible to the first confirmed track
    ttf = None
    if visible and tracked:
        ttf = tracked[0]["rec"]["t"] - visible[0]["rec"]["t"]

    # longest run of visible frames with no confirmed track
    longest_gap, cur = 0, 0
    for r in visible:
        cur = 0 if r["matched"] else cur + 1
        longest_gap = max(longest_gap, cur)

    # Velocity: compare against ground-truth centre motion. Ground truth is
    # differenced over a short window rather than a single frame pair, because
    # a single pair at 10 Hz is mostly quantisation noise on a 4 px target.
    MOVING_PX_S = 20.0
    verrs_abs, verrs_rel = [], []
    idx = {id(r): i for i, r in enumerate(visible)}
    for r in visible:
        if not r["matched"]:
            continue
        i = idx[id(r)]
        lo, hi = max(0, i - 2), min(len(visible) - 1, i + 2)
        dt = visible[hi]["rec"]["t"] - visible[lo]["rec"]["t"]
        if dt <= 0 or not (visible[lo]["rec"].get("centre")
                           and visible[hi]["rec"].get("centre")):
            continue
        gvx = (visible[hi]["rec"]["centre"][0] - visible[lo]["rec"]["centre"][0]) / dt
        gvy = (visible[hi]["rec"]["centre"][1] - visible[lo]["rec"]["centre"][1]) / dt
        tvx, tvy = r["matched"]["velocity"]
        err = math.hypot(tvx - gvx, tvy - gvy)
        gmag = math.hypot(gvx, gvy)
        verrs_abs.append(err)
        if gmag > MOVING_PX_S:
            verrs_rel.append(err / gmag)
    verrs = verrs_rel

    print(f"\n=== tracking / {clip.name} / {args.tracker} ===")
    print(f"frames: {len(rows)}  ({len(visible)} with the drone visible)")
    print(f"tracking recall           : {recall:.3f}  ({len(tracked)}/{len(visible)})")
    print(f"distinct IDs for 1 drone  : {len(set(ids))}  (ideal 1)")
    print(f"ID switches               : {switches}")
    print(f"spurious tracks           : {len(spurious)}")
    if ttf is not None:
        print(f"time to first track       : {ttf:.2f} s")
    print(f"longest gap without track : {longest_gap} frames")
    if verrs_abs:
        print(f"velocity error (absolute) : median {np.median(verrs_abs):.1f} px/s "
              f"({len(verrs_abs)} frames)")
    if verrs:
        print(f"velocity error (relative) : median {np.median(verrs):.2f}, "
              f"p90 {np.percentile(verrs, 90):.2f}  "
              f"({len(verrs)} frames moving >{MOVING_PX_S:.0f} px/s)")

    det_file = clip / "detections_full.jsonl"
    if det_file.exists():
        dets = {r["frame"]: r["detections"] for r in
                (json.loads(l) for l in open(det_file))}
        hits = sum(1 for r in visible
                   if any(d["conf"] >= args.conf and is_hit(d["xyxy"], r["rec"])
                          for d in dets.get(r["rec"]["frame"], [])))
        dr = hits / len(visible) if visible else 0.0
        print(f"\ndetection-only recall     : {dr:.3f}")
        print(f"tracking recall           : {recall:.3f}  "
              f"({'+' if recall >= dr else ''}{(recall - dr) * 100:.1f} points)")

    summary = {
        "clip": clip.name, "tracker": args.tracker,
        "frames": len(rows), "visible": len(visible),
        "tracking_recall": round(recall, 4), "distinct_ids": len(set(ids)),
        "id_switches": switches, "spurious_tracks": len(spurious),
        "time_to_first_track_s": round(ttf, 3) if ttf is not None else None,
        "longest_gap_frames": longest_gap,
        "velocity_rel_err_median": round(float(np.median(verrs)), 4) if verrs else None,
        "velocity_abs_err_median_px_s": (round(float(np.median(verrs_abs)), 2)
                                         if verrs_abs else None),
    }
    out = clip / f"summary_tracking_{args.tracker}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
