#!/usr/bin/env python3
"""EO/IR track-level fusion evaluation.

Centralised measurement fusion, per the design that already carries this
project's results: ONE centre-distance Kalman tracker consumes both sensors'
detections; evidence is fused per TRACK, not per frame. Pixel-level fusion was
measured in the literature to add nothing; per-frame appearance is saturated
at these target sizes.

Per frame:
  RGB detections (cached detections_full.jsonl, both classes, conf floor)
  IR hot spots   (cached detections_ir.jsonl, already in RGB pixel space)
    -> co-located RGB+IR pairs merge into one measurement (sources recorded)
    -> tracker.update()
    -> per-track evidence: motion P(drone), RGB class votes, IR persistence

Policies reported:
  rgb-only   : motion >= thr on RGB-fed tracks (the existing system)
  or-fusion  : tracks fed by EITHER sensor; motion >= thr
  and-confirm: or-fusion AND IR persistence >= 0.3 on the track

    python camera/fuse_eval.py --clip data/clips/terrain_ir_canopy
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera.classify import (MotionClassifier, drone_vote_fraction,  # noqa: E402
                             features_from_history)
from camera.detector import Detection, merge_close  # noqa: E402
from camera.evaluate import hits_object, is_hit  # noqa: E402
from camera.tracking import CentroidTracker  # noqa: E402

MERGE_TOL_PX = 12.0


def _centre_dist(a: Detection, b: Detection) -> float:
    (ax, ay), (bx, by) = a.centre, b.centre
    return math.hypot(ax - bx, ay - by)


def fuse_measurements(rgb, ir):
    """Merge co-located RGB+IR detections into single measurements.

    Returns (detections, sources) where sources[i] is {'rgb'}, {'ir'} or both.
    The RGB box/class wins when both sensors see the target (finer pixel
    scale); the IR sensor's narrower FOV means 'no IR' outside its cone is
    expected, not evidence of absence.
    """
    dets, sources = [], []
    used_ir = set()
    for d in rgb:
        src = {"rgb"}
        for j, h in enumerate(ir):
            if j in used_ir:
                continue
            tol = max(MERGE_TOL_PX, 2.0 * max(d.width, h.width))
            if _centre_dist(d, h) <= tol:
                used_ir.add(j)
                src.add("ir")
                break
        dets.append(d)
        sources.append(src)
    for j, h in enumerate(ir):
        if j not in used_ir:
            dets.append(h)
            sources.append({"ir"})
    return dets, sources


def load_clip(clip: Path, rgb_conf: float, ir_conf: float):
    labels = {r["frame"]: r for r in
              (json.loads(l) for l in open(clip / "labels.jsonl"))}
    rgb = {r["frame"]: r["detections"] for r in
           (json.loads(l) for l in open(clip / "detections_full.jsonl"))}
    ir_path = clip / "detections_ir.jsonl"
    ir = {}
    if ir_path.exists():
        ir = {r["frame"]: r["detections"] for r in
              (json.loads(l) for l in open(ir_path))}
    frames = sorted(rgb.keys())
    out = []
    for f in frames:
        r_dets = [Detection(*d["xyxy"], confidence=d["conf"],
                            cls_name=d.get("cls", "drone"))
                  for d in rgb[f] if d["conf"] >= rgb_conf]
        i_dets = [Detection(*d["xyxy"], confidence=d["conf"], cls_name="hotspot")
                  for d in ir.get(f, []) if d["conf"] >= ir_conf]
        out.append((f, labels[f], merge_close(r_dets), i_dets))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--rgb-conf", type=float, default=0.05)
    ap.add_argument("--ir-conf", type=float, default=0.2,
                    help="IR confidence = peak contrast / 10 K")
    ap.add_argument("--motion-thr", type=float, default=0.5)
    ap.add_argument("--ir-persist", type=float, default=0.3)
    ap.add_argument("--min-travel", type=float, default=8.0,
                    help="min net track displacement in px before it may "
                         "alarm; static warm clutter never travels")
    args = ap.parse_args()
    clip = Path(args.clip)

    clf = MotionClassifier.load()
    if clf.w is None:
        raise SystemExit("no trained motion classifier")

    modes = {
        "rgb-only": lambda rd, id_: (rd, []),
        "fused": lambda rd, id_: (rd, id_),
    }
    results = {}
    for mode, select in modes.items():
        tracker = CentroidTracker()
        ir_seen = defaultdict(lambda: deque(maxlen=90))
        obs = []
        rows = load_clip(clip, args.rgb_conf, args.ir_conf)
        for f, gt, rgb_dets, ir_dets in rows:
            rd, id_ = select(rgb_dets, ir_dets)
            dets, sources = fuse_measurements(rd, id_)
            tracks = tracker.update(dets, timestamp=gt["t"])
            boxmap = {tuple(round(v, 1) for v in d.xyxy): s
                      for d, s in zip(dets, sources)}
            for tr in tracks:
                src = boxmap.get(tuple(round(v, 1) for v in tr.box))
                if tr.misses == 0 and src is not None:
                    ir_seen[tr.track_id].append("ir" in src)
                is_drone = bool(gt["visible"] and is_hit(tr.box, gt))
                on_bird = any(hits_object(tr.box, b)
                              for b in (gt.get("birds") or []))
                feats = features_from_history(tr.history)
                seen = ir_seen[tr.track_id]
                # Net travel over the track's history: a sun-warmed bush or a
                # bright rock never moves; even a hovering drone wanders a few
                # pixels and flies legs between hovers.
                xs = [h[1] for h in tr.history]
                ys = [h[2] for h in tr.history]
                travel = (math.hypot(max(xs) - min(xs), max(ys) - min(ys))
                          if len(xs) >= 2 else 0.0)
                obs.append({
                    "is_drone": is_drone, "on_bird": on_bird,
                    "clutter": not is_drone and not on_bird,
                    "p": clf.probability(feats) if feats is not None else None,
                    "vote": drone_vote_fraction(
                        [h for h in tr.history if len(h) < 6 or h[5] != "hotspot"]),
                    "ir_frac": (sum(seen) / len(seen)) if seen else 0.0,
                    "travel_px": travel,
                })
        results[mode] = obs

    n_frames = sum(1 for _ in open(clip / "labels.jsonl"))
    meta = json.loads((clip / "meta.json").read_text())
    minutes = n_frames * meta.get("interval_s", 0.5) / 60.0
    vis = sum(1 for l in open(clip / "labels.jsonl")
              if json.loads(l)["visible"])

    print(f"\n=== EO/IR fusion on {clip.name} "
          f"({n_frames} frames, {vis} drone-visible, {minutes:.1f} min) ===")
    print(f"{'policy':>12} {'drone cover':>12} {'alarms/min':>11} "
          f"{'on birds':>9} {'clutter':>8}")

    def report(name, obs, need_ir=False):
        keep = [o for o in obs
                if o["p"] is not None and o["p"] >= args.motion_thr
                and o["travel_px"] >= args.min_travel
                and (not need_ir or o["ir_frac"] >= args.ir_persist)]
        cover = sum(1 for o in keep if o["is_drone"])
        birds = sum(1 for o in keep if o["on_bird"])
        clutter = sum(1 for o in keep if o["clutter"])
        alarms = birds + clutter
        print(f"{name:>12} {cover:>5}/{vis:<5} {alarms / minutes:>10.1f} "
              f"{birds:>9} {clutter:>8}")

    report("rgb-only", results["rgb-only"])
    report("or-fusion", results["fused"])
    report("and-confirm", results["fused"], need_ir=True)


if __name__ == "__main__":
    main()
