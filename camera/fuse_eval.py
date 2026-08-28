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

import numpy as np

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


def fuse_measurements(rgb, aux):
    """Merge co-located detections from RGB and auxiliary channels (IR hot
    spots, motion candidates) into single measurements.

    Returns (detections, sources); sources[i] ⊆ {'rgb', 'ir', 'mv'}. The RGB
    box/class wins when several channels see the target (finest pixel scale);
    a narrow-FOV channel's silence outside its cone is expected, not evidence
    of absence.
    """
    def tag(d):
        return "ir" if d.cls_name == "hotspot" else "mv"
    dets, sources = [], []
    used = set()
    for d in rgb:
        src = {"rgb"}
        for j, h in enumerate(aux):
            if j in used:
                continue
            tol = max(MERGE_TOL_PX, 2.0 * max(d.width, h.width))
            if _centre_dist(d, h) <= tol:
                used.add(j)
                src.add(tag(h))
        dets.append(d)
        sources.append(src)
    for j, h in enumerate(aux):
        if j not in used:
            # a motion candidate co-located with an unused IR hotspot merges too
            merged = False
            for k, (d0, s0) in enumerate(zip(dets, sources)):
                if "rgb" not in s0 and _centre_dist(d0, h) <= max(
                        MERGE_TOL_PX, 2.0 * max(d0.width, h.width)):
                    s0.add(tag(h))
                    merged = True
                    break
            if not merged:
                dets.append(h)
                sources.append({tag(h)})
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
    mv_path = clip / "detections_motion.jsonl"
    mv = {}
    if mv_path.exists():
        mv = {r["frame"]: r["detections"] for r in
              (json.loads(l) for l in open(mv_path))}
    print(f"streams: rgb{' + ir' if ir else ''}{' + motion' if mv else ''}")
    frames = sorted(rgb.keys())
    out = []
    for f in frames:
        r_dets = [Detection(*d["xyxy"], confidence=d["conf"],
                            cls_name=d.get("cls", "drone"))
                  for d in rgb[f] if d["conf"] >= rgb_conf]
        aux = [Detection(*d["xyxy"], confidence=d["conf"], cls_name="hotspot")
               for d in ir.get(f, []) if d["conf"] >= ir_conf]
        aux += [Detection(*d["xyxy"], confidence=d["conf"], cls_name="mover")
                for d in mv.get(f, []) if d["conf"] >= 0.10]
        out.append((f, labels[f], merge_close(r_dets), aux))
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
        tracker = CentroidTracker(class_consistent=True,
                                  suppress_spawn_near_coasting=True)
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
                        [h for h in tr.history
                         if len(h) < 6 or h[5] not in ("hotspot", "mover")]),
                    # bird-mute evidence: share of RGB entries the detector
                    # called bird; aux-only tracks have no opinion (None).
                    # med_w gates trust: below ~8 px the class votes are known
                    # to flip on real drones (the measured bird-flip band), so
                    # appearance opinions only count on big-enough targets.
                    "bird_vote": (lambda rgbh: (sum(1 for h in rgbh
                                                    if len(h) > 5 and h[5] == "bird")
                                                / len(rgbh)) if rgbh else None)(
                        [h for h in tr.history
                         if len(h) < 6 or h[5] not in ("hotspot", "mover")]),
                    "med_w": (lambda ws: float(np.median(ws)) if ws else 0.0)(
                        [h[3] for h in tr.history
                         if len(h) < 6 or h[5] not in ("hotspot", "mover")]),
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

    def report(name, obs, need_ir=False, need_vote=False, bird_mute=False):
        keep = [o for o in obs
                if o["p"] is not None and o["p"] >= args.motion_thr
                and o["travel_px"] >= args.min_travel
                and (not need_ir or o["ir_frac"] >= args.ir_persist)
                and (not need_vote or o["vote"] >= 0.5)
                and (not bird_mute or o["bird_vote"] is None
                     or o["bird_vote"] < 0.5 or o["med_w"] < 8.0)]
        cover = sum(1 for o in keep if o["is_drone"])
        birds = sum(1 for o in keep if o["on_bird"])
        clutter = sum(1 for o in keep if o["clutter"])
        alarms = birds + clutter
        print(f"{name:>12} {cover:>5}/{vis:<5} {alarms / minutes:>10.1f} "
              f"{birds:>9} {clutter:>8}")

    report("rgb-only", results["rgb-only"])
    report("or-fusion", results["fused"])
    report("and-confirm", results["fused"], need_ir=True)
    # full gate: motion + travel + RGB class votes agree it is a drone —
    # the vote gate is what collapsed sky-bird alarms to 0.2/min
    report("vote-gated", results["fused"], need_vote=True)
    # bird-mute: suppress only tracks the detector actively calls bird —
    # aux-held tracks (no RGB opinion) stay alive; the asymmetric gate
    report("bird-mute", results["fused"], bird_mute=True)


if __name__ == "__main__":
    main()
