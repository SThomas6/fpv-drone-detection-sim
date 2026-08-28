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
from camera.clutter_map import StaticClutterSuppressor  # noqa: E402
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


def _load_dets(path: Path) -> dict:
    return {r["frame"]: r["detections"] for r in
            (json.loads(l) for l in open(path))}


def load_clip(clip: Path, rgb_conf: float, ir_conf: float,
              rgb_mode: str = "full", rgb_tag: str | None = None):
    """Cached streams for one clip.

    rgb_mode picks which detector pass feeds the RGB stream:
      full : detections_full.jsonl  (one native-scale pass, 46 FPS)
      sahi : detections_sahi.jsonl  (tiled pass, 5 FPS, ~2x recall on
             sub-canopy targets - measured, see docs/phase2-results.md)
      both : the union of the two, deduplicated by merge_close. Tiling and
             full-frame miss different targets, so the union is a genuine
             detector-level OR, not just the better of the two.
    """
    labels = {r["frame"]: r for r in
              (json.loads(l) for l in open(clip / "labels.jsonl"))}
    # rgb_tag reads a TAGGED detector pass (evaluate.py run --tag X), so a
    # candidate model can be evaluated end-to-end without overwriting the
    # deployed model's untagged cache - the landmine this repo already hit.
    sfx = f"_{rgb_tag}" if rgb_tag else ""
    full_path = clip / f"detections_full{sfx}.jsonl"
    sahi_path = clip / f"detections_sahi{sfx}.jsonl"
    if not full_path.exists():
        raise SystemExit(f"{full_path} missing - run: python camera/evaluate.py"
                         f" run --clip {clip} --mode full"
                         + (f" --tag {rgb_tag}" if rgb_tag else ""))
    if rgb_mode in ("sahi", "both") and not sahi_path.exists():
        raise SystemExit(
            f"{sahi_path} missing - run: "
            f"python camera/evaluate.py run --clip {clip} --mode sahi"
            + (f" --tag {rgb_tag}" if rgb_tag else ""))
    if rgb_mode == "full":
        rgb = _load_dets(full_path)
    elif rgb_mode == "sahi":
        rgb = _load_dets(sahi_path)
    else:
        rgb = _load_dets(full_path)
        for f, dets in _load_dets(sahi_path).items():
            rgb[f] = rgb.get(f, []) + dets
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
    print(f"streams: rgb[{rgb_mode}]{' + ir' if ir else ''}"
          f"{' + motion' if mv else ''}")
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
    ap.add_argument("--rgb-mode", default="full",
                    choices=["full", "sahi", "both"],
                    help="which cached RGB detector pass feeds the fusion")
    ap.add_argument("--rgb-tag", default=None,
                    help="read detections_<mode>_<tag>.jsonl instead of the "
                         "untagged cache (candidate-model evaluation)")
    ap.add_argument("--class-consistent", dest="class_consistent",
                    action="store_true", default=True)
    ap.add_argument("--no-class-consistent", dest="class_consistent",
                    action="store_false",
                    help="let a detection join a track whose majority class "
                         "contradicts it. The consistency gate protects "
                         "terrain tracks from bird pollution, but when the "
                         "detector misnames the TARGET (m3 flips the sky "
                         "drone to 'bird' in 25%% of frames near birds) it "
                         "locks the target's own detections out of its track")
    ap.add_argument("--coast-alarms", default="all",
                    choices=["all", "judged", "none"],
                    help="may a track declare on a frame where it was NOT "
                         "detected (coasting)? 'all' = legacy behaviour; "
                         "'judged' = only tracks the classifier has actually "
                         "passed (young tracks must be seen to declare) - "
                         "measured on canopy this is FREE, 235->170 alarms/min "
                         "at identical coverage, because the drone's coverage "
                         "almost never depends on young coasting frames while "
                         "57%% of alarm frames were coasting; 'none' = every "
                         "declaration needs a detection this frame")
    ap.add_argument("--young-tracks", default="drop",
                    choices=["drop", "pass"],
                    help="what to do with a track too short for the motion "
                         "classifier to have an opinion (<8 samples). 'drop' "
                         "is the original alarm-discipline choice; 'pass' "
                         "declares it, which is what a coverage target wants")
    ap.add_argument("--classifier", default=None,
                    help="motion-classifier JSON to use (default: the "
                         "installed camera/motion_classifier.json)")
    ap.add_argument("--clutter", action="store_true",
                    help="suppress measurements at established static-clutter "
                         "locations (camera/clutter_map.py) before tracking")
    ap.add_argument("--clutter-radius", type=float, default=25.0)
    ap.add_argument("--clutter-window", type=float, default=15.0,
                    help="persistence window, seconds")
    ap.add_argument("--clutter-persist", type=int, default=8)
    ap.add_argument("--clutter-extent", type=float, default=15.0)
    ap.add_argument("--exempt-travel", type=float, default=40.0,
                    help="net track travel that exempts a measurement from "
                         "clutter suppression")
    args = ap.parse_args()
    clip = Path(args.clip)

    clf = (MotionClassifier.load(args.classifier) if args.classifier
           else MotionClassifier.load())
    if clf.w is None:
        raise SystemExit("no trained motion classifier")

    modes = {
        "rgb-only": lambda rd, id_: (rd, []),
        "fused": lambda rd, id_: (rd, id_),
    }
    results = {}
    for mode, select in modes.items():
        tracker = CentroidTracker(class_consistent=args.class_consistent,
                                  suppress_spawn_near_coasting=True)
        ir_seen = defaultdict(lambda: deque(maxlen=90))
        obs = []
        supp = (StaticClutterSuppressor(
            radius=args.clutter_radius, window_s=args.clutter_window,
            min_persist=args.clutter_persist,
            max_extent=args.clutter_extent) if args.clutter else None)
        n_supp = 0

        def _exempt(d, _tracker=tracker):
            """Measurements owned by a track that has demonstrably travelled.

            Only currently-detected tracks vouch: a coasting track's Kalman
            prediction keeps flying at its last velocity and would exempt
            half the frame.
            """
            cx, cy = d.centre
            for tr in _tracker.active:
                if tr.misses > 0 or len(tr.history) < 2:
                    continue
                xs = [h[1] for h in tr.history]
                ys = [h[2] for h in tr.history]
                if math.hypot(max(xs) - min(xs),
                              max(ys) - min(ys)) < args.exempt_travel:
                    continue
                if math.hypot(cx - tr.mean[0],
                              cy - tr.mean[1]) <= max(25.0, 2.0 * d.width):
                    return True
            return False

        rows = load_clip(clip, args.rgb_conf, args.ir_conf, args.rgb_mode,
                         args.rgb_tag)
        for f, gt, rgb_dets, ir_dets in rows:
            rd, id_ = select(rgb_dets, ir_dets)
            dets, sources = fuse_measurements(rd, id_)
            if supp is not None:
                keep_set, dropped = supp.step(dets, gt["t"], exempt=_exempt)
                n_supp += len(dropped)
                keep_ids = {id(d) for d in keep_set}
                dets, sources = zip(*[(d, s) for d, s in zip(dets, sources)
                                      if id(d) in keep_ids]) \
                    if keep_set else ([], [])
                dets, sources = list(dets), list(sources)
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
                feats = clf.features(tr.history)
                seen = ir_seen[tr.track_id]
                # Net travel over the track's history: a sun-warmed bush or a
                # bright rock never moves; even a hovering drone wanders a few
                # pixels and flies legs between hovers.
                xs = [h[1] for h in tr.history]
                ys = [h[2] for h in tr.history]
                travel = (math.hypot(max(xs) - min(xs), max(ys) - min(ys))
                          if len(xs) >= 2 else 0.0)
                obs.append({
                    "frame": f,
                    "coasting": tr.misses > 0,
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
        if supp is not None:
            print(f"[{mode}] clutter map suppressed {n_supp} measurements")

    n_frames = sum(1 for _ in open(clip / "labels.jsonl"))
    meta = json.loads((clip / "meta.json").read_text())
    minutes = n_frames * meta.get("interval_s", 0.5) / 60.0
    vis = sum(1 for l in open(clip / "labels.jsonl")
              if json.loads(l)["visible"])

    print(f"\n=== EO/IR fusion on {clip.name} "
          f"[rgb={args.rgb_mode}{'/' + args.rgb_tag if args.rgb_tag else ''}"
          f"{', clutter' if args.clutter else ''}] "
          f"({n_frames} frames, {vis} drone-visible, {minutes:.1f} min) ===")
    print(f"{'policy':>12} {'drone cover':>12} {'alarms/min':>11} "
          f"{'on birds':>9} {'clutter':>8}")

    def report(name, obs, need_ir=False, need_vote=False, bird_mute=False):
        undecided = args.young_tracks == "pass"
        keep = [o for o in obs
                if (o["p"] >= args.motion_thr if o["p"] is not None
                    else undecided)
                and (args.coast_alarms == "all"
                     or not o.get("coasting")
                     or (args.coast_alarms == "judged"
                         and o["p"] is not None))
                and o["travel_px"] >= args.min_travel
                and (not need_ir or o["ir_frac"] >= args.ir_persist)
                and (not need_vote or o["vote"] >= 0.5)
                and (not bird_mute or o["bird_vote"] is None
                     or o["bird_vote"] < 0.5 or o["med_w"] < 8.0)]
        # Coverage is per FRAME, not per track-frame. Two tracks sitting on
        # the same drone in one frame is one frame covered, not two — counting
        # observations here inflated every fused coverage number this project
        # recorded before 2026-08-28 (canopy or-fusion read 79.3%; it is
        # 68.8%). Alarms stay per track-frame, which is the convention
        # camera/evaluate.py already uses for FP/min.
        cover = len({o["frame"] for o in keep if o["is_drone"]})
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
    # thermal confirmation AND appearance mute together - measured as the
    # canopy frontier: the bird alarms are real birds the detector can name,
    # the clutter alarms are IR-co-located, so the two gates cut different
    # populations and stack almost losslessly
    report("and+mute", results["fused"], need_ir=True, bird_mute=True)


if __name__ == "__main__":
    main()
