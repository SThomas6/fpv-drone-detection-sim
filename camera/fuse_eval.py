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
        if d.cls_name == "hotspot":
            return "ir"
        if d.cls_name.startswith("radar_"):
            return d.cls_name          # radar_drone / radar_bird / radar_unknown
        return "mv"
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
              rgb_mode: str = "full", rgb_tag: str | None = None,
              use_radar: bool = False):
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
    rd_path = clip / "detections_radar.jsonl"
    rd = {}
    # Opt-in: a new sensor stream changes every fused row (it spawns its own
    # tracks - measured: the sweep's and-confirm fell 377->302 just from the
    # file existing), so benchmarks stay exact unless --radar is passed.
    if use_radar and rd_path.exists():
        rd = {r["frame"]: r["detections"] for r in
              (json.loads(l) for l in open(rd_path))}
    print(f"streams: rgb[{rgb_mode}]{' + ir' if ir else ''}"
          f"{' + motion' if mv else ''}{' + radar' if rd else ''}")
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
        aux += [Detection(*d["xyxy"], confidence=d["conf"],
                          cls_name=d["cls"])
                for d in rd.get(f, [])]
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
    ap.add_argument("--radar", nargs="?", const="on", default="off",
                    choices=["off", "on", "cued"],
                    help="off (default): stream ignored, benchmarks exact. "
                         "on: constant scan - an UPPER BOUND, not the "
                         "architecture. cued: the user design - radar off "
                         "while searching; a passive-confident track cues a "
                         "brief aimed dwell, emissions are accounted, and "
                         "only dwell returns exist")
    ap.add_argument("--radar-dwell", type=float, default=3.0,
                    help="cued: seconds the beam stays on a cued track")
    ap.add_argument("--radar-cue-latency", type=float, default=0.5,
                    help="cued: slew/spin-up delay before returns start")
    ap.add_argument("--radar-beam-px", type=float, default=120.0,
                    help="cued: beam association radius around the cued track")
    ap.add_argument("--radar-recue-s", type=float, default=10.0,
                    help="cued: cooldown before an unconfirmed track may cue "
                         "again (bounds emissions)")
    ap.add_argument("--zoom-confirm", action="store_true",
                    help="passive zoom-camera check BEFORE any radar cue: "
                         "slew the zoom lens onto the suspect, classify at "
                         "~2.5x pixel scale. Verdict stats are MEASURED from "
                         "this project's own zoom-lens clips with the "
                         "installed detector (687/687 drones and 141/141 "
                         "birds named correctly when resolved); a target too "
                         "small even zoomed comes back unknown and only THEN "
                         "may the radar be cued. Completely silent.")
    ap.add_argument("--zoom-latency", type=float, default=1.0,
                    help="zoom: slew + settle before the look starts")
    ap.add_argument("--zoom-dwell", type=float, default=1.5,
                    help="zoom: seconds of magnified observation")
    ap.add_argument("--radar-certainty", type=float, default=0.9,
                    help="passive P(drone) above which a persistently "
                         "zoom-unresolvable speck may earn ONE brief "
                         "identification dwell (the user exception: very "
                         "certain, but limit emissions)")
    ap.add_argument("--zoom-relook-s", type=float, default=5.0,
                    help="speck cadence: how often the zoom re-checks an "
                         "unresolved suspect it is silently tracking")
    ap.add_argument("--zoom-see-px", type=float, default=3.0,
                    help="wide-camera target width above which the zoomed "
                         "view resolves reliably (measured: the zoom sweep "
                         "resolves 62%% of 30-250 m overall, near-certain "
                         "when the wide width is >= ~3 px)")
    ap.add_argument("--radar-persist", type=float, default=0.3,
                    help="radar-gate: min fraction of track samples the "
                         "radar classified 'drone'")
    ap.add_argument("--radar-bird-thr", type=float, default=0.5,
                    help="radar-mute: min 'bird' fraction that mutes")
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
        rd_seen = defaultdict(lambda: deque(maxlen=90))
        # cued-radar state: per-track dwell windows and global emissions
        cue_start = {}          # track_id -> dwell start (after latency)
        radar_denied = set()    # micro-Doppler said bird: never re-cue
        radar_spurious = set()  # dwell completed with ZERO returns: no
                                # airframe there - never re-cue
        radar_confirmed = set() # returned + not bird: confirmation is STICKY
                                # for the track lifetime (the user design is
                                # one double-check per object, not periodic
                                # re-painting; deny still overrides)
        # CONFIRMATION INHERITANCE - the fix for "confirmation dies with the
        # track id". Canopy occlusion and sky crossings fragment tracks; the
        # physical object persists. A new track born where a status-bearing
        # track died moments ago inherits that status (denied > confirmed >
        # spurious), so a verified drone stays verified across fragments and
        # a radar-denied bird cannot burn a fresh dwell every fragment.
        prev_pos = {}           # track_id -> (x, y) last frame
        morgue = []             # (t_dead, x, y, status)
        inherit_n = {}          # diagnostics
        # zoom-confirm stage: ONE zoom camera, one look at a time, silent
        import random as _random
        zoom_rng = _random.Random(11)
        zoom_start = {}         # track_id -> look start (after slew)
        zoom_busy_until = 0.0
        zoom_next = {}          # speck cadence: earliest next silent re-look
        zoom_fails = defaultdict(int)   # unresolved looks per track
        dwell_until = {}        # rare certainty-dwell (see --radar-certainty)
        zoom_looks = [0, 0.0]   # count, busy seconds
        lock_err = []           # (frame, px error) of locked track vs GT
        coloc_streak = {}       # track_id -> consecutive co-located frames
        INHERIT_RADIUS = 40.0
        INHERIT_S = 5.0
        dwell_returns = defaultdict(int)   # returns seen during current dwell
        cue_block = {}          # track_id -> no re-cue before this time
        emissions = []          # (on, off) transmit windows, for duty cycle
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
                         args.rgb_tag, args.radar != "off")
        for f, gt, rgb_dets, ir_dets in rows:
            if args.radar == "cued":
                t_now = gt["t"]
                live = {tid: st for tid, st in cue_start.items()
                        if tid in radar_confirmed}
                live.update({t_: 0.0 for t_, u_ in dwell_until.items()
                             if t_now <= u_})
                radar_aux = [d for d in ir_dets
                             if d.cls_name.startswith("radar_")]
                other_aux = [d for d in ir_dets
                             if not d.cls_name.startswith("radar_")]
                kept_radar = []
                if live:
                    pos = {tr.track_id: (float(tr.mean[0]), float(tr.mean[1]))
                           for tr in tracker.active}
                    for d in radar_aux:
                        cx, cy = d.centre
                        if any(tid in pos and math.hypot(
                                cx - pos[tid][0], cy - pos[tid][1])
                                <= args.radar_beam_px for tid in live):
                            kept_radar.append(d)
                ir_dets = other_aux + kept_radar
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
            if args.radar == "cued":
                t_now2 = gt["t"]
                cur_pos = {tr.track_id: (float(tr.mean[0]),
                                         float(tr.mean[1]))
                           for tr in tracker.active}
                cur_w = {tr.track_id: max(float(tr.width), 4.0)
                         for tr in tracker.active}
                for tid_d, pos_d in prev_pos.items():
                    if tid_d in cur_pos:
                        continue
                    st = ("denied" if tid_d in radar_denied else
                          "confirmed" if tid_d in radar_confirmed else
                          "spurious" if tid_d in radar_spurious else None)
                    if st:
                        morgue.append((t_now2, pos_d[0], pos_d[1], st))
                morgue = [m for m in morgue if t_now2 - m[0] <= INHERIT_S]
                for tid_n, pos_n in cur_pos.items():
                    if tid_n in prev_pos or tid_n in radar_denied                             or tid_n in radar_confirmed                             or tid_n in radar_spurious:
                        continue
                    best = None
                    for (td, mx, my, st) in morgue:
                        d2 = (pos_n[0] - mx) ** 2 + (pos_n[1] - my) ** 2
                        if d2 <= INHERIT_RADIUS ** 2:
                            rank = {"denied": 0, "confirmed": 1,
                                    "spurious": 2}[st]
                            if best is None or rank < best[0]:
                                best = (rank, st)
                    if best is not None:
                        {"denied": radar_denied,
                         "confirmed": radar_confirmed,
                         "spurious": radar_spurious}[best[1]].add(tid_n)
                        inherit_n[best[1]] = inherit_n.get(best[1], 0) + 1
                # CO-LOCATION PROPAGATION - the measured sky pathology is
                # CONCURRENT duplicate tracks on one object (2 simultaneous
                # long tracks + ~18 one-frame flickers on the drone), which
                # a death-morgue cannot touch. Any track within the gate of
                # a living status-bearing track is the same physical object:
                # it inherits that verdict immediately. Denied wins over
                # confirmed (a flicker beside a denied bird is that bird).
                # CONFIRMED-only, and at same-object scale (a duplicate
                # spawns within a few px of its twin). Denied must NOT
                # propagate by proximity - measured: at 90 px it spread from
                # birds to the drone flying among them and coverage
                # collapsed to 39%. A wasted duplicate dwell costs 1.5 s of
                # emission; a propagated deny costs the target.
                # same-object scale = the boxes effectively overlap
                # (merge_close geometry), NOT a fixed neighborhood: at 25 px
                # birds brushing past the confirmed drone inherited its
                # verdict in the canopy furball (alarms 10 -> 38 ev/min)
                carriers = [(cur_pos[t_], cur_w[t_])
                            for t_ in radar_confirmed if t_ in cur_pos]
                if carriers:
                    for tid_c, pos_c in cur_pos.items():
                        if tid_c in radar_denied or tid_c in radar_confirmed:
                            coloc_streak.pop(tid_c, None)
                            continue
                        wc = cur_w.get(tid_c, 4.0)
                        near = any(
                            (pos_c[0] - cp[0]) ** 2 + (pos_c[1] - cp[1]) ** 2
                            <= max(8.0, 1.3 * max(wc, wr)) ** 2
                            for cp, wr in carriers)
                        if near:
                            # a bird CROSSING the confirmed drone overlaps
                            # for a frame or two; a duplicate of the same
                            # object stays put - demand a sustained overlap
                            coloc_streak[tid_c] = coloc_streak.get(tid_c, 0) + 1
                            if coloc_streak[tid_c] >= 3:
                                radar_confirmed.add(tid_c)
                                inherit_n["coloc"] =                                     inherit_n.get("coloc", 0) + 1
                        else:
                            coloc_streak.pop(tid_c, None)
                prev_pos = cur_pos
            boxmap = {tuple(round(v, 1) for v in d.xyxy): s
                      for d, s in zip(dets, sources)}
            for tr in tracks:
                src = boxmap.get(tuple(round(v, 1) for v in tr.box))
                if tr.misses == 0 and src is not None:
                    ir_seen[tr.track_id].append("ir" in src)
                    rd_seen[tr.track_id].append(
                        "radar_drone" if "radar_drone" in src else
                        "radar_bird" if "radar_bird" in src else None)
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
                rgbh_ = [h for h in tr.history
                         if len(h) < 6 or (h[5] not in ("hotspot", "mover")
                                           and not h[5].startswith("radar_"))]
                bird_vote_ = ((sum(1 for h in rgbh_
                                   if len(h) > 5 and h[5] == "bird")
                               / len(rgbh_)) if rgbh_ else None)
                med_w_ = (float(np.median([h[3] for h in rgbh_]))
                          if rgbh_ else 0.0)
                if args.radar == "cued":
                    p_now = (clf.probability(feats)
                             if feats is not None else None)
                    # cue only what the FULL passive stack would declare:
                    # motion-classifier pass + travelled + detected-now +
                    # not something the camera itself keeps calling a bird
                    # (bird-mute logic) - emissions are precious
                    cam_bird = (bird_vote_ is not None and bird_vote_ >= 0.5
                                and med_w_ >= 8.0)
                    # young (unjudged) tracks may cue when the young policy
                    # declares them - otherwise every track fragment sits
                    # unconfirmed until it grows old enough, which is where
                    # the sky coverage went
                    p_ok = (p_now >= args.motion_thr if p_now is not None
                            else args.young_tracks == "pass")
                    passive_ok = (p_ok
                                  and travel >= args.min_travel
                                  and tr.misses == 0
                                  and not cam_bird)
                    tid = tr.track_id
                    # ---- stage 1: the silent zoom look ----
                    if args.zoom_confirm and tid in zoom_start:
                        if gt["t"] >= zoom_start[tid] + args.zoom_dwell:
                            truth_d = bool(gt["visible"]
                                           and is_hit(tr.box, gt))
                            truth_b = any(hits_object(tr.box, b)
                                          for b in (gt.get("birds") or []))
                            resolves = zoom_rng.random() < (
                                0.90 if tr.width >= args.zoom_see_px
                                else 0.35)
                            if not (truth_d or truth_b):
                                # zoomed onto empty blur: no aircraft there
                                if zoom_rng.random() < 0.95:
                                    radar_spurious.add(tid)
                                else:
                                    zoom_next[tid] = (gt["t"]
                                                      + args.zoom_relook_s)
                            elif resolves:
                                correct = zoom_rng.random() < 0.99
                                if truth_d == correct:
                                    radar_confirmed.add(tid)
                                else:
                                    radar_denied.add(tid)
                            else:
                                # still a speck even zoomed: keep it under
                                # SILENT zoom surveillance on a cadence
                                zoom_fails[tid] += 1
                                zoom_next[tid] = (gt["t"]
                                                  + args.zoom_relook_s)
                            del zoom_start[tid]
                    # REVOCATION - no verdict outlives contradicting
                    # evidence. A confirmed track the wide camera keeps
                    # naming bird loses its confirmation and queues for a
                    # fresh look (the camera names birds ~95% at these
                    # sizes; the zoom errs ~1% - the camera catches the
                    # zoom mistakes). Symmetrically a denied track the
                    # camera insists is a drone re-queues.
                    # revocation demands OVERWHELMING contradiction: it
                    # exists to catch the zoom's ~1% mistakes, and at 0.5 it
                    # ate the canopy drone alive (its own wide-camera votes
                    # run ~20-50% bird in the furball; revoke->rezoom->
                    # confirm loops crushed coverage to 37%)
                    if (tid in radar_confirmed and bird_vote_ is not None
                            and bird_vote_ >= 0.8 and med_w_ >= 10.0):
                        radar_confirmed.discard(tid)
                        zoom_next.pop(tid, None)
                        inherit_n["revoked"] = inherit_n.get("revoked", 0) + 1
                    if (tid in radar_denied and bird_vote_ is not None
                            and bird_vote_ < 0.05 and med_w_ >= 10.0):
                        radar_denied.discard(tid)
                        zoom_next.pop(tid, None)
                        inherit_n["undenied"] = inherit_n.get("undenied", 0) + 1
                    zoom_eligible = (args.zoom_confirm
                                     and tid not in zoom_start
                                     and gt["t"] >= zoom_next.get(tid, 0.0))
                    # verdicts from any radar returns (lock upkeep, or the
                    # rare certainty-dwell below)
                    rets = [v for v in rd_seen[tid] if v is not None]
                    if rets and (sum(1 for v in rets if v == "radar_bird")
                                 / len(rets)) >= args.radar_bird_thr:
                        radar_denied.add(tid)
                    elif rets:
                        radar_confirmed.add(tid)
                    radar_confirmed -= radar_denied
                    confirmed_now = tid in radar_confirmed
                    if confirmed_now and tr.misses == 0:
                        # FIRE-CONTROL LOCK: a confirmed drone is painted
                        # CONTINUOUSLY - its position must be laser-holding
                        # grade at every instant. This is the only
                        # circumstance in which the system ever transmits;
                        # the moment the track is lost, the radar goes dark.
                        interval_ = 0.5
                        emissions.append((gt["t"], gt["t"] + interval_))
                        cue_start[tid] = gt["t"]     # marks the lock live
                    elif (passive_ok and not confirmed_now
                            and tid not in radar_denied
                            and tid not in radar_spurious):
                        if zoom_eligible and gt["t"] >= zoom_busy_until:
                            # silent zoom look - the DEFAULT identification
                            # path; the radar stays dark
                            zoom_start[tid] = gt["t"] + args.zoom_latency
                            zoom_busy_until = (zoom_start[tid]
                                               + args.zoom_dwell)
                            zoom_looks[0] += 1
                            zoom_looks[1] += (args.zoom_latency
                                              + args.zoom_dwell)
                        elif (p_now is not None
                                and p_now >= args.radar_certainty
                                and zoom_fails.get(tid, 0) >= 2
                                and tid not in dwell_until
                                and gt["t"] >= cue_block.get(tid, 0.0)):
                            # CERTAINTY DWELL - the user's exception: the
                            # passive stack is very sure this speck is a
                            # drone but the zoom cannot resolve it. One
                            # brief emission is allowed to settle it.
                            on = gt["t"] + args.radar_cue_latency
                            dwell_until[tid] = on + args.radar_dwell
                            cue_block[tid] = (dwell_until[tid]
                                              + args.radar_recue_s)
                            emissions.append((on, dwell_until[tid]))
                    if tid in dwell_until and gt["t"] > dwell_until[tid]:
                        del dwell_until[tid]
                if (args.radar == "cued" and gt.get("visible")
                        and tr.track_id in radar_confirmed
                        and is_hit(tr.box, gt)):
                    gx = (gt["bbox"][0] + gt["bbox"][2]) / 2
                    gy = (gt["bbox"][1] + gt["bbox"][3]) / 2
                    lock_err.append((f, math.hypot(
                        float(tr.mean[0]) - gx, float(tr.mean[1]) - gy)))
                obs.append({
                    "frame": f,
                    "track_id": tr.track_id,
                    "coasting": tr.misses > 0,
                    "is_drone": is_drone, "on_bird": on_bird,
                    "clutter": not is_drone and not on_bird,
                    "p": clf.probability(feats) if feats is not None else None,
                    "vote": drone_vote_fraction(
                        [h for h in tr.history
                         if len(h) < 6 or (h[5] not in ("hotspot", "mover")
                                           and not h[5].startswith("radar_"))]),
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
                    # fractions among actual radar RETURNS; a None entry
                    # means the radar was silent that frame (off, beam
                    # elsewhere, or a miss) and is not evidence either way
                    "rd_drone": (lambda q: (sum(1 for v in q
                                                if v == "radar_drone")
                                            / len(q)) if q else 0.0)(
                        [v for v in rd_seen[tr.track_id] if v is not None]),
                    "rd_bird": (lambda q: (sum(1 for v in q
                                               if v == "radar_bird")
                                           / len(q)) if q else 0.0)(
                        [v for v in rd_seen[tr.track_id] if v is not None]),
                    "rd_n": sum(1 for v in rd_seen[tr.track_id]
                                if v is not None),
                    "rd_conf": (args.radar == "cued"
                                and tr.track_id in radar_confirmed),
                    "travel_px": travel,
                })
        results[mode] = obs
        if args.radar == "cued" and mode == "fused":
            if args.zoom_confirm:
                print(f"[zoom] {zoom_looks[0]} silent looks, "
                      f"{zoom_looks[1]:.1f}s of zoom-camera time")
            if lock_err:
                best = {}
                for f_, e_ in lock_err:
                    best[f_] = min(best.get(f_, 1e9), e_)
                errs = sorted(best.values())
                n_vis = sum(1 for l in open(clip / "labels.jsonl")
                            if json.loads(l)["visible"])
                within = sum(1 for e_ in errs if e_ <= 25.0)
                print(f"[laser-lock] locked {len(best)}/{n_vis} "
                      f"drone-frames ({100 * len(best) / max(n_vis, 1):.1f}%); "
                      f"aim error px: p50 {errs[len(errs) // 2]:.1f}, "
                      f"p90 {errs[int(len(errs) * 0.9)]:.1f}; "
                      f"within 25 px: {100 * within / len(errs):.1f}%")
            print(f"[cued radar] inheritance events: {inherit_n}")
            iv = sorted(emissions)
            merged_iv = []
            for a_, b_ in iv:
                if merged_iv and a_ <= merged_iv[-1][1]:
                    merged_iv[-1] = (merged_iv[-1][0],
                                     max(merged_iv[-1][1], b_))
                else:
                    merged_iv.append((a_, b_))
            on_s = sum(b_ - a_ for a_, b_ in merged_iv)
            clip_s = sum(1 for _ in open(clip / "labels.jsonl")) * \
                json.loads((clip / "meta.json").read_text()).get(
                    "interval_s", 0.5)
            print(f"[cued radar] {len(iv)} dwells, transmitting "
                  f"{on_s:.1f}s of {clip_s:.0f}s = "
                  f"{100 * on_s / max(clip_s, 1e-9):.1f}% duty cycle")
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
          f"{'on birds':>9} {'clutter':>8} {'events/min':>11}")

    def report(name, obs, need_ir=False, need_vote=False, bird_mute=False,
               radar_gate=False, radar_mute=False, need_return=False):
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
                     or o["bird_vote"] < 0.5 or o["med_w"] < 8.0)
                and (not need_return
                     or o.get("rd_conf") or o.get("rd_n", 0) >= 1)
                and (not radar_gate
                     or o.get("rd_drone", 0.0) >= args.radar_persist)
                and (not radar_mute
                     or o.get("rd_bird", 0.0) < args.radar_bird_thr)]
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
        # Distinct alarm EVENTS: consecutive surviving frames of one track =
        # one event (an operator acknowledges a track once, not per frame).
        # Frame-rate-invariant, unlike alarm track-frames, whose per-minute
        # count scales with capture rate (7.5x between the 2 Hz and 15 Hz
        # benchmarks). Both currencies are printed; neither replaces the other.
        by_track = {}
        for o in keep:
            if o["is_drone"]:
                continue
            by_track.setdefault(o["track_id"], []).append(o["frame"])
        events = 0
        order = {f: i for i, f in enumerate(sorted({o["frame"] for o in obs}))}
        for frames_ in by_track.values():
            idx = sorted(order[f] for f in frames_)
            events += 1 + sum(1 for a_, b_ in zip(idx, idx[1:]) if b_ - a_ > 3)
        print(f"{name:>12} {cover:>5}/{vis:<5} {alarms / minutes:>10.1f} "
              f"{birds:>9} {clutter:>8} {events / minutes:>10.2f}")

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
    # micro-Doppler rows (rows print only when a radar stream exists):
    # radar-mute keeps everything except tracks the radar persistently calls
    # bird; radar-gate additionally demands persistent radar drone opinion.
    if any(o.get("rd_drone") or o.get("rd_bird") or o.get("rd_n")
           for o in results["fused"]):
        report("radar-mute", results["fused"], radar_mute=True)
        report("rdr+mute", results["fused"], radar_mute=True, bird_mute=True)
        report("radar-gate", results["fused"], radar_gate=True,
               radar_mute=True)
    if args.radar == "cued":
        # The user architecture row: passive declaration, then the brief
        # dwell must (a) RETURN something - a passive track with no radar
        # echo during its dwell is not a physical airframe - and (b) not be
        # classified bird by micro-Doppler.
        report("cued-confirm", results["fused"], radar_mute=True,
               need_return=True)


if __name__ == "__main__":
    main()
