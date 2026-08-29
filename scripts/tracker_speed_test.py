#!/usr/bin/env python3
"""Does the tracker hold a 150 kph target, or does the gate lose it?

Association is distance-gated (camera/tracking.py:_gate_for):

    gate = base_gate_px + min(4*width, 250) + speed*dt*1.5 + coast*base/2

The motion term uses the track's OWN estimated speed, so it only helps once
the track is already established and moving. On the first association after
a spawn the estimate is zero and the gate is just base + size - about 42 px
for a small target. That is the number that decides whether a fast drone can
ever get a track started.

Per-frame image displacement is what actually matters, and it scales with
focal length as well as speed: 41.7 m/s crossing at 500 m is 1018 px/s
through the 6 deg lens, which is 68 px/frame at 15 Hz and 509 px/frame at
2 Hz. So the same drone is trivially trackable in one configuration and
untrackable in another, and the clip's capture rate is part of the answer.

Reported per displacement bin:
  held      fraction of frames where SOME track sits on the drone
  switches  how many DIFFERENT track ids the drone was given - a track that
            breaks and respawns still scores well on `held` but is useless
            for a hand-off to a zoom or an effector, so count it separately

    python scripts/tracker_speed_test.py --clip data/clips/range_fast_sky
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera.detector import Detection, merge_close  # noqa: E402
from camera.evaluate import is_hit  # noqa: E402
from camera.fuse_eval import fuse_measurements  # noqa: E402
from camera.tracking import CentroidTracker  # noqa: E402

BINS = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 120), (120, 1e9)]


def load(clip: Path, name: str):
    p = clip / f"detections_{name}.jsonl"
    if not p.exists():
        return {}
    return {r["frame"]: r["detections"]
            for r in (json.loads(l) for l in open(p))}


def run(clip: Path, base_gate: float, conf: float):
    labels = [json.loads(l) for l in open(clip / "labels.jsonl")]
    streams = {n: load(clip, n) for n in ("full", "ir", "motion", "point")}
    tracker = CentroidTracker(class_consistent=True,
                              suppress_spawn_near_coasting=True,
                              base_gate_px=base_gate)
    prev = None
    rows = {b: [0, 0, set()] for b in BINS}      # n, held, ids
    order = []
    for rec in labels:
        f = rec["frame"]
        rgb = [Detection(*d["xyxy"], confidence=d["conf"],
                         cls_name=d.get("cls", "drone"))
               for d in streams["full"].get(f, []) if d["conf"] >= 0.05]
        aux = [Detection(*d["xyxy"], confidence=d["conf"], cls_name="hotspot")
               for d in streams["ir"].get(f, []) if d["conf"] >= 0.2]
        aux += [Detection(*d["xyxy"], confidence=d["conf"], cls_name="mover")
                for d in streams["motion"].get(f, []) if d["conf"] >= 0.10]
        aux += [Detection(*d["xyxy"], confidence=d["conf"], cls_name="point")
                for d in streams["point"].get(f, [])]
        dets, _ = fuse_measurements(merge_close(rgb), aux)
        tracks = tracker.update(dets, timestamp=rec["t"])
        if not rec.get("visible"):
            prev = None
            continue
        c = np.array([(rec["bbox"][0] + rec["bbox"][2]) / 2,
                      (rec["bbox"][1] + rec["bbox"][3]) / 2])
        disp = float(np.linalg.norm(c - prev)) if prev is not None else 0.0
        prev = c
        b = next((b for b in BINS if b[0] <= disp < b[1]), None)
        if b is None:
            continue
        rows[b][0] += 1
        on = [t for t in tracks if is_hit(t.box, rec)]
        if on:
            rows[b][1] += 1
            ids = {t.track_id for t in on}
            rows[b][2].update(ids)
            # A handover is when the id we were HOLDING stops being on the
            # target - not merely when a different one sorts first. Taking
            # on[0] counted list-order flips between two co-located tracks
            # and reported 158 handovers where there were 6.
            if order and order[-1] in ids:
                pass
            else:
                order.append(sorted(ids)[0])
    return rows, order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--gates", type=float, nargs="+", default=[30.0])
    args = ap.parse_args()

    clip = Path(args.clip)
    meta = json.loads((clip / "meta.json").read_text())
    print(f"\n{clip.name}: fx={meta.get('fx'):.0f}, "
          f"interval {meta.get('interval_s')} s "
          f"({1/meta.get('interval_s', 1):.0f} Hz)")

    for g in args.gates:
        rows, order = run(clip, g, args.conf)
        n_tot = sum(r[0] for r in rows.values())
        held = sum(r[1] for r in rows.values())
        print(f"\nbase_gate_px = {g:.0f}   overall held "
              f"{held}/{n_tot} = {held / max(n_tot,1):.1%}   "
              f"track handovers on the drone: {max(0, len(order) - 1)}")
        print(f"{'px/frame':>12} {'frames':>7} {'held':>7} {'distinct ids':>13}")
        for b in BINS:
            n, h, ids = rows[b]
            if not n:
                continue
            lab = f"{b[0]}-{b[1]:.0f}" if b[1] < 1e8 else f"{b[0]}+"
            print(f"{lab:>12} {n:>7} {h / n:>6.0%} {len(ids):>13}")


if __name__ == "__main__":
    main()
