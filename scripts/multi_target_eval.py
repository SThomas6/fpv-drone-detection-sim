#!/usr/bin/env python3
"""Several drones at once: does the tracker keep them apart?

The tracker has only ever been shown ONE drone. Everything it does that
could break with more - nearest-neighbour association, the class-consistency
rule, spawn suppression near a coasting track - is untested at N > 1, and the
failure would not look like an error. It would look like slightly lower
recall, which is easy to read as a hard scene.

Three things are reported, and only the last two are new:

  found      per-drone detection: is each airframe detected at all
  held       per-drone tracking: does SOME track sit on each one
  identity   how many distinct track ids each drone was given over the clip,
             and how often two drones were covered by the SAME id at the same
             instant. A swap is invisible to recall - both drones look
             tracked - but it is fatal to a hand-off, because the thing you
             point at is no longer the thing you decided to point at.

    python scripts/multi_target_eval.py --clip data/clips/multi_drone
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera.detector import Detection, merge_close  # noqa: E402
from camera.evaluate import is_hit  # noqa: E402
from camera.fuse_eval import fuse_measurements  # noqa: E402
from camera.tracking import CentroidTracker  # noqa: E402


def load(clip: Path, name: str):
    p = clip / f"detections_{name}.jsonl"
    if not p.exists():
        return {}
    return {r["frame"]: r["detections"]
            for r in (json.loads(l) for l in open(p))}


def targets_in(rec):
    """Every drone in this frame as (name, label-like dict)."""
    out = []
    if rec.get("visible") and rec.get("bbox"):
        out.append(("target_drone", rec))
    for d in rec.get("other_drones") or []:
        out.append((d["name"], {"bbox": d["bbox"], "range_m": d.get("range_m")}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--conf", type=float, default=0.10)
    args = ap.parse_args()

    clip = Path(args.clip)
    labels = [json.loads(l) for l in open(clip / "labels.jsonl")]
    st = {n: load(clip, n) for n in ("full", "sahi", "ir", "motion", "point")}
    tracker = CentroidTracker(class_consistent=True,
                              suppress_spawn_near_coasting=True)

    seen = defaultdict(int)
    held = defaultdict(int)
    total = defaultdict(int)
    ids = defaultdict(set)
    current = {}
    switches = defaultdict(int)
    shared_frames = 0
    n_frames = 0

    for rec in labels:
        f = rec["frame"]
        rgb = [Detection(*d["xyxy"], confidence=d["conf"],
                         cls_name=d.get("cls", "drone"))
               for d in st["full"].get(f, []) if d["conf"] >= 0.05]
        rgb += [Detection(*d["xyxy"], confidence=d["conf"],
                          cls_name=d.get("cls", "drone"))
                for d in st["sahi"].get(f, []) if d["conf"] >= 0.05]
        aux = [Detection(*d["xyxy"], confidence=d["conf"], cls_name="hotspot")
               for d in st["ir"].get(f, []) if d["conf"] >= 0.2]
        aux += [Detection(*d["xyxy"], confidence=d["conf"], cls_name="mover")
                for d in st["motion"].get(f, []) if d["conf"] >= 0.10]
        dets, _ = fuse_measurements(merge_close(rgb), aux)
        tracks = tracker.update(dets, timestamp=rec["t"])

        tg = targets_in(rec)
        if not tg:
            continue
        n_frames += 1
        owner = {}
        for name, lab in tg:
            total[name] += 1
            if any(is_hit(d.xyxy, lab) for d in dets):
                seen[name] += 1
            on = [t for t in tracks if is_hit(t.box, lab)]
            if on:
                held[name] += 1
                tset = {t.track_id for t in on}
                ids[name].update(tset)
                if current.get(name) not in tset:
                    if name in current:
                        switches[name] += 1
                    current[name] = sorted(tset)[0]
                owner[name] = tset
        # one id covering two different drones in the same frame
        names = list(owner)
        if any(owner[a] & owner[b]
               for i, a in enumerate(names) for b in names[i + 1:]):
            shared_frames += 1

    print(f"\n{clip.name}: {n_frames} frames with at least one drone, "
          f"{len(total)} drones")
    print(f"{'drone':>16} {'frames':>7} {'detected':>9} {'tracked':>8} "
          f"{'track ids':>10} {'id switches':>12}")
    for name in sorted(total):
        n = total[name]
        print(f"{name:>16} {n:>7} {seen[name]/n:>8.0%} {held[name]/n:>7.0%} "
              f"{len(ids[name]):>10} {switches[name]:>12}")
    print(f"\nframes where one track id covered TWO drones at once: "
          f"{shared_frames} ({shared_frames/max(n_frames,1):.1%})")
    if shared_frames:
        print("  -> identity is being confused between targets; recall looks "
              "fine and the hand-off would point at the wrong drone")


if __name__ == "__main__":
    main()
