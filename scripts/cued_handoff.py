#!/usr/bin/env python3
"""The full chain, end to end, on ONE flight: cue -> slew -> track -> range.

Every link has been measured separately. PCL detects 76-87% across
300-1000 m where every camera channel reads 0-2%. The telephoto detects 100%
to 1.4 km against sky. Neither closes the loop alone: PCL has no elevation so
it may never spawn a track, and the telephoto sees 1% of the sky so it does
not know where to look. The chain has never actually run.

It could not, because the station carried one camera, so a flight was either
wide or telephoto, never both. `detection_world_terrain_dual.sdf` puts wide,
telephoto and thermal on one mount, and `capture_dataset.py --zoom` records
the second view, so both are frame-synchronised on the same drone.

What this measures:
  1. PCL reports a bearing and a range on a target nothing is tracking.
  2. The telephoto is COMMANDED to that bearing. It is unavailable for
     --slew-s while it moves and settles: a cue is not a free look, and
     pretending otherwise is the main way a demo like this cheats.
  3. Once settled, the point-target detector runs on the telephoto frame. A
     detection there spawns a real track.
  4. PCL's range attaches to that track.

Reported: how long from first cue to a tracked target carrying a range, and
what fraction of the flight the system holds one. The wide-camera chain runs
alongside so the contribution is attributable rather than assumed.

    python scripts/cued_handoff.py --clip data/clips/range_dual
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--slew-s", type=float, default=0.4,
                    help="slew + settle before a cued look is usable")
    ap.add_argument("--zoom-hfov", type=float, default=0.10472)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--confirm-hits", type=int, default=3,
                    help="hits that make a track count as established, so a "
                         "PCL cue behind it no longer needs a zoom look")
    ap.add_argument("--relook-s", type=float, default=1.0,
                    help="how long a settled look stays valid before the "
                         "zoom is re-commanded")
    args = ap.parse_args()

    clip = Path(args.clip)
    labels = [json.loads(l) for l in open(clip / "labels.jsonl")]
    meta = json.loads((clip / "meta.json").read_text())
    fx = float(meta.get("fx", 1108.77))
    zfx = (meta.get("zoom") or {}).get("fx", 640 / math.tan(0.10472 / 2))
    st = {n: load(clip, n) for n in ("full", "sahi", "ir", "motion",
                                     "pcl", "zoompoint")}
    print(f"\n{clip.name}: wide fx={fx:.0f}, telephoto fx={zfx:.0f}, "
          f"{len(labels)} frames")
    print(f"streams: {', '.join(n for n, v in st.items() if v)}")

    tracker = CentroidTracker(class_consistent=True,
                              suppress_spawn_near_coasting=True)
    zoom_ready_at = None      # when the commanded look becomes usable
    zoom_cmd_bearing = None
    first_cue_t = first_ranged_t = None
    n_ranged = n_tracked = n_vis = 0
    cue_frames = zoom_looks = zoom_hits = 0

    for rec in labels:
        f, t = rec["frame"], rec["t"]
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

        # ---- 3. a settled telephoto look becomes a real measurement ----
        if zoom_ready_at is not None and t >= zoom_ready_at:
            zoom_looks += 1
            for d in st["zoompoint"].get(f, []):
                # telephoto pixels -> wide pixels, about the principal point
                cx = (d["xyxy"][0] + d["xyxy"][2]) / 2
                cy = (d["xyxy"][1] + d["xyxy"][3]) / 2
                w = (d["xyxy"][2] - d["xyxy"][0]) * fx / zfx
                h = (d["xyxy"][3] - d["xyxy"][1]) * fx / zfx
                u = 640 + (cx - 640) * fx / zfx
                v = 360 + (cy - 360) * fx / zfx
                aux.append(Detection(u - w / 2, v - h / 2, u + w / 2,
                                     v + h / 2, confidence=0.9,
                                     cls_name="point"))
                if rec.get("visible") and is_hit(
                        [u - w / 2, v - h / 2, u + w / 2, v + h / 2], rec):
                    zoom_hits += 1
                break

        dets, _ = fuse_measurements(merge_close(rgb), aux)
        live = tracker.update(dets, timestamp=t)

        # ---- 1 + 4. PCL: bearing update, range attached ----
        cues = [((d["xyxy"][0] + d["xyxy"][2]) / 2,
                 max(4.0, fx * math.tan(math.radians(
                     d.get("bearing_sigma_deg", 3.0)))),
                 d.get("range_m", 0.0), d.get("sigma_range_m", 20.0))
                for d in st["pcl"].get(f, [])]
        gate = fx * math.tan(math.radians(7.0))
        if cues:
            cue_frames += 1
            if first_cue_t is None:
                first_cue_t = t
            tracker.fuse_pcl(cues, gate_px=gate)

        # ---- 2. a cue with no CONFIRMED track behind it commands the zoom ----
        # An earlier version triggered only on cues fuse_pcl left UNMATCHED,
        # and the zoom was never commanded once in 2804 cue frames: a cue
        # matches any track within the gate, and clutter tracks are always
        # somewhere near. Matching a clutter track is not the same as knowing
        # what is out there, so the trigger is "no ESTABLISHED track at this
        # bearing", which is the question the zoom actually answers.
        if cues and (zoom_ready_at is None or t >= zoom_ready_at + args.relook_s):
            need = [c for c in cues
                    if not any(abs(c[0] - float(tr.mean[0])) <= gate
                               and tr.hits >= args.confirm_hits
                               for tr in tracker._tracks)]
            if need:
                zoom_cmd_bearing = need[0][0]
                zoom_ready_at = t + args.slew_s

        if rec.get("visible"):
            n_vis += 1
            on = [tr for tr in live if is_hit(tr.box, rec)]
            if on:
                n_tracked += 1
                if any(tr.range_m is not None and tr.range_age <= 3
                       for tr in on):
                    n_ranged += 1
                    if first_ranged_t is None:
                        first_ranged_t = t

    print(f"\nPCL produced a cue on {cue_frames} frames")
    print(f"telephoto took {zoom_looks} settled looks "
          f"(slew+settle {args.slew_s:.2f} s), hit the target on {zoom_hits}")
    print(f"tracked            {n_tracked}/{n_vis} = {n_tracked/max(n_vis,1):.1%}")
    print(f"tracked WITH RANGE {n_ranged}/{n_vis} = {n_ranged/max(n_vis,1):.1%}")
    if first_cue_t is not None and first_ranged_t is not None:
        print(f"first PCL cue at t={first_cue_t:.2f}s -> first ranged track "
              f"at t={first_ranged_t:.2f}s "
              f"({first_ranged_t - first_cue_t:.2f}s to close the chain)")
    elif first_ranged_t is None:
        print("NO ranged track was ever formed - the chain did not close")


if __name__ == "__main__":
    main()
