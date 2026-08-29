#!/usr/bin/env python3
"""Can a steerable telephoto FIND the drone by scanning, or must it be cued?

The measured position: against sky the 6 deg telephoto detects a drone 100%
of the time out to 1366 m (scripts/point_target_detect.py on range_tele_sky),
but it sees 1% of the wide camera's field at any instant. So the range is
there and the coverage is not - unless the lens is swept.

That turns a sensitivity question into a SCHEDULING one: a scan finds the
target only if the beam happens to be pointed at it while it is in range, so
what matters is revisit time against closing speed. This models that
directly - raster the beam over a search volume, and at each pointing check
whether the target was inside the footprint at that moment, rolling against
the MEASURED detection curve.

Detection probability is interpolated from real measurements, never assumed:

  sky  (data/clips/range_tele_sky) 100% at every size measured, 11.4 -> 3.5 px
  rock (data/clips/range_tele)     88% @ 11.6, 95% @ 8.7, 47% @ 7.2, 0% @ 5.9

Note the honest edge: 3.5 px is the smallest target the sky clip contains, so
Pd below that is UNMEASURED. --floor-px controls what happens there and
defaults to treating it as a miss, which understates the sensor rather than
flattering it.

    python scripts/raster_scan.py --mode approach --background sky
    python scripts/raster_scan.py --mode clip --clip data/clips/range_tele_sky
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np

DRONE_M = 0.34
W_PX = 1280
CAM = np.array([0.1, 0.0, 2.5])

# (target px, Pd) - measured, see module docstring
PD_CURVE = {
    "sky":  [(3.5, 1.00), (5.0, 1.00), (7.2, 1.00), (11.4, 1.00)],
    "rock": [(5.9, 0.00), (7.2, 0.47), (8.7, 0.95), (11.6, 0.88)],
}


def fx_of(fov_deg):
    return (W_PX / 2) / math.tan(math.radians(fov_deg) / 2)


def pd_of(px, background, floor_px):
    """Detection probability at this target size, from measured points."""
    if px < floor_px:
        return 0.0
    pts = PD_CURVE[background]
    if px <= pts[0][0]:
        return pts[0][1]
    if px >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= px <= x1:
            return y0 + (y1 - y0) * (px - x0) / (x1 - x0)
    return 0.0


class Raster:
    """A steerable narrow lens sweeping a search volume, boustrophedon."""

    def __init__(self, hfov, az_span, el_lo, el_hi, overlap,
                 slew_deg_s, settle_s, dwell_s):
        self.hfov, self.vfov = hfov, hfov * 720.0 / 1280.0
        astep = hfov * (1 - overlap)
        estep = self.vfov * (1 - overlap)
        azs = np.arange(-az_span / 2 + astep / 2, az_span / 2, astep)
        els = np.arange(el_lo + estep / 2, el_hi, estep)
        self.points = []
        for i, el in enumerate(els):
            row = azs if i % 2 == 0 else azs[::-1]
            for az in row:
                self.points.append((float(az), float(el)))
        # time to step from one pointing to the next, plus settle and look
        step_deg = math.hypot(astep, 0.0)
        self.per_point = step_deg / slew_deg_s + settle_s + dwell_s
        self.dwell = dwell_s
        self.period = self.per_point * len(self.points)

    def pointing_at(self, t):
        """Where the beam is looking at time t, and whether it is dwelling."""
        k = int((t % self.period) / self.per_point)
        into = (t % self.period) - k * self.per_point
        looking = into >= (self.per_point - self.dwell)
        return self.points[k % len(self.points)], looking

    def sees(self, az, el, pointing):
        paz, pel = pointing
        daz = (az - paz + 180) % 360 - 180
        return abs(daz) <= self.hfov / 2 and abs(el - pel) <= self.vfov / 2


def sweep_from_args(a):
    return Raster(a.hfov, a.az_span, a.el_lo, a.el_hi, a.overlap,
                  a.slew, a.settle, a.dwell)


def run_approach(a, scan, rng):
    """Monte-Carlo: a drone flies straight in from max range, random bearing."""
    fx = fx_of(a.hfov)
    got = []
    for _ in range(a.trials):
        az = rng.uniform(-a.az_span / 2, a.az_span / 2)
        el = rng.uniform(a.el_lo, a.el_hi)
        t = rng.uniform(0, scan.period)       # scan phase is not synchronised
        r = a.start_m
        found = None
        while r > a.stop_m:
            (pt, looking) = scan.pointing_at(t)
            if looking and scan.sees(az, el, pt):
                px = DRONE_M * fx / r
                if rng.random() < pd_of(px, a.background, a.floor_px):
                    found = r
                    break
            t += a.tick
            r -= a.speed * a.tick
        got.append(found)
    return got


def run_clip(a, scan, rng):
    """Replay a captured trajectory through the same scan schedule."""
    recs = [json.loads(l) for l in open(Path(a.clip) / "labels.jsonl")]
    fx = fx_of(a.hfov)
    # A clip sampled slower than the dwell cannot resolve a look at all: the
    # beam opens and closes between two frames and the replay scores a miss
    # that never happened. Measured: the 2 Hz sky clip reported 0.2% detected
    # against 99.8% for the same scan in the approach model - an artefact of
    # the clip's interval, not a property of the scanner.
    interval = json.loads((Path(a.clip) / "meta.json").read_text()).get(
        "interval_s", 0.5)
    if interval > a.dwell:
        print(f"\nWARNING: {Path(a.clip).name} samples every {interval:.3f} s "
              f"but each look lasts {a.dwell:.3f} s.\nMost dwells fall BETWEEN "
              f"frames, so the rates below are a sampling artefact.\nUse "
              f"--mode approach, or a clip captured faster than the dwell.")
    seen, inbeam, total = 0, 0, 0
    first = None
    for rec in recs:
        if not rec.get("visible"):
            continue
        p = np.array(rec["pos"]) - CAM
        rng_m = float(np.linalg.norm(p))
        az = math.degrees(math.atan2(p[1], p[0]))
        el = math.degrees(math.atan2(p[2], math.hypot(p[0], p[1])))
        t = rec["t"]
        (pt, looking) = scan.pointing_at(t)
        total += 1
        if looking and scan.sees(az, el, pt):
            inbeam += 1
            px = DRONE_M * fx / rng_m
            if rng.random() < pd_of(px, a.background, a.floor_px):
                seen += 1
                if first is None:
                    first = (rec["t"], rng_m)
    print(f"\nclip {Path(a.clip).name}: {total} drone-visible frames")
    print(f"  in the scanned beam : {inbeam / max(total,1):.1%} of frames")
    print(f"  detected            : {seen / max(total,1):.1%} of frames")
    if first:
        print(f"  first detection     : t={first[0]:.1f}s at {first[1]:.0f} m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="approach", choices=["approach", "clip"])
    ap.add_argument("--clip", default="data/clips/range_tele_sky")
    ap.add_argument("--background", default="sky", choices=["sky", "rock"])
    ap.add_argument("--hfov", type=float, default=6.0)
    ap.add_argument("--az-span", type=float, default=60.0,
                    help="azimuth sector to search, degrees")
    ap.add_argument("--el-lo", type=float, default=0.0)
    ap.add_argument("--el-hi", type=float, default=24.0)
    ap.add_argument("--overlap", type=float, default=0.10)
    ap.add_argument("--slew", type=float, default=60.0, help="deg/s")
    ap.add_argument("--settle", type=float, default=0.10)
    ap.add_argument("--dwell", type=float, default=0.20,
                    help="seconds of look per pointing; the point detector "
                         "needs a single frame, so this is a few frames")
    ap.add_argument("--speed", type=float, default=14.0, help="closing m/s")
    ap.add_argument("--start-m", type=float, default=1600.0)
    ap.add_argument("--stop-m", type=float, default=50.0)
    ap.add_argument("--tick", type=float, default=0.05)
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--floor-px", type=float, default=3.5,
                    help="target size below which detection is scored ZERO. "
                         "3.5 px is the smallest the sky clip measured, so "
                         "anything smaller is unmeasured, not proven blind")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    scan = sweep_from_args(args)
    rng = random.Random(args.seed)
    print(f"\n{args.hfov:.0f} deg beam over {args.az_span:.0f}x"
          f"{args.el_hi - args.el_lo:.0f} deg: {len(scan.points)} pointings, "
          f"{scan.per_point*1000:.0f} ms each")
    print(f"full sweep (revisit) = {scan.period:.1f} s; at {args.speed:.0f} m/s "
          f"the target closes {scan.period * args.speed:.0f} m between looks")

    if args.mode == "clip":
        run_clip(args, scan, rng)
        return

    got = run_approach(args, scan, rng)
    hits = [g for g in got if g is not None]
    print(f"\napproach Monte-Carlo, {args.trials} runs, {args.background} "
          f"background, from {args.start_m:.0f} m")
    print(f"  detected at all     : {len(hits) / len(got):.1%}")
    if hits:
        h = np.array(sorted(hits)[::-1])
        print(f"  detection range     : median {np.median(h):.0f} m, "
              f"p90 {np.percentile(h, 90):.0f} m, worst {h.min():.0f} m")
        for gate in (1000, 700, 550, 300):
            print(f"  found beyond {gate:>4} m : "
                  f"{(h >= gate).sum() / len(got):.1%}")


if __name__ == "__main__":
    main()
