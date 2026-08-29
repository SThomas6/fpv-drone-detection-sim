#!/usr/bin/env python3
"""Micro-Doppler radar sensor model - Phase-4-lite.

Every fielded counter-UAS stack this campaign researched solves the
bird-false-alarm problem with a radar that reads MICRO-DOPPLER: a
multirotor's spinning blades put sidebands on the return that no bird can
produce, and a bird's wingbeat puts a slow periodic signature there that no
rigid airframe can produce. Cameras cannot see either at range (measured
here: sub-pixel), which is why the optical wingbeat arc concluded marginal.

This models such a radar the same way camera/ir_detector.py models thermal
physics: as a SENSOR MODEL over simulator ground truth, with published
performance figures and honest error modelling - never as an oracle.

Parameters and their sources (docstring numbers = defaults below):
  - Detection envelope: small-drone instrumented range ~1000 m, blind under
    30 m, Pd falling with range (Robin Radar ELVIRA-class public specs;
    airsight.com/blog/can-radar-detect-drones).
  - Micro-Doppler classification, per ~1 s dwell: drone correctly called
    drone 93%, bird correctly called bird 92%, remainder 'unknown' or
    confused - mid-range of published results (Nature s41598-018-35880-9,
    K/W-band signatures; Frontiers frsip.2021.781777; Robin Radar's own
    material claims higher, the academic mid-range is used).
  - Bearing noise ~0.6 deg, range noise ~1.5 m -> pixel jitter after
    projection through the station camera.
  - Clutter false tracks ~0.5/min inside the camera FOV share of the scan
    (modern MTI radar suppresses static clutter; birds ARE its residual
    clutter, which is the whole point of the classification stage).

Output: detections_radar.jsonl in RGB pixel space, cls one of
'radar_drone' / 'radar_bird' / 'radar_unknown', conf = modelled class
confidence. Class decisions are stable within a dwell (re-drawn each dwell),
so a track accumulates radar OPINION over seconds exactly the way the real
system's operator display does.

    python scripts/radar_sim.py run --clip data/clips/terrain_ir_canopy --seed 5
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# -------- envelope --------
MAX_RANGE_M = 1000.0
MIN_RANGE_M = 30.0
PD_NEAR = 0.98            # detection probability inside half range
PD_FAR = 0.80             # at max range (linear in between)
# -------- micro-Doppler classification, per dwell --------
DWELL_S = 1.0
P_DRONE_AS_DRONE = 0.93   # remainder: 0.05 unknown, 0.02 bird
P_DRONE_AS_BIRD = 0.02
P_BIRD_AS_BIRD = 0.92     # remainder: 0.05 unknown, 0.03 drone
P_BIRD_AS_DRONE = 0.03
# -------- noise --------
BEARING_SIGMA_DEG = 0.6
RANGE_SIGMA_M = 1.5
CLUTTER_PER_MIN = 0.5     # false radar tracks appearing in-FOV
CLUTTER_DWELLS = 3        # how many dwells a clutter track lives


def _class_draw(rng: random.Random, truth: str) -> tuple[str, float]:
    r = rng.random()
    if truth == "drone":
        if r < P_DRONE_AS_DRONE:
            return "radar_drone", 0.85 + 0.10 * rng.random()
        if r < P_DRONE_AS_DRONE + P_DRONE_AS_BIRD:
            return "radar_bird", 0.55 + 0.2 * rng.random()
        return "radar_unknown", 0.4
    else:
        if r < P_BIRD_AS_BIRD:
            return "radar_bird", 0.85 + 0.10 * rng.random()
        if r < P_BIRD_AS_BIRD + P_BIRD_AS_DRONE:
            return "radar_drone", 0.55 + 0.2 * rng.random()
        return "radar_unknown", 0.4


def run(clip: Path, seed: int):
    labels = [json.loads(l) for l in open(clip / "labels.jsonl")]
    meta = json.loads((clip / "meta.json").read_text())
    fx = meta.get("fx", 1108.77)
    W = meta.get("width", 1280)
    H = meta.get("height", 720)
    interval = meta.get("interval_s", 0.5)
    rng = random.Random(seed)

    # per-object dwell state: (dwell_index -> (cls, conf))
    dwell_state: dict[str, tuple[int, str, float]] = {}

    def dwell_class(key: str, truth: str, t: float):
        d = int(t / DWELL_S)
        st = dwell_state.get(key)
        if st is None or st[0] != d:
            cls, conf = _class_draw(rng, truth)
            dwell_state[key] = (d, cls, conf)
        return dwell_state[key][1], dwell_state[key][2]

    def pd_at(rng_m: float) -> float:
        if rng_m < MIN_RANGE_M or rng_m > MAX_RANGE_M:
            return 0.0
        half = MAX_RANGE_M / 2
        if rng_m <= half:
            return PD_NEAR
        return PD_NEAR + (PD_FAR - PD_NEAR) * (rng_m - half) / half

    def jitter_px(cx: float, cy: float, rng_m: float, box_w: float):
        # bearing error -> lateral pixels: fx * tan(sigma) at any range;
        # range error barely moves pixels for a distant target
        sx = fx * math.tan(math.radians(BEARING_SIGMA_DEG))
        return (cx + rng.gauss(0.0, sx * 0.5),
                cy + rng.gauss(0.0, sx * 0.25))

    out = []
    clutter: list[dict] = []
    for rec in labels:
        t = rec["t"]
        dets = []

        def emit(key, truth, bbox, rng_m):
            if rng.random() > pd_at(rng_m):
                return
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            w = max(bbox[2] - bbox[0], 3.0)
            cx, cy = jitter_px(cx, cy, rng_m, w)
            if not (0 <= cx < W and 0 <= cy < H):
                return
            cls, conf = dwell_class(key, truth, t)
            dets.append({"xyxy": [round(cx - w / 2, 1), round(cy - w / 2, 1),
                                  round(cx + w / 2, 1), round(cy + w / 2, 1)],
                         "conf": round(conf, 3), "cls": cls})

        if rec.get("visible") and rec.get("bbox"):
            emit("drone", "drone", rec["bbox"],
                 rec.get("range_m") or 150.0)
        for i, b in enumerate(rec.get("birds") or []):
            emit(f"bird{i}", "bird", b["bbox"], b.get("range_m") or 120.0)

        # clutter births
        if rng.random() < CLUTTER_PER_MIN * interval / 60.0:
            clutter.append({"cx": rng.uniform(0, W), "cy": rng.uniform(0, H),
                            "until": t + CLUTTER_DWELLS * DWELL_S,
                            "key": f"cl{t:.2f}"})
        clutter = [c for c in clutter if c["until"] > t]
        for c in clutter:
            cls, conf = dwell_class(c["key"], "bird" if rng.random() < 0.5
                                    else "drone", t)
            # clutter classification is near-uniform garbage: overwrite
            dets.append({"xyxy": [round(c["cx"] - 4, 1), round(c["cy"] - 4, 1),
                                  round(c["cx"] + 4, 1), round(c["cy"] + 4, 1)],
                         "conf": 0.5, "cls": "radar_unknown"
                         if rng.random() < 0.6 else cls})

        out.append({"frame": rec["frame"], "detections": dets})

    dst = clip / "detections_radar.jsonl"
    with open(dst, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    n = sum(len(r["detections"]) for r in out)
    print(f"wrote {dst} ({len(out)} frames, {n} radar detections)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run"])
    ap.add_argument("--clip", required=True)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    run(Path(args.clip), args.seed)


if __name__ == "__main__":
    main()
